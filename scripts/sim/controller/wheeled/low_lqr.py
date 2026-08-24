"""하위 배분 계층 (balancing 2륜 diff): 몸체 속도 명령 -> 균형 유지 바퀴 토크.

2륜 역진자(세그웨이) 표준 모델을 정본 URDF의 질량·관성에서 직접 유도해
LQR 게인을 로봇당 1회 산출하고, 매 스텝은 게인 행렬곱(배치)만 수행한다.

모델 유도 (몸체 좌표 +x 전진, pitch 양수 = 앞으로 숙임 — core.base 규약):
기호: M = 몸체(바퀴 제외) 질량, l = 바퀴 축-몸체 질량중심 거리,
I_b = 몸체 질량중심 기준 피치(y축) 관성, m_w·I_w = 바퀴 1개 질량·축 관성,
r = 바퀴 반지름, tau = 좌우 바퀴 토크 합 (+ = 전진 굴림), v = 축 전진 속도.

라그랑주 식을 직립 근방(sin th -> th)에서 선형화하면:
  (I_b + M l^2) th'' + M l v' - M g l th = -tau   ... (i) 몸체 회전
  (M + 2 m_w + 2 I_w / r^2) v' + M l th'' = tau/r ... (ii) 병진 (굴림 구속)
(i)의 -tau는 모터 반작용(전진 토크가 몸체를 뒤로 젖힘), (ii)의 tau/r는
접지 추진력. I = I_b + M l^2, h = M l, Mt = 병진 유효 질량으로 두고 풀면
  th'' = ( Mt M g l th - (Mt + h/r) tau ) / D,   D = I Mt - h^2 > 0
  v''자리 v' = ( -h M g l th + (I/r + h) tau ) / D
상태 x = [th - th0, th', v], 입력 u = tau 의 선형계 (A, B)가 나온다.
th0 = 질량중심이 축 연직 위에 오는 평형 피치 (생성기의 축선 배치 잔차 보정).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_continuous_are

from ..core.base import RobotCtrlParams, parse_urdf, zero_pose_frame

_GRAVITY = 9.81


@dataclass
class BalanceModel:
    """역진자 모델 파라미터 (모듈 docstring 기호, base_link 기준·SI 단위)."""

    body_mass: float
    lever: float
    body_inertia_y: float
    wheel_mass: float
    wheel_inertia: float
    wheel_radius: float
    # 평형 피치 오프셋 (rad, 질량중심 x 잔차 보정)
    pitch_eq: float


def derive_balance_model(urdf_path: Path, params: RobotCtrlParams) -> BalanceModel:
    """정본 URDF에서 역진자 파라미터를 합성한다.

    몸체 = 구동 바퀴 링크를 제외한 전 링크 (balancing은 캐스터가 없어
    사실상 base_link 하나). 관성은 링크별 질량중심 기준 텐서를 조인트 0
    자세로 회전·평행축 합성해 몸체 질량중심 기준 피치 관성으로 만든다.
    """
    model = parse_urdf(urdf_path)
    wheel_links = {model.joints[w.joint].child for w in params.wheels}

    # 바퀴 축 위치: 좌우 바퀴 중심의 평균 (base 프레임)
    axle = np.mean([w.pos for w in params.wheels], axis=0)

    # 몸체 질량·질량중심 합성 (base 프레임)
    mass_sum, moment = 0.0, np.zeros(3)
    frames = {}
    for name, link in model.links.items():
        if name in wheel_links or link.mass <= 0.0:
            continue
        R, p = zero_pose_frame(model, name)
        frames[name] = (R, p)
        com_b = p + R @ link.com
        mass_sum += link.mass
        moment += link.mass * com_b
    body_com = moment / mass_sum

    # 몸체 피치 관성: 링크 텐서를 회전(R I R^T) 후 평행축 정리로 몸체
    # 질량중심 y축에 모은다 (y축 관성의 평행축 항 = m (dx^2 + dz^2))
    inertia_y = 0.0
    for name, (R, p) in frames.items():
        link = model.links[name]
        ixx, iyy, izz, ixy, ixz, iyz = link.inertia
        tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        rotated = R @ tensor @ R.T
        com_b = p + R @ link.com
        d = com_b - body_com
        inertia_y += rotated[1, 1] + link.mass * (d[0] ** 2 + d[2] ** 2)

    # 바퀴: 질량 평균·자체 축(y) 관성 평균 (좌우 동일 샘플이지만 일반화)
    wheel_mass = float(np.mean([model.links[n].mass for n in wheel_links]))
    wheel_inertia = float(np.mean([model.links[n].inertia[1] for n in wheel_links]))

    # 평형 피치: 축 -> 질량중심 벡터 (cx, cz)가 연직이 되는 각.
    # 수평 잔차 cx cos(th) + cz sin(th) = 0 -> th0 = -atan2(cx, cz)
    cx, cz = body_com[0] - axle[0], body_com[2] - axle[2]
    return BalanceModel(
        body_mass=mass_sum, lever=math.hypot(cx, cz), body_inertia_y=inertia_y,
        wheel_mass=wheel_mass, wheel_inertia=wheel_inertia,
        wheel_radius=params.wheel_radius, pitch_eq=-math.atan2(cx, cz),
    )


def lqr_gain(model: BalanceModel, lqr_cfg: dict) -> np.ndarray:
    """선형계 (A, B)에 CARE를 풀어 LQR 게인 K (1,3)를 산출한다.

    u = -K [th - th0, th', v - v_cmd]. Q·R은 configs/controller.yaml lqr 섹션.
    """
    M, l, r = model.body_mass, model.lever, model.wheel_radius
    I = model.body_inertia_y + M * l ** 2
    h = M * l
    Mt = M + 2.0 * model.wheel_mass + 2.0 * model.wheel_inertia / r ** 2
    D = I * Mt - h ** 2

    # 모듈 docstring에서 유도한 선형계 (상태 [th, th', v], 입력 tau)
    A = np.array([
        [0.0, 1.0, 0.0],
        [Mt * M * _GRAVITY * l / D, 0.0, 0.0],
        [-h * M * _GRAVITY * l / D, 0.0, 0.0],
    ])
    B = np.array([[0.0], [-(Mt + h / r) / D], [(I / r + h) / D]])
    Q = np.diag(lqr_cfg["q_diag"])
    R = np.array([[lqr_cfg["r_effort"]]])
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


class BalancingLQR:
    """balancing 저수준 제어기: LQR 종방향 토크 + yaw 차동 토크 (배치 상태 보유)."""

    def __init__(self, urdf_path: Path, params: RobotCtrlParams,
                 lqr_cfg: dict, device: str):
        """모델 유도·게인 산출 후 좌/우 바퀴와 토크 한계를 고정한다."""
        self._model = derive_balance_model(urdf_path, params)
        self._gain = torch.tensor(lqr_gain(self._model, lqr_cfg)[0],
                                  dtype=torch.float32, device=device)
        # 좌(+y)/우(-y) 바퀴 조인트와 개별 토크 한계
        left = max(params.wheels, key=lambda w: w.pos[1])
        right = min(params.wheels, key=lambda w: w.pos[1])
        self.left_joint, self.right_joint = left.joint, right.joint
        self._limit_l = params.wheel_effort_limit[left.joint]
        self._limit_r = params.wheel_effort_limit[right.joint]
        # yaw 차동 P 게인: 절대 상수는 질량 1-265kg 분포에서 규모 부정합
        # -> 바퀴 토크 한계 평균에 비례시켜 로봇 크기와 함께 스케일한다
        self._yaw_kp = float(lqr_cfg["yaw_kp_ratio"]) \
            * 0.5 * (self._limit_l + self._limit_r)

    @property
    def model(self) -> BalanceModel:
        """유도된 역진자 모델 (진입점 로그·보고용)."""
        return self._model

    def wheel_torques(self, pitch: torch.Tensor, pitch_rate: torch.Tensor,
                      v: torch.Tensor, yaw_rate: torch.Tensor,
                      v_cmd: torch.Tensor, w_cmd: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """상태·명령 (전부 (N,))에서 (좌, 우) 바퀴 토크를 계산한다.

        종방향: tau = -K x (합계) -> 반씩 배분. yaw: 좌우 차동 토크
        (+w 명령 = 반시계 -> 좌 바퀴 감속·우 바퀴 가속). 각 바퀴는 URDF
        토크 한계로 클램프한다.
        """
        # LQR 상태 편차: 평형 피치·속도 명령 기준
        err = torch.stack([pitch - self._model.pitch_eq, pitch_rate, v - v_cmd],
                          dim=1)
        total = -(err @ self._gain)
        # yaw 차동: P 제어 (균형과 분리 — 좌우 대칭 모델이라 상호 간섭 없음)
        diff = self._yaw_kp * (w_cmd - yaw_rate)
        tau_l = torch.clamp(0.5 * total - diff, -self._limit_l, self._limit_l)
        tau_r = torch.clamp(0.5 * total + diff, -self._limit_r, self._limit_r)
        return tau_l, tau_r

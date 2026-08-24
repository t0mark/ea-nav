"""하위 배분 계층 (일반 wheeled): 몸체 속도 명령 -> 바퀴 각속도·조향각.

전부 상태 없는 순수 함수 (torch 배치, N = env 수). 구동 바퀴의 역기구학이
명령 (vx, vy, w)에 선형이라는 점을 이용해 로봇당 행렬 A (바퀴 수 x 3)를
한 번만 만들고, 매 스텝은 행렬곱 한 번으로 배분한다.

행 유도 (몸체 좌표, 바퀴 접지점 = 바퀴 중심의 평면 투영 (x, y)):
- 접지점의 몸체 기준 평면 속도 u = (vx - w*y, vy + w*x)  [강체 속도장]
- 일반 바퀴: 굴림 전진 방향 t = axis x z_hat (회전축 +y 규약 -> t가 수평).
  롤러가 없으므로 u의 t 성분이 굴림 속도 전부: r*wh = t . u
  -> 행 = [t_x, t_y, t_y*x - t_x*y] / r
- 매커넘 바퀴: 접지 롤러 축의 지면 투영 a = (1, s)/sqrt(2), 자유 미끄럼은
  a에 수직 방향뿐 -> 구속은 a 성분: a . u = a . (r*wh, 0)
  -> r*wh = vx + s*vy + w*(s*x - y), 행 = [1, s, s*x - y] / r
  (s = -sign(x*y) 표준 X 배치 — yaw 모멘트 팔 |x|+|y|, 1단계 생성 규약)
- 옴니휠(방사 배치)은 일반 바퀴 행으로 자동 처리된다: t = (sin phi, -cos phi)
  -> 행 = [sin phi, -cos phi, -ring] / r (배치 반지름 ring이 세 번째 항)
"""
from __future__ import annotations

import numpy as np
import torch

from ..core.base import RobotCtrlParams


def build_wheel_matrix(params: RobotCtrlParams, device: str,
                       skid_yaw_scale: float = 1.0) \
        -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """구동 바퀴 배분 행렬을 만든다 (로봇당 1회).

    반환: (조인트 이름 목록, A (n,3), 조인트 속도 한계 (n,)).
    바퀴 각속도 = cmd @ A^T, cmd = (vx, vy, w). 모듈 docstring의 행 유도 참조.
    skid_yaw_scale: skid 전용 옆미끄럼 보상 — 고정축 다륜은 선회 시 접지
    미끄럼 저항 때문에 명령 yaw보다 실제 yaw가 작으므로, 유효 트랙을
    키운 것처럼 yaw 항(세 번째 열)을 배율해 바퀴 차동을 키운다.
    """
    names, rows, limits = [], [], []
    r = params.wheel_radius
    yaw_scale = skid_yaw_scale if params.base_tag == "skid" else 1.0
    for w in params.wheels:
        x, y = float(w.pos[0]), float(w.pos[1])
        if w.joint in params.mecanum_sign:
            # 매커넘: 롤러 구속 행 (모듈 docstring). 행이 굴림 방향 +x를
            # 전제하므로 회전축 = 몸체 +y 규약을 명시 확인한다 (일반 바퀴는
            # 축에서 t를 계산해 규약 위반에 강건하지만 매커넘은 아님)
            # 부호 포함 검사 — 반평행(-y)은 양의 회전이 후진 굴림이 되어
            # 배분 행 전체 부호가 뒤집히는 치명 케이스라 절댓값으로는 부족
            if float(w.axis[1]) < 0.99:
                raise ValueError(f"{w.joint}: 매커넘 바퀴 축이 몸체 +y가 아님 "
                                 f"(axis={w.axis}) — 배분 행 전제 위반")
            s = params.mecanum_sign[w.joint]
            rows.append([1.0 / r, s / r, (s * x - y) / r])
        else:
            # 일반 바퀴: 회전축의 수평 성분에서 굴림 방향 t = axis x z_hat
            t = np.array([w.axis[1], -w.axis[0]])
            t /= max(np.linalg.norm(t), 1e-9)
            rows.append([t[0] / r, t[1] / r,
                         yaw_scale * (t[1] * x - t[0] * y) / r])
        names.append(w.joint)
        limits.append(params.wheel_vel_limit[w.joint])
    return (names,
            torch.tensor(rows, dtype=torch.float32, device=device),
            torch.tensor(limits, dtype=torch.float32, device=device))


def wheel_speeds(A: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    """몸체 속도 명령 (N,3)을 바퀴 각속도 목표 (N,n)로 배분한다."""
    return cmd @ A.T


def feasible_scale(A: torch.Tensor, limits: torch.Tensor,
                   cmd: torch.Tensor) -> torch.Tensor:
    """바퀴 속도 한계를 넘는 명령을 env별로 비율 축소한다.

    개별 바퀴만 잘라내면 선회 반경이 왜곡되므로 (vx, vy, w)를 통째로
    스케일해 경로 형상을 보존한다. 반환: 조정된 cmd (N,3).
    """
    speeds = torch.abs(cmd @ A.T)
    # env별 최대 위반 비율의 역수 (위반 없으면 1.0 유지)
    worst = torch.amax(speeds / limits.unsqueeze(0), dim=1)
    scale = torch.clamp(1.0 / torch.clamp(worst, min=1e-9), max=1.0)
    return cmd * scale.unsqueeze(1)


def ackermann_steer(delta: torch.Tensor, params: RobotCtrlParams) \
        -> dict[str, torch.Tensor]:
    """중심 조향각 -> 좌우 조향 조인트 각 배분 (애커먼 기하).

    delta = 자전거 모델 중심 조향각 (N,), 반환 = {조인트 이름: (N,) 각도}.
    유도: 곡률 kappa = tan(delta)/wb, ICR은 뒤 축 연장선 위 측방 거리
    1/kappa. 조향축이 중심선에서 d만큼 옆(steer_y)에 있으면 그 바퀴의
    측방 팔은 1/kappa - d -> tan(delta_i) = wb*kappa / (1 - d*kappa)
    (선회 안쪽 바퀴가 더 큰 각을 받는다). 가동 범위로 클램프한다.
    """
    kappa = torch.tan(delta) / params.wheelbase
    out = {}
    for name in params.steer_joints:
        d = params.steer_y[name]
        angle = torch.atan2(params.wheelbase * kappa, 1.0 - d * kappa)
        out[name] = torch.clamp(angle, -params.steer_range, params.steer_range)
    return out

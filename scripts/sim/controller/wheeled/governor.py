from __future__ import annotations

import torch

from ..core.base import ControlObs, RobotCtrlParams
from ..core.govern_cbf import project_interval

_GRAVITY = 9.81

class Governor:
    """전복(rollover) 방지 CBF 세이프티 필터.

    Rollover Prevention for Mobile Robots with CBF (arXiv 2403.08916)의 ZMP 기반 barrier:
        h1 = v*w - (b/l_cg)*g_z - g_y >= 0   (v*w 하한)
        h2 = -v*w - (b/l_cg)*g_z + g_y >= 0  (v*w 상한)
    여기서 g_y, g_z는 body frame에 투영된 "중력 가속도"(m/s^2, 평지에서 g_z~-9.81) —
    반면 이 코드베이스의 ControlObs.gravity_b는 방향만 담은 단위벡터(평지에서 (0,0,-1))다.
    또한 b/l_cg(로봇 반폭/무게중심 높이)는 RobotCtrlParams.tip_accel과 같은 양
    (tip_accel = 9.81 * b/l_cg, core/base.py::extract_ctrl_params 참고)이므로,
    g_z = 9.81*gravity_b_z를 그대로 대입하면 (b/l_cg)*g_z = tip_accel*gravity_b_z로 9.81이
    상쇄되지만, g_y = 9.81*gravity_b_y 항은 tip_accel과 짝지어질 인수가 없어 9.81이 남는다:
        h1 = v*w - tip_accel*gravity_b_z - 9.81*gravity_b_y >= 0
        h2 = -v*w - tip_accel*gravity_b_z + 9.81*gravity_b_y >= 0
    기존 controller.py의 즉석 tip_accel 속도 캡을, 실시간 자세를 반영하는 정식 barrier로
    대체한다. v는 고정하고(속도 조절은 PP가 이미 맡음) w만 두 barrier가 허용하는 구간으로
    투영하는 닫힌 해 — 두 barrier가 v*w에 대해 쌍곡선(비아핀) 제약이라 일반 QP 대신,
    v를 주어진 값으로 고정해 w에 대한 구간 제약으로 접어 푼다.

    w 구간(tip_accel/v)은 v가 작을수록 넓어져 v->0에서 사실상 무제한이 된다 — 그런데
    diff_0000 slope 실측에서 로봇이 거의 정지(v~0.003)한 채로 급격한 회전을 하다 넘어지는
    사례가 확인됨: v*w(원심 성분)는 작아도, 그 w를 내려면 두 바퀴 토크를 크게 반대로 걸어야
    하고 그 반작용 자체가 별도의 전복 모멘트를 만든다 — ZMP-v*w 모델이 포착 못 하는
    물리(바퀴 토크 반작용)라 barrier 확장 대신, v_max(설계 순항 속도) 기준으로 계산한
    절대 상한을 바닥으로 깔아 "정지 상태에서 더 자유로워지는" 구멍을 막는다 — 이러면 저속에서도
    "순항 속도에서 안전했을 만큼"보다 더 세게는 못 돈다. v=0 특이점도 이 상한으로 자연히
    처리돼 근처 우회 로직이 필요 없다.
    """

    def __init__(self, params: RobotCtrlParams, margin: float, v_max: float):

        self._tip_accel = float(params.tip_accel) * margin
        self._w_cap = self._tip_accel / max(float(v_max), 1e-3)

    def filter(self, cmd: torch.Tensor, obs: ControlObs) -> torch.Tensor:
        """cmd: (num_envs, 3) = (vx, vy, w) — wheeled 컨트롤러가 이미 이 3성분 형태로 낸다."""

        if self._tip_accel <= 0.0:
            return cmd

        v, w = cmd[:, 0], cmd[:, 2]
        gravity_b_y, gravity_b_z = obs.gravity_b[:, 1], obs.gravity_b[:, 2]

        v_safe = torch.where(v.abs() > 1e-3, v, torch.ones_like(v) * 1e-3)
        bound_a = (self._tip_accel * gravity_b_z + _GRAVITY * gravity_b_y) / v_safe
        bound_b = (-self._tip_accel * gravity_b_z + _GRAVITY * gravity_b_y) / v_safe
        lo = torch.clamp(torch.minimum(bound_a, bound_b), -self._w_cap, self._w_cap)
        hi = torch.clamp(torch.maximum(bound_a, bound_b), -self._w_cap, self._w_cap)

        w_safe = project_interval(w, lo, hi)
        return torch.stack([cmd[:, 0], cmd[:, 1], w_safe], dim=1)

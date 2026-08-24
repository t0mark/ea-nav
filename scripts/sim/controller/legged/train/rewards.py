"""보행 RL 보상 항 (상태 없는 모듈 함수 — 전부 (N,) 반환).

Isaac Lab velocity 환경(velocity_env_cfg)의 표준 보상 구성을 이식하되,
로봇 간 규모(질량 1-265kg, 토크 수-수천 Nm)가 극단적으로 달라지는 다중
임바디먼트 학습에 맞게 토크 페널티는 토크 한계 정규화 비율로 바꾼다
(원본의 절대 토크 L2는 대형 로봇만 벌점이 커져 형태 간 학습 불균형).

가중치·시그마는 configs/rl.yaml rewards 섹션 (호출측 train_env가 적용).
입력 규약: vel_b/ang_b = 몸체 좌표 (m/s, rad/s), cmd = (vx, vy, wz),
슬롯 텐서 (N,S)는 마스크 적용 후 값 (빈 슬롯 0).
"""
from __future__ import annotations

import torch


def track_lin_vel_exp(vel_b: torch.Tensor, cmd: torch.Tensor,
                      sigma: float) -> torch.Tensor:
    """평면 선속도 명령 추종: exp(-|오차|^2 / sigma) (1 = 완전 추종)."""
    err = torch.sum((cmd[:, :2] - vel_b[:, :2]) ** 2, dim=1)
    return torch.exp(-err / sigma)


def track_ang_vel_exp(ang_b: torch.Tensor, cmd: torch.Tensor,
                      sigma: float) -> torch.Tensor:
    """yaw 각속도 명령 추종: exp(-오차^2 / sigma)."""
    err = (cmd[:, 2] - ang_b[:, 2]) ** 2
    return torch.exp(-err / sigma)


def lin_vel_z_l2(vel_b: torch.Tensor) -> torch.Tensor:
    """수직 속도 페널티 항 (튀는 보행 억제) — 가중치는 음수로 적용."""
    return vel_b[:, 2] ** 2


def ang_vel_xy_l2(ang_b: torch.Tensor) -> torch.Tensor:
    """roll/pitch 각속도 페널티 항 (몸통 요동 억제)."""
    return torch.sum(ang_b[:, :2] ** 2, dim=1)


def flat_orientation_l2(gravity_b: torch.Tensor) -> torch.Tensor:
    """몸통 기울기 페널티 항: 중력 방향의 수평 성분 크기 (직립 = 0)."""
    return torch.sum(gravity_b[:, :2] ** 2, dim=1)


def torque_ratio_l2(tau: torch.Tensor, effort: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """토크 사용률 페널티 항: sum((tau / 토크 한계)^2) (모듈 docstring —
    절대 토크 대신 한계 정규화로 로봇 규모 불변)."""
    ratio = tau / torch.clamp(effort, min=1e-6) * mask
    return torch.sum(ratio ** 2, dim=1)


def dof_acc_l2(qd: torch.Tensor, qd_prev: torch.Tensor,
               ctrl_dt: float) -> torch.Tensor:
    """관절 가속 페널티 항: |(qd - qd_prev) / dt|^2 (거친 동작 억제)."""
    return torch.sum(((qd - qd_prev) / ctrl_dt) ** 2, dim=1)


def action_rate_l2(action: torch.Tensor,
                   prev_action: torch.Tensor) -> torch.Tensor:
    """액션 변화율 페널티 항 (액추에이터 채터링 억제)."""
    return torch.sum((action - prev_action) ** 2, dim=1)


def feet_air_time(last_air_time: torch.Tensor, first_contact: torch.Tensor,
                  cmd: torch.Tensor, threshold: float,
                  deadband: float) -> torch.Tensor:
    """발 체공 시간 보상 (legged_gym 표준, 4족·6족용): 접지 순간 (체공-문턱) 합산.

    질질 끄는 보행(체공 짧음)에 벌점, 성큼 걷기(문턱 이상)에 보상.
    last_air_time/first_contact = 발별 (N,F) (ContactSensor 집계), 정지
    명령(수평 명령 크기 < deadband)에서는 0 — 제자리 기립에 걸음 강요 방지.
    2족에는 쓰지 않는다 — 체공이 길수록 보상이라 두 발 동시 점프가 치트가
    된다 (실측 실패 사례). 2족은 feet_air_time_biped 사용.
    """
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward * (torch.norm(cmd[:, :2], dim=1) > deadband)


def feet_air_time_biped(air_time: torch.Tensor, contact_time: torch.Tensor,
                        cmd: torch.Tensor, threshold: float,
                        deadband: float) -> torch.Tensor:
    """2족 보행 보상 (Isaac H1/G1 표준 positive-biped 변형): 한 발 지지 유도.

    정확히 한 발만 접지(single stance)일 때만 현재 상태(체공 또는 접지)
    지속 시간을 보상하고 문턱에서 클램프한다 — 두 발 동시 체공(점프)과
    두 발 동시 접지(끌기)는 보상 0이라 교대 걷기만 이득이 된다.
    air_time/contact_time = 발별 (N,2) 현재 지속 시간.
    """
    in_contact = contact_time > 0.0
    in_mode = torch.where(in_contact, contact_time, air_time)
    single_stance = in_contact.int().sum(dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode,
                                   torch.zeros_like(in_mode)), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    return reward * (torch.norm(cmd[:, :2], dim=1) > deadband)


def undesired_contacts(forces: torch.Tensor, threshold: float) -> torch.Tensor:
    """다리 중간 링크 접촉 페널티 항: 접촉 중인 링크 수 (무릎 보행 억제).

    forces = 대상 링크별 순 접촉력 (N,B,3). 발이 아닌 링크로 땅을 짚는
    보행(무릎·정강이 보행 — 실측 실패 사례)을 벌점으로 차단한다.
    """
    return (torch.norm(forces, dim=-1) > threshold).float().sum(dim=1)


def feet_slide(feet_vel_xy: torch.Tensor,
               in_contact: torch.Tensor) -> torch.Tensor:
    """발 미끄럼 페널티 항 (Isaac H1 표준): 접지 중 발의 수평 속도 합산.

    feet_vel_xy = 발 링크 수평 속도 (N,F,2), in_contact = 접지 여부 (N,F).
    접지한 발이 미끄러지며 이동하는 스케이팅 보행을 억제한다.
    """
    return (torch.norm(feet_vel_xy, dim=-1) * in_contact.float()).sum(dim=1)


def joint_deviation_l1(q_err: torch.Tensor,
                       dev_mask: torch.Tensor) -> torch.Tensor:
    """지정 슬롯의 기립 자세 이탈 L1 (Isaac H1/G1 joint_deviation 계열).

    q_err = 슬롯 관절각 - 기립 자세 (N,S), dev_mask = 대상 슬롯 (S,) 1/0
    (humanoid 고관절 yaw/roll — 다리를 측방으로 벌리는 탐색을 억제해
    single-stance 걸음에 도달하기 전의 전도를 줄인다).
    """
    return torch.sum(torch.abs(q_err) * dev_mask, dim=1)


def dof_pos_limits(q: torch.Tensor, soft_lower: torch.Tensor,
                   soft_upper: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """관절 soft 한계 침범량 합산 (legged_gym/Isaac 표준 dof_pos_limits).

    q = 슬롯 관절각 절대값 (N,S), soft_* = 가동 범위를 중심 기준으로
    soft_ratio만큼 좁힌 한계 (S,). 한계 근방 상시 체류(하드 스톱 보행)를
    억제한다.
    """
    low = torch.clamp(soft_lower - q, min=0.0)
    high = torch.clamp(q - soft_upper, min=0.0)
    return torch.sum((low + high) * mask, dim=1)

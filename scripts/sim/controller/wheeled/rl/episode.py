from __future__ import annotations

import math

import torch

def group_termination(base_sensor, gravity_z: torch.Tensor, contact_thresh: float,
                      tilt_cos: float) -> torch.Tensor:
    """base 접촉력 초과 또는 전복 여부로 로봇 그룹의 물리 종료를 판정한다."""

    hist = base_sensor.data.net_forces_w_history[:, :, 0]
    base_force = torch.norm(hist, dim=-1).max(dim=1).values
    tilted = gravity_z > -tilt_cos
    return (base_force > contact_thresh) | tilted

def stuck_mask(dist: torch.Tensor, vel_xy: torch.Tensor, active: torch.Tensor,
              stuck_steps: torch.Tensor, goal_scale: float, reach_radius: float,
              speed_thresh: float, tick_limit: int) -> torch.Tensor:
    """목표에서 멀리 떨어진 채 오래 멈춰 있는 env를 stuck으로 판정하고 카운터를 갱신한다."""

    far = dist > goal_scale * reach_radius
    slow = torch.norm(vel_xy, dim=1) < speed_thresh
    stuck_now = far & slow & active
    stuck_steps.copy_(torch.where(stuck_now, stuck_steps + 1, torch.zeros_like(stuck_steps)))
    return stuck_steps >= tick_limit

def effective_goal_radius(goal_radius: tuple[float, float],
                          terrain_size: tuple[float, ...] | None,
                          reach_radius: float, goal_margin: float) -> tuple[float, float]:
    """목표가 지형 칸 안에 들어오도록 goal_radius 상한을 보정한다."""

    lo, hi = float(goal_radius[0]), float(goal_radius[1])
    if terrain_size is not None:
        cell = min(float(v) for v in terrain_size)
        cap = max(reach_radius * 2.0, 0.5 * cell - goal_margin)
        hi = min(hi, cap)
        lo = min(lo, 0.8 * hi)
    if hi <= 0.0 or lo <= 0.0 or lo > hi:
        raise ValueError(f"잘못된 wheeled goal_radius: [{lo}, {hi}]")
    return lo, hi

def sample_goals(origins_xy: torch.Tensor, goal_radius: tuple[float, float],
                 rng: torch.Generator, device: str) -> torch.Tensor:
    """원점 주변 반지름 범위 안에서 env별 목표 위치를 무작위로 뽑는다."""

    lo, hi = goal_radius
    u = torch.rand(origins_xy.shape[0], 2, generator=rng, device=device)
    radius = lo + u[:, 0] * (hi - lo)
    theta = u[:, 1] * 2.0 * math.pi
    offset = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)
    return origins_xy + offset

def curriculum_move(walked: torch.Tensor, reached: torch.Tensor, failed: torch.Tensor,
                    min_reaches: int, walked_up_ratio: float, walked_down_ratio: float,
                    goal_radius_hi: float) -> tuple[torch.Tensor, torch.Tensor]:
    """이동 거리·목표 도달 횟수·실패 여부로 다음 지형 난이도(상/하) 이동을 정한다."""

    move_up = ~failed & ((reached >= min_reaches)
                         | (walked > walked_up_ratio * goal_radius_hi))
    move_down = failed | ((reached == 0) & (walked < walked_down_ratio * goal_radius_hi))
    return move_up, move_down

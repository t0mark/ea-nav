from __future__ import annotations

import torch

def progress_reward(prev_pos_xy: torch.Tensor, pos_xy: torch.Tensor,
                    goal_xy: torch.Tensor) -> torch.Tensor:
    """목표까지의 거리 감소량을 reward로 계산한다."""

    before = torch.norm(goal_xy - prev_pos_xy, dim=1)
    after = torch.norm(goal_xy - pos_xy, dim=1)
    return before - after

def upright_reward(gravity_b: torch.Tensor) -> torch.Tensor:
    """body z축이 중력 반대 방향을 유지하는 정도를 reward로 계산한다."""

    return torch.clamp(-gravity_b[:, 2], min=0.0, max=1.0)

def fall_penalty(terminated: torch.Tensor) -> torch.Tensor:
    """전복 등 물리 종료 상태를 penalty tensor로 변환한다."""

    return terminated.float()

def parameter_regularization(scales: torch.Tensor, active: torch.Tensor,
                             param_min: torch.Tensor,
                             param_max: torch.Tensor) -> torch.Tensor:
    """controller 기본 설정(배율 1.0)에서 벗어난 정도를 penalty로 계산한다.

    편차는 각 parameter가 갈 수 있는 최대 편차로 나눠 무차원화한다 — 범위가 넓은
    parameter가 penalty를 독식하지 않게 한다. schema 밖 차원은 배율이 정확히 1.0이라
    편차 0을 낸다.
    """

    span = torch.clamp(torch.maximum(param_max - 1.0, 1.0 - param_min), min=1e-6)
    deviation = (scales - 1.0) / span
    return torch.sum(deviation ** 2 * active.float(), dim=1)

def step_reward(prev_pos_xy: torch.Tensor, pos_xy: torch.Tensor, goal_xy: torch.Tensor,
                vel_xy: torch.Tensor, gravity_b: torch.Tensor, base_hit: torch.Tensor,
                terminated: torch.Tensor, reached: torch.Tensor, ctrl_dt: float,
                weights: dict, stall_dist_thresh: float,
                stall_speed_thresh: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """이번 control tick의 progress/goal/upright/fall/stall reward를 계산한다."""

    progress = progress_reward(prev_pos_xy, pos_xy, goal_xy)
    far_from_goal = torch.norm(goal_xy - pos_xy, dim=1) > stall_dist_thresh
    stalled = far_from_goal & (torch.norm(vel_xy, dim=1) < stall_speed_thresh)

    terms = {
        "progress": float(weights["progress"]) * progress,
        "goal": float(weights["goal"]) * reached.float(),
        "upright": float(weights["upright"]) * ctrl_dt * upright_reward(gravity_b),
        "fall": float(weights["fall"]) * fall_penalty(terminated),
        "base_contact": float(weights["base_contact"]) * base_hit.float(),
        "stall": float(weights["stall"]) * ctrl_dt * stalled.float(),
    }
    return sum(terms.values()), terms

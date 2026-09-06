"""속도 추종 보상 - 목표 이동 속도(commands.base_velocity)를 얼마나 잘 따라가는지에 대한 보상.

전 로봇 공용이다 - 4족이든 2족이든 바퀴-다리 혼합이든 "명령 속도를 따라가야 한다"는 요구 자체는
로봇 형태와 무관하게 동일하다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import RewardTermCfg as RewTerm

if TYPE_CHECKING:
    from ..robot_profile import RobotProfile


def track_lin_vel_xy_exp(weight: float, profile: RobotProfile, std: float = 0.5, **_unused) -> RewTerm:
    """수평(xy) 선속도 추종 - 오차가 클수록 exp(-오차^2/std^2)로 급격히 줄어드는 보상."""
    return RewTerm(
        func=core_mdp.track_lin_vel_xy_exp, weight=weight, params={"command_name": "base_velocity", "std": std}
    )


def track_ang_vel_z_exp(weight: float, profile: RobotProfile, std: float = 0.5, **_unused) -> RewTerm:
    """요(yaw) 각속도 추종 - track_lin_vel_xy_exp와 같은 형태의 지수 보상."""
    return RewTerm(
        func=core_mdp.track_ang_vel_z_exp, weight=weight, params={"command_name": "base_velocity", "std": std}
    )

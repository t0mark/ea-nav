"""몸통 기울기 초과 종료 - Isaac Lab 공식 digit 세팅이 쓰는 bad_orientation 종료조건.

base_contact(접촉 기반)만으로는 "넘어지는 중이지만 아직 바닥에 안 닿은" 구간을 못 잡는다 - 기울기
자체를 기준으로 삼으면 더 빨리(물리적으로 덜 격렬한 상태에서) 에피소드를 끝낼 수 있다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm

if TYPE_CHECKING:
    from ..robot_profile import RobotProfile


def bad_orientation(profile: RobotProfile, limit_angle: float = 0.7, **_unused) -> DoneTerm:
    """투영 중력 벡터 기준 몸통이 limit_angle(rad) 이상 기울면 종료."""
    return DoneTerm(func=core_mdp.bad_orientation, params={"limit_angle": limit_angle})

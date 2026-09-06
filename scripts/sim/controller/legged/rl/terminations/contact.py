"""몸통(base) 접촉 종료 - 낙상을 조기에 감지해 에피소드를 끝내고 GPU 시간을 더 낭비하지 않는다."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

if TYPE_CHECKING:
    from ..robot_profile import RobotProfile


def base_contact(profile: RobotProfile) -> DoneTerm:
    """profile.base_body_name이 threshold 이상의 힘으로 지면(또는 다른 물체)에 닿으면 종료."""
    return DoneTerm(
        func=core_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.base_body_name), "threshold": 1.0},
    )

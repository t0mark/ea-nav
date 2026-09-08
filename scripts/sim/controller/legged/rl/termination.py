"""종료조건 세트 - preset의 `terminations:` 키로 고른다.

time_out(시간 초과)은 어느 세트든 항상 포함되는 기반 조건이라 여기서 다루지 않고 loco_rl_env.py가 직접 추가한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .robot_profile import RobotProfile


@configclass
class TerminationsCfg:
    """빈 컨테이너 - build()가 base_contact를 채우고, loco_rl_env.py가 time_out을 추가한다."""


def _build_base_contact(profile: RobotProfile) -> TerminationsCfg:
    """profile.base_body_name이 threshold 이상의 힘으로 지면(또는 다른 물체)에 닿으면 종료되는 조건을 만든다."""
    cfg = TerminationsCfg()
    cfg.base_contact = DoneTerm(
        func=core_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.base_body_name), "threshold": 1.0},
    )
    return cfg


TERMINATION_SETS = {
    "base_contact": _build_base_contact,
}


def build(profile: RobotProfile, set_name: str) -> TerminationsCfg:
    """preset.terminations 이름으로 종료조건 세트를 만든다(time_out은 loco_rl_env.py가 따로 추가)."""
    return TERMINATION_SETS[set_name](profile)

"""액션 세트 - preset의 `action:` 키로 고른다.

clip=(-100.0, 100.0)은 fan-ziqi/robot_lab·DeepRoboticsLab/rl_training 원본이 거는 안전장치를 그대로
반영한 것이다 - 평소 액션 값에는 전혀 걸리지 않는 느슨한 값이지만, 정책이 학습 중 순간적으로
극단값을 내는 경우 관절 목표가 무한대로 튀는 것을 막아준다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp as core_mdp
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .robot_profile import RobotProfile

_ACTION_CLIP = {".*": (-100.0, 100.0)}


@configclass
class ActionsCfg:
    """빈 컨테이너 - 빌더가 joint_pos 속성 하나만 채운다."""


def _build_joint_position(action_scale: float, joint_names: str) -> ActionsCfg:
    """action_scale(정책 출력에 곱해 기본 자세 오프셋으로 더할 비율)로 joint_names 관절군의 위치제어 액션을 만든다."""
    cfg = ActionsCfg()
    cfg.joint_pos = core_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[joint_names],
        scale=action_scale,
        use_default_offset=True,
        clip=_ACTION_CLIP,
    )
    return cfg


ACTION_BUILDERS = {
    "joint_position": _build_joint_position,
}


def build(profile: RobotProfile, set_name: str) -> ActionsCfg:
    """preset.action 이름으로 액션 세트를 만든다."""
    return ACTION_BUILDERS[set_name](profile.action_scale, profile.controlled_joint_names)

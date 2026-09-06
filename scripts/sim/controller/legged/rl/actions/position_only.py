"""관절 부분집합(또는 전체)을 위치제어로 구동한다.

joint_names(선택, 기본 ".*")로 액션에 포함할 관절을 고르고(다리만 액션에 넣고 팔은 액션 밖에 두는
Unitree 공식 h1/g1/h1_2 방식을 재현), scale은 스칼라 1개이거나 관절 패턴별 값이 다른 딕셔너리일 수
있다({".*_hipx_joint": 0.125, "^(?!.*_hipx_joint).*": 0.25} 형태). 이 딕셔너리 지원은 우리가
지어낸 게 아니라 Isaac Lab의 `JointPositionActionCfg.scale`이 원래 두 형태를 그대로 받는 기능이다
(DeepRobotics M20 공식 rough_env_cfg.py에서 실제로 이렇게 쓰는 것을 클론해 확인).
"""

from __future__ import annotations

from isaaclab.envs import mdp as core_mdp
from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    """빈 컨테이너 - build()가 joint_pos 속성 하나만 채운다."""


def build(action_cfg: dict) -> ActionsCfg:
    """joint_names(문자열 또는 리스트)와 scale(스칼라 또는 딕셔너리)로 위치제어 액션을 만든다."""
    joint_names = action_cfg.get("joint_names", ".*")
    if isinstance(joint_names, str):
        joint_names = [joint_names]
    cfg = ActionsCfg()
    cfg.joint_pos = core_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=joint_names, scale=action_cfg["scale"], use_default_offset=True
    )
    return cfg

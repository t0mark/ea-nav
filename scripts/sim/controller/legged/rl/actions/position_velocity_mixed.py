"""다리는 위치제어, 바퀴는 속도제어로 나누는 액션 구성.

go2w/m20/tron2a_wf처럼 다리-바퀴 혼합 구조인 로봇 전용이다 - 위치 목표로는 바퀴가 연속 회전할 수 없으므로
(목표 각도에 도달하면 정지) 바퀴 관절만 반드시 속도제어로 분리해야 실제로 굴러갈 수 있다. 다족(go2w/m20)과
2족(tron2a_wf) 양쪽에서 로봇 타입과 무관하게 동일하게 쓰인다.

position_joints/velocity_joints 각각의 pattern은 문자열(하나의 정규식) 또는 리스트(정확한 관절
이름들, DeepRobotics M20 공식 rough_env_cfg.py가 실제로 쓰는 방식)를 받고, scale은 스칼라 또는
관절 패턴별 값이 다른 딕셔너리(M20 공식의 "HipX만 0.125, 나머지 0.25")를 받는다.
"""

from __future__ import annotations

from isaaclab.envs import mdp as core_mdp
from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    """빈 컨테이너 - build()가 joint_pos(다리)/joint_vel(바퀴) 두 속성을 채운다."""


def _as_joint_names(pattern: str | list[str]) -> list[str]:
    """문자열 하나로 온 정규식은 리스트로 감싸고, 이미 리스트면 그대로 둔다."""
    return [pattern] if isinstance(pattern, str) else pattern


def build(action_cfg: dict) -> ActionsCfg:
    """position_joints(다리)/velocity_joints(바퀴) 각각의 관절 선택·스케일로 액션 그룹을 나눠 만든다."""
    cfg = ActionsCfg()
    position = action_cfg["position_joints"]
    velocity = action_cfg["velocity_joints"]
    cfg.joint_pos = core_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_as_joint_names(position["pattern"]),
        scale=position["scale"],
        use_default_offset=True,
    )
    cfg.joint_vel = core_mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=_as_joint_names(velocity["pattern"]), scale=velocity["scale"]
    )
    return cfg

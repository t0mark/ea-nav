"""자세 관련 보상 - 몸통 기울기·목표 높이·관절 리밋·기본 자세 이탈·정지 유지·생존/조기 종료 보상.

원래 quadruped/humanoid 형태 그룹에 종속된 개념이 아니다 - h1/g1(2족)과 lite3/m20(4족)이 다리 개수와
무관하게 base_height_l2·joint_deviation_l1·stand_still을 똑같이 쓴다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ..robot_profile import RobotProfile


def flat_orientation_l2(weight: float, profile: RobotProfile, **_unused) -> RewTerm:
    """몸통이 수평에서 벗어난 정도(투영 중력 벡터의 xy 성분 크기) 페널티."""
    return RewTerm(func=core_mdp.flat_orientation_l2, weight=weight)


def dof_pos_limits(weight: float, profile: RobotProfile, joint_names: str = ".*", **_unused) -> RewTerm:
    """관절이 소프트 리밋에 가까워질수록 커지는 페널티 - joint_names로 특정 관절군만 적용할 수 있다
    (예: h1/g1은 발목 관절에만 적용)."""
    return RewTerm(
        func=core_mdp.joint_pos_limits, weight=weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)}
    )


def base_height_l2(weight: float, profile: RobotProfile, target_height: float = 0.5, **_unused) -> RewTerm:
    """몸통 높이가 target_height에서 벗어난 정도의 제곱합 페널티 - 주저앉거나 과도하게 서는 것을 억제."""
    return RewTerm(
        func=core_mdp.base_height_l2,
        weight=weight,
        params={"target_height": target_height, "asset_cfg": SceneEntityCfg("robot", body_names=profile.base_body_name)},
    )


def joint_deviation_l1(weight: float, profile: RobotProfile, joint_names: list[str], **_unused) -> RewTerm:
    """지정한 관절이 기본 자세에서 벗어난 정도의 L1 페널티 - 보행에 직접 필요 없는 관절(팔·허리 등)이
    제멋대로 움직이지 않도록 잡아둔다."""
    return RewTerm(
        func=core_mdp.joint_deviation_l1, weight=weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)}
    )


def stand_still(weight: float, profile: RobotProfile, joint_names: str = ".*", **_unused) -> RewTerm:
    """정지 명령(속도 명령이 거의 0)일 때 관절이 기본 자세에서 벗어나면 주는 페널티 - 제자리에서 불필요하게
    움직이는 것을 억제."""
    return RewTerm(
        func=core_mdp.stand_still_joint_deviation_l1,
        weight=weight,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)},
    )


def termination_penalty(weight: float, profile: RobotProfile, **_unused) -> RewTerm:
    """에피소드가 조기 종료(낙상 등)되면 주는 일회성 페널티 - 낙상 리스크가 큰 2족보행에 주로 쓴다."""
    return RewTerm(func=core_mdp.is_terminated, weight=weight)


def alive(weight: float, profile: RobotProfile, **_unused) -> RewTerm:
    """매 스텝 살아있으면(조기 종료되지 않았으면) 주는 보상 - Unitree 공식 g1/h1/h1_2의 "alive" 항목.
    조기 종료가 곧 손해라는 걸 매 스텝 누적으로 체감하게 해 낙상을 회피하도록 유도한다."""
    return RewTerm(func=core_mdp.is_alive, weight=weight)

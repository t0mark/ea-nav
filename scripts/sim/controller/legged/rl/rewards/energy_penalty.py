"""에너지·안정성 페널티 - 과도한 토크·속도·가속도·액션 변화·수직/각속도 움직임을 억제한다. 전 로봇 공용.

dof_torques_l2/dof_acc_l2는 joint_names로 특정 관절군만 걸러 적용할 수 있다 - Unitree 공식(g1)이
"다리·무릎만 토크·가속도 페널티 대상으로 삼는다"처럼 관절군을 가리는 경우를 재현한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ..robot_profile import RobotProfile


def lin_vel_z_l2(weight: float, profile: RobotProfile, **_unused) -> RewTerm:
    """수직(z) 속도 페널티 - 위아래로 튀는 움직임을 억제(이족보행의 자연스러운 상하 진동과는 상충할 수 있어
    이족보행 로봇은 보통 이 항목을 빼거나 weight를 0으로 둔다)."""
    return RewTerm(func=core_mdp.lin_vel_z_l2, weight=weight)


def ang_vel_xy_l2(weight: float, profile: RobotProfile, **_unused) -> RewTerm:
    """롤·피치 각속도 페널티 - 몸통이 앞뒤·좌우로 흔들리는 것을 억제."""
    return RewTerm(func=core_mdp.ang_vel_xy_l2, weight=weight)


def dof_torques_l2(weight: float, profile: RobotProfile, joint_names: str = ".*", **_unused) -> RewTerm:
    """관절 토크 제곱합 페널티 - joint_names로 특정 관절군만 적용할 수 있다."""
    return RewTerm(
        func=core_mdp.joint_torques_l2, weight=weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)}
    )


def dof_acc_l2(weight: float, profile: RobotProfile, joint_names: str = ".*", **_unused) -> RewTerm:
    """관절 가속도 제곱합 페널티 - joint_names로 특정 관절군만 적용할 수 있다."""
    return RewTerm(
        func=core_mdp.joint_acc_l2, weight=weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)}
    )


def dof_vel_l2(weight: float, profile: RobotProfile, joint_names: str = ".*", **_unused) -> RewTerm:
    """관절 속도 제곱합 페널티 - Unitree 공식 h1/g1/h1_2가 쓰는 항목(dof_vel), joint_names로 필터 가능."""
    return RewTerm(
        func=core_mdp.joint_vel_l2, weight=weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)}
    )


def action_rate_l2(weight: float, profile: RobotProfile, **_unused) -> RewTerm:
    """연속된 두 액션 사이의 변화량 페널티 - 관절이 떨리는(jitter) 것을 억제."""
    return RewTerm(func=core_mdp.action_rate_l2, weight=weight)

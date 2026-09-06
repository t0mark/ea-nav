"""3점 이상 동시 지지가 가능한 보행(4족 등)의 발 접촉 보상.

feet_air_time_multi는 Isaac Lab 코어 패키지(isaaclab.envs.mdp)가 아니라 예제 패키지(isaaclab_tasks)에만
있는 함수라, 예제 코드를 런타임 의존성으로 끌어오는 대신 이 프로젝트 안에 그대로 옮겨와 관리한다
(isaaclab_tasks의 locomotion/velocity/mdp/rewards.py 원본 로직 그대로).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..robot_profile import RobotProfile


def _feet_air_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """발이 임계 시간 이상 공중에 머문 스텝만큼 보상 - 발을 끌지 않고 제대로 들어 걷도록 유도.

    Ref: Isaac Lab legged_gym 계열 표준 보상 - 정지 명령(속도 명령이 거의 0)일 때는 보상을 0으로 죽여,
    가만히 서 있어야 하는 상황에서 불필요하게 발을 드는 행동을 학습하지 않게 한다.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_multi(weight: float, profile: RobotProfile, threshold: float = 0.5, **_unused) -> RewTerm:
    """발마다 독립적으로 공중 체류 시간을 보상 - 3점 이상 동시 지지가 가능한 보행(4족 등)에 사용."""
    return RewTerm(
        func=_feet_air_time,
        weight=weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names),
            "command_name": "base_velocity",
            "threshold": threshold,
        },
    )


def undesired_contacts(weight: float, profile: RobotProfile, threshold: float = 1.0, **_unused) -> RewTerm:
    """발이 아닌 지정 부위(허벅지·정강이 등, profile.undesired_contact_body_names)가 지면에 닿으면 주는 페널티."""
    return RewTerm(
        func=core_mdp.undesired_contacts,
        weight=weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.undesired_contact_body_names),
            "threshold": threshold,
        },
    )

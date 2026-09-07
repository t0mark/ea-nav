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


def _feet_air_time_lin_xy(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float, cmd_threshold: float
) -> torch.Tensor:
    """전후·좌우 이동 명령이 있을 때만 공중 체류 시간을 보상 - DeepRoboticsLab/rl_training의
    feet_air_time_lin_xy_cmd 원본 그대로(지형 난이도 커리큘럼 배율은 우리 쪽에 없어 뺐다)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    cmd_lin_xy = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return reward * (cmd_lin_xy > cmd_threshold)


def feet_air_time_lin_xy(
    weight: float, profile: RobotProfile, threshold: float = 0.5, cmd_threshold: float = 0.1, **_unused
) -> RewTerm:
    """직진·횡이동 명령이 있을 때만 걸리는 공중 체류 보상 - feet_air_time_multi와 달리 정지 명령이 아니라
    "전후좌우 이동" 명령으로만 게이팅한다(회전 전용 명령일 땐 무시)."""
    return RewTerm(
        func=_feet_air_time_lin_xy,
        weight=weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names),
            "command_name": "base_velocity",
            "threshold": threshold,
            "cmd_threshold": cmd_threshold,
        },
    )


def _feet_air_time_ang_z(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float, cmd_threshold: float
) -> torch.Tensor:
    """회전 명령이 있을 때만 공중 체류 시간을 보상 - DeepRoboticsLab/rl_training의
    feet_air_time_ang_z_cmd_lite3 원본 그대로(지형 난이도 커리큘럼 배율 제외)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    cmd_ang_z = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    return reward * (cmd_ang_z > cmd_threshold)


def feet_air_time_ang_z_lite3(
    weight: float, profile: RobotProfile, threshold: float = 0.5, cmd_threshold: float = 0.1, **_unused
) -> RewTerm:
    """제자리 회전 명령이 있을 때만 걸리는 공중 체류 보상 - feet_air_time_lin_xy와 짝을 이뤄, 이동이든
    회전이든 명령이 있으면 발을 들어 걷게(제자리에서 끌지 않게) 유도한다."""
    return RewTerm(
        func=_feet_air_time_ang_z,
        weight=weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names),
            "command_name": "base_velocity",
            "threshold": threshold,
            "cmd_threshold": cmd_threshold,
        },
    )


def _feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """정지 명령일 때 발이 접지 상태를 유지하면 주는 보상 - DeepRoboticsLab/rl_training의
    feet_contact_without_cmd 원본 그대로."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact.float(), dim=-1)
    return reward * (torch.norm(env.command_manager.get_command(command_name), dim=1) < 0.5)


def feet_contact_without_cmd(weight: float, profile: RobotProfile, **_unused) -> RewTerm:
    """정지 명령일 때의 접지 유지 보상 - stand_still(관절 자세 페널티)과 짝을 이뤄, 정지 중엔 발도
    가만히 붙어 있도록 유도한다."""
    return RewTerm(
        func=_feet_contact_without_cmd,
        weight=weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names),
            "command_name": "base_velocity",
        },
    )


def _contact_force_violation(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """접지 충격력이 threshold를 넘는 만큼 페널티 - DeepRoboticsLab/rl_training의 contact_forces 원본
    그대로(지형 난이도 커리큘럼 배율 제외). undesired_contacts(부위 자체가 닿으면 안 됨)와 달리, 발이
    닿는 것 자체는 정상이고 "얼마나 세게 닿는지"만 문제 삼는다."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    return torch.sum(violation.clip(min=0.0), dim=1)


def contact_forces(weight: float, profile: RobotProfile, threshold: float = 1.0, **_unused) -> RewTerm:
    """발의 접지 충격력이 threshold를 넘는 만큼 물리는 페널티 - 착지 충격을 줄여 부드럽게 걷도록 유도."""
    return RewTerm(
        func=_contact_force_violation,
        weight=weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names),
            "threshold": threshold,
        },
    )

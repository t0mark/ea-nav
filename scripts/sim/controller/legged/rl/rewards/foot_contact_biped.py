"""단일 지지 구간이 의미 있는 2족보행 전용 발 접촉 보상 - 양발을 동시에 띄우는 hopping을 억제한다.

feet_air_time_biped/feet_slide는 Isaac Lab 코어가 아니라 예제 패키지(isaaclab_tasks)의 humanoid 보행
설정에서 쓰는 함수를 그대로 옮겨왔다. feet_contact/feet_swing_height는 Unitree 공식(unitree_rl_gym)
g1/h1/h1_2 legged_gym 세팅의 contact/feet_swing_height 항목을 재현한 것이다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from ..robot_profile import RobotProfile


def _feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """정확히 한 발만 지면에 닿아 있는(단일 지지) 구간에만 보상을 준다.

    두 발이 동시에 지면에서 떨어진 구간(single_stance=False)은 보상이 0이 되므로, 속도 추종 보상만
    있을 때 흔히 나오는 "양발을 동시에 띄워 뛰어서 이동" 하는 퇴화 해(解)를 원천적으로 막는다 - 알려진
    2족보행 RL의 hopping 문제에 대한 표준 대응책(Isaac Lab H1/G1 설정이 채택한 방식).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)

    # 두 발 중 정확히 하나만 지면에 닿아 있을 때만(single_stance) 그 지지/스윙 지속 시간을 보상으로 인정
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def _feet_slide(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """지면에 닿아 있는 발이 미끄러지는 만큼 페널티 - 접촉 중 수평 속도가 있으면 발을 끌고 있다는 뜻."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_velocity.norm(dim=-1) * contacts, dim=1)


def _feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """발이 지면에 접촉해 있는 개수를 그대로 보상 - 아무 발도 안 닿은 채(공중에 뜬 채) 있는 걸 억제."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return torch.sum((forces > threshold).float(), dim=1)


def _feet_swing_height(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, target_height: float
) -> torch.Tensor:
    """스윙 중(접촉 없음)인 발의 높이가 target_height에서 벗어난 정도의 제곱 페널티.

    Ref: Boston Dynamics AI Institute의 foot_clearance_reward와 동일한 개념 - 발이 스윙 중일 때만
    목표 들어올림 높이를 강제해, 발을 끌며 걷는 것과 과도하게 높이 드는 것 둘 다 억제한다.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_error = torch.square(foot_height - target_height)
    return torch.sum(height_error * (~in_contact).float(), dim=1)


def feet_air_time_biped(weight: float, profile: RobotProfile, threshold: float = 0.4, **_unused) -> RewTerm:
    """2족보행 단일지지 보상 - profile.foot_body_names(발 2개)를 그대로 접촉 센서 대상으로 쓴다."""
    return RewTerm(
        func=_feet_air_time_positive_biped,
        weight=weight,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names),
            "threshold": threshold,
        },
    )


def feet_slide(weight: float, profile: RobotProfile, **_unused) -> RewTerm:
    """발 미끄러짐 페널티 - 접촉 센서와 발 링크 둘 다 profile.foot_body_names를 그대로 쓴다."""
    return RewTerm(
        func=_feet_slide,
        weight=weight,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names),
            "asset_cfg": SceneEntityCfg("robot", body_names=profile.foot_body_names),
        },
    )


def feet_contact(weight: float, profile: RobotProfile, threshold: float = 1.0, **_unused) -> RewTerm:
    """접촉해 있는 발 개수 보상 - Unitree 공식 g1/h1/h1_2의 "contact" 항목."""
    return RewTerm(
        func=_feet_contact,
        weight=weight,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names), "threshold": threshold},
    )


def no_jumps(weight: float, profile: RobotProfile, threshold: float = 1.0, **_unused) -> RewTerm:
    """두 발이 동시에 지면에서 떨어지면(접촉이 하나도 없으면) 주는 페널티 - Isaac Lab 공식 digit
    세팅의 "no_jumps" 항목(core_mdp.desired_contacts: 지정 센서에 접촉이 하나도 없을 때만 1.0)."""
    return RewTerm(
        func=core_mdp.desired_contacts,
        weight=weight,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names), "threshold": threshold},
    )


def feet_swing_height(weight: float, profile: RobotProfile, target_height: float = 0.08, **_unused) -> RewTerm:
    """스윙 중인 발 높이 페널티 - Unitree 공식 g1/h1/h1_2의 "feet_swing_height" 항목."""
    return RewTerm(
        func=_feet_swing_height,
        weight=weight,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=profile.foot_body_names),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names),
            "target_height": target_height,
        },
    )

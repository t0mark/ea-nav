"""legged 로봇 보상 항목 - 형태 그룹(quadruped/humanoid)별로 다른 보상 세트를 정의한다.

feet_air_time 계열(feet_air_time, feet_air_time_positive_biped)과 feet_slide는 Isaac Lab 코어
패키지(isaaclab.envs.mdp)가 아니라 예제 패키지(isaaclab_tasks)에만 있는 함수라, 예제 코드를 런타임
의존성으로 끌어오는 대신 이 프로젝트 안에 그대로 옮겨와 관리한다(Isaac Lab 프로젝트들의 통상적인
관례 - isaaclab_tasks의 locomotion/velocity/mdp/rewards.py 원본 로직 그대로).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
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


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """이족보행 전용 - 정확히 한 발만 지면에 닿아 있는(단일 지지) 구간에만 보상을 준다.

    두 발이 동시에 지면에서 떨어진 구간(single_stance=False)은 보상이 0이 되므로, 속도 추종 보상만
    있을 때 흔히 나오는 "양발을 동시에 띄워 뛰어서 이동" 하는 퇴화 해(解)를 원천적으로 막는다 - 알려진
    휴머노이드 RL 보행의 hopping 문제에 대한 표준 대응책(Isaac Lab H1/G1 설정이 채택한 방식).
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


def feet_slide(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """지면에 닿아 있는 발이 미끄러지는 만큼 페널티 - 접촉 중 수평 속도가 있으면 발을 끌고 있다는 뜻."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_velocity.norm(dim=-1) * contacts, dim=1)


@configclass
class BaseLocoRewardsCfg:
    """quadruped/humanoid가 공유하는 기본 보상 - 속도 추종 + 에너지·안정성 페널티."""

    track_lin_vel_xy_exp = RewTerm(
        func=core_mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=core_mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": 0.5}
    )
    lin_vel_z_l2 = RewTerm(func=core_mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=core_mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=core_mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=core_mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=core_mdp.action_rate_l2, weight=-0.01)
    flat_orientation_l2 = RewTerm(func=core_mdp.flat_orientation_l2, weight=-1.0)
    dof_pos_limits = RewTerm(func=core_mdp.joint_pos_limits, weight=-1.0)


@configclass
class QuadrupedRewardsCfg(BaseLocoRewardsCfg):
    """4족보행 로봇 보상 - 발을 들며 걷게 하는 feet_air_time과, 몸통 이외 부위 접촉 금지."""

    feet_air_time = RewTerm(
        func=feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=core_mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 1.0},
    )


@configclass
class HumanoidRewardsCfg(BaseLocoRewardsCfg):
    """이족보행(휴머노이드) 로봇 보상 - hopping 억제용 단일 지지 보상 + 미끄러짐 페널티로 대체."""

    termination_penalty = RewTerm(func=core_mdp.is_terminated, weight=-200.0)
    # 이족보행은 걸음마다 상하 진동이 자연스러워, quadruped용 z속도 페널티를 그대로 두면 정상 보행과 충돌한다
    lin_vel_z_l2 = None
    feet_air_time = RewTerm(
        func=feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        },
    )

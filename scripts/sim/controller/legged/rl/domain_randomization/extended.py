"""basic보다 넓은 도메인 랜덤화 - 액추에이터 게인·링크별 질량·마찰/반발력 범위까지 랜덤화한다.

LimX tron2_rl_lab·Booster booster_gym 공식 세팅에서 실제로 확인된 범위(질량 ±5kg, 링크별 질량 스케일
0.8~1.2배, 마찰 0.4~1.2/0.7~0.9, 반발력 0~0.5, 액추에이터 게인 0.8~1.2배)를 기본값으로 쓰되, 로봇 yaml
에서 각 범위를 덮어쓸 수 있다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from ..robot_profile import RobotProfile


@configclass
class EventCfg:
    """basic의 5개 이벤트에 링크별 질량 스케일·액추에이터 게인 랜덤화 2개를 더한 것."""

    physics_material = EventTerm(
        func=core_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.7, 0.9),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=core_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_distribution_params": (-5.0, 5.0), "operation": "add"},
    )
    randomize_link_mass = EventTerm(
        func=core_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    randomize_actuator_gains = EventTerm(
        func=core_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    reset_base = EventTerm(
        func=core_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=core_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=core_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


def build(profile: RobotProfile, params: dict) -> EventCfg:
    """base_body_name을 반영하고, params로 넘어온 범위 오버라이드를 적용한다."""
    cfg = EventCfg()
    cfg.add_base_mass.params["asset_cfg"].body_names = profile.base_body_name
    reset_range = tuple(params.get("reset_joint_position_range", (0.5, 1.5)))
    cfg.reset_robot_joints.params["position_range"] = reset_range
    if "mass_distribution_params" in params:
        cfg.add_base_mass.params["mass_distribution_params"] = tuple(params["mass_distribution_params"])
    if "actuator_gain_range" in params:
        gain_range = tuple(params["actuator_gain_range"])
        cfg.randomize_actuator_gains.params["stiffness_distribution_params"] = gain_range
        cfg.randomize_actuator_gains.params["damping_distribution_params"] = gain_range
    return cfg

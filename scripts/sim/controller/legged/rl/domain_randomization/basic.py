"""sim-to-real 격차를 줄이기 위한 마찰·질량·초기 상태·외란 이벤트 - 지금까지의 기본 방식.

reset_joint_position_range로 관절 초기 자세 랜덤화 범위를 로봇별로 좁힐 수 있다 - Isaac Lab 공식
digit/h1/g1 세팅은 전부 (1.0, 1.0)(랜덤화 끔)을 쓴다. digit은 다리에 닫힌 루프(로드 구속) 구조가 있어
관절을 무작위로 벌려놓으면 그 루프가 물리적으로 깨질 수 있고("Don't randomize the initial joint
positions because we have closed loops" - Isaac Lab 원본 주석), h1/g1은 정확한 초기 자세가 필요해서다.
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
    """도메인 랜덤화 - sim-to-real 격차를 줄이기 위한 마찰·질량·초기 상태·외란 이벤트."""

    physics_material = EventTerm(
        func=core_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=core_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            # 곱연산 ±25% 로그균등 - Isaac Lab 공식 velocity_env_cfg.py 그대로(재클론해 재검증).
            # 로봇 크기와 무관하게 비율로 스케일되고(로봇별 kg 오버라이드가 필요 없음), 기하평균이
            # 1.0이라 무거워지는 방향/가벼워지는 방향이 대칭이다.
            "mass_distribution_params": (1 / 1.25, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
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
    """add_base_mass 대상 body를 base_body_name으로 맞추고, reset_joint_position_range를 반영한다."""
    cfg = EventCfg()
    cfg.add_base_mass.params["asset_cfg"].body_names = profile.base_body_name
    reset_range = tuple(params.get("reset_joint_position_range", (0.5, 1.5)))
    cfg.reset_robot_joints.params["position_range"] = reset_range
    return cfg

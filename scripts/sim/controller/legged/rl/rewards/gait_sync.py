"""보행 위상 강제 보상 - 지정한 발 쌍이 서로 동기화되도록, 좌우 대칭 관절 쌍이 벌어지지 않도록 유도한다.

lite3·m20·spot 등 트로팅(대각선 두 발이 짝지어 움직이는) 계열 보행에서 쓰는 개념이고, 발 개수·다리
형태와 무관해 로봇 타입을 가리지 않는다. synced_feet_pair_names/mirror_joints는 로봇마다 발·관절
이름이 달라 자동으로 만들 수 없으므로 로봇 yaml의 params에서 직접 받는다.

원본(DeepRobotics rl_training의 feet_gait/phase_foot_trajectory_exp, Isaac Lab Spot의 GaitReward)은
위상 클래스 기반으로 더 정교하지만, 이 구현은 "발 쌍의 최근 공중 시간이 비슷할수록/관절 쌍의 각도
차이가 작을수록"로 단순화한 근사치다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from ..robot_profile import RobotProfile


def _feet_gait_sync(env: ManagerBasedRLEnv, synced_feet_pair_names: list[tuple[str, str]], sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """지정한 발 쌍마다 마지막 공중 시간 차이가 작을수록 exp(-차이^2)로 커지는 보상을 평균낸다."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.last_air_time
    body_names = contact_sensor.body_names
    reward = torch.zeros(env.num_envs, device=env.device)
    for name_a, name_b in synced_feet_pair_names:
        index_a, index_b = body_names.index(name_a), body_names.index(name_b)
        reward += torch.exp(-torch.square(air_time[:, index_a] - air_time[:, index_b]))
    return reward / len(synced_feet_pair_names)


def feet_gait(weight: float, profile: RobotProfile, synced_feet_pair_names: list[list[str]], **_unused) -> RewTerm:
    """대각선(또는 지정한) 발 쌍이 같은 위상으로 움직이도록 유도 - 트로팅 보행 강제."""
    return RewTerm(
        func=_feet_gait_sync,
        weight=weight,
        params={
            "synced_feet_pair_names": [tuple(pair) for pair in synced_feet_pair_names],
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )


def _joint_mirror(env: ManagerBasedRLEnv, mirror_joints: list[tuple[str, str]], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """좌우 대칭이어야 할 관절 패턴 쌍의 위치 차이 제곱합 - 트로팅 보행의 좌우 비대칭을 억제."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    for pattern_a, pattern_b in mirror_joints:
        ids_a = asset.find_joints(pattern_a)[0]
        ids_b = asset.find_joints(pattern_b)[0]
        reward += torch.sum(torch.square(asset.data.joint_pos[:, ids_a] - asset.data.joint_pos[:, ids_b]), dim=1)
    return reward


def joint_mirror(weight: float, profile: RobotProfile, mirror_joints: list[list[str]], **_unused) -> RewTerm:
    """대각선 다리끼리(예: FL-HR) 관절 패턴을 짝지어 좌우 비대칭을 페널티로 억제."""
    return RewTerm(
        func=_joint_mirror,
        weight=weight,
        params={"mirror_joints": [tuple(pair) for pair in mirror_joints], "asset_cfg": SceneEntityCfg("robot")},
    )

"""RL 학습용 커리큘럼 지형 씬 구성과 난이도 승급·강등 로직.

지형 형상 자체(계단·경사 파라미터)는 configs/sim_rl.yaml에 두고, 파싱은 scripts/sim/env/terrain.py의
load_terrain_generator_cfg()를 그대로 재사용한다(포맷이 이미 config-agnostic이라 중복 구현할 이유가
없음). 이 파일이 추가로 맡는 책임은 두 가지뿐이다.
    1. 그 지형을 커리큘럼(난이도별 행 배치) 모드로 씬에 임포트하는 TerrainImporterCfg 조립
    2. 로봇이 한 에피소드를 얼마나 잘 통과했는지에 따라 다음 에피소드에 더 어렵거나 더 쉬운 행으로
       옮기는 판단 로직(이름은 "커리큘럼"이지만 실제로는 씬 안의 로봇 배치를 바꾸는 씬 레벨 동작이라
       controller가 아니라 env에 둔다)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

from scripts.sim.env.terrain import load_terrain_generator_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def build_rl_terrain_importer_cfg(yaml_path: str | Path, prim_path: str = "/World/ground") -> TerrainImporterCfg:
    """configs/sim_rl.yaml을 읽어 커리큘럼 모드 TerrainImporterCfg를 만든다.

    max_init_terrain_level=0으로 고정한다 - 보행을 처음부터 어려운 지형에서 배우면 학습이 발산하기
    쉬우므로, 항상 가장 쉬운 행(낮은 단차·완만한 경사)에서 시작해 아래
    promote_terrain_levels_by_travel_distance()의 판단에 따라서만 점진적으로 올라가게 한다.
    """
    terrain_generator_cfg = load_terrain_generator_cfg(yaml_path)
    return TerrainImporterCfg(
        prim_path=prim_path,
        terrain_type="generator",
        terrain_generator=terrain_generator_cfg,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )


def promote_terrain_levels_by_travel_distance(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """에피소드 동안 이동한 거리와 명령 속도를 비교해 다음 에피소드의 지형 난이도(행)를 조정한다.

    Ref: Isaac Lab 표준 지형 커리큘럼(legged_gym 계열 방식)을 그대로 채택 - 절반 이상 이동하면 한 단계
    승급, 명령 속도 기준 이동량의 절반에도 못 미치면 한 단계 강등, 그 사이는 유지. 이렇게 해야 "어느
    정도 걸으면 경사·단차를 조금씩 올리는" 커리큘럼이 로봇 성능에 맞춰 자동으로 진행된다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command(command_name)

    # base_link가 에피소드 시작 위치(env_origin) 대비 xy 평면에서 실제로 이동한 거리
    travelled_distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = travelled_distance > terrain.cfg.terrain_generator.size[0] / 2.0
    move_down = travelled_distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())

"""파쿠르 단차 지형.

Extreme Parkour(ICRA 2024, chengxuxin/extreme-parkour)의 parkour_step_terrain을 참고해,
연속 계단이 아니라 단차 하나마다 평지 회복 구간을 두고 반복 배치한다. 시각 정보 없이
고유수용감각·접촉 피드백만으로도 넘을 수 있는 "단일 최대 단차 높이"를 로봇별로 커리큘럼
난이도(row)에 따라 탐색하기 위한 지형이다.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .mesh_utils import make_full_width_obstacles, make_ground_plane


def parkour_step_terrain(difficulty: float, cfg: ParkourStepTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """난이도에 비례한 높이의 단차를 평지 회복 구간과 번갈아 배치한다."""
    # difficulty(0~1)를 step_height_range에 선형 보간 - 공식 mesh_pyramid_stairs와 동일한 보간 규약
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    meshes = [make_ground_plane(cfg.size)]
    meshes += make_full_width_obstacles(
        size=cfg.size,
        obstacle_height=step_height,
        obstacle_length=cfg.step_length,
        gap_length=cfg.platform_length,
        num_obstacles=cfg.num_steps,
    )

    # 스폰 지점은 타일 중심(평지 회복 구간에 오도록 step_length/platform_length로 배치를 맞춤)
    origin = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])
    return meshes, origin


@configclass
class ParkourStepTerrainCfg(SubTerrainBaseCfg):
    """파쿠르 단차 지형 설정."""

    function = parkour_step_terrain

    step_height_range: tuple[float, float] = MISSING
    """단차 높이의 최소·최대값 (m). difficulty(row)에 따라 선형 보간된다."""

    step_length: float = 0.4
    """단차 하나의 진행 방향(x) 두께 (m)."""

    platform_length: float = 1.6
    """단차 사이 평지 회복 구간의 길이 (m)."""

    num_steps: int = 3
    """타일 하나에 배치할 단차 개수."""

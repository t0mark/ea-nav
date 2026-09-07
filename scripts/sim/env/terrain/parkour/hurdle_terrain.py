"""파쿠르 허들 지형.

Extreme Parkour(ICRA 2024, chengxuxin/extreme-parkour)의 parkour_hurdle_terrain을 참고한
지형이다. step 지형과 동일하게 "장애물 하나 - 평지 회복" 구조를 쓰지만, 장애물이 얇은 벽
형태라 로봇이 위로 올라서기보다 다리를 들어 넘어가야 한다는 점이 다르다.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .mesh_utils import make_full_width_obstacles, make_ground_plane


def parkour_hurdle_terrain(difficulty: float, cfg: ParkourHurdleTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """난이도에 비례한 높이의 허들을 평지 회복 구간과 번갈아 배치한다."""
    # difficulty(0~1)를 hurdle_height_range에 선형 보간
    hurdle_height = cfg.hurdle_height_range[0] + difficulty * (cfg.hurdle_height_range[1] - cfg.hurdle_height_range[0])

    meshes = [make_ground_plane(cfg.size)]
    meshes += make_full_width_obstacles(
        size=cfg.size,
        obstacle_height=hurdle_height,
        obstacle_length=cfg.hurdle_thickness,
        gap_length=cfg.platform_length,
        num_obstacles=cfg.num_hurdles,
    )

    # 스폰 지점은 타일 중심(평지 회복 구간에 오도록 hurdle_thickness/platform_length로 배치를 맞춤)
    origin = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])
    return meshes, origin


@configclass
class ParkourHurdleTerrainCfg(SubTerrainBaseCfg):
    """파쿠르 허들 지형 설정."""

    function = parkour_hurdle_terrain

    hurdle_height_range: tuple[float, float] = MISSING
    """허들 높이의 최소·최대값 (m). difficulty(row)에 따라 선형 보간된다."""

    hurdle_thickness: float = 0.1
    """허들 하나의 진행 방향(x) 두께 (m). 얇게 두어 위로 올라서지 않고 다리로 넘게 만든다."""

    platform_length: float = 1.6
    """허들 사이 평지 회복 구간의 길이 (m)."""

    num_hurdles: int = 3
    """타일 하나에 배치할 허들 개수."""

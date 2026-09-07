"""configs/*.yaml 의 서브 지형 정의를 isaaclab TerrainGeneratorCfg로 변환한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from .parkour import PARKOUR_SUB_TERRAIN_TYPES

# yaml의 sub_terrains[].type 문자열 -> isaaclab 서브 지형 설정 클래스 매핑
_SUB_TERRAIN_TYPES: dict[str, type] = {
    "mesh_pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg,
    "mesh_pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg,
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg,
    "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg,
    "mesh_random_grid": terrain_gen.MeshRandomGridTerrainCfg,
    "hf_random_uniform": terrain_gen.HfRandomUniformTerrainCfg,
    **PARKOUR_SUB_TERRAIN_TYPES,
}


def load_terrain_generator_cfg(yaml_path: str | Path) -> TerrainGeneratorCfg:
    """configs/sim_terrain.yaml, configs/sim_rl.yaml을 읽어 TerrainGeneratorCfg를 생성한다."""
    # yaml은 순수 데이터만 담고, isaaclab 클래스 조립은 여기서만 처리한다
    with open(yaml_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    # 서브 지형 항목마다 지정된 타입 클래스에 나머지 파라미터를 그대로 전달
    # (size는 TerrainGenerator가 부모 cfg.size로 덮어쓰므로 여기서 지정하지 않는다)
    sub_terrains = {}
    for name, raw_params in raw["sub_terrains"].items():
        params = dict(raw_params)
        sub_terrain_cls = _SUB_TERRAIN_TYPES[params.pop("type")]
        params = {key: (tuple(value) if isinstance(value, list) else value) for key, value in params.items()}
        sub_terrains[name] = sub_terrain_cls(**params)

    return TerrainGeneratorCfg(
        seed=raw.get("seed"),
        curriculum=raw["curriculum"],
        size=tuple(raw["size"]),
        border_width=raw["border_width"],
        num_rows=raw["num_rows"],
        num_cols=raw["num_cols"],
        color_scheme=raw["color_scheme"],
        horizontal_scale=raw["horizontal_scale"],
        vertical_scale=raw["vertical_scale"],
        slope_threshold=raw["slope_threshold"],
        sub_terrains=sub_terrains,
    )

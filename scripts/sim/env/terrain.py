"""TerrainGenerator 지형을 Isaac Sim 스테이지에 구성하는 모듈."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg

# yaml의 sub_terrains[].type 문자열 -> isaaclab 서브 지형 설정 클래스 매핑
_SUB_TERRAIN_TYPES: dict[str, type] = {
    "mesh_pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg,
    "mesh_pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg,
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg,
    "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg,
}


def load_terrain_generator_cfg(yaml_path: str | Path) -> TerrainGeneratorCfg:
    """configs/sim_terrain.yaml을 읽어 TerrainGeneratorCfg를 생성한다."""
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


class TerrainSceneBuilder:
    """TerrainGeneratorCfg를 받아 스테이지에 지형을 임포트하는 단일 책임 클래스."""

    def __init__(
        self,
        terrain_generator_cfg: TerrainGeneratorCfg,
        num_envs: int,
        env_spacing: float,
        prim_path: str = "/World/ground",
    ) -> None:
        """빌더가 사용할 지형 설정과 환경 배치 정보를 저장한다."""
        self._terrain_generator_cfg = terrain_generator_cfg
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._prim_path = prim_path
        self._importer: TerrainImporter | None = None

    def build(self, debug_vis: bool = False) -> TerrainImporter:
        """TerrainImporterCfg를 구성해 지형을 스테이지에 임포트한다."""
        # 서브 지형 원점 표시 여부(debug_vis)만 외부에서 제어하고 나머지는 고정
        importer_cfg = TerrainImporterCfg(
            num_envs=self._num_envs,
            env_spacing=self._env_spacing,
            prim_path=self._prim_path,
            terrain_type="generator",
            terrain_generator=self._terrain_generator_cfg,
            debug_vis=debug_vis,
        )
        self._importer = TerrainImporter(importer_cfg)
        num_rows = self._terrain_generator_cfg.num_rows
        num_cols = self._terrain_generator_cfg.num_cols
        print(f"[TerrainSceneBuilder] 지형 임포트 완료: {num_rows}x{num_cols} sub-terrain, prim_path={self._prim_path}")
        return self._importer

    @property
    def importer(self) -> TerrainImporter:
        """빌드된 TerrainImporter 인스턴스를 반환한다."""
        if self._importer is None:
            raise RuntimeError("build()를 먼저 호출해야 합니다.")
        return self._importer

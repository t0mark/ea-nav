"""TerrainGeneratorCfg를 Isaac Sim 스테이지에 임포트하는 모듈."""

from __future__ import annotations

from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg


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

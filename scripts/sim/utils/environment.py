"""시뮬 환경 1개(스테이지·물리 컨텍스트·지면/지형·로봇)의 수명 관리.

Isaac 앱이 기동된 뒤에만 임포트할 수 있다 (tools/utils/sim.py launch_app 참고).

사용 순서: SimEnvironment(...) -> add_ground() 또는 add_terrain() ->
spawn_robot(...) -> reset() -> step()/hold_step() 반복. 다음 씬은 새
SimEnvironment 생성이 이전 정리를 겸한다 (프로세스당 활성 환경 1개).

Isaac Lab 2.3.0 실측 특성 반영:
- 타임라인 STOP 콜백이 헤드리스에서 무한 렌더 루프가 됨 -> stop 직전마다
  _disable_app_control_on_stop_handle=True (reset()이 False로 되돌리므로
  생성 시점 설정은 무효 — 정리 직전에 다시 켠다)
- GroundPlaneCfg는 Nucleus 클라우드 에셋 참조라 오프라인에서 실패
  -> physicsUtils.add_ground_plane 프로시저럴 평면 사용
- 스포너는 프림만 배치하고 관절 초기각·루트 상태를 물리에 기록하지 않음
  -> reset()에서 write_*_to_sim으로 명시 기록 (빠지면 legged가 영점 자세 시작)
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrains
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaacsim.core.cloner import Cloner, GridCloner
from omni.physx.scripts import physicsUtils
from pxr import Gf, Usd, UsdGeom, UsdShade

from scripts.sim.utils.robot_spawn import make_articulation_cfg

logger = logging.getLogger(__name__)


class SimEnvironment:
    """시뮬 환경 1개의 생성·스폰·스텝·정리를 책임진다.

    좌표계: 월드 Z-up, 지면/지형 기준면 = z 0 (m). 단일 스폰(spawn_robot)은
    평지 격자, 다중 그룹 스폰(spawn_robot_groups)은 평지 격자 또는 지형
    원점(terrain.env_origins) 위 배치를 지원한다 (RL 학습·3단계 롤아웃).
    """

    # 평지·지형 공용 지면 경로 — RL 원본(velocity_env_cfg)과 동일 경로라
    # 2단계 height_scanner(mesh_prim_paths=/World/ground)를 그대로 쓸 수 있다
    _GROUND_PATH = "/World/ground"
    _ENV_NS = "/World/envs"

    def __init__(self, physics_dt: float, device: str):
        """이전 환경을 정리한 뒤 새 스테이지와 물리 컨텍스트를 만든다.

        physics_dt: 물리 스텝 (s). device: 물리 파이프라인 장치 (예: "cuda:0").
        """
        self._teardown_previous()
        self._sim = SimulationContext(SimulationCfg(dt=physics_dt, device=device))
        self._dt = physics_dt
        # 로봇 그룹 목록 + env 구간: 단일 스폰(spawn_robot) = 그룹 1개,
        # 다중 스폰(spawn_robot_groups) = 로봇 종류당 그룹 1개 (RL 학습용)
        self._groups: list[Articulation] = []
        self._slices: list[slice] = []
        self._origins: torch.Tensor | None = None
        self._terrain: terrains.TerrainImporter | None = None

    @property
    def sim(self) -> SimulationContext:
        """내부 SimulationContext (렌더·카메라 등 외부 유틸 접근용)."""
        return self._sim

    @property
    def robot(self) -> Articulation:
        """단일 스폰 로봇 articulation. spawn_robot() 이전 접근은 오류."""
        if len(self._groups) != 1:
            raise RuntimeError("robot은 spawn_robot() 단일 스폰 후에만 유효하다"
                               " (다중 그룹은 robots 사용)")
        return self._groups[0]

    @property
    def robots(self) -> list[Articulation]:
        """스폰된 로봇 그룹 목록 (spawn_robot_groups 순서)."""
        if not self._groups:
            raise RuntimeError("스폰 이전에는 robots에 접근할 수 없다")
        return self._groups

    def group_slice(self, g: int) -> slice:
        """그룹 g가 차지하는 전역 env 구간 (origins·전역 버퍼 인덱싱용)."""
        return self._slices[g]

    @property
    def origins(self) -> torch.Tensor:
        """env별 원점 (N, 3). 루트 상태 기록·관측 상대화에 사용."""
        if self._origins is None:
            raise RuntimeError("스폰 이전에는 origins에 접근할 수 없다")
        return self._origins

    @staticmethod
    def _teardown_previous():
        """이전 시뮬 인스턴스와 스테이지를 정리한다 (생성자가 호출).

        STOP 콜백 무한 렌더 루프 방지 플래그는 reset()이 매번 False로
        되돌리므로 반드시 stop 직전에 다시 켠다 (모듈 docstring 참고).
        """
        prev = SimulationContext.instance()
        if prev is not None:
            prev._disable_app_control_on_stop_handle = True
            prev.stop()
            prev.clear_all_callbacks()
            prev.clear_instance()
        stage_utils.create_new_stage()

    @staticmethod
    def _add_light():
        """렌더 체크용 조명을 만든다 (물리 무영향, 씬당 1회).

        태양광은 비스듬히(약 40도) 기울인다 — 수직광은 경사·계단 면의
        음영 차이가 없어 무채색 지형이 흰 판처럼 보인다 (시각 확인 실측).
        돔광은 그림자 완전 암부를 밝히는 보조광.
        """
        # Ry(40도) 쿼터니언 (w,0,sin20,0) — 라이트 -z축이 기울어 비춘다
        light = sim_utils.DistantLightCfg(intensity=2500.0)
        light.func("/World/light", light,
                   orientation=(0.9397, 0.0, 0.342, 0.0))
        # 보조광은 약하게 — 400은 로봇 색이 씻겨 보이는 과노출 실측
        dome = sim_utils.DomeLightCfg(intensity=150.0)
        dome.func("/World/dome_light", dome)

    def add_ground(self, size: float = 20.0, friction: float = 1.0):
        """평지 지면 충돌 평면(z=0)과 조명을 만든다.

        size: 평면 반크기 (m). friction: 지면 마찰 (기본 1.0 = 3단계 지형
        마찰과 정합 — 미명시 시 PhysX 기본 재질(0.5)로 돌아 평지 기준선
        (WVN 추종 오차 정규화)이 지형과 다른 마찰 위에서 만들어진다).
        Nucleus 미사용 프로시저럴 평면 (모듈 docstring).
        """
        stage = stage_utils.get_current_stage()
        physicsUtils.add_ground_plane(stage, self._GROUND_PATH,
                                      "Z", size, Gf.Vec3f(0.0), Gf.Vec3f(0.2))
        mat_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=friction, dynamic_friction=friction,
            friction_combine_mode="multiply", restitution_combine_mode="multiply")
        mat_cfg.func("/World/Materials/ground", mat_cfg)
        material = UsdShade.Material(stage.GetPrimAtPath("/World/Materials/ground"))
        plane = stage.GetPrimAtPath(f"{self._GROUND_PATH}/CollisionPlane")
        # 대상 프림명이 헬퍼 내부 규약이라 유효성을 확인하고 로그로 남긴다
        # (조용한 바인딩 실패 재발 탐지 — 이 코드베이스에서 2회 실측된 계열)
        if not plane.IsValid():
            logger.warning("지면 충돌 프림 미발견 — 마찰 재질 미적용 (기본 마찰로 동작)")
        else:
            UsdShade.MaterialBindingAPI.Apply(plane).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics")
            logger.info("평지 마찰 재질 바인딩: mu %.2f (%s)", friction,
                        plane.GetPath())
        self._add_light()

    @property
    def terrain(self) -> terrains.TerrainImporter:
        """지형 임포터 (add_terrain 이후 유효 — env 원점·커리큘럼 API 접근용).

        terrain.env_origins = env별 서브지형 원점, terrain.terrain_levels =
        env별 난이도 행, terrain.update_env_origins(...) = 커리큘럼 승급·강등.
        """
        if self._terrain is None:
            raise RuntimeError("add_terrain() 이전에는 terrain에 접근할 수 없다")
        return self._terrain

    def add_terrain(self, terrain_cfg: dict, num_envs: int = 1,
                    color_scheme: str = "height") -> tuple[float, float]:
        """TerrainGenerator 지형과 조명을 만든다 (add_ground의 대안).

        구성은 Isaac Lab 보행 RL 표준(velocity_env_cfg의 ROUGH_TERRAINS_CFG)과
        정합: 피라미드 계단/역계단/박스/러프/경사/역경사 6종 + 난이도 커리큘럼
        (행 방향 오름차순) + 지면 마찰 명시. 격자 규모만 config로 축소·확대한다.
        지형은 원점 중심 배치, 높이 색상(color_scheme height)으로 탑뷰 판독성 확보.

        num_envs: 지형 위 스폰할 env 수 — 임포터가 env별 원점을 커리큘럼
        규칙(난이도 행 = 랜덤 [0, max_init], 종류 열 = env 인덱스 분할)으로
        배정한다 (terrain 프로퍼티로 접근). 지형만 쓰는 호출(탑뷰 렌더)은 1.
        color_scheme: "height" = 높이 색상 (탑뷰 판독용 기본) / "none" =
        무채색 + 조명 음영만 (주행 비디오용 — 높이 무지개색이 화면을
        지배하는 것 시각 확인 피드백).
        반환: 지형 전체 크기 (x, y) (m) — 탑뷰 카메라 고도 산정용.
        """
        sub_terrains = {
            "pyramid_stairs": terrains.MeshPyramidStairsTerrainCfg(
                proportion=terrain_cfg["stairs_proportion"],
                step_height_range=tuple(terrain_cfg["step_height"]),
                step_width=terrain_cfg["step_width"],
                platform_width=terrain_cfg["stairs_platform_width"],
                border_width=terrain_cfg["stairs_border"], holes=False),
            "pyramid_stairs_inv": terrains.MeshInvertedPyramidStairsTerrainCfg(
                proportion=terrain_cfg["stairs_inv_proportion"],
                step_height_range=tuple(terrain_cfg["step_height"]),
                step_width=terrain_cfg["step_width"],
                platform_width=terrain_cfg["stairs_platform_width"],
                border_width=terrain_cfg["stairs_border"], holes=False),
            "boxes": terrains.MeshRandomGridTerrainCfg(
                proportion=terrain_cfg["boxes_proportion"],
                grid_width=terrain_cfg["grid_width"],
                grid_height_range=tuple(terrain_cfg["grid_height"]),
                platform_width=terrain_cfg["platform_width"]),
            "random_rough": terrains.HfRandomUniformTerrainCfg(
                proportion=terrain_cfg["rough_proportion"],
                noise_range=tuple(terrain_cfg["rough_noise"]), noise_step=0.02,
                border_width=terrain_cfg["hf_border"]),
            "hf_pyramid_slope": terrains.HfPyramidSlopedTerrainCfg(
                proportion=terrain_cfg["slope_proportion"],
                slope_range=tuple(terrain_cfg["slope_range"]),
                platform_width=terrain_cfg["platform_width"],
                border_width=terrain_cfg["hf_border"]),
            "hf_pyramid_slope_inv": terrains.HfInvertedPyramidSlopedTerrainCfg(
                proportion=terrain_cfg["slope_inv_proportion"],
                slope_range=tuple(terrain_cfg["slope_range"]),
                platform_width=terrain_cfg["platform_width"],
                border_width=terrain_cfg["hf_border"]),
        }
        generator_cfg = terrains.TerrainGeneratorCfg(
            size=tuple(terrain_cfg["size"]),
            border_width=terrain_cfg["border_width"],
            num_rows=terrain_cfg["num_rows"],
            num_cols=terrain_cfg["num_cols"],
            curriculum=terrain_cfg["curriculum"],
            sub_terrains=sub_terrains,
            color_scheme=color_scheme,
            use_cache=False,
        )
        # visual_material 기본값(검정 PreviewSurface) 적용 시 이 오프라인 환경에서
        # 프로세스가 무한 대기하는 것 실측 (원인 미규명) -> None + 높이 색상으로 회피
        importer_cfg = terrains.TerrainImporterCfg(
            prim_path=self._GROUND_PATH,
            terrain_type="generator",
            terrain_generator=generator_cfg,
            visual_material=None,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=terrain_cfg["static_friction"],
                dynamic_friction=terrain_cfg["dynamic_friction"]),
            max_init_terrain_level=terrain_cfg["max_init_terrain_level"],
            num_envs=num_envs,
            collision_group=-1,
        )
        self._terrain = terrains.TerrainImporter(importer_cfg)
        if color_scheme == "none":
            self._paint_terrain_by_slope()
        self._add_light()

        # 전체 크기 = 격자(행 x 열) x 칸 크기 + 양쪽 테두리 (원점 중심 배치)
        size_x = terrain_cfg["num_rows"] * terrain_cfg["size"][0] + 2 * terrain_cfg["border_width"]
        size_y = terrain_cfg["num_cols"] * terrain_cfg["size"][1] + 2 * terrain_cfg["border_width"]
        logger.info("지형 생성: %d x %d 칸, 전체 %.1f x %.1f m",
                    terrain_cfg["num_rows"], terrain_cfg["num_cols"], size_x, size_y)
        return size_x, size_y

    def _paint_terrain_by_slope(self):
        """지형 메시를 면 단위로 칠한다: 평지 회색, 경사면은 방향별 색조.

        무채색 단일 재질은 경사 시작점·방향이 화면에서 안 읽히는 것 실측
        (시각 확인 피드백 — "경사마다 바닥색 다르게"). 면 법선으로 분류:
        - 평지(법선 z > 0.98) = 기본 회색
        - 수직면(z < 0.5, 계단 챌면 등) = 진회색
        - 경사면 = 법선 수평 성분의 4방위(+x/-x/+y/-y)별 저채도 색조
        displayColor 면 단위 primvar 사용 — 높이 색상(color_scheme height)과
        같은 렌더 경로라 재질 없이도 표시된다 (재질 바인딩은 지형 메시에
        안 먹는 것 실측).
        """
        from pxr import Sdf

        # 저채도 회색 기반 팔레트. 태양광+돔광 아래 displayColor가 크게
        # 밝아지는 것 실측(0.46 -> 화면 0.9) — 어둡게 잡아 보정한다
        flat_c = (0.30, 0.30, 0.32)
        steep_c = (0.15, 0.15, 0.17)
        quad_c = {
            "+x": (0.40, 0.30, 0.22), "-x": (0.22, 0.30, 0.40),
            "+y": (0.25, 0.37, 0.25), "-y": (0.37, 0.26, 0.37),
        }
        stage = stage_utils.get_current_stage()
        ground = stage.GetPrimAtPath(self._GROUND_PATH)
        for prim in Usd.PrimRange(ground, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get())
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            if pts.size == 0 or counts.size == 0 or not (counts == 3).all():
                # 지형 생성기는 삼각 메시만 만든다 — 예외 프림은 건너뜀
                continue
            idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            # 면 법선 (정규화) -> 기울기·방위 분류
            v0, v1, v2 = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
            n = np.cross(v1 - v0, v2 - v0)
            n /= np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)
            # 위쪽 정렬 (winding 혼재 대비)
            n[n[:, 2] < 0.0] *= -1.0

            # 삼각형별 분류는 하이트필드 높이 양자화(계단화된 경사) 때문에
            # 평지/경사가 면마다 널뛰어 체커 얼룩이 되는 것 실측 -> 0.5m
            # XY 격자 빈의 평균 법선으로 분류해 매크로 기울기를 읽는다
            centers = (v0 + v1 + v2) / 3.0
            bin_size = 0.5
            bx = np.floor(centers[:, 0] / bin_size).astype(np.int64)
            by = np.floor(centers[:, 1] / bin_size).astype(np.int64)
            bx -= bx.min()
            by -= by.min()
            bin_id = bx * (by.max() + 1) + by
            n_bins = int(bin_id.max()) + 1
            avg = np.zeros((n_bins, 3))
            np.add.at(avg, bin_id, n)
            avg /= np.clip(np.linalg.norm(avg, axis=1, keepdims=True), 1e-9, None)

            # 빈 단위 분류 -> 면에 전파. 개별 수직면(계단 챌면)만 진회색 유지
            bin_colors = np.tile(np.array(flat_c), (n_bins, 1))
            bnz = avg[:, 2]
            sloped = bnz <= 0.997
            east = np.abs(avg[:, 0]) >= np.abs(avg[:, 1])
            bin_colors[sloped & east & (avg[:, 0] >= 0)] = quad_c["+x"]
            bin_colors[sloped & east & (avg[:, 0] < 0)] = quad_c["-x"]
            bin_colors[sloped & ~east & (avg[:, 1] >= 0)] = quad_c["+y"]
            bin_colors[sloped & ~east & (avg[:, 1] < 0)] = quad_c["-y"]
            colors = bin_colors[bin_id]
            colors[n[:, 2] < 0.35] = steep_c
            primvar = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.uniform)
            primvar.Set([Gf.Vec3f(*c) for c in colors])

    def spawn_robot(self, usd_dir: Path, drive_cfg: dict, num_envs: int = 1,
                    spacing: float = 3.0, spawn_margin: float = 0.01,
                    color: tuple | None = None,
                    self_collision: bool = True,
                    gain_overrides: dict | None = None,
                    contact_cfg: dict | None = None,
                    origin: tuple | None = None,
                    friction_links: list[str] | None = None,
                    friction_links_mu: float = 1.0,
                    solver_iters: tuple[int, int] | None = None,
                    actuator_model: str = "implicit") -> Articulation:
        """로봇 1종을 단일/다중 env 격자로 스폰한다.

        usd_dir: 변환 산출 폴더 (robot.usd + meta.json + joints.json).
        drive_cfg: configs/sim.yaml drive 규칙 (스폰 시점 게인 결정).
        spawn_margin: 기립 높이 위 여유 낙하 (m) — 초기 지면 관통 방지.
        color: 로봇 표시 색 RGB 0-1 (렌더 체크용, None = 원본 회색).
          스테이지 프림 직접 수정(displayColor)이 렌더에 반영되지 않는 것을
          실측 -> 링크별 개별 재질 바인딩 (_bind_link_palette: 구동부
          바퀴·다리 = 검정 2톤 고정, 몸통 = form색 팔레트 — 마디 경계가
          보여야 비디오에서 관절 움직임·걸음새를 판독할 수 있다).
        self_collision: 셀프충돌 활성 (기본 켬). 1단계 정적 검사가 임의 자세
          셀프충돌을 "시뮬 접촉으로 해소" 전제로 허용했으므로 켜야 정합
          (게인과 동일하게 USD에 굽지 않고 스폰 시점에 결정).
        gain_overrides: {조인트 이름: (강성, 감쇠)} — drive 규칙 결과를 개별
          덮어쓰기 (balancing 바퀴 게인 0 등 제어기 요구 반영).
        contact_cfg: 수동 접촉 링크 마찰 상수 (configs/sim.yaml contact —
          볼 캐스터·롤러의 URDF에 없는 물리를 전 로봇 공통으로 고정).
        origin: env 원점 오프셋 (x, y, z) — 지형 위 스폰용 (지형 셀 원점의
          z 포함. 프림 격자는 원점대로 두고 reset()의 루트 기록만 옮긴다).
        반환: 생성된 Articulation (reset() 이후 유효).
        """
        usd_dir = Path(usd_dir)

        # env_0에 원본을 배치하고 격자로 복제 (물리 복제 = env 간 상호작용 차단).
        # 색은 스포너 루트 재질을 쓰지 않는다 — 루트 바인딩이
        # strongerThanDescendants라 링크 팔레트 바인딩을 덮는 것 실측
        # -> 전 링크 개별 바인딩(_bind_link_palette)만 쓴다
        prim_utils.define_prim(f"{self._ENV_NS}/env_0")
        material = None
        # sleep_threshold 0 = PhysX 절전 비활성 — 저속·정지 순간 articulation이
        # 잠들면 조인트 드라이브 목표로는 깨어나지 않아 명령을 보내도 영구
        # 동결되는 것 실측 (제자리 선회·지형 위 일시 정지에서 발생)
        # solver_iters = 보행 RL 표준 정렬 (spawn_robot_groups와 동일 규약)
        art_props = dict(enabled_self_collisions=self_collision,
                         sleep_threshold=0.0, stabilization_threshold=0.0)
        if solver_iters is not None:
            art_props.update(solver_position_iteration_count=solver_iters[0],
                             solver_velocity_iteration_count=solver_iters[1])
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=str(usd_dir / "robot.usd"),
            visual_material=material,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(**art_props))
        spawn_cfg.func(f"{self._ENV_NS}/env_0/Robot", spawn_cfg)
        # 마찰 바인딩은 복제 전 env_0 원본에만 하면 전 env에 복제된다
        if contact_cfg is not None:
            self._bind_passive_friction(f"{self._ENV_NS}/env_0/Robot", contact_cfg)
        # 접촉 링크(legged 발) 마찰 명시 — 학습 스폰과 동일 규약 (기본 0.5
        # 결함 부류 교정)
        if friction_links:
            self.bind_link_friction(f"{self._ENV_NS}/env_0/Robot",
                                    friction_links, friction_links_mu,
                                    "/World/Materials/contact_links")
        # 링크 팔레트도 복제 전 원본에만 (docstring — 걸음새 판독성)
        if color is not None:
            self._bind_link_palette(f"{self._ENV_NS}/env_0/Robot", color)
        cloner = GridCloner(spacing=spacing)
        cloner.define_base_env(self._ENV_NS)
        env_paths = cloner.generate_paths(f"{self._ENV_NS}/env", num_envs)
        positions = cloner.clone(f"{self._ENV_NS}/env_0", env_paths,
                                 replicate_physics=True, base_env_path=self._ENV_NS)
        # env끼리는 충돌 제외하되 공용 지면과는 충돌 유지
        cloner.filter_collisions(self._sim.cfg.physics_prim_path, "/World/collisions",
                                 env_paths, [self._GROUND_PATH])
        self._origins = torch.tensor(positions, dtype=torch.float32, device=self._sim.device)
        if origin is not None:
            self._origins += torch.tensor(origin, dtype=torch.float32,
                                          device=self._sim.device)

        robot_cfg, meta = make_articulation_cfg(usd_dir, drive_cfg,
                                                f"{self._ENV_NS}/env_.*/Robot", spawn_margin,
                                                gain_overrides=gain_overrides,
                                                actuator_model=actuator_model)
        self._groups = [Articulation(robot_cfg)]
        self._slices = [slice(0, num_envs)]
        logger.info("스폰 %s: env %d개, 스폰 높이 %.3fm, 셀프충돌 %s",
                    meta["name"], num_envs, robot_cfg.init_state.pos[2],
                    self._applied_self_collision())
        return self._groups[0]

    def spawn_robot_groups(self, usd_dirs: list[Path], drive_cfg: dict,
                           envs_per_robot: int, spacing: float = 4.0,
                           spawn_margin: float = 0.01,
                           self_collision: bool = True,
                           gain_overrides_list: list[dict | None] | None = None,
                           origins: torch.Tensor | None = None,
                           activate_contact_sensors: bool = False,
                           friction_links_list: list[list[str] | None] | None = None,
                           friction_links_mu: float = 1.0,
                           solver_iters: tuple[int, int] | None = None,
                           actuator_model: str = "implicit") \
            -> list[Articulation]:
        """로봇 K종을 각 M개 env로 스폰한다 (RL 학습용 이종 다중 스폰).

        PhysX articulation 뷰는 동일 DoF만 묶을 수 있어, 로봇 종류(그룹)마다
        전용 프림 이름(Robot_gXXX)으로 스폰하고 그룹별 Articulation을 만든다
        — 정규식이 해당 그룹 env에만 매칭되어 이종 구조가 분리된다.
        그룹별 클론은 명시 위치의 base Cloner를 쓴다 (GridCloner의 자동
        격자는 호출마다 원점 중심이라 그룹끼리 겹친다). replicate_physics는
        이종 소스 다중 등록의 검증이 없어 끄고 일반 파싱으로 간다.

        env 배치 (그룹 g = 전역 구간 [g*M, (g+1)*M)):
        - origins=None: 평지용 XY 격자 (원점 중심, spacing 간격)
        - origins=(K*M, 3) 텐서: 지형 위 스폰 초기 위치. 팬시 인덱싱 결과를
          넘기면 복사본이라 이후 커리큘럼의 임포터 제자리 갱신이 여기엔
          반영되지 않는다 — 최신 원점이 필요한 호출측(train_env)은 매번
          임포터에서 직접 읽는다 (LeggedTrainEnv._origins_of)

        activate_contact_sensors: 전 강체에 접촉 보고 API 활성 (RL 발 접촉
        센서 전제 — ContactSensor는 이 플래그 없이 생성이 실패한다).
        usd_dirs: 그룹별 변환 산출 폴더. gain_overrides_list: 그룹별 게인
        덮어쓰기 (None = 규칙값 — RL 로코모션 게인은 호출측이 전달).
        반환: 그룹별 Articulation 목록 (usd_dirs 순서).
        """
        K, M = len(usd_dirs), envs_per_robot
        total = K * M
        overrides = gain_overrides_list or [None] * K

        if origins is None:
            # 평지: 전역 격자 위치 (행 우선, 원점 중심 — GridCloner와 동일 구도)
            cols = max(1, math.ceil(math.sqrt(total)))
            rows = math.ceil(total / cols)
            positions = np.zeros((total, 3))
            for i in range(total):
                positions[i, 0] = (i % cols - 0.5 * (cols - 1)) * spacing
                positions[i, 1] = (i // cols - 0.5 * (rows - 1)) * spacing
            self._origins = torch.tensor(positions, dtype=torch.float32,
                                         device=self._sim.device)
        else:
            # 지형: 임포터 배정 원점 (별칭 유지 — docstring)
            if origins.shape[0] != total:
                raise ValueError(f"origins 수 {origins.shape[0]} != env 수 {total}")
            positions = origins.detach().cpu().numpy()
            self._origins = origins

        cloner = Cloner()
        self._groups, self._slices = [], []
        all_paths = []
        for g, usd_dir in enumerate(usd_dirs):
            usd_dir = Path(usd_dir)
            base = g * M
            src = f"{self._ENV_NS}/env_{base}"
            prim_utils.define_prim(src)
            # 절전 비활성 사유는 spawn_robot과 동일 (저속·정지 동결 실측).
            # solver_iters = 보행 RL 표준 정렬용 (Isaac 로봇 자산 cfg 전 계열
            # = 위치 4 / 속도 0. None = 변환 USD 기본 유지)
            art_props = dict(enabled_self_collisions=self_collision,
                             sleep_threshold=0.0, stabilization_threshold=0.0)
            if solver_iters is not None:
                art_props.update(solver_position_iteration_count=solver_iters[0],
                                 solver_velocity_iteration_count=solver_iters[1])
            spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=str(usd_dir / "robot.usd"),
                activate_contact_sensors=activate_contact_sensors,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(**art_props))
            spawn_cfg.func(f"{src}/Robot_g{g:03d}", spawn_cfg)
            # 접촉 링크 마찰 바인딩은 복제 전 원본에 (클론 상속 규약 —
            # 로봇 링크는 재질 미보유라 PhysX 기본 0.5로 도는 결함 부류)
            if friction_links_list is not None and friction_links_list[g]:
                self.bind_link_friction(f"{src}/Robot_g{g:03d}",
                                        friction_links_list[g],
                                        friction_links_mu,
                                        f"/World/Materials/links_g{g:03d}")
            paths = [f"{self._ENV_NS}/env_{base + k}" for k in range(M)]
            cloner.clone(src, paths, positions=positions[base:base + M],
                         replicate_physics=False)
            all_paths += paths

            robot_cfg, meta = make_articulation_cfg(
                usd_dir, drive_cfg, f"{self._ENV_NS}/env_.*/Robot_g{g:03d}",
                spawn_margin, gain_overrides=overrides[g],
                actuator_model=actuator_model)
            self._groups.append(Articulation(robot_cfg))
            self._slices.append(slice(base, base + M))
            logger.info("그룹 %d/%d 스폰 %s: env %d개, 스폰 높이 %.3fm",
                        g + 1, K, meta["name"], M, robot_cfg.init_state.pos[2])

        # env끼리는 충돌 제외하되 공용 지면과는 충돌 유지 (단일 스폰과 동일)
        cloner.filter_collisions(self._sim.cfg.physics_prim_path,
                                 "/World/collisions", all_paths,
                                 [self._GROUND_PATH])
        return self._groups

    # 검정 계열로 칠할 구동부 링크 이름 토큰 (바퀴·조향·다리 — 생성기 명명
    # 규약 + 실로봇 관례 토큰. 시각 확인 피드백: 구동부는 검정 고정)
    _DRIVE_LINK_TOKENS = ("wheel", "caster", "ball", "roller", "knuckle",
                          "steer", "leg", "hip", "thigh", "shin", "calf",
                          "knee", "foot", "ankle", "coxa", "femur", "tibia")

    def _bind_link_palette(self, robot_path: str, color: tuple):
        """링크별 시각 재질 바인딩: 구동부(바퀴·다리) 검정 2톤 + 몸통 팔레트.

        구동부는 이름 토큰 매칭으로 검정 계열 고정 (인접 마디는 두 검정
        톤을 번갈아 받아 관절 경계가 보인다), 나머지(몸통·머리 등)는 form색
        기반 [기본, 밝음, 보색] 순환. 링크 Xform 개별 바인딩 — 루트 일괄
        바인딩(strongerThanDescendants)은 이를 덮으므로 spawn_robot이 루트
        재질을 쓰지 않는 것과 짝이다 (물리 재질과 같은 경로, 프록시 상속).
        """
        palette = {
            "base": tuple(color),
            "light": tuple(min(1.0, 0.5 + 0.5 * c) for c in color),
            "accent": tuple(min(1.0, 1.15 - c) for c in color),
            "black1": (0.06, 0.06, 0.07),
            "black2": (0.20, 0.20, 0.22),
        }
        stage = stage_utils.get_current_stage()
        materials = {}
        for name, rgb in palette.items():
            cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=rgb)
            cfg.func(f"/World/Materials/robot_{name}", cfg)
            materials[name] = UsdShade.Material(
                stage.GetPrimAtPath(f"/World/Materials/robot_{name}"))
        root = stage.GetPrimAtPath(robot_path)
        body_order = ["base", "light", "accent"]
        links = sorted((p for p in root.GetChildren() if p.GetName() != "Looks"),
                       key=lambda p: p.GetName())
        n_drive, n_body = 0, 0
        for prim in links:
            name = prim.GetName().lower()
            if any(tok in name for tok in self._DRIVE_LINK_TOKENS):
                kind = "black1" if n_drive % 2 == 0 else "black2"
                n_drive += 1
            else:
                kind = body_order[n_body % len(body_order)]
                n_body += 1
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                materials[kind],
                bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    def bind_link_friction(self, robot_path: str, link_names: list[str],
                           friction: float, material_path: str):
        """지정 링크들에 마찰 재질을 직접 바인딩한다 (발 등 접촉 링크용).

        로봇 링크에는 물리 재질이 없어 PhysX 기본(마찰 0.5)으로 동작하는
        것이 기록된 결함 부류 (바퀴·캐스터에서 2회 실측) — legged 발도
        동일해 발-지면 유효 마찰 0.5가 보행을 미끄럼 쪽으로 기울인다.
        바인딩 방식은 _bind_passive_friction과 동일 (프록시 콜라이더 상속).
        """
        cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=friction, dynamic_friction=friction,
            friction_combine_mode="multiply", restitution_combine_mode="multiply")
        cfg.func(material_path, cfg)
        stage = stage_utils.get_current_stage()
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        root = stage.GetPrimAtPath(robot_path)
        names = set(link_names)
        bound = 0
        for prim in Usd.PrimRange(root):
            if prim.GetName() in names:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics")
                bound += 1
        logger.info("링크 마찰 바인딩: %d개 (mu %.2f, %s)", bound, friction,
                    robot_path)

    def _bind_passive_friction(self, robot_path: str, contact_cfg: dict):
        """수동 접촉 링크(볼 캐스터·롤러)에 공통 마찰 재질을 바인딩한다.

        URDF에 마찰 필드가 없어 시뮬 물리가 임의값이 되는 부분을 전 로봇
        공통 상수로 고정한다 (plan 제어기 섹션 원칙):
        - ball_* 링크: 낮은 마찰 — fixed 병합된 볼 캐스터의 전방향 구름을
          미끄럼 접촉으로 근사 (1단계 생성기 docstring 전제)
        - *_roller_* 링크: 표준 마찰 명시 — 매커넘·옴니 접지 롤러
        링크 이름 규칙은 1단계 생성기 고정 규약이라 프림 이름 매칭으로 충분.
        """
        # 재질은 씬당 1회 생성 (동일 경로 재정의는 무해 — 같은 값)
        mats = {
            "ball": ("/World/Materials/passive_ball",
                     contact_cfg["ball_friction"]),
            "roller": ("/World/Materials/passive_roller",
                       contact_cfg["roller_friction"]),
        }
        for path, friction in mats.values():
            cfg = sim_utils.RigidBodyMaterialCfg(
                static_friction=friction, dynamic_friction=friction,
                friction_combine_mode="multiply", restitution_combine_mode="multiply")
            cfg.func(path, cfg)

        # 변환 USD의 콜라이더는 인스턴스 프록시라 (1) 링크 Xform에는
        # CollisionAPI가 없고 (2) 하위 순회로도 닿지 않아 Isaac
        # bind_physics_material이 조용히 실패한다 (유효 재질 None 실측
        # — 캐스터가 PhysX 기본 마찰 0.5로 굴러 차동 선회를 짓눌렀다)
        # -> 링크 프림에 UsdShade 물리 바인딩을 직접 건다 (상위 바인딩은
        # 프록시 콜라이더에 상속 적용된다)
        stage = stage_utils.get_current_stage()
        root = stage.GetPrimAtPath(robot_path)
        bound = {"ball": 0, "roller": 0}
        for prim in Usd.PrimRange(root):
            name = prim.GetName()
            if name.startswith("ball_"):
                kind = "ball"
            elif "_roller_" in name:
                kind = "roller"
            else:
                continue
            material = UsdShade.Material(stage.GetPrimAtPath(mats[kind][0]))
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(material,
                         bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                         materialPurpose="physics")
            bound[kind] += 1
        if bound["ball"] or bound["roller"]:
            logger.info("수동 접촉 마찰 바인딩: 볼 캐스터 %d개(mu %.2f), 롤러 %d개(mu %.2f)",
                        bound["ball"], mats["ball"][1], bound["roller"], mats["roller"][1])

    def _applied_self_collision(self):
        """스테이지에 실제 기록된 셀프충돌 값을 읽는다 (적용 확인 로그용).

        articulation_props는 API 미발견 시 조용히 무시되므로, 요청값이 아니라
        스테이지 실값을 로그에 남겨 적용 실패를 드러낸다. 반환: True/False/None(미발견).
        """
        root = stage_utils.get_current_stage().GetPrimAtPath(f"{self._ENV_NS}/env_0/Robot")
        for prim in Usd.PrimRange(root):
            attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
            if attr and attr.HasAuthoredValue():
                return attr.Get()
        return None

    def reset(self):
        """물리 초기화 후 전 그룹의 기립 자세·루트 상태를 명시 기록한다.

        스포너가 물리 상태를 기록하지 않으므로 default 상태(= init_state 규약)를
        env 원점 오프셋과 함께 직접 써 넣는다 (모듈 docstring 참고).
        """
        self._sim.reset()
        for art, sl in zip(self._groups, self._slices):
            root_state = art.data.default_root_state.clone()
            root_state[:, :3] += self.origins[sl]
            art.write_root_pose_to_sim(root_state[:, :7])
            art.write_root_velocity_to_sim(root_state[:, 7:])
            art.write_joint_state_to_sim(art.data.default_joint_pos.clone(),
                                         art.data.default_joint_vel.clone())
            art.reset()

    def step(self, joint_pos_target: torch.Tensor | None = None,
             joint_vel_target: torch.Tensor | None = None,
             joint_effort_target: torch.Tensor | None = None,
             render: bool = False):
        """드라이브 목표를 기록하고 물리를 1스텝 진행한다 (단일 스폰용).

        joint_pos_target/joint_vel_target/joint_effort_target: (N, DoF) 목표.
        None이면 해당 목표 생략 (직전 목표 유지 — 제어 주기가 물리 스텝보다
        느릴 때 사이 스텝은 인자 없이 호출한다). effort는 드라이브 게인 0인
        조인트의 직접 토크 입력용 (balancing LQR). render: 렌더 프레임 동반.
        """
        art = self.robot
        if joint_pos_target is not None:
            art.set_joint_position_target(joint_pos_target)
        if joint_vel_target is not None:
            art.set_joint_velocity_target(joint_vel_target)
        if joint_effort_target is not None:
            art.set_joint_effort_target(joint_effort_target)
        art.write_data_to_sim()
        self._sim.step(render)
        art.update(self._dt)

    def step_multi(self, joint_pos_targets: list[torch.Tensor] | None = None,
                   render: bool = False):
        """전 그룹 드라이브 목표 기록 + 물리 1스텝 (다중 그룹 스폰용).

        joint_pos_targets: 그룹별 (M, DoF_g) 위치 목표 목록 (None = 전 그룹
        직전 목표 유지 — RL decimation 사이 물리 스텝). 그룹별 DoF가 달라
        전역 텐서 하나로 못 묶으므로 목록으로 받는다.
        """
        for g, art in enumerate(self._groups):
            if joint_pos_targets is not None:
                art.set_joint_position_target(joint_pos_targets[g])
            art.write_data_to_sim()
        self._sim.step(render)
        for art in self._groups:
            art.update(self._dt)

    def hold_step(self, render: bool = False):
        """기립 자세 홀드로 1스텝 진행한다 (제어기 이전의 시뮬 실행 체크용).

        position 드라이브는 기립 자세, velocity 드라이브(바퀴)는 속도 0을
        목표로 유지한다 (passive는 게인 0이라 목표 무관).
        """
        art = self.robot
        self.step(joint_pos_target=art.data.default_joint_pos,
                  joint_vel_target=torch.zeros_like(art.data.default_joint_vel),
                  render=render)

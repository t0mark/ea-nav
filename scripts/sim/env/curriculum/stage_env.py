"""단계형 커리큘럼 지형 - stage 번호 -> 그 단계 난이도의 지형 cfg.

"언제 다음 단계로 넘어갈지"(수렴/정체 판정)는 scripts/.../rl/rl_trainer.py의 StageTrainer가 정하고,
이 파일은 "stage N의 지형이 어떻게 생겼는지"만 만든다.

Rudin et al.(CoRL 2021/2022, arXiv:2109.11978)의 지형 유형(계단·경사)에 Extreme Parkour(ICRA 2024,
chengxuxin/extreme-parkour)의 parkour_step(단일 단차 + 평지 회복 반복)을 더해 5종을 동일 비율로 섞는다.

Rudin의 "게임식 커리큘럼"은 이동 거리 기반으로 난이도(row)를 자동 승급·강등하는데, 이 판정은
회전 위주 명령에서 체계적으로 실패한다 - 직선 이동 거리로 재므로, 제자리 회전이 섞인 명령은
정책이 완벽히 추종해도 원점 근처로 돌아와 강등 조건에 걸린다. 그래서 ETH-PBL/elmap-rl-controller
(Plozza et al., ICRA 2025, arXiv:2505.12537)처럼 한 단계 안에서는 난이도 폭을 좁게 고정하고,
추종 정확도로 다음 단계 전환을 판정한다(커리큘럼 루프가 자동으로).

stage=0은 완전 평지(제대로 서기 + 평지 주행 겸용, 낮은 속도 명령도 범위 안에 있어 별도 단계가
필요 없다). stage>=1부터 아래 5종 지형이 그 단계의 난이도 구간으로 생성된다.

난이도 구간의 기준값·증가폭은 preset yaml의 curriculum.stage 딕셔너리에서 온다(step_height_base_m,
step_height_increment_m, slope_base_ratio, slope_increment_ratio) - stage 1은 [base, base+inc],
stage 2는 [base+inc, base+2*inc] 식으로 선형 증가한다.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg

from scripts.sim.env.terrain.parkour import ParkourStepTerrainCfg

_TERRAIN_PROPORTION = 0.2  # 5종 동일 비율


def _stage_step_height_range(stage: int, stage_cfg: dict) -> tuple[float, float]:
    """stage번째 계단/단차 난이도 구간(m)."""
    base = stage_cfg["step_height_base_m"]
    inc = stage_cfg["step_height_increment_m"]
    low = base + (stage - 1) * inc
    return (low, low + inc)


def _stage_slope_range(stage: int, stage_cfg: dict) -> tuple[float, float]:
    """stage번째 경사 난이도 구간(수평 대비 수직 비율, tan(각도))."""
    base = stage_cfg["slope_base_ratio"]
    inc = stage_cfg["slope_increment_ratio"]
    low = base + (stage - 1) * inc
    return (low, low + inc)


def _build_flat_terrain_importer_cfg() -> TerrainImporterCfg:
    """stage=0용 완전 평지."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply", static_friction=1.0, dynamic_friction=1.0
        ),
        debug_vis=False,
    )


def _build_stage_terrain_generator_cfg(stage: int, stage_cfg: dict) -> TerrainGeneratorCfg:
    """stage>=1용 5종 지형(계단 오름·내림, 경사 오름·내림, parkour_step) - 그 단계 난이도로 고정."""
    step_height_range = _stage_step_height_range(stage, stage_cfg)
    slope_range = _stage_slope_range(stage, stage_cfg)
    return TerrainGeneratorCfg(
        seed=42,
        curriculum=False,  # 단계 내부에서는 난이도를 안 바꾼다 - 단계 전환은 커리큘럼 루프가 stage 번호로 한다
        size=(8.0, 8.0),
        border_width=10.0,
        num_rows=1,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        color_scheme="height",
        sub_terrains={
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=_TERRAIN_PROPORTION,
                step_height_range=step_height_range,
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=_TERRAIN_PROPORTION,
                step_height_range=step_height_range,
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=_TERRAIN_PROPORTION, slope_range=slope_range, platform_width=2.0, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=_TERRAIN_PROPORTION, slope_range=slope_range, platform_width=2.0, border_width=0.25
            ),
            "parkour_step": ParkourStepTerrainCfg(
                proportion=_TERRAIN_PROPORTION,
                step_height_range=step_height_range,
                step_length=0.4,
                platform_length=1.6,
                num_steps=3,
            ),
        },
    )


def build_stage_terrain_importer_cfg(stage: int, stage_cfg: dict) -> TerrainImporterCfg:
    """stage 번호로 학습 지형을 만든다 - stage=0은 평지, stage>=1은 그 단계 난이도의 5종 지형."""
    if stage == 0:
        return _build_flat_terrain_importer_cfg()
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=_build_stage_terrain_generator_cfg(stage, stage_cfg),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply", static_friction=1.0, dynamic_friction=1.0
        ),
        debug_vis=False,
    )


def stage_terrain_summary(stage: int, stage_cfg: dict) -> dict:
    """stage의 난이도 구간을 사람이 읽을 수 있는 dict로 - 커리큘럼 결과 메타데이터에 그대로 넣는다."""
    if stage == 0:
        return {"stage": 0, "terrain": "flat"}
    sh = _stage_step_height_range(stage, stage_cfg)
    sl = _stage_slope_range(stage, stage_cfg)
    return {
        "stage": stage,
        "step_height_range_m": [round(sh[0], 4), round(sh[1], 4)],
        "slope_range": [round(sl[0], 4), round(sl[1], 4)],
        "max_step_height_m": round(sh[1], 4),
        "max_slope_ratio": round(sl[1], 4),
        "max_slope_deg": round(math.degrees(math.atan(sl[1])), 2),
    }

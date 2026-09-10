"""단계형 커리큘럼 지형 - 축별 레벨 -> 그 난이도의 지형 cfg.

"언제 난이도를 올릴지"(수렴/정체 판정)는 scripts/.../rl/rl_trainer.py의 CurriculumTrainer가 정하고,
이 파일은 "이 레벨의 지형이 어떻게 생겼는지"와 "어느 타일이 어느 축인지"만 만든다.

Rudin et al.(CoRL 2021/2022, arXiv:2109.11978)의 지형 유형(계단·경사·랜덤 굴곡)에 Extreme
Parkour(ICRA 2024, chengxuxin/extreme-parkour)의 parkour_step(단일 단차 + 평지 회복 반복)을 더해
6종을 섞는다.

난이도 축은 오르막 단차·내리막 단차·오르막 경사·내리막 경사·파쿠르 단차 다섯이고 각자 레벨을
가진다. 오르내림이 갈리는 것은 지형 종류 자체가 갈려 있기 때문이다 - Isaac Lab의 서브지형은 스폰
지점이 타일 중심인데, pyramid_stairs는 중심이 꼭대기라(origin_z = +(num_steps+1)*step_height) 밖으로
나가는 모든 방향이 내리막이고, inverted 쪽은 중심이 구덩이 바닥이라 오르막이다. 경사면도 같은
구조다(inverted가 기울기 부호를 뒤집는다). 그래서 오르내림은 지형을 새로 만들지 않고도 열만 나누면
분리된다.

방향을 네 축으로 나누는 이유는 로봇의 한계가 방향마다 다르기 때문이다. 축이 둘이면 "단차 0.185m
통과"가 오르막 0.25m와 내리막 0.12m의 평균일 수 있는데, 그 둘은 서로 다른 능력이고 결과에 따로 남아야
한다. 파쿠르 단차가 다섯 번째 축인 것도 같은 이유다 - 고립된 블록을 넘는 능력은 연속 계단을 오르는
능력과 다르므로, 한 축에 섞으면 한쪽의 실패가 다른 쪽의 승급을 막는다. 지형에는 다섯 축이 항상 함께
들어가되(학습 분포를 쪼개지 않는다) 승급은 축마다 따로 판정하므로, 어디서 막혔는지가 그대로 결과가
된다.

축별 판정을 하려면 "어느 env가 어느 축의 타일 위에 있는가"를 알아야 한다. TerrainGenerator는
curriculum=True일 때만 서브지형을 열 단위로 결정적으로 배치하므로(비율 누적합으로 열 인덱스를
가른다) 그 모드를 쓰고, 같은 식을 여기서 재현해 열->축 표를 만든다. 행은 난이도 구간 안의 세부
난이도라 그대로 둔다(TerrainImporter가 env를 행에 무작위 배정한다).

비율은 방향 네 축이 같은 열 수(21열 중 4열씩)를 갖도록 잡았다 - 축마다 표본 수가 다르면 표본이 적은
축의 점수가 더 크게 흔들려 축끼리 비교가 성립하지 않는다. 파쿠르 축이 2열, 어느 축도 아닌
random_rough가 2열, 레벨 0 전용 평지가 1열을 쓴다.

Rudin의 "게임식 커리큘럼"은 이동 거리 기반으로 난이도(row)를 자동 승급·강등하는데, 이 판정은
회전 위주 명령에서 체계적으로 실패한다 - 직선 이동 거리로 재므로, 제자리 회전이 섞인 명령은
정책이 완벽히 추종해도 원점 근처로 돌아와 강등 조건에 걸린다. 그래서 ETH-PBL/elmap-rl-controller
(Plozza et al., ICRA 2025, arXiv:2505.12537)처럼 한 레벨 안에서는 난이도 폭을 좁게 고정하고,
추종 정확도로 다음 레벨 전환을 판정한다(커리큘럼 루프가 자동으로).

난이도 구간의 기준값·증가폭은 preset yaml의 curriculum.stage 딕셔너리에서 온다 - 레벨 1이
[base, base+increment], 레벨 2가 [base+increment, base+2*increment] 식으로 선형 증가한다. 오르막과
내리막은 같은 사다리를 쓰고 지형 모양만 반대다 - 같은 눈금으로 재야 두 방향의 한계를 비교할 수 있다.
"""

from __future__ import annotations

import math

import numpy as np

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg

from scripts.sim.env.terrain.parkour import ParkourStepTerrainCfg

# 난이도 축 - 계단과 경사면을 각각 오르막/내리막으로 나눈 넷에, 파쿠르 단차를 따로 둔 다섯.
# rough는 어느 축도 아닌 바닥 거칠기, flat은 레벨 0 전용 평지라 둘 다 축이 아니다.
DIFFICULTY_AXES = ("step_up", "step_down", "slope_up", "slope_down", "parkour")
# 오르내림을 가리는 네 축 - 로봇의 주파 한계를 "방향을 가리지 않는 값"으로 요약할 때 이 넷만 쓴다.
# parkour는 방향이 아니라 장애물 종류라 그 요약에 섞이면 안 된다.
DIRECTIONAL_AXES = ("step_up", "step_down", "slope_up", "slope_down")
NEUTRAL_AXIS = "rough"
FLAT_AXIS = "flat"

# 축 -> 그 축의 난이도를 정하는 파라미터 사다리. 오르막·내리막은 같은 사다리를 공유한다.
_AXIS_DIFFICULTY_KIND = {
    "step_up": "step",
    "step_down": "step",
    "slope_up": "slope",
    "slope_down": "slope",
    "parkour": "step",
}

# 서브지형 이름 -> (열 수, 축). 열 수를 그대로 비율로 넘긴다(TerrainGenerator가 정규화한 뒤 누적합으로
# 열을 가르므로, 합이 num_cols와 같으면 각 종류가 정확히 그 수만큼의 열을 차지한다).
# flat은 레벨 0 전용 평지 열이다 - 랜덤 정책은 메시 지형 위에서 곧바로 걸음을 배우지 못하고 "빨리
# 넘어져 벌점 누적을 끊는" 국소 최적에 먼저 갇히므로, 평지에서 보행을 세운 뒤 지형으로 올려보낸다.
# 한 축에 한 종류의 지형만 넣는다. 축의 점수는 그 축 열들의 평균이고 승급은 그 평균으로 판정하므로,
# 난이도가 다른 지형을 한 축에 섞으면 쉬운 지형을 이미 통과한 축이 어려운 지형 때문에 영영 승급하지
# 못한다. parkour_step은 고립된 블록을 넘는 과제라 연속 계단과 난이도가 다르고, 그래서 오르막 계단이
# 아니라 자기 축을 갖는다.
_SUB_TERRAINS = (
    ("flat", 1, FLAT_AXIS),
    ("pyramid_stairs", 4, "step_down"),
    ("pyramid_stairs_inv", 4, "step_up"),
    ("parkour_step", 2, "parkour"),
    ("hf_pyramid_slope", 4, "slope_down"),
    ("hf_pyramid_slope_inv", 4, "slope_up"),
    ("random_rough", 2, NEUTRAL_AXIS),
)
# 난이도 레벨 수 = 지형의 행 수. Isaac Lab은 difficulty=(row+eta)/num_rows를 서브지형 파라미터
# 구간에 선형 보간하므로(terrain_generator.py 주석, mesh_terrains.py의 step_height 계산), 구간을
# base - base+NUM_LEVELS*increment로 잡으면 행 r이 기존 "레벨 r+1"과 정확히 같은 구간이 된다.
# 그래서 난이도 사다리를 바꾸지 않고도 전 레벨을 한 지형에 담을 수 있다.
NUM_LEVELS = 20
_NUM_COLS = sum(columns for _, columns, _ in _SUB_TERRAINS)
# 계단 디딤판의 진행 방향 폭(m). 난이도는 단차 높이만 올리므로, 폭이 좁으면 상위 레벨이 계단이 아니라
# 사다리가 된다 - 폭 0.3m에서는 단차 0.26m가 41도, 0.31m가 46도로 실제 건축 계단의 상한(약 32도)을
# 넘는다. 0.5m로 두면 로봇들이 실제로 도달하는 구간이 전부 실제 계단 범위에 들어간다
# (단차 0.185m -> 20도, 0.235m -> 25도, 0.31m -> 32도). 그래야 레벨이 "얼마나 가파른가"가 아니라
# "얼마나 높은 단차인가"를 잰다.
_STEP_WIDTH = 0.5
# 타일 중심의 평지 구간 폭(m) - 스폰 지점에서 지형까지의 거리를 정한다. 승급 판정은 리셋 직후
# 짧은 롤아웃으로 재는데, 이 구간이 그 롤아웃의 이동거리보다 넓으면 로봇이 계단·경사에 닿기도 전에
# 채점이 끝나 "평지 보행 점수"로 승급하게 된다. 명령 속도 하한(preset의 lin_vel_max_floor_mps)에서도
# 평가 창 안에 지형에 닿도록 반폭을 그 이동거리보다 좁게 잡는다.
_STAIRS_PLATFORM_WIDTH = 2.0
_SLOPE_PLATFORM_WIDTH = 1.5


def _stage_range(level: int, base: float, increment: float) -> tuple[float, float]:
    """level번째 난이도 구간 - level 1이 [base, base+increment]인 선형 사다리(level 0은 평지라 제외)."""
    low = base + (level - 1) * increment
    return (low, low + increment)


def _step_height_range(level: int, stage_cfg: dict) -> tuple[float, float]:
    """단차 축 level의 계단 높이 구간(m)."""
    return _stage_range(level, stage_cfg["step_height_base_m"], stage_cfg["step_height_increment_m"])


def _slope_range(level: int, stage_cfg: dict) -> tuple[float, float]:
    """경사 축 level의 기울기 구간(수평 대비 수직 비율, tan(각도))."""
    return _stage_range(level, stage_cfg["slope_base_ratio"], stage_cfg["slope_increment_ratio"])


def _noise_range(level: int, stage_cfg: dict) -> tuple[float, float]:
    """랜덤 굴곡 높이 구간(m) - 어느 축에도 속하지 않는 바닥 거칠기라 전 축의 최대 레벨을 따른다."""
    return _stage_range(level, stage_cfg["noise_base_m"], stage_cfg["noise_increment_m"])


def _full_ladder_range(base: float, increment: float) -> tuple[float, float]:
    """전 레벨을 덮는 파라미터 구간 - 행 r이 [base+r*inc, base+(r+1)*inc]를 담당하게 된다."""
    return (base, base + NUM_LEVELS * increment)


def terrain_column_axes() -> list[str]:
    """열 인덱스 -> 그 열이 속한 축. TerrainGenerator._generate_curriculum_terrains의 배치식을 그대로 재현한다.

    env가 어느 열에 있는지는 TerrainImporter.terrain_types가 알려주므로, 이 표와 합치면 env마다
    "지금 밟고 있는 지형이 어느 축인지"가 정해진다. 승급 판정을 축별로 나누는 근거가 이 표다.
    """
    proportions = np.array([c for _, c, _ in _SUB_TERRAINS], dtype=float)
    proportions /= proportions.sum()
    cumulative = np.cumsum(proportions)
    axes = []
    for column in range(_NUM_COLS):
        index = int(np.min(np.where(column / _NUM_COLS + 0.001 < cumulative)[0]))
        axes.append(_SUB_TERRAINS[index][2])
    return axes


def flat_terrain_column() -> int:
    """레벨 0(평지) env를 세울 열. 평면은 행에 상관없이 평지라 행은 아무 값이나 써도 된다."""
    return terrain_column_axes().index(FLAT_AXIS)


def _build_flat_terrain_importer_cfg() -> TerrainImporterCfg:
    """모든 축이 레벨 0일 때 쓰는 완전 평지."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply", static_friction=1.0, dynamic_friction=1.0
        ),
        debug_vis=False,
    )


def _build_terrain_generator_cfg(stage_cfg: dict) -> TerrainGeneratorCfg:
    """전 난이도 레벨을 한 번에 담은 6종 지형 - 행이 난이도, 열이 지형 종류다.

    행마다 난이도가 갈리므로(curriculum=True) 행 r의 계단 높이는 [base+r*inc, base+(r+1)*inc]가
    되어 기존 "레벨 r+1" 구간과 같다. 승급은 지형을 다시 만드는 대신 env를 다음 행으로 옮겨서
    한다(TerrainImporter.terrain_levels) - Isaac Lab의 terrain_levels_vel 커리큘럼과 같은 방식이다.

    오르막·내리막은 같은 파라미터 구간을 쓴다. 지형 종류가 스폰 지점의 높낮이를 뒤집으므로,
    같은 step_height_range로도 한쪽은 내려가는 계단이 되고 다른 쪽은 올라가는 계단이 된다.
    """
    step_height_range = _full_ladder_range(stage_cfg["step_height_base_m"], stage_cfg["step_height_increment_m"])
    slope_range = _full_ladder_range(stage_cfg["slope_base_ratio"], stage_cfg["slope_increment_ratio"])
    noise_range = _full_ladder_range(stage_cfg["noise_base_m"], stage_cfg["noise_increment_m"])
    # TerrainGeneratorCfg는 proportion을 float으로 정규화하므로(int 배열이면 in-place 나눗셈이 캐스팅
    # 에러를 낸다) 열 수를 float으로 바꿔 넘긴다
    proportions = {name: float(columns) for name, columns, _ in _SUB_TERRAINS}
    return TerrainGeneratorCfg(
        seed=42,
        # 열 단위로 서브지형을 결정적으로 배치하려면 이 모드여야 한다(축별 판정의 전제).
        # 행은 난이도 순으로 배치되므로 행 인덱스가 곧 레벨이 된다.
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        size=(8.0, 8.0),
        border_width=10.0,
        num_rows=NUM_LEVELS,
        num_cols=_NUM_COLS,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        color_scheme="height",
        sub_terrains={
            # 레벨 0 전용 - 여기서 평지 보행을 세운 뒤 나머지 열로 올려보낸다
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=proportions["flat"]),
            # 중심이 꼭대기라 밖으로 나가는 모든 방향이 내리막 계단이다
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=proportions["pyramid_stairs"],
                step_height_range=step_height_range,
                step_width=_STEP_WIDTH,
                platform_width=_STAIRS_PLATFORM_WIDTH,
                border_width=1.0,
                holes=False,
            ),
            # 중심이 구덩이 바닥이라 밖으로 나가는 모든 방향이 오르막 계단이다
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=proportions["pyramid_stairs_inv"],
                step_height_range=step_height_range,
                step_width=_STEP_WIDTH,
                platform_width=_STAIRS_PLATFORM_WIDTH,
                border_width=1.0,
                holes=False,
            ),
            "parkour_step": ParkourStepTerrainCfg(
                proportion=proportions["parkour_step"],
                step_height_range=step_height_range,
                step_length=0.4,
                platform_length=1.6,
                num_steps=3,
            ),
            # 중심이 봉우리라 내리막 경사, inverted는 분지 바닥이라 오르막 경사다
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=proportions["hf_pyramid_slope"], slope_range=slope_range, platform_width=_SLOPE_PLATFORM_WIDTH, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=proportions["hf_pyramid_slope_inv"], slope_range=slope_range, platform_width=_SLOPE_PLATFORM_WIDTH, border_width=0.25
            ),
            # 계단·경사가 없는 완만한 굴곡 - 어느 레벨에서도 height scan이 상수가 되지 않게 한다
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=proportions["random_rough"],
                noise_range=noise_range,
                noise_step=stage_cfg["noise_step_m"],
                border_width=0.25,
            ),
        },
    )


def build_curriculum_terrain_importer_cfg(stage_cfg: dict) -> TerrainImporterCfg:
    """커리큘럼 전 구간을 담은 학습 지형 - 한 번만 만들고 승급은 env를 옮겨서 한다."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=_build_terrain_generator_cfg(stage_cfg),
        # 처음에는 모든 env를 가장 쉬운 행에 둔다 - 승급은 CurriculumTrainer가 축별로 올린다
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply", static_friction=1.0, dynamic_friction=1.0
        ),
        debug_vis=False,
    )


def build_flat_terrain_importer_cfg() -> TerrainImporterCfg:
    """끝없는 평면 - 구동 확인처럼 지형 굴곡이 필요 없을 때 쓴다."""
    return _build_flat_terrain_importer_cfg()


def stage_command_speed_limits(level: int, command_cfg: dict) -> tuple[float, float]:
    """level에서 쓸 (선속도 상한 m/s, 기준 대비 배율) - 레벨이 오를수록 낮아지고 하한에서 멈춘다."""
    base = command_cfg["lin_vel_max_base_mps"]
    limit = max(command_cfg["lin_vel_max_floor_mps"], base - level * command_cfg["lin_vel_decay_per_stage_mps"])
    return limit, limit / base


def _axis_difficulty(axis: str, level: int, stage_cfg: dict) -> dict:
    """축 하나의 레벨을 물리 단위로 푼다 - 레벨 0은 그 축에서 평지도 못 벗어났다는 뜻이다."""
    if _AXIS_DIFFICULTY_KIND[axis] == "step":
        low, high = _step_height_range(max(level, 1), stage_cfg)
        return {
            "level": level,
            "step_height_range_m": [round(low, 4), round(high, 4)],
            "max_step_height_m": round(high, 4) if level else 0.0,
        }
    low, high = _slope_range(max(level, 1), stage_cfg)
    return {
        "level": level,
        "slope_range": [round(low, 4), round(high, 4)],
        "max_slope_ratio": round(high, 4) if level else 0.0,
        "max_slope_deg": round(math.degrees(math.atan(high)), 2) if level else 0.0,
    }


def stage_terrain_summary(levels: dict, stage_cfg: dict) -> dict:
    """축별 레벨의 난이도를 사람이 읽을 수 있는 dict로 - 커리큘럼 결과 메타데이터에 그대로 넣는다."""
    highest_level = max((int(levels.get(axis, 0)) for axis in DIFFICULTY_AXES), default=0)
    noise_low, noise_high = _noise_range(max(highest_level, 1), stage_cfg)
    return {
        "axes": {axis: _axis_difficulty(axis, int(levels.get(axis, 0)), stage_cfg) for axis in DIFFICULTY_AXES},
        "noise_range_m": [round(noise_low, 4), round(noise_high, 4)] if highest_level else [0.0, 0.0],
    }

"""파쿠르 스타일(단일 장애물 + 평지 회복) 커스텀 서브 지형 타입 모음.

Extreme Parkour(ICRA 2024, chengxuxin/extreme-parkour)의 지형 생성 방식을 참고해, 연속
계단/경사 대신 로봇이 한 번에 넘을 수 있는 최대 단차를 직접 겨냥해 탐색하기 위한 지형이다.
"""

from .hurdle_terrain import ParkourHurdleTerrainCfg, parkour_hurdle_terrain
from .step_terrain import ParkourStepTerrainCfg, parkour_step_terrain

# generator_cfg.py의 _SUB_TERRAIN_TYPES에 병합되는 yaml type 문자열 -> Cfg 클래스 매핑
PARKOUR_SUB_TERRAIN_TYPES: dict[str, type] = {
    "parkour_step": ParkourStepTerrainCfg,
    "parkour_hurdle": ParkourHurdleTerrainCfg,
}

__all__ = [
    "PARKOUR_SUB_TERRAIN_TYPES",
    "ParkourHurdleTerrainCfg",
    "ParkourStepTerrainCfg",
    "parkour_hurdle_terrain",
    "parkour_step_terrain",
]

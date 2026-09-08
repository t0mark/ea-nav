"""파쿠르 스타일(단일 장애물 + 평지 회복) 커스텀 서브 지형 타입.

Extreme Parkour(ICRA 2024, chengxuxin/extreme-parkour)의 지형 생성 방식을 참고해, 연속 계단/경사
대신 로봇이 한 번에 넘을 수 있는 최대 단차를 직접 겨냥해 탐색하기 위한 지형이다.
"""

from .step_terrain import ParkourStepTerrainCfg, parkour_step_terrain

__all__ = ["ParkourStepTerrainCfg", "parkour_step_terrain"]

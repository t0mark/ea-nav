"""목표 지점(goal) 하나를 씬에 시각화하는 유틸리티.

전체 웨이포인트를 한 번에 늘어놓지 않고, "지금 로봇이 향하고 있는 목표" 하나만 보여준다 - 경로
추종 중에는 목표가 계속 바뀌므로, 마커를 매번 새로 스폰하지 않고 하나만 만든 뒤 목표가 바뀔 때마다
그 위치로 옮겨서 재사용한다(여러 번 호출해 위치만 갱신).

pxr는 Kit 프로세스가 뜬 뒤에만 임포트할 수 있으므로, 이 모듈의 함수는 AppLauncher 부팅이 끝난
tools/ 진입점에서만 호출해야 한다(capture.py와 동일한 전제).
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def spawn_goal_marker(prim_path: str = "/World/GoalMarker") -> VisualizationMarkers:
    """목표 지점 마커 하나를 만든다 - 이후 update_goal_marker()로 위치만 계속 바꿔가며 재사용한다."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            )
        },
    )
    return VisualizationMarkers(marker_cfg)


def update_goal_marker(marker: VisualizationMarkers, goal_xy: tuple[float, float], height: float = 0.05) -> None:
    """마커를 목표 지점(x, y) 위치로 옮긴다. 지면 바로 위로 살짝 띄워 바닥과 겹쳐 안 보이는 걸 막는다."""
    translation = torch.tensor([[goal_xy[0], goal_xy[1], height]])
    marker.visualize(translations=translation)

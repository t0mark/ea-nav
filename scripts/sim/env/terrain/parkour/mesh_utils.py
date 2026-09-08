"""파쿠르 지형(단차)이 쓰는 trimesh 생성 헬퍼."""

from __future__ import annotations

import trimesh

from isaaclab.terrains.trimesh.utils import make_plane


def make_ground_plane(size: tuple[float, float]) -> trimesh.Trimesh:
    """타일 전체(0,0)~(size[0], size[1])를 덮는 z=0 평면 지면 메시를 만든다."""
    return make_plane(size, height=0.0, center_zero=False)


def make_full_width_obstacles(
    size: tuple[float, float],
    obstacle_height: float,
    obstacle_length: float,
    gap_length: float,
    num_obstacles: int,
) -> list[trimesh.Trimesh]:
    """진행 방향(x)을 따라 폭(y) 전체를 가로막는 박스 장애물을 일정 간격으로 배치한다.

    장애물 사이를 gap_length만큼 평지로 비워 두어 "장애물 하나 - 평지 회복 - 장애물 하나"
    구조를 만든다. 연속 계단과 달리 매 장애물마다 로봇이 자세를 회복할 여지를 주는 구성으로,
    Extreme Parkour(ICRA 2024, chengxuxin/extreme-parkour)의 parkour_step 지형 생성 방식을 참고했다.
    """
    meshes: list[trimesh.Trimesh] = []
    span = obstacle_length + gap_length
    start_x = gap_length / 2.0  # 첫 장애물 앞에도 회복 구간을 둔다
    for index in range(num_obstacles):
        center_x = start_x + index * span + obstacle_length / 2.0
        box_dims = (obstacle_length, size[1], obstacle_height)
        box_pos = (center_x, size[1] / 2.0, obstacle_height / 2.0)
        meshes.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))
    return meshes

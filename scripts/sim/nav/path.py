"""2D 웨이포인트 경로 표현과 경로 생성기.

Isaac Lab/torch에 의존하지 않는 순수 파이썬이다 - "다음 목표가 뭐고 도달했는지"만 관리하는
상태 머신이라 어떤 시뮬레이터·프레임워크에서도 그대로 재사용할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Path2D:
    """평지 위 2D 웨이포인트 목록 - 도달하면 자동으로 다음 목표로 넘어가는 상태 머신."""

    waypoints: list[tuple[float, float]]
    # 실제 로봇은 관성·마찰 때문에 이상적인 유니사이클처럼 목표를 정확히 지나가지 못하고 살짝
    # 비껴가며 돈다 - 0.3m는 그 오차를 흡수할 도달 판정 여유다
    reach_radius: float = 0.3

    def __post_init__(self) -> None:
        """추종 진행 상태(현재 목표 인덱스, 완주 여부)를 초기화한다."""
        self._current_index = 0
        self._finished = False

    @property
    def current_goal(self) -> tuple[float, float]:
        """지금 추종해야 할 목표 웨이포인트 (완주했으면 마지막 웨이포인트를 계속 반환)."""
        return self.waypoints[self._current_index]

    @property
    def is_finished(self) -> bool:
        """마지막 웨이포인트까지 전부 도달했는지."""
        return self._finished

    def update(self, current_position: tuple[float, float]) -> bool:
        """현재 위치가 목표에 도달했으면 다음 웨이포인트로 넘어간다. 목표가 바뀌면 True를 반환한다."""
        if self._finished:
            return False
        goal_x, goal_y = self.current_goal
        distance = ((current_position[0] - goal_x) ** 2 + (current_position[1] - goal_y) ** 2) ** 0.5
        if distance >= self.reach_radius:
            return False
        if self._current_index == len(self.waypoints) - 1:
            self._finished = True
            return False
        self._current_index += 1
        return True


def generate_straight_path(goal: tuple[float, float]) -> Path2D:
    """목적지 하나로 바로 가는 직선 경로 - 웨이포인트 1개(목적지 자체)."""
    return Path2D(waypoints=[goal])


def generate_l_shaped_path(start: tuple[float, float], leg_length: float) -> Path2D:
    """start에서 +x로 leg_length만큼 간 뒤, 그 지점에서 +y로 leg_length만큼 가는 ㄱ자 경로.

    직진만으로는 조향(steering)이 전혀 검증되지 않으므로, 중간에 90도 꺾이는 코너를 웨이포인트로
    넣어 조향 컨트롤러가 실제로 방향을 바꾸는지 확인하는 용도다.
    """
    corner = (start[0] + leg_length, start[1])
    end = (corner[0], corner[1] + leg_length)
    return Path2D(waypoints=[corner, end])

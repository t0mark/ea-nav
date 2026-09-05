"""웨이포인트 경로를 (선속도, 각속도) 유니사이클 명령으로 바꾸는 경로 추종기.

Isaac Lab/torch에 의존하지 않는 순수 파이썬이다 - 호출부가 로봇의 pose를 순수 숫자(position, yaw)로
넘겨주고, 반환값도 순수 숫자라 어떤 컨트롤러 command 텐서로 바꿀지는 호출부가 결정한다.
"""

from __future__ import annotations

import math

from scripts.sim.nav.path import Path2D


class PurePursuitTracker:
    """목표 웨이포인트 방향으로 정렬될 때까지 최대 각속도로 도는 bang-bang 방식 추종기.

    선속도는 고정값을 쓴다(저속 지상 로봇 조향 검증용이라 속도 프로파일까지는 필요 없음). 웨이포인트
    간 구간이 짧고 직선이라, 원래 pure pursuit의 lookahead 원-경로 교차 계산 없이 "지금 목표
    웨이포인트 하나"를 향한 헤딩 정렬만으로 동일한 추종 결과를 낸다.

    heading error에 비례해 각속도를 줄이는 P 제어 대신 bang-bang(정렬 전엔 항상 최대 각속도, 정렬되면
    0)을 쓴다 - 접지 마찰이 큰 로봇은 좌우 바퀴 속도차가 어떤 임계값을 넘어야 실제 회전이 나오는
    비선형 구간을 가진다. P 제어는 정렬에 가까워질수록 명령을 줄이는데, 그러면 그 임계값 아래로
    떨어져 회전이 멈추고 코너를 완주하지 못한다. 정렬될 때까지 항상 검증된 최대 각속도를 유지해야
    이 임계값을 계속 넘을 수 있다.
    """

    def __init__(
        self, linear_velocity: float = 0.5, max_angular_velocity: float = 4.5, heading_deadband: float = 0.05
    ) -> None:
        """직진 속도, 최대 각속도, "이 안이면 정렬됐다고 보는" 헤딩 오차 데드밴드를 저장한다."""
        self._linear_velocity = linear_velocity
        self._max_angular_velocity = max_angular_velocity
        self._heading_deadband = heading_deadband

    def compute_command(self, position: tuple[float, float], yaw: float, path: Path2D) -> tuple[float, float]:
        """현재 pose와 path.current_goal로 (선속도, 각속도)를 계산한다. 경로를 마쳤으면 정지(0, 0)."""
        path.update(position)
        if path.is_finished:
            return 0.0, 0.0

        goal_x, goal_y = path.current_goal
        target_yaw = math.atan2(goal_y - position[1], goal_x - position[0])
        heading_error = self._wrap_to_pi(target_yaw - yaw)

        if abs(heading_error) < self._heading_deadband:
            angular_velocity = 0.0
        else:
            angular_velocity = math.copysign(self._max_angular_velocity, heading_error)
        return self._linear_velocity, angular_velocity

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        """각도를 [-pi, pi] 범위로 접어준다 - 헤딩 오차가 항상 최단 회전 방향을 가리키게 한다."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

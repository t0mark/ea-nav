"""차동 구동(differential drive) 로봇의 기구학 계산."""

from __future__ import annotations

import torch

from scripts.sim.controller.base import RobotController


class DifferentialDriveKinematics(RobotController):
    """좌우 바퀴 쌍으로 구동되는 로봇의 (선속도, 각속도) 명령 -> 바퀴 각속도 변환.

    좌우 바퀴가 여러 개(예: 6륜 스키드 스티어)여도 같은 쪽은 항상 같은 각속도로 돈다고 가정하고
    좌/우 그룹 전체에 동일 목표값을 복제한다.
    """

    def __init__(
        self,
        wheel_radius: float,
        wheel_base: float,
        num_left_wheels: int,
        num_right_wheels: int,
        direction_sign: float = 1.0,
        rotation_sign: float = 1.0,
    ) -> None:
        """바퀴 반지름(m), 좌우 바퀴 중심 간 거리(m), 좌/우 바퀴 개수를 저장한다.

        direction_sign(전진)과 rotation_sign(회전)을 따로 둔다 - USD에 authored된 바퀴 관절 회전축
        방향은 "양수 각속도 = 전진"과 "양수 각속도 차이 = 좌회전"을 독립적으로 어긋나게 만들 수 있다
        (관측됨: fraunhofer_evobot - direction_sign만 보정해도 회전은 여전히 반대 방향으로 나옴,
        w=+1을 줬는데 실제 yaw는 -0.91rad로 반대 방향 회전). 실측 기반으로 usd_export_config.py가
        둘을 따로 재서 config에 저장한다.
        """
        self._wheel_radius = wheel_radius
        self._wheel_base = wheel_base
        self._num_left_wheels = num_left_wheels
        self._num_right_wheels = num_right_wheels
        self._direction_sign = direction_sign
        self._rotation_sign = rotation_sign

    def reset(self) -> None:
        """기구학 계산은 내부 상태가 없어 아무 것도 하지 않는다."""

    def compute_joint_targets(
        self, command: torch.Tensor, observation: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """command = [선속도 v, 각속도 w] -> 좌/우 바퀴 각속도(velocity) 목표값.

        미분 구동 기구학(differential drive kinematics): 좌우 바퀴 접지선까지의 거리가 wheel_base/2일 때,
        로봇 중심의 선속도 v, 각속도 w는 순간 회전 중심을 기준으로 v_left = v - w*wheel_base/2,
        v_right = v + w*wheel_base/2로 분배되고, 접지 미끄러짐이 없다고 가정하면 바퀴 각속도는
        v_wheel = v_side / wheel_radius 로 변환된다. direction_sign은 v 항에, rotation_sign은 w 항에
        독립적으로 곱해 전진·회전 방향을 각각 보정한다.
        """
        linear_velocity, angular_velocity = command[0], command[1]
        translation = self._direction_sign * linear_velocity
        rotation = self._rotation_sign * angular_velocity * self._wheel_base / 2.0
        left_wheel_speed = (translation - rotation) / self._wheel_radius
        right_wheel_speed = (translation + rotation) / self._wheel_radius

        # 좌/우 각각 여러 바퀴가 있어도 같은 쪽은 전부 같은 각속도로 돈다고 보고 그대로 복제한다
        left_targets = left_wheel_speed.expand(self._num_left_wheels)
        right_targets = right_wheel_speed.expand(self._num_right_wheels)
        return {"velocity": torch.cat([left_targets, right_targets])}

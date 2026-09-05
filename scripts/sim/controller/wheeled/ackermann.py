"""애커먼 조향(ackermann steering) 로봇의 기구학 계산."""

from __future__ import annotations

import torch

from scripts.sim.controller.base import RobotController


class AckermannKinematics(RobotController):
    """자전거 모델(bicycle model) 기준 조향각을 좌우 앞바퀴 조향각으로 분배하고, 구동 바퀴 각속도를 계산한다.

    좌우 앞바퀴가 같은 각도로 꺾이면 안쪽 바퀴가 도는 원의 반지름이 더 작아 미끄러짐이 생기므로,
    애커먼 조향 기하는 두 바퀴가 공통 회전 중심을 공유하도록 좌우 조향각을 다르게 준다.
    """

    def __init__(
        self,
        wheelbase: float,
        track_width: float,
        wheel_radius: float,
        num_drive_wheels: int,
        direction_sign: float = 1.0,
        steering_sign: float = 1.0,
    ) -> None:
        """앞뒤 차축 간 거리(wheelbase), 좌우 바퀴 간 거리(track_width), 바퀴 반지름, 구동 바퀴 개수를 저장한다.

        direction_sign(구동)과 steering_sign(조향)을 따로 둔다 - USD에 authored된 구동 바퀴 축
        방향과 조향 관절 축 방향은 서로 다른 문제라 독립적으로 어긋날 수 있다(diff/omni에서도 이미
        확인된 패턴 - 전진 방향과 회전 방향의 부호가 서로 무관하게 반대로 나올 수 있었다). 조향각
        계산 자체(steering_angle)는 목표 곡률에서 나오는 순수 기하량이라 그대로 두고, 실제 관절에
        내보내는 최종 값(구동 속도·조향각)에만 각 부호를 곱해 보정한다.
        """
        self._wheelbase = wheelbase
        self._track_width = track_width
        self._wheel_radius = wheel_radius
        self._num_drive_wheels = num_drive_wheels
        self._direction_sign = direction_sign
        self._steering_sign = steering_sign

    def reset(self) -> None:
        """기구학 계산은 내부 상태가 없어 아무 것도 하지 않는다."""

    def compute_joint_targets(
        self, command: torch.Tensor, observation: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """command = [선속도 v, 각속도 w] -> 좌우 조향각(position) + 구동 바퀴 각속도(velocity).

        command를 다른 wheeled 컨트롤러(diff/omni)와 동일하게 각속도 w로 통일한다 - 그래야 경로
        추종기(pure pursuit) 같은 상위 레이어가 로봇 종류를 몰라도 동일한 (v, w)를 넘길 수 있다.
        자전거 모델에서 각속도 w = v * tan(delta) / wheelbase 이므로, 여기서 먼저 delta =
        atan(wheelbase * w / v)로 조향각을 역산한 뒤, 좌우 앞바퀴는 자전거 모델의 순간 회전 반지름
        R = wheelbase / tan(delta) (원 중심은 뒤 차축 연장선 위, delta>0을 좌회전으로 둔다)를 공유하도록
        atan(wheelbase / (R -+ track_width/2))로 조향각을 다시 나눈다 - 안쪽 바퀴가 더 크게, 바깥쪽
        바퀴가 더 작게 꺾여야 두 바퀴 모두 같은 중심을 도는 순수 구름이 된다.
        """
        linear_velocity, angular_velocity = command[0], command[1]

        # v가 0에 가까우면 delta = atan(wheelbase*w/v)가 발산하므로, 아주 작은 값으로 바닥을 깐다
        safe_linear_velocity = torch.where(
            torch.abs(linear_velocity) < 1e-3, torch.full_like(linear_velocity, 1e-3), linear_velocity
        )
        steering_angle = torch.atan(self._wheelbase * angular_velocity / safe_linear_velocity)

        # tan(delta)가 0에 가까우면 회전 반지름이 발산하므로 직진(steering=0) 케이스는 따로 처리한다
        straight_ahead = torch.abs(steering_angle) < 1e-6
        turning_radius = self._wheelbase / torch.tan(steering_angle.clamp(min=-1.5, max=1.5))
        left_steering_angle = torch.atan(self._wheelbase / (turning_radius - self._track_width / 2.0))
        right_steering_angle = torch.atan(self._wheelbase / (turning_radius + self._track_width / 2.0))
        left_steering_angle = torch.where(straight_ahead, torch.zeros_like(steering_angle), left_steering_angle)
        right_steering_angle = torch.where(straight_ahead, torch.zeros_like(steering_angle), right_steering_angle)

        # 구동 바퀴는 조향각과 무관하게 선속도만큼만 굴러가면 된다고 가정한다(전륜/후륜 구동 모두 공통)
        drive_wheel_speed = (self._direction_sign * linear_velocity / self._wheel_radius).expand(
            self._num_drive_wheels
        )
        return {
            "position": self._steering_sign * torch.stack([left_steering_angle, right_steering_angle]),
            "velocity": drive_wheel_speed,
        }

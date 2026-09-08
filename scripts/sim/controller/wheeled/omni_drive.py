"""전방향(mecanum omni-wheel) 구동 로봇의 기구학 계산."""

from __future__ import annotations

import torch

from scripts.sim.controller.base import RobotController


class OmniDriveKinematics(RobotController):
    """4륜 메카넘 휠(mecanum wheel, 롤러 45도) 배치를 가정한 전방향 구동 기구학.

    바퀴가 3개인 트라이크형 전방향 구동(예: pepper)이나 캐스터 기반 구동(예: pr2)은 이 기구학이
    맞지 않으므로 대상에서 제외한다 - 조향 방식 자체가 달라 별도 클래스가 필요하다.
    """

    def __init__(
        self,
        wheel_radius: float,
        half_wheelbase: float,
        half_track_width: float,
        direction_sign: float = 1.0,
        rotation_sign: float = 1.0,
    ) -> None:
        """바퀴 반지름과, 로봇 중심에서 앞/뒤 축까지·좌/우 바퀴까지의 절반 거리를 저장한다.

        direction_sign(전진/횡이동)과 rotation_sign(회전)은 따로 둔다 - 메카넘 롤러 배치(X자형/O자형)에
        따라 전진은 맞는데 회전 반응만 반대로 나올 수 있어, 둘을 하나의 부호로 묶으면 한쪽을
        맞추다 다른 쪽이 깨진다. usd_export_config.py가 각각 독립적으로 실측해 config에 저장한다.
        """
        self._wheel_radius = wheel_radius
        self._lateral_sum = half_wheelbase + half_track_width
        self._direction_sign = direction_sign
        self._rotation_sign = rotation_sign

    def reset(self) -> None:
        """기구학 계산은 내부 상태가 없어 아무 것도 하지 않는다."""

    def compute_joint_targets(
        self, command: torch.Tensor, observation: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """command = [vx, vy, wz](로봇 base 좌표계) -> [front_left, front_right, rear_left, rear_right] 바퀴 각속도.

        메카넘 휠은 롤러가 45도로 붙어 있어 구동력이 대각선 방향으로 나뉘는데, 이 역기구학은 그 대각선
        성분을 목표 vx, vy, wz로 역산해 4바퀴 각속도로 분배하는 표준식이다(Ref: KUKA youBot 계열 메카넘
        역기구학 - lateral_sum = half_wheelbase + half_track_width는 회전(wz) 성분이 각 바퀴에 실리는
        팔 길이).
        """
        vx, vy, wz = command[0], command[1], command[2]
        translation = self._direction_sign
        rotation = self._rotation_sign * self._lateral_sum
        front_left = translation * (vx - vy) - rotation * wz
        front_right = translation * (vx + vy) + rotation * wz
        rear_left = translation * (vx + vy) - rotation * wz
        rear_right = translation * (vx - vy) + rotation * wz

        # 대각선 성분(vx, vy, wz)을 바퀴 각속도로 바꾸는 마지막 단계는 접지 미끄러짐이 없다는 가정
        wheel_speeds = torch.stack([front_left, front_right, rear_left, rear_right]) / self._wheel_radius
        return {"velocity": wheel_speeds}

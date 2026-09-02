from __future__ import annotations

import torch

from .controller import WheeledControllerBase

class UnicycleController(WheeledControllerBase):
    """차동 구동(diff)과 스키드 스티어(skid) 공용 controller.

    두 타입은 tracker 출력 (v, w)를 body command (vx, 0, wz)로 펴고 공통 wheel matrix로
    휠 속도를 만드는 과정이 완전히 같다. 유일한 차이인 회전 실효 반경 보정은 wheel matrix의
    yaw 열 배율(ctrl.wheel_yaw_scale)로 표현되므로 클래스가 아니라 설정으로 구분한다.
    """

    MODEL = "unicycle"

    def _control_bounds(self) -> tuple[list, float]:
        """차동·스키드 구동의 입력 경계와 명목 선회반경을 반환한다."""

        return self._unicycle_bounds()

    def _to_body_cmd(self, u: torch.Tensor) -> torch.Tensor:
        """tracker 출력 (v, w)를 body command (vx, 0, wz)로 변환한다."""

        return torch.stack([u[:, 0], torch.zeros_like(u[:, 0]), u[:, 1]], dim=1)

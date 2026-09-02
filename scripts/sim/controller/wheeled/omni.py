from __future__ import annotations

import torch

from .controller import WheeledControllerBase

class OmniController(WheeledControllerBase):
    """omni/mecanum controller.

    생성기 규약상 base_tag가 omni면 항상 홀로노믹이라 tracker 출력 (vx, vy, wz)를 그대로
    body command로 쓴다. mecanum4의 45도 롤러 방향은 wheel matrix 부호에 이미 반영돼 있다.
    """

    MODEL = "holonomic"

    def _control_bounds(self) -> tuple[list, float]:
        """홀로노믹 구동의 입력 경계와 명목 선회반경을 반환한다."""

        r_turn = self._v_max / max(self._w_max, 1e-6)
        return ([[-self._v_max, self._v_max], [-self._v_max, self._v_max],
                 [-self._w_max, self._w_max]], r_turn)

    def _to_body_cmd(self, u: torch.Tensor) -> torch.Tensor:
        """tracker 출력 (vx, vy, wz)를 그대로 body command로 쓴다."""

        return u

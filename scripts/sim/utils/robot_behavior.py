from __future__ import annotations

import math

STATUS_PASS = "pass"
STATUS_FAIL = "fail"

def tilt_deg(quat_wxyz) -> float:

    zz = 1.0 - 2.0 * (float(quat_wxyz[1]) ** 2 + float(quat_wxyz[2]) ** 2)
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))

class StandingCriteria:

    def __init__(self, success_cfg: dict, meta: dict):

        form = meta.get("form", "")

        self._base_height = float(meta["metrics"]["base_height"])
        self._height_ratio = float(success_cfg["height_ratio_min"])
        by_form = success_cfg.get("tilt_max_deg_by_form", {}) or {}
        self._tilt_max = float(by_form.get(form, success_cfg["tilt_max_deg"]))

    @property
    def height_min(self) -> float:

        return self._height_ratio * self._base_height

    @property
    def tilt_max_deg(self) -> float:

        return self._tilt_max

    def evaluate(self, pos, quat) -> dict:

        finite = all(math.isfinite(float(v)) for v in list(pos) + list(quat))
        height = float(pos[2])
        tilt = tilt_deg(quat) if finite else float("nan")

        reasons = []
        if not finite:
            reasons.append("nonfinite")
        else:
            if height < self.height_min:
                reasons.append("fallen_height")
            if tilt > self._tilt_max:
                reasons.append("tilted")

        return {
            "status": STATUS_PASS if not reasons else STATUS_FAIL,
            "ok": not reasons,
            "final_height": height,
            "tilt_deg": tilt,
            "height_min": self.height_min,
            "tilt_max_deg": self._tilt_max,
            "fail_reasons": reasons,
        }

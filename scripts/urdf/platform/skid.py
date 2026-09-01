from __future__ import annotations

import numpy as np

from ..core.base import RobotSpec
from .wheeled_base import WheeledBase

class SkidGenerator(WheeledBase):

    FAMILY = "skid"
    FORMS = ("skid",)

    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:

        spec = RobotSpec(name="skid", family=self.FAMILY, form=form, control_tag="skid")

        dims = self._sample_body_dims(rng)
        n_axle = int(rng.choice([2, 3]))
        wheelbase = dims["length"] * self._u(rng, "wheelbase_factor")

        gap_cap = 0.19 if n_axle == 3 else 0.45
        radius = min(self._u(rng, "wheel_radius"), 0.5 * dims["length"], gap_cap * wheelbase)
        radius = max(radius, 0.015)
        wheel_w = radius * self._u(rng, "wheel_width_factor")

        clearance = max(radius * rng.uniform(0.3, 1.2), 0.02)
        geo = self._build_base(spec, rng, dims, clearance)
        geo["wheel_density"] = self._u(rng, "wheel_density")

        exposed = bool(rng.random() < 0.5)
        track_half = self._track_half(rng, geo, wheel_w, exposed)
        ratio_max = self._cfg[self.FAMILY]["wb_track_ratio_max"]
        wheelbase = min(wheelbase, ratio_max * 2 * track_half)
        radius = min(radius, gap_cap * wheelbase)

        xs = [wheelbase / 2, -wheelbase / 2]
        if n_axle == 3:
            xs.insert(1, rng.uniform(-0.08, 0.08) * wheelbase)
        axle_z = radius - geo["body_z"]
        for i, x in enumerate(xs):
            for sy in (1, -1):
                self._add_wheel(spec, rng, geo, f"wheel_{i}_{'l' if sy > 0 else 'r'}",
                                (x, sy * track_half, axle_z), drive=True,
                                radius=radius, width=wheel_w)

        self._offset_com(spec, (rng.uniform(-0.2, 0.2) * wheelbase / 2,
                                rng.uniform(-0.04, 0.04) * geo["width"],
                                rng.uniform(-0.15, 0.15) * geo["height"]))

        self._set_drive_limits(spec, rng, radius)
        self._clamp_mass_ratio(spec)
        spec.params.update({
            "n_wheels": 2 * n_axle, "wheelbase": wheelbase, "track_width": 2 * track_half,
            "wheel_width": wheel_w, "wheel_exposed": exposed,
            "wb_track_ratio": wheelbase / (2 * track_half),
            "total_mass": spec.total_mass(),
            "has_wheels": True, "has_legs": False,
            "est_step_height": radius * 0.6,
        })
        return spec

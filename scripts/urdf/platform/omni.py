from __future__ import annotations

import math

import numpy as np

from ..core.base import RobotSpec
from .wheeled_base import WheeledBase

class OmniGenerator(WheeledBase):

    FAMILY = "omni"
    FORMS = ("omni",)

    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:

        spec = RobotSpec(name="omni", family=self.FAMILY, form=form, control_tag="omni")
        subtype = str(rng.choice(self._cfg[self.FAMILY]["subtypes"]))
        if subtype == "mecanum4":
            self._build_mecanum(spec, rng)
        else:
            self._build_omniwheel(spec, rng, k=3 if subtype == "omni3" else 4)
        spec.params.update({
            "subtype": subtype, "total_mass": spec.total_mass(),
            "has_wheels": True, "has_legs": False,
        })
        return spec

    def _build_mecanum(self, spec, rng):

        dims = self._sample_body_dims(rng)
        wheelbase = dims["length"] * self._u(rng, "wheelbase_factor")
        radius = min(self._u(rng, "wheel_radius"), 0.5 * dims["length"], 0.45 * wheelbase)
        radius = max(radius, 0.02)
        wheel_w = radius * self._u(rng, "wheel_width_factor")
        clearance = max(radius * rng.uniform(0.3, 1.0), 0.02)
        geo = self._build_base(spec, rng, dims, clearance)
        geo["wheel_density"] = self._u(rng, "wheel_density")

        n_roller = int(rng.integers(10, 15))
        roller_r = radius * rng.uniform(0.2, 0.3)
        hub_r = radius - roller_r
        l_need = 2.3 * hub_r * math.sin(math.pi / n_roller) / math.cos(math.pi / 4)
        roller_l = float(np.clip(l_need, 2.0 * roller_r, 3.0 * roller_r))

        protrusion = (roller_l / 2) * math.sin(math.pi / 4) + roller_r * math.cos(math.pi / 4)
        track = (geo["width"] / 2 + rng.uniform(0.005, 0.03) + wheel_w / 2
                 + max(0.0, protrusion - wheel_w / 2) + 0.008)

        axle_z = radius - geo["body_z"]
        for key, (sx, sy) in dict(fl=(1, 1), fr=(1, -1), rl=(-1, 1), rr=(-1, -1)).items():
            wname = f"wheel_{key}"
            self._add_wheel(spec, rng, geo, wname,
                            (sx * wheelbase / 2, sy * track, axle_z),
                            drive=True, radius=hub_r, width=wheel_w)

            tilt = -float(sx * sy) * math.pi / 4
            phase = rng.uniform(0, 2 * math.pi)
            for k in range(n_roller):
                theta = 2 * math.pi * k / n_roller + phase
                self._add_roller(spec, geo, wname, k, theta, tilt, hub_r, roller_r, roller_l)

        spec.contact_links = [l.name for l in spec.links if "_roller_" in l.name]

        self._offset_com(spec, (rng.uniform(-0.2, 0.2) * wheelbase / 2,
                                rng.uniform(-0.04, 0.04) * geo["width"],
                                rng.uniform(-0.15, 0.15) * geo["height"]))
        self._set_drive_limits(spec, rng, radius)
        self._clamp_mass_ratio(spec)
        spec.params.update({
            "wheelbase": wheelbase, "track_width": 2 * track, "wheel_width": wheel_w,
            "n_rollers_per_wheel": n_roller, "roller_radius": roller_r,
            "roller_length": roller_l, "mecanum_pattern": "X_standard",
            "n_wheels": 4, "est_step_height": radius * 0.6,
        })

    def _build_omniwheel(self, spec, rng, k: int):

        dims = self._sample_body_dims(rng)
        if k == 3:
            dims["shape"] = "cylinder"
            dims["length"] = dims["width"]
        radius = max(min(self._u(rng, "wheel_radius"), 0.35 * dims["width"]), 0.02)
        wheel_w = radius * self._u(rng, "wheel_width_factor")
        clearance = max(radius * rng.uniform(0.3, 1.0), 0.02)
        geo = self._build_base(spec, rng, dims, clearance)
        geo["wheel_density"] = self._u(rng, "wheel_density")

        n_roller = int(rng.integers(12, 17))
        roller_r = radius * rng.uniform(0.2, 0.3)
        hub_r = radius - roller_r
        roller_l = float(np.clip(2.3 * hub_r * math.sin(math.pi / n_roller),
                                 1.6 * roller_r, 3.0 * roller_r))

        gap = max(rng.uniform(0.005, 0.03), roller_r - wheel_w / 2 + 0.008)
        ring = geo["width"] / 2 + gap + wheel_w / 2
        axle_z = radius - geo["body_z"]

        phase0 = rng.uniform(0, 2 * math.pi / k)
        for i in range(k):
            phi = 2 * math.pi * i / k + phase0
            wname = f"wheel_{i}"
            self._add_wheel(spec, rng, geo, wname,
                            (ring * math.cos(phi), ring * math.sin(phi), axle_z),
                            drive=True, radius=hub_r, width=wheel_w,
                            joint_yaw=phi - math.pi / 2)
            phase = rng.uniform(0, 2 * math.pi)
            for j in range(n_roller):
                theta = 2 * math.pi * j / n_roller + phase
                self._add_roller(spec, geo, wname, j, theta, 0.0, hub_r, roller_r, roller_l)

        spec.contact_links = [l.name for l in spec.links if "_roller_" in l.name]

        self._offset_com(spec, (rng.uniform(-0.15, 0.15) * ring,
                                rng.uniform(-0.15, 0.15) * ring,
                                rng.uniform(-0.15, 0.15) * geo["height"]))
        self._set_drive_limits(spec, rng, radius)
        self._clamp_mass_ratio(spec)
        spec.params.update({
            "ring_radius": ring, "wheel_width": wheel_w,
            "n_rollers_per_wheel": n_roller, "roller_radius": roller_r,
            "roller_length": roller_l,
            "n_wheels": k, "est_step_height": radius * 0.5,
        })

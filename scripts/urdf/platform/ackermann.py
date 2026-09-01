from __future__ import annotations

import math

import numpy as np

from ..core.base import GeomSpec, GeomType, JointSpec, LinkSpec, RobotSpec
from .wheeled_base import WheeledBase

class AckermannGenerator(WheeledBase):

    FAMILY = "ackermann"
    FORMS = ("ackermann",)

    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:

        spec = RobotSpec(name="ackermann", family=self.FAMILY, form=form, control_tag="ackermann")

        dims = self._sample_body_dims(rng)
        wheelbase = dims["length"] * self._u(rng, "wheelbase_factor")
        radius = min(self._u(rng, "wheel_radius"), 0.5 * dims["length"], 0.45 * wheelbase)
        radius = max(radius, 0.015)
        wheel_w = radius * self._u(rng, "wheel_width_factor")
        clearance = max(radius * rng.uniform(0.3, 1.2), 0.02)
        geo = self._build_base(spec, rng, dims, clearance)
        geo["wheel_density"] = self._u(rng, "wheel_density")

        steer = rng.uniform(0.3, 0.7)
        steer = min(max(steer, math.atan(wheelbase / (self._cfg[self.FAMILY]["turn_radius_factor"]
                                                      * geo["length"]))), 0.75)

        drive_mode = str(rng.choice(["rear", "front", "all"]))
        kingpin = wheel_w * rng.uniform(0.6, 1.2) if rng.random() < 0.5 else 0.0

        track_r = geo["width"] / 2 + rng.uniform(0.005, 0.03) + wheel_w / 2
        axle_z = radius - geo["body_z"]
        for sy in (1, -1):
            self._add_wheel(spec, rng, geo, f"wheel_rear_{'l' if sy > 0 else 'r'}",
                            (-wheelbase / 2, sy * track_r, axle_z),
                            drive=drive_mode in ("rear", "all"), radius=radius, width=wheel_w)

        encroach = ((kingpin - wheel_w / 2) * (1 - math.cos(steer))
                    + radius * math.sin(steer))
        track_f = geo["width"] / 2 + rng.uniform(0.005, 0.03) + wheel_w / 2 + max(encroach, 0) + 0.008
        for sy in (1, -1):
            side = "l" if sy > 0 else "r"

            kr = radius * 0.25
            knuckle = LinkSpec(f"knuckle_{side}", [GeomSpec(GeomType.CYLINDER, (kr, kr * 2, 0))])
            knuckle.mass = knuckle.geoms[0].volume * geo["wheel_density"]
            spec.links.append(knuckle)
            spec.joints.append(JointSpec(
                f"steer_{side}", "revolute", "base_link", f"knuckle_{side}",
                origin_xyz=(wheelbase / 2, sy * (track_f - kingpin), axle_z),
                axis=(0, 0, 1), lower=-steer, upper=steer,
                effort=1.0, velocity=rng.uniform(2.0, 6.0),
            ))
            spec.standing_pose[f"steer_{side}"] = 0.0

            self._add_wheel(spec, rng, geo, f"wheel_front_{side}",
                            (0.0, sy * kingpin, 0.0),
                            drive=drive_mode in ("front", "all"),
                            radius=radius, width=wheel_w, parent=f"knuckle_{side}")

        com_dx = self._setup_com_load(spec, rng, geo, wheelbase, drive_mode)

        spec.check_poses += [{"steer_l": s * steer, "steer_r": s * steer} for s in (1, -1)]

        self._set_drive_limits(spec, rng, radius)

        front_share = float(np.clip(0.5 + com_dx / wheelbase, 0.05, 0.95))
        steer_arm = kingpin + wheel_w * 0.5
        for j in spec.joints:
            if j.name.startswith("steer_"):
                j.effort = (spec.total_mass() * 9.81 * front_share / 2
                            * steer_arm * rng.uniform(1.5, 4.0))
        self._clamp_mass_ratio(spec)
        spec.params.update({
            "wheelbase": wheelbase, "track_front": 2 * track_f, "track_rear": 2 * track_r,
            "wheel_width": wheel_w, "steer_range": steer, "kingpin_offset": kingpin,
            "drive_mode": drive_mode, "front_track_outset": max(encroach, 0),
            "min_turn_radius": wheelbase / math.tan(steer),
            "total_mass": spec.total_mass(), "n_wheels": 4,
            "has_wheels": True, "has_legs": False,
            "est_step_height": radius * 0.35,
        })
        return spec

    def _setup_com_load(self, spec, rng, geo, wheelbase, drive_mode) -> float:

        if drive_mode == "all":
            dx = wheelbase * rng.uniform(-0.25, 0.25)
        else:
            drive_x = -wheelbase / 2 if drive_mode == "rear" else wheelbase / 2
            other_x = -drive_x
            share = rng.uniform(self._cfg["validation"]["load_share_min"] + 0.05, 0.9)
            dx = other_x + share * (drive_x - other_x)
            spec.special["load_share"] = {"drive_x": drive_x, "other_x": other_x}
        self._offset_com(spec, (dx, rng.uniform(-0.04, 0.04) * geo["width"],
                                rng.uniform(-0.15, 0.15) * geo["height"]))
        return dx

from __future__ import annotations

import math

import numpy as np

from ..core.base import GeomSpec, GeomType, JointSpec, LinkSpec, RobotSpec
from .wheeled_base import WheeledBase

class DiffGenerator(WheeledBase):

    FAMILY = "diff"
    FORMS = ("diff",)

    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:

        spec = RobotSpec(name="diff", family=self.FAMILY, form=form, control_tag="diff")

        caster_kind = str(rng.choice(self._cfg[self.FAMILY]["caster_kinds"]))

        dims = self._sample_body_dims(rng)
        radius = min(self._u(rng, "wheel_radius"), 0.5 * dims["length"])
        radius = max(radius, 0.015)
        wheel_w = radius * self._u(rng, "wheel_width_factor")

        if caster_kind == "swivel":

            clearance = max(radius * rng.uniform(0.7, 1.2), 0.03)
        else:
            clearance = max(radius * rng.uniform(0.4, 1.2), 0.02)
        geo = self._build_base(spec, rng, dims, clearance)
        geo["wheel_density"] = self._u(rng, "wheel_density")

        L = geo["length"]
        axle_x = L * rng.uniform(-0.35, 0.35)

        exposed = bool(rng.random() < 0.5)
        track_half = self._track_half(rng, geo, wheel_w, exposed)
        axle_z = radius - geo["body_z"]
        for sy in (1, -1):
            self._add_wheel(spec, rng, geo, f"wheel_{'l' if sy > 0 else 'r'}",
                            (axle_x, sy * track_half, axle_z), drive=True,
                            radius=radius, width=wheel_w)

        caster_xs = self._add_casters(spec, rng, geo, caster_kind, axle_x, radius)
        self._setup_com_load(spec, rng, geo, axle_x, caster_xs)

        com_x = spec.links[0].com_xyz[0]
        fore_aft = min([abs(axle_x - com_x)] + [abs(cx - com_x) for cx in caster_xs])
        spec.params["wheelbase"] = 2.0 * fore_aft

        self._set_drive_limits(spec, rng, radius)
        self._clamp_mass_ratio(spec)
        spec.params.update({
            "caster_kind": caster_kind, "axle_x": axle_x, "track_width": 2 * track_half,
            "wheel_width": wheel_w, "wheel_exposed": exposed,
            "total_mass": spec.total_mass(), "n_wheels": 2,
            "has_wheels": True, "has_legs": False,
            "est_step_height": radius * 0.35,
        })
        return spec

    def _add_casters(self, spec, rng, geo, kind, axle_x, radius) -> list[float]:

        placements = ["opp1", "opp2"]
        if abs(axle_x) < 0.12 * geo["length"]:
            placements.append("both")
        placement = str(rng.choice(placements))

        s = -math.copysign(1.0, axle_x) if abs(axle_x) > 1e-6 else float(rng.choice([1, -1]))
        L = geo["length"]
        xs = []
        if placement == "both":

            d = L * rng.uniform(0.3, 0.42)
            spots = [(axle_x + d, 0.0), (axle_x - d, 0.0)]
        elif placement == "opp1":
            cx = float(np.clip(axle_x + s * L * rng.uniform(0.4, 0.8), -0.46 * L, 0.46 * L))
            spots = [(cx, 0.0)]
        else:

            cx = float(np.clip(axle_x + s * L * rng.uniform(0.4, 0.8), -0.46 * L, 0.46 * L))
            wy = geo["width"] * rng.uniform(0.2, 0.35)
            spots = [(cx, wy), (cx, -wy)]

        for i, (cx, cy) in enumerate(spots):
            name = f"c{i}"
            if kind == "ball":
                self._add_ball_caster(spec, rng, geo, name, cx, cy, radius)
            else:
                self._add_swivel_caster(spec, rng, geo, name, cx, cy)
            xs.append(cx)

        if kind == "swivel":
            swivels = [j.name for j in spec.joints if j.name.startswith("swivel_")]
            spec.check_poses += [{n: ang for n in swivels} for ang in (math.pi / 2, math.pi)]
        spec.params["caster_placement"] = placement
        return xs

    def _setup_com_load(self, spec, rng, geo, axle_x, caster_xs):

        if len(caster_xs) == 2 and (caster_xs[0] - axle_x) * (caster_xs[1] - axle_x) < 0:

            dx = axle_x + rng.uniform(-0.05, 0.05) * geo["length"]
        else:

            other_x = float(np.mean(caster_xs))
            t = rng.uniform(0.05, 1.0 - self._cfg["validation"]["load_share_min"] - 0.05)
            dx = axle_x + (other_x - axle_x) * t
            spec.special["load_share"] = {"drive_x": axle_x, "other_x": other_x}
        dy = rng.uniform(-0.04, 0.04) * geo["width"]
        dz = rng.uniform(-0.15, 0.15) * geo["height"]
        self._offset_com(spec, (dx, dy, dz))

    def _add_ball_caster(self, spec, rng, geo, name, cx, cy, radius):

        r_s = max(min(radius * rng.uniform(0.3, 0.7), geo["clearance"] * 0.95), 0.012)
        geoms = [GeomSpec(GeomType.SPHERE, (r_s, 0, 0))]

        gap = geo["clearance"] - 2 * r_s
        if gap > 0.008:
            geoms.append(GeomSpec(GeomType.CYLINDER, (r_s * 0.45, gap, 0),
                                  origin_xyz=(0, 0, r_s + gap / 2)))
        link = LinkSpec(f"ball_{name}", geoms, mass=0.05 + 200 * r_s ** 3)
        spec.links.append(link)
        spec.joints.append(JointSpec(
            f"fix_ball_{name}", "fixed", "base_link", f"ball_{name}",
            origin_xyz=(cx, cy, r_s - geo["body_z"]),
        ))
        spec.contact_links.append(f"ball_{name}")

    def _add_swivel_caster(self, spec, rng, geo, name, cx, cy):

        r_c = geo["clearance"] * rng.uniform(0.25, 0.45)
        drop = geo["clearance"] - r_c

        mount = LinkSpec(f"caster_mount_{name}",
                         [GeomSpec(GeomType.CYLINDER, (max(r_c * 0.35, 0.008), max(drop, 0.01), 0),
                                   origin_xyz=(0, 0, -drop / 2))])
        mount.mass = 0.05 + 100 * r_c ** 3
        spec.links.append(mount)
        spec.joints.append(JointSpec(
            f"swivel_{name}", "continuous", "base_link", f"caster_mount_{name}",
            origin_xyz=(cx, cy, geo["clearance"] - geo["body_z"]), axis=(0, 0, 1),
            effort=0.0, velocity=rng.uniform(5.0, 20.0),
        ))

        trail = r_c * rng.uniform(0.3, 0.8)
        wheel = LinkSpec(f"caster_wheel_{name}",
                         [GeomSpec(GeomType.CYLINDER, (r_c, r_c * 0.5, 0),
                                   origin_rpy=(math.pi / 2, 0, 0))])
        wheel.mass = wheel.geoms[0].volume * geo["wheel_density"]
        spec.links.append(wheel)
        spec.joints.append(JointSpec(
            f"spin_caster_{name}", "continuous", f"caster_mount_{name}", f"caster_wheel_{name}",
            origin_xyz=(-trail, 0, -drop), axis=(0, 1, 0),
            effort=0.0, velocity=rng.uniform(10.0, 40.0),
        ))
        spec.contact_links.append(f"caster_wheel_{name}")
        spec.standing_pose[f"swivel_{name}"] = 0.0
        spec.standing_pose[f"spin_caster_{name}"] = 0.0
        spec.params["caster_trail"] = trail
        spec.params["caster_radius"] = r_c

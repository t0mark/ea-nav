from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from ..core.base import BaseGenerator, GeomSpec, GeomType, JointSpec, LinkSpec, RobotSpec

class WheeledBase(BaseGenerator):

    def _sample_body_dims(self, rng: np.random.Generator) -> dict:

        """wheeled base의 몸통 치수를 안정성 비율 안에서 샘플링한다."""

        shape = str(rng.choice(self._cfg[self.FAMILY]["body_shapes"]))
        length = self._u(rng, "body_length")
        width = min(self._u(rng, "body_width"), length * 1.2)
        height = self._u(rng, "body_height")
        rules = self._cfg.get("validation", {}).get("wheeled", {})
        width = max(width, length * float(rules.get("body_width_length_min", 0.0)))
        height = min(height, width * float(rules.get("body_height_width_max", float("inf"))))

        if shape.startswith("cylinder"):
            length = width
        return {"shape": shape, "length": length, "width": width, "height": height}

    def _min_drive_radius(self, dims: dict) -> float:

        """몸통 높이에 비해 지나치게 작은 구동 바퀴가 나오지 않게 하한을 계산한다."""

        rules = self._cfg.get("validation", {}).get("wheeled", {})
        return dims["height"] * float(rules.get("wheel_radius_body_height_min", 0.0))

    def _build_base(self, spec: RobotSpec, rng: np.random.Generator,
                    dims: dict, clearance: float) -> dict:

        L, W, H = dims["length"], dims["width"], dims["height"]
        shape = dims["shape"]
        geoms: list[GeomSpec] = []

        if shape in ("stack", "cylinder_stack"):
            h1 = H * rng.uniform(0.5, 0.75)
            h2 = H - h1
            top = GeomSpec(GeomType.BOX,
                           (L * rng.uniform(0.4, 0.85), W * rng.uniform(0.4, 0.85), h2),
                           origin_xyz=(rng.uniform(-0.15, 0.15) * L, 0, -H / 2 + h1 + h2 / 2))
            if shape == "stack":
                geoms.append(GeomSpec(GeomType.BOX, (L, W, h1), origin_xyz=(0, 0, -H / 2 + h1 / 2)))
            else:
                geoms.append(GeomSpec(GeomType.CYLINDER, (W / 2, h1, 0),
                                      origin_xyz=(0, 0, -H / 2 + h1 / 2)))
            geoms.append(top)
        elif shape == "cylinder":
            geoms.append(GeomSpec(GeomType.CYLINDER, (W / 2, H, 0)))
        else:
            geoms.append(GeomSpec(GeomType.BOX, (L, W, H)))

        body = LinkSpec("base_link", geoms)
        body.mass = sum(g.volume for g in geoms) * self._u(rng, "body_density")
        spec.links.append(body)

        geo = {"length": L, "width": W, "height": H, "shape": shape,
               "clearance": clearance, "body_z": clearance + H / 2, "top_z": H / 2}
        spec.params.update({"body_shape": shape, "body_length": L, "body_width": W,
                            "body_height": H, "ground_clearance": clearance})
        return geo

    def _offset_com(self, spec: RobotSpec, offset: tuple[float, float, float]):

        from ..core.base import compute_com
        base = spec.links[0]
        natural = compute_com(base)
        base.com_xyz = tuple(float(natural[i] + offset[i]) for i in range(3))
        spec.params["com_offset"] = [float(v) for v in offset]

    def _add_wheel(self, spec: RobotSpec, rng: np.random.Generator, geo: dict, name: str,
                   xyz: tuple[float, float, float], drive: bool, radius: float, width: float,
                   parent: str = "base_link", joint_yaw: float = 0.0):

        link = LinkSpec(name, [GeomSpec(GeomType.CYLINDER, (radius, width, 0),
                                        origin_rpy=(math.pi / 2, 0, 0))])
        link.mass = link.geoms[0].volume * geo["wheel_density"]
        spec.links.append(link)

        jname = f"drive_{name}" if drive else f"spin_{name}"
        spec.joints.append(JointSpec(
            jname, "continuous", parent, name,
            origin_xyz=xyz, origin_rpy=(0, 0, joint_yaw), axis=(0, 1, 0),
            effort=0.0, velocity=rng.uniform(10.0, 30.0),
        ))
        spec.contact_links.append(name)
        spec.standing_pose[jname] = 0.0

    def _add_roller(self, spec: RobotSpec, geo: dict, wheel_name: str, idx: int,
                    theta: float, tilt: float, hub_r: float, roller_r: float, roller_l: float):

        pos = (hub_r * math.cos(theta), 0.0, hub_r * math.sin(theta))
        tangent = np.array([-math.sin(theta), 0.0, math.cos(theta)])
        axis = tangent * math.cos(tilt) + np.array([0.0, math.sin(tilt), 0.0])
        axis /= np.linalg.norm(axis)

        rot, _ = Rotation.align_vectors(axis[None, :], np.array([[0.0, 0.0, 1.0]]))
        rpy = tuple(rot.as_euler("xyz"))

        name = f"{wheel_name}_roller_{idx}"
        link = LinkSpec(name, [GeomSpec(GeomType.CYLINDER, (roller_r, roller_l, 0))])
        link.mass = link.geoms[0].volume * geo["wheel_density"]
        spec.links.append(link)
        spec.joints.append(JointSpec(
            f"passive_{name}", "continuous", wheel_name, name,
            origin_xyz=pos, origin_rpy=rpy, axis=(0, 0, 1),
            effort=0.0, velocity=50.0,
        ))
        spec.standing_pose[f"passive_{name}"] = 0.0

    def _track_half(self, rng: np.random.Generator, geo: dict, wheel_w: float,
                    exposed: bool) -> float:

        rules = self._cfg.get("validation", {}).get("wheeled", {})
        min_track_half = 0.5 * geo["width"] * float(rules.get("track_body_width_min", 0.0))
        if exposed:
            return max(geo["width"] / 2 + rng.uniform(0.005, 0.03) + wheel_w / 2,
                       min_track_half)

        upper = geo["width"] / 2 * 0.95 - wheel_w / 2
        return max(min(geo["width"] / 2 * rng.uniform(0.5, 0.85), upper),
                   wheel_w * 0.6, min_track_half)

    def _set_drive_limits(self, spec: RobotSpec, rng: np.random.Generator, radius: float):

        m = spec.total_mass()
        drives = [j for j in spec.joints if j.name.startswith("drive_")]
        v_max = rng.uniform(0.5, 3.0) * spec.params["body_length"]
        for j in drives:
            j.effort = m * 9.81 * radius / max(len(drives), 1) * rng.uniform(0.6, 2.5)
            j.velocity = v_max / radius
        spec.params["max_lin_vel"] = v_max
        spec.params["wheel_radius"] = radius

    def _mark_wheeled_static_checks(self, spec: RobotSpec, base_type: str | None = None):

        """시뮬레이션 없이 검사할 wheeled 전용 준정적 규칙을 예약한다."""

        spec.special["overturn"] = {}
        spec.special["slope_static"] = {}
        spec.special["wheeled_geometry"] = {
            "base_type": base_type or spec.control_tag.removeprefix("wheeled_humanoid_"),
            **spec.params,
        }

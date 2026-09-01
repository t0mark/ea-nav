from __future__ import annotations

import math

import numpy as np

from ..core.base import BaseGenerator, GeomSpec, GeomType, JointSpec, LinkSpec, RobotSpec
from .diff import DiffGenerator
from .omni import OmniGenerator
from .skid import SkidGenerator

_ARM_CHAIN = [
    ("shoulder_pitch", (0, 1, 0)), ("shoulder_roll", (1, 0, 0)), ("shoulder_yaw", (0, 0, 1)),
    ("elbow", (0, 1, 0)), ("wrist_pitch", (0, 1, 0)), ("wrist_roll", (1, 0, 0)),
    ("wrist_yaw", (0, 0, 1)),
]

class WheeledHumanoidGenerator(BaseGenerator):

    FAMILY = "wheeled_humanoid"
    FORMS = ("wheeled_humanoid",)

    def __init__(self, cfg: dict):

        super().__init__(cfg)
        self._bases = {"diff": DiffGenerator(cfg), "skid": SkidGenerator(cfg),
                       "omni": OmniGenerator(cfg)}

    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:

        base_type = str(rng.choice(self._cfg[self.FAMILY]["base_types"]))
        spec = self._bases[base_type].sample(base_type, rng)
        base_tag = spec.control_tag

        spec.name = "wheeled_humanoid"
        spec.family = self.FAMILY
        spec.form = form
        spec.control_tag = f"wheeled_humanoid_{base_tag}"

        geo_top = spec.params["body_height"] / 2
        torso, th = self._add_torso(spec, rng, geo_top)
        self._add_arms(spec, rng, torso, th)
        if rng.random() < 0.7:
            self._add_head(spec, rng, torso, th)

        spec.special["overturn"] = {}
        self._bases[base_type]._set_drive_limits(spec, rng, spec.params["wheel_radius"])
        self._clamp_mass_ratio(spec)
        spec.params.update({
            "base_type": base_type, "total_mass": spec.total_mass(),
            "has_upper_body": True,
        })
        return spec

    def _add_torso(self, spec, rng, base_top: float) -> tuple[str, float]:

        W, L = spec.params["body_width"], spec.params["body_length"]
        tw = W * rng.uniform(0.5, 0.9)
        td = L * rng.uniform(0.2, 0.5)
        th = self._u(rng, "torso_height")
        torso = LinkSpec("torso", [GeomSpec(GeomType.BOX, (td, tw, th),
                                            origin_xyz=(0, 0, th / 2))])
        torso.mass = torso.geoms[0].volume * self._u(rng, "body_density")
        spec.links.append(torso)

        mount_x = rng.uniform(-0.2, 0.2) * L
        use_lift = bool(rng.random() < 0.5)
        if use_lift:
            stroke = self._u(rng, "lift_stroke")
            spec.joints.append(JointSpec(
                "torso_lift", "prismatic", "base_link", "torso",
                origin_xyz=(mount_x, 0, base_top), axis=(0, 0, 1),
                lower=0.0, upper=stroke,
                effort=torso.mass * 9.81 * rng.uniform(2.0, 4.0),
                velocity=rng.uniform(0.05, 0.2),
            ))

            spec.standing_pose["torso_lift"] = 0.0
            spec.check_poses.append({"torso_lift": stroke})
            spec.params["lift_stroke"] = stroke
        else:
            spec.joints.append(JointSpec(
                "fix_torso", "fixed", "base_link", "torso",
                origin_xyz=(mount_x, 0, base_top),
            ))
        spec.params.update({"torso": [td, tw, th], "has_lift": use_lift})
        return "torso", th

    def _add_arms(self, spec, rng, torso: str, th: float):

        n_arms = int(rng.choice([1, 2]))
        arm_dof = int(rng.integers(2, 8))
        tw = spec.params["torso"][1]
        td = spec.params["torso"][0]

        if arm_dof == 2:
            chain = [_ARM_CHAIN[0], _ARM_CHAIN[3]]
        elif arm_dof == 3:
            chain = [_ARM_CHAIN[0], _ARM_CHAIN[1], _ARM_CHAIN[3]]
        else:
            chain = list(_ARM_CHAIN[:arm_dof])

        z_sh = th * rng.uniform(0.75, 0.9)
        la, lf = self._u(rng, "upper_arm_length"), self._u(rng, "forearm_length")
        reach_max = z_sh - 0.03
        if la + lf > reach_max:
            k = reach_max / (la + lf)
            la, lf = max(la * k, 0.05), max(lf * k, 0.05)
        arm_r = float(np.clip((la + lf) * rng.uniform(0.06, 0.1), 0.015, 0.05))
        blk = max(arm_r * 1.1, 0.02)
        off = blk + 0.004

        if n_arms == 2:
            mounts = [(1, "l", (0, tw / 2 + arm_r * 1.5, z_sh)),
                      (-1, "r", (0, -(tw / 2 + arm_r * 1.5), z_sh))]
        else:
            mounts = [(0, "c", (td / 2 + arm_r * 1.5, 0, z_sh))]

        out_roll = rng.uniform(0.05, 0.2)
        elbow_bend = rng.uniform(0.4, 1.0)
        for sy, side, mount in mounts:
            parent, origin = torso, mount
            for i, (jname, axis) in enumerate(chain):
                is_elbow = jname == "elbow"
                is_last = i == len(chain) - 1

                if is_elbow:
                    child, link = f"{side}_forearm", self._limb(f"{side}_forearm", lf, arm_r)
                elif i + 1 < len(chain) and chain[i + 1][0] == "elbow":
                    child, link = f"{side}_upper_arm", self._limb(f"{side}_upper_arm", la, arm_r)
                elif is_last and jname.startswith("wrist"):
                    child = f"{side}_hand"
                    link = LinkSpec(child, [GeomSpec(GeomType.SPHERE, (arm_r * 1.2, 0, 0))])
                elif is_last:
                    child, link = f"{side}_upper_arm", self._limb(f"{side}_upper_arm", la, arm_r)
                else:
                    child = f"{side}_{jname}_link"
                    link = LinkSpec(child, [GeomSpec(GeomType.SPHERE, (blk, 0, 0))])
                if link.mass == 0:
                    link.mass = sum(g.volume for g in link.geoms) * 800.0
                spec.links.append(link)
                spec.joints.append(JointSpec(
                    f"{side}_{jname}", "revolute", parent, child, origin_xyz=origin, axis=axis,
                ))

                if jname == "shoulder_roll":
                    spec.standing_pose[f"{side}_{jname}"] = sy * out_roll
                elif is_elbow:
                    spec.standing_pose[f"{side}_{jname}"] = -elbow_bend
                else:
                    spec.standing_pose[f"{side}_{jname}"] = 0.0

                if child.endswith("upper_arm"):
                    origin = (0, 0, -la)
                elif child.endswith("forearm"):
                    origin = (0, 0, -lf)
                else:
                    origin = (0, 0, -off)
                parent = child

            spec.check_poses.append({f"{side}_shoulder_pitch": -1.2})

        m_arm = sum(l.mass for l in spec.links
                    if l.name.split("_")[0] in ("l", "r", "c") and
                    any(k in l.name for k in ("arm", "hand", "shoulder", "wrist", "elbow")))
        for j in spec.joints:
            if any(j.name.endswith(k) or k in j.name for k in
                   ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                    "wrist_pitch", "wrist_roll", "wrist_yaw")) and j.jtype == "revolute":
                center = spec.standing_pose.get(j.name, 0.0)
                j.lower = float(np.clip(center - rng.uniform(0.6, 1.5), -2.9, 2.9))
                j.upper = float(np.clip(center + rng.uniform(0.6, 1.5), -2.9, 2.9))
                j.effort = max(m_arm, 0.5) * 9.81 * (la + lf) * rng.uniform(1.0, 3.0)
                j.velocity = rng.uniform(1.0, 4.0)
        spec.params.update({"n_arms": n_arms, "arm_dof": arm_dof,
                            "upper_arm_length": la, "forearm_length": lf})

    def _add_head(self, spec, rng, torso: str, th: float):

        head_r = self._u(rng, "head_radius")
        head = LinkSpec("head", [GeomSpec(GeomType.SPHERE, (head_r, 0, 0))],
                        mass=0.5 + 30 * head_r ** 3)
        spec.links.append(head)
        spec.joints.append(JointSpec(
            "neck", "fixed", torso, "head",
            origin_xyz=(0, 0, th + head_r * 1.1),
        ))
        spec.params["has_head"] = True

    def _limb(self, name, length, radius):

        link = LinkSpec(name, [GeomSpec(GeomType.CYLINDER, (radius, length, 0),
                                        origin_xyz=(0, 0, -length / 2))])
        return link

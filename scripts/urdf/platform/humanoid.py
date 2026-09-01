from __future__ import annotations

import math
import re

import numpy as np

from ..core.base import BaseGenerator, GeomSpec, GeomType, JointSpec, LinkSpec, RobotSpec

_ARM_CHAIN = [
    ("shoulder_pitch", (0, 1, 0)), ("shoulder_roll", (1, 0, 0)), ("shoulder_yaw", (0, 0, 1)),
    ("elbow", (0, 1, 0)), ("wrist_pitch", (0, 1, 0)), ("wrist_roll", (1, 0, 0)),
    ("wrist_yaw", (0, 0, 1)),
]

class HumanoidGenerator(BaseGenerator):

    FAMILY = "humanoid"
    FORMS = ("humanoid",)

    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:

        pd, pw, ph = self._u(rng, "pelvis_depth"), self._u(rng, "pelvis_width"), self._u(rng, "pelvis_height")
        td, tw, th = self._u(rng, "torso_depth"), self._u(rng, "torso_width"), self._u(rng, "torso_height")
        l1, l2 = self._u(rng, "thigh_length"), self._u(rng, "shin_length")
        leg_r = float(np.clip((l1 + l2) * rng.uniform(0.05, 0.09), 0.02, 0.07))
        torso_split = str(rng.choice(["merged", "waist"]))
        waist_axis = str(rng.choice(["yaw", "pitch"]))
        foot_shape = str(rng.choice(["box", "disc", "dual"]))
        arm_dof = int(rng.choice([0, 3, 4, 5, 6, 7]))

        spec = RobotSpec(name="humanoid", family=self.FAMILY, form=form, control_tag="humanoid")
        body_density = self._u(rng, "body_density")
        limb_density = self._u(rng, "limb_density")
        pelvis = LinkSpec("base_link", [GeomSpec(GeomType.BOX, (pd, pw, ph))])
        pelvis.mass = pelvis.geoms[0].volume * body_density
        spec.links.append(pelvis)
        torso_parent = self._add_torso(spec, rng, torso_split, waist_axis,
                                       (td, tw, th), ph, body_density)

        hip_sep = max(pw * rng.uniform(0.55, 0.95), 2 * leg_r + 0.06)
        knee = rng.uniform(0.08, 0.45)
        gamma = math.atan2(l2 * math.sin(knee), l1 + l2 * math.cos(knee))
        ankle_h = rng.uniform(0.03, 0.08)
        foot_dims, ankle_h, hip_off = self._add_legs(spec, rng, hip_sep, ph, l1, l2, leg_r,
                                                     ankle_h, foot_shape, gamma, knee, limb_density)
        if arm_dof > 0:
            self._add_arms(spec, rng, torso_parent, tw, th, arm_dof, limb_density)

        stance = (ph / 2 + 2 * hip_off
                  + l1 * math.cos(gamma) + l2 * math.cos(knee - gamma) + ankle_h)

        self._set_limits(spec, rng, l1, l2, gamma, knee, foot_dims, stance)
        self._clamp_mass_ratio(spec)
        spec.params.update({
            "pelvis": [pd, pw, ph], "torso": [td, tw, th],
            "thigh_length": l1, "shin_length": l2, "leg_radius": leg_r,
            "hip_separation": hip_sep, "ankle_height": ankle_h,
            "torso_split": torso_split, "waist_axis": waist_axis,
            "foot_shape": foot_shape, "foot_dims": foot_dims, "arm_dof": arm_dof,
            "stance_height": stance, "knee_stand": knee,
            "total_mass": spec.total_mass(),
            "has_wheels": False, "has_legs": True,
            "est_step_height": 0.3 * (l1 + l2),
        })
        return spec

    def _add_torso(self, spec, rng, torso_split, waist_axis, dims, ph, density) -> str:

        td, tw, th = dims
        torso_geom = GeomSpec(GeomType.BOX, (td, tw, th))
        head_r = self._u(rng, "head_radius")

        if torso_split == "merged":
            torso_geom.origin_xyz = (0, 0, ph / 2 + th / 2)
            base = spec.links[0]
            base.geoms.append(torso_geom)
            base.mass += torso_geom.volume * density
            self._add_head(spec, rng, "base_link", ph / 2 + th, head_r)
            return "base_link"

        torso = LinkSpec("torso", [torso_geom])
        torso_geom.origin_xyz = (0, 0, th / 2)
        torso.mass = torso_geom.volume * density
        spec.links.append(torso)
        axis = (0, 0, 1) if waist_axis == "yaw" else (0, 1, 0)
        rng_lim = rng.uniform(0.35, 1.0)
        spec.joints.append(JointSpec(
            "waist", "revolute", "base_link", "torso",
            origin_xyz=(0, 0, ph / 2), axis=axis,
            lower=-rng_lim, upper=rng_lim,
        ))
        spec.standing_pose["waist"] = 0.0
        self._add_head(spec, rng, "torso", th, head_r)
        return "torso"

    def _add_head(self, spec, rng, parent, top_z, head_r):

        head = LinkSpec("head", [GeomSpec(GeomType.SPHERE, (head_r, 0, 0))], mass=0.5 + 30 * head_r ** 3)
        spec.links.append(head)
        spec.joints.append(JointSpec(
            "neck", "fixed", parent, "head",
            origin_xyz=(0, 0, top_z + head_r * 1.1),
        ))

    def _add_legs(self, spec, rng, hip_sep, ph, l1, l2, leg_r, ankle_h,
                  foot_shape, gamma, knee, density):

        blk = max(leg_r * 1.1, 0.025)

        off = blk + 0.004

        ankle_h = max(ankle_h, blk + 0.018)
        foot_dims = self._sample_foot(rng, hip_sep, ankle_h)

        base_chain = [("hip_yaw", (0, 0, 1)), ("hip_roll", (1, 0, 0)), ("hip_pitch", (0, 1, 0))]
        order = [int(i) for i in rng.permutation(3)]
        chain = [base_chain[i] for i in order]
        spec.params["hip_axis_order"] = "_".join(c[0].split("_")[1] for c in chain)

        for sy, side in ((1, "l"), (-1, "r")):

            parent = "base_link"
            origin = (0, sy * hip_sep / 2, -ph / 2)
            for i, (jname, axis) in enumerate(chain):
                child = f"{side}_{jname}_link" if i < 2 else f"{side}_thigh"
                if i < 2:
                    link = LinkSpec(child, [GeomSpec(GeomType.SPHERE, (blk, 0, 0))])
                    link.mass = link.geoms[0].volume * density
                else:
                    link = self._limb(child, l1, leg_r, density)
                spec.links.append(link)
                spec.joints.append(JointSpec(
                    f"{side}_{jname}", "revolute", parent, child,
                    origin_xyz=origin, axis=axis,
                ))

                spec.standing_pose[f"{side}_{jname}"] = -gamma if jname == "hip_pitch" else 0.0
                parent, origin = child, (0, 0, -off)

            shin = self._limb(f"{side}_shin", l2, leg_r, density)
            spec.links.append(shin)
            spec.joints.append(JointSpec(
                f"{side}_knee", "revolute", f"{side}_thigh", f"{side}_shin",
                origin_xyz=(0, 0, -l1), axis=(0, 1, 0),
            ))
            spec.standing_pose[f"{side}_knee"] = knee

            ankle_blk = LinkSpec(f"{side}_ankle_link", [GeomSpec(GeomType.SPHERE, (blk, 0, 0))])
            ankle_blk.mass = ankle_blk.geoms[0].volume * density
            spec.links.append(ankle_blk)
            spec.joints.append(JointSpec(
                f"{side}_ankle_pitch", "revolute", f"{side}_shin", f"{side}_ankle_link",
                origin_xyz=(0, 0, -l2), axis=(0, 1, 0),
            ))
            spec.standing_pose[f"{side}_ankle_pitch"] = gamma - knee

            foot = LinkSpec(f"{side}_foot", self._foot_geoms(foot_shape, foot_dims, ankle_h))
            foot.mass = sum(g.volume for g in foot.geoms) * density
            spec.links.append(foot)
            spec.joints.append(JointSpec(
                f"{side}_ankle_roll", "revolute", f"{side}_ankle_link", f"{side}_foot",
                origin_xyz=(0, 0, 0), axis=(1, 0, 0),
            ))
            spec.standing_pose[f"{side}_ankle_roll"] = 0.0
            spec.contact_links.append(f"{side}_foot")
        return foot_dims, ankle_h, off

    def _sample_foot(self, rng, hip_sep, ankle_h):

        lf = self._u(rng, "foot_length")
        wf = min(self._u(rng, "foot_width"), hip_sep - 0.03)
        hf = min(rng.uniform(0.02, 0.05), ankle_h * 0.8)
        return [lf, wf, hf]

    def _foot_geoms(self, foot_shape, foot_dims, ankle_h):

        lf, wf, hf = foot_dims
        z = -ankle_h + hf / 2
        if foot_shape == "box":
            return [GeomSpec(GeomType.BOX, (lf, wf, hf), origin_xyz=(lf * 0.2, 0, z))]
        if foot_shape == "disc":
            r = max(lf, wf) / 2
            return [GeomSpec(GeomType.CYLINDER, (r, hf, 0), origin_xyz=(lf * 0.1, 0, z))]

        return [
            GeomSpec(GeomType.BOX, (lf * 0.45, wf, hf), origin_xyz=(-lf * 0.2, 0, z)),
            GeomSpec(GeomType.BOX, (lf * 0.45, wf * 0.9, hf), origin_xyz=(lf * 0.35, 0, z)),
        ]

    def _add_arms(self, spec, rng, torso_parent, tw, th, arm_dof, density):

        la, lf = self._u(rng, "upper_arm_length"), self._u(rng, "forearm_length")
        arm_r = float(np.clip((la + lf) * rng.uniform(0.05, 0.09), 0.015, 0.05))
        blk = max(arm_r * 1.1, 0.02)

        off = blk + 0.004

        chain = list(_ARM_CHAIN[:arm_dof])
        if arm_dof == 3:
            chain = [_ARM_CHAIN[0], _ARM_CHAIN[1], _ARM_CHAIN[3]]

        z_sh = th * rng.uniform(0.75, 0.95) if torso_parent == "torso"            else spec.links[0].geoms[-1].origin_xyz[2] + th * rng.uniform(0.25, 0.45)
        out_roll = rng.uniform(0.05, 0.2)
        elbow_bend = rng.uniform(0.05, 0.35)
        for sy, side in ((1, "l"), (-1, "r")):
            parent = torso_parent
            origin = (0, sy * (tw / 2 + arm_r * 1.5), z_sh)

            for i, (jname, axis) in enumerate(chain):
                is_elbow = jname == "elbow"
                is_last = i == len(chain) - 1

                if is_elbow:
                    child = f"{side}_forearm"
                    link = self._limb(child, lf, arm_r, density)
                elif i + 1 < len(chain) and chain[i + 1][0] == "elbow":
                    child = f"{side}_upper_arm"
                    link = self._limb(child, la, arm_r, density)
                elif is_last and jname.startswith("wrist"):
                    child = f"{side}_hand"
                    link = LinkSpec(child, [GeomSpec(GeomType.SPHERE, (arm_r * 1.2, 0, 0))])
                    link.mass = link.geoms[0].volume * density
                elif is_last:
                    child = f"{side}_upper_arm"
                    link = self._limb(child, la, arm_r, density)
                else:
                    child = f"{side}_{jname}_link"
                    link = LinkSpec(child, [GeomSpec(GeomType.SPHERE, (blk, 0, 0))])
                    link.mass = link.geoms[0].volume * density
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
        spec.params.update({"upper_arm_length": la, "forearm_length": lf})

    def _limb(self, name, length, radius, density):

        link = LinkSpec(name, [GeomSpec(GeomType.CYLINDER, (radius, length, 0),
                                        origin_xyz=(0, 0, -length / 2))])
        link.mass = link.geoms[0].volume * density
        return link

    def _set_limits(self, spec, rng, l1, l2, gamma, knee, foot_dims, stance):

        g = self._cfg["gait"]
        m = spec.total_mass()
        weight = m * 9.81
        leg_len = l1 + l2

        tilt = abs(math.sin(knee - gamma))
        arms = {"hip_yaw": 0.0, "hip_roll": 0.0, "hip_pitch": 0.0,
                "knee": l2 * tilt, "ankle_pitch": foot_dims[0] / 2, "ankle_roll": foot_dims[1] / 2}
        reach = {"hip_yaw": leg_len, "hip_roll": leg_len, "hip_pitch": leg_len, "knee": l2}

        v_max = g["froude_biped"] * math.sqrt(9.81 * stance)
        w_min = g["vel_margin"] * math.pi * v_max / leg_len
        w_hi = max(g["vel_hi_biped"], g["vel_span"] * w_min)

        arm_scale = {"shoulder": 0.25, "elbow": 0.15, "wrist": 0.08}

        drawn: dict[str, dict] = {}
        stance_tau = spec.params.setdefault("stance_torque", {})
        for j in spec.actuated_joints():
            if j.name == "waist":
                continue
            role = re.sub(r"^[lr]_", "", j.name)
            if role not in drawn:
                if role in arms:

                    arm = max(arms[role], g["arm_min_frac"] * reach.get(role, 0.0))
                    effort = weight * arm * rng.uniform(*g["stance_margin"])
                    velocity = rng.uniform(w_min, w_hi)
                    tau_req = weight * arm
                else:

                    k = next((v for key, v in arm_scale.items() if key in role), 0.1)
                    effort = weight * leg_len * k * rng.uniform(0.5, 2.5)
                    velocity = rng.uniform(4.0, 12.0)
                    tau_req = None
                drawn[role] = {"lo": rng.uniform(0.45, 1.3), "up": rng.uniform(0.45, 1.3),
                               "effort": effort, "velocity": velocity, "tau_req": tau_req}
            d = drawn[role]

            if d["tau_req"] is not None:
                stance_tau[j.name] = d["tau_req"]

            lo, up = d["lo"], d["up"]
            if j.name.startswith("r_") and (j.axis[0] != 0 or j.axis[2] != 0):
                lo, up = up, lo
            center = spec.standing_pose.get(j.name, 0.0)
            j.lower = float(np.clip(center - lo, -2.9, 2.9))
            j.upper = float(np.clip(center + up, -2.9, 2.9))
            j.effort = d["effort"]
            j.velocity = d["velocity"]

        if any(j.name == "waist" for j in spec.joints):
            waist = next(j for j in spec.joints if j.name == "waist")
            waist.effort = weight * leg_len * rng.uniform(0.5, 1.5)
            waist.velocity = rng.uniform(3.0, 8.0)

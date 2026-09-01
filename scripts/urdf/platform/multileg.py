from __future__ import annotations

import math
import re

import numpy as np

from ..core.base import BaseGenerator, GeomSpec, GeomType, JointSpec, LinkSpec, RobotSpec

_LEG_JOINT = re.compile(r"^leg(\d+)([lr])_(.+)$")

class MultilegGenerator(BaseGenerator):

    FAMILY = "multileg"
    FORMS = ("quad", "hex")

    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:

        n_legs = 4 if form == "quad" else 6
        n_rows = n_legs // 2

        length = self._u(rng, "body_length")
        width = min(self._u(rng, "body_width"), length)
        height = self._u(rng, "body_height")
        mount = str(rng.choice(["mammal", "sprawl"]))
        n_seg = int(rng.choice([2, 3]))

        seg_lens = [length * self._u(rng, "seg_length_factor") for _ in range(n_seg)]
        leg_r = float(np.clip(sum(seg_lens) * rng.uniform(0.05, 0.1), 0.012, 0.05))
        foot_r = leg_r * rng.uniform(1.2, 1.8)

        spec = RobotSpec(name=f"multileg_{form}", family=self.FAMILY, form=form, control_tag=form)
        body = LinkSpec("base_link", [GeomSpec(GeomType.BOX, (length, width, height))])
        body.mass = body.geoms[0].volume * self._u(rng, "body_density")
        spec.links.append(body)

        limb_density = self._u(rng, "limb_density")
        pose, seg_lens, stance = self._solve_standing(rng, mount, n_seg, seg_lens, height, foot_r)

        knee_dirs = [int(rng.choice([1, -1])) for _ in range(n_rows)]
        axis_order = str(rng.choice(["roll_pitch", "pitch_roll"])) if mount == "mammal" else "yaw_pitch"

        x_nom = self._row_positions(n_rows, length)
        d_mounts = []
        for row in range(n_rows):
            x = x_nom[row] + rng.uniform(-0.06, 0.06) * length
            if mount == "mammal":
                d_ab = max(width * rng.uniform(0.08, 0.25), leg_r + 0.015)
                d_mounts.append(d_ab)
                for sy in (1, -1):
                    leg = f"leg{row}{'l' if sy > 0 else 'r'}"
                    self._add_leg_mammal(spec, leg, x, sy, width, seg_lens, leg_r, foot_r,
                                         limb_density, knee_dirs[row], axis_order, pose, d_ab)
            else:
                yaw_jit = rng.uniform(-0.25, 0.25)
                d_cox = width * rng.uniform(0.06, 0.15)
                d_mounts.append(d_cox)
                for sy in (1, -1):
                    leg = f"leg{row}{'l' if sy > 0 else 'r'}"
                    self._add_leg_sprawl(spec, leg, x, sy, width, seg_lens, leg_r, foot_r,
                                         limb_density, knee_dirs[row], pose, yaw_jit, d_cox)

        self._set_limits(spec, rng, seg_lens, n_legs, mount, axis_order, pose, stance, d_mounts)
        self._clamp_mass_ratio(spec)
        spec.params.update({
            "body_length": length, "body_width": width, "body_height": height,
            "n_legs": n_legs, "mount": mount, "n_segments": n_seg,
            "seg_lengths": seg_lens, "leg_radius": leg_r, "foot_radius": foot_r,
            "stance_height": stance, "knee_dirs": knee_dirs, "axis_order": axis_order,
            "total_mass": spec.total_mass(),
            "has_wheels": False, "has_legs": True,
            "est_step_height": 0.45 * stance,
        })
        return spec

    def _solve_standing(self, rng, mount, n_seg, seg_lens, height, foot_r):

        if mount == "mammal":

            l1 = seg_lens[0]
            l2 = seg_lens[1] if len(seg_lens) == 3 else sum(seg_lens[1:])
            l3 = seg_lens[2] if len(seg_lens) == 3 else 0.0

            h_min = max(height / 2 - foot_r + 0.05 - l3, 1.05 * abs(l1 - l2))
            if h_min > 0.93 * (l1 + l2):
                scale = h_min / (0.93 * (l1 + l2)) * 1.15
                seg_lens = [s * scale for s in seg_lens]
                l1, l2, l3 = l1 * scale, l2 * scale, l3 * scale
            h2 = float(np.clip((l1 + l2) * rng.uniform(0.6, 0.88), h_min, 0.93 * (l1 + l2)))

            psi = math.acos(np.clip((h2 * h2 - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1, 1))

            gamma = math.atan2(l2 * math.sin(psi), l1 + l2 * math.cos(psi))

            ankle = gamma - psi if l3 > 0 else 0.0

            stance = h2 + l3 + foot_r
            return {"pitch": gamma, "knee": psi, "ankle": ankle}, seg_lens, stance

        femur = rng.uniform(0.25, 0.7)
        beta = rng.uniform(0.15, 0.6) if n_seg == 3 else 0.0

        drop = self._sprawl_drop(seg_lens, femur, beta, foot_r)
        need = height / 2 + 0.05
        if drop < need:
            scale = need / max(drop, 1e-3) * 1.1
            seg_lens = [seg_lens[0]] + [s * scale for s in seg_lens[1:]]
            drop = self._sprawl_drop(seg_lens, femur, beta, foot_r)

        stance = drop
        return {"femur": femur, "beta": beta}, seg_lens, stance

    def _sprawl_drop(self, seg_lens, femur, beta, foot_r):

        if len(seg_lens) == 2:
            return seg_lens[0] * math.sin(femur) + seg_lens[1] + foot_r
        return seg_lens[0] * math.sin(femur) + seg_lens[1] * math.cos(beta) + seg_lens[2] + foot_r

    def _row_positions(self, n_rows, length):

        if n_rows == 2:
            return [0.4 * length, -0.4 * length]
        return [0.42 * length, 0.0, -0.42 * length]

    def _add_leg_mammal(self, spec, leg, x, sy, width, seg_lens, leg_r, foot_r,
                        density, knee_dir, axis_order, pose, d_ab):

        blk = max(leg_r * 1.4, 0.02)
        hip = LinkSpec(f"{leg}_hip", [GeomSpec(GeomType.BOX, (blk, d_ab + blk, blk),
                                               origin_xyz=(0, sy * d_ab / 2, 0))])
        hip.mass = hip.geoms[0].volume * density
        spec.links.append(hip)

        axes = {"roll": (1, 0, 0), "pitch": (0, 1, 0)}
        order = axis_order.split("_")

        spec.joints.append(JointSpec(
            f"{leg}_hip_{order[0]}", "revolute", "base_link", f"{leg}_hip",
            origin_xyz=(x, sy * width / 2, 0), axis=axes[order[0]],
        ))
        spec.standing_pose[f"{leg}_hip_{order[0]}"] =            0.0 if order[0] == "roll" else -knee_dir * pose["pitch"]

        upper = self._seg_link(f"{leg}_upper", seg_lens[0], leg_r, density, None)
        spec.links.append(upper)
        spec.joints.append(JointSpec(
            f"{leg}_hip_{order[1]}", "revolute", f"{leg}_hip", f"{leg}_upper",
            origin_xyz=(0, sy * d_ab, 0), axis=axes[order[1]],
        ))
        spec.standing_pose[f"{leg}_hip_{order[1]}"] =            0.0 if order[1] == "roll" else -knee_dir * pose["pitch"]

        names = ["upper", "mid", "lower"] if len(seg_lens) == 3 else ["upper", "lower"]
        angles = [knee_dir * pose["knee"], knee_dir * pose["ankle"]]
        parent = f"{leg}_upper"
        for i in range(1, len(seg_lens)):
            child = f"{leg}_{names[i]}"
            is_last = i == len(seg_lens) - 1
            link = self._seg_link(child, seg_lens[i], leg_r, density, foot_r if is_last else None)
            spec.links.append(link)
            jname = f"{leg}_{'knee' if i == 1 else 'ankle'}"
            spec.joints.append(JointSpec(
                jname, "revolute", parent, child,
                origin_xyz=(0, 0, -seg_lens[i - 1]), axis=(0, 1, 0),
            ))
            spec.standing_pose[jname] = angles[i - 1]
            parent = child

        spec.contact_links.append(parent)

    def _add_leg_sprawl(self, spec, leg, x, sy, width, seg_lens, leg_r, foot_r,
                        density, knee_dir, pose, yaw_jit, d_cox):

        yaw = sy * (math.pi / 2 + yaw_jit)
        blk = max(leg_r * 1.4, 0.02)
        coxa = LinkSpec(f"{leg}_coxa", [GeomSpec(GeomType.BOX, (d_cox + blk, blk, blk),
                                                 origin_xyz=(d_cox / 2, 0, 0))])
        coxa.mass = coxa.geoms[0].volume * density
        spec.links.append(coxa)
        spec.joints.append(JointSpec(
            f"{leg}_coxa_yaw", "revolute", "base_link", f"{leg}_coxa",
            origin_xyz=(x, sy * width / 2, 0), origin_rpy=(0, 0, yaw), axis=(0, 0, 1),
        ))
        spec.standing_pose[f"{leg}_coxa_yaw"] = 0.0

        femur = LinkSpec(f"{leg}_femur",
                         [GeomSpec(GeomType.CYLINDER, (leg_r, seg_lens[0], 0),
                                   origin_xyz=(seg_lens[0] / 2, 0, 0), origin_rpy=(0, math.pi / 2, 0))])
        femur.mass = femur.geoms[0].volume * density
        spec.links.append(femur)
        spec.joints.append(JointSpec(
            f"{leg}_femur_pitch", "revolute", f"{leg}_coxa", f"{leg}_femur",
            origin_xyz=(d_cox, 0, 0), axis=(0, 1, 0),
        ))
        spec.standing_pose[f"{leg}_femur_pitch"] = pose["femur"]

        names = ["femur", "mid", "lower"] if len(seg_lens) == 3 else ["femur", "lower"]
        if len(seg_lens) == 2:
            angles = [-pose["femur"]]
        else:
            angles = [knee_dir * pose["beta"] - pose["femur"], -knee_dir * pose["beta"]]
        parent = f"{leg}_femur"
        for i in range(1, len(seg_lens)):
            child = f"{leg}_{names[i]}"
            is_last = i == len(seg_lens) - 1
            link = self._seg_link(child, seg_lens[i], leg_r, density, foot_r if is_last else None)
            spec.links.append(link)
            jname = f"{leg}_{'knee' if i == 1 else 'ankle'}"

            origin = (seg_lens[0], 0, 0) if i == 1 else (0, 0, -seg_lens[i - 1])
            spec.joints.append(JointSpec(
                jname, "revolute", parent, child, origin_xyz=origin, axis=(0, 1, 0),
            ))
            spec.standing_pose[jname] = angles[i - 1]
            parent = child

        spec.contact_links.append(parent)

    def _seg_link(self, name, seg_len, leg_r, density, foot_r):

        geoms = [GeomSpec(GeomType.CYLINDER, (leg_r, seg_len, 0), origin_xyz=(0, 0, -seg_len / 2))]
        if foot_r is not None:
            geoms.append(GeomSpec(GeomType.SPHERE, (foot_r, 0, 0), origin_xyz=(0, 0, -seg_len)))
        link = LinkSpec(name, geoms)
        link.mass = sum(g.volume for g in geoms) * density
        return link

    def _stance_arms(self, mount, axis_order, pose, seg_lens, d_ab):

        if mount == "mammal":
            tilt = abs(math.sin(pose["pitch"] - pose["knee"]))
            first, second = axis_order.split("_")
            arms = {
                f"hip_{first}": d_ab if first == "roll" else 0.0,
                f"hip_{second}": 0.0,
                "knee": seg_lens[1] * tilt,
            }
            if len(seg_lens) == 3:
                arms["ankle"] = 0.0
            return arms
        horiz = seg_lens[0] * math.cos(pose["femur"])
        mid = seg_lens[1] * math.sin(pose["beta"]) if len(seg_lens) == 3 else 0.0
        arms = {"coxa_yaw": 0.0, "femur_pitch": horiz + mid, "knee": mid}
        if len(seg_lens) == 3:
            arms["ankle"] = 0.0
        return arms

    def _set_limits(self, spec, rng, seg_lens, n_legs, mount, axis_order, pose, stance, d_mounts):

        g = self._cfg["gait"]
        m = spec.total_mass()
        support = m * 9.81 / (n_legs // 2)
        leg_len = sum(seg_lens)

        reach = {"knee": sum(seg_lens[1:]), "ankle": sum(seg_lens[2:]) or 0.0}

        v_max = g["froude"] * math.sqrt(9.81 * stance)
        w_min = g["vel_margin"] * math.pi * v_max / leg_len
        w_hi = max(g["vel_hi"], g["vel_span"] * w_min)

        drawn: dict[tuple, dict] = {}
        stance_tau = spec.params.setdefault("stance_torque", {})
        for j in spec.actuated_joints():
            row, side, role = _LEG_JOINT.match(j.name).groups()
            key = (row, role)
            if key not in drawn:
                arms = self._stance_arms(mount, axis_order, pose, seg_lens, d_mounts[int(row)])
                arm = max(arms[role], g["arm_min_frac"] * reach.get(role, leg_len))
                drawn[key] = {
                    "lo": rng.uniform(0.45, 1.3), "up": rng.uniform(0.45, 1.3),
                    "tau_req": support * arm,
                    "effort": support * arm * rng.uniform(*g["stance_margin"]),
                    "velocity": rng.uniform(w_min, w_hi),
                }
            d = drawn[key]

            stance_tau[j.name] = d["tau_req"]

            lo, up = d["lo"], d["up"]
            if side == "r" and (j.axis[0] != 0 or j.axis[2] != 0):
                lo, up = up, lo
            center = spec.standing_pose.get(j.name, 0.0)
            j.lower = float(np.clip(center - lo, -2.9, 2.9))
            j.upper = float(np.clip(center + up, -2.9, 2.9))
            j.effort = d["effort"]
            j.velocity = d["velocity"]

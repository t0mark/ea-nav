"""휴머노이드 로봇 생성기. (예: Unitree G1·H1, Atlas, Talos)

기본 구조: 골반(base_link) + 몸통 + 머리 + 다리 2개(고관절 3자유도 + 무릎 +
발목 2자유도) + 팔 2개(선택).
구조 축: 상체 분할(일체형 / waist 조인트 yaw·pitch), 발 형상(박스/원반/앞뒤 분할 —
발 링크 1개에 충돌 형상 2개), 팔 없음/3-7 DoF (Atlas·Talos급 7자유도 팔 커버),
고관절 축 순서(yaw-roll-pitch 순열).
물리 제약: 무릎 굽힘 기립 2링크 IK + 발바닥 수평(피치각 합 0), 발 폭 < 고관절 간격.

기하 규약: base_link 원점 = 골반 중심. 다리·팔 세그먼트는 자기 프레임 -z로 뻗고,
+y축 양의 회전은 말단을 -x(뒤)로 보낸다. 즉 고관절 피치 -gamma = 다리 앞 굽힘.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..core.base import BaseGenerator, GeomSpec, GeomType, JointSpec, LinkSpec, RobotSpec

# 팔 관절 체인 정의 (dof 3-7은 이 목록의 앞에서 자른다. 3 = pitch/roll/elbow 구성)
_ARM_CHAIN = [
    ("shoulder_pitch", (0, 1, 0)), ("shoulder_roll", (1, 0, 0)), ("shoulder_yaw", (0, 0, 1)),
    ("elbow", (0, 1, 0)), ("wrist_pitch", (0, 1, 0)), ("wrist_roll", (1, 0, 0)),
    ("wrist_yaw", (0, 0, 1)),
]


class HumanoidGenerator(BaseGenerator):
    """상체 분할·발 형상·팔 DoF를 랜덤화하는 휴머노이드 생성기."""

    FAMILY = "humanoid"
    FORMS = ("humanoid",)

    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:
        """휴머노이드 스펙 하나를 샘플링한다.

        샘플링 순서: 골반(base) -> 몸통 -> 다리(무릎 굽힘 -> IK) -> 팔 -> 관절 한계.
        무릎 굽힘각을 먼저 뽑고 기립고는 그 결과로 정해진다.
        """
        # 치수·구조 축 샘플 (모든 길이는 독립, 인체 비례에 묶지 않음)
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

        # 다리: 무릎 굽힘각 -> 발목이 고관절 바로 아래 오는 2링크 IK
        # gamma = atan2(l2 sin(knee), l1 + l2 cos(knee)) = 수직선-대퇴 각 (표준 2R 해)
        hip_sep = max(pw * rng.uniform(0.55, 0.95), 2 * leg_r + 0.06)
        knee = rng.uniform(0.08, 0.45)
        gamma = math.atan2(l2 * math.sin(knee), l1 + l2 * math.cos(knee))
        ankle_h = rng.uniform(0.03, 0.08)
        foot_dims, ankle_h, hip_off = self._add_legs(spec, rng, hip_sep, ph, l1, l2, leg_r,
                                                     ankle_h, foot_shape, gamma, knee, limb_density)
        if arm_dof > 0:
            self._add_arms(spec, rng, torso_parent, tw, th, arm_dof, limb_density)

        # 기립고 = 골반 반높이 + 고관절 연쇄 낙차 + 다리 수직 성분 + 발목 높이
        # (관절 한계의 속도 하한이 기립고를 쓰므로 _set_limits보다 먼저 계산)
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

    # ---------- 상체 ----------

    def _add_torso(self, spec, rng, torso_split, waist_axis, dims, ph, density) -> str:
        """상체(몸통+머리) 조립. 팔을 매달 부모 링크 이름을 반환한다.

        merged: 몸통 박스를 base_link의 추가 형상으로 합침 (진짜 단일 링크).
        waist: 골반 상단에 revolute 조인트(yaw=z / pitch=y)로 몸통 링크 분리.
        """
        td, tw, th = dims
        torso_geom = GeomSpec(GeomType.BOX, (td, tw, th))
        head_r = self._u(rng, "head_radius")

        # 단일 상체: 몸통을 base_link 형상에 합친다 (질량도 합산)
        if torso_split == "merged":
            torso_geom.origin_xyz = (0, 0, ph / 2 + th / 2)
            base = spec.links[0]
            base.geoms.append(torso_geom)
            base.mass += torso_geom.volume * density
            self._add_head(spec, rng, "base_link", ph / 2 + th, head_r)
            return "base_link"

        # 분할 상체: waist 조인트로 몸통 링크를 골반 위에 연결
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
        """머리 구체를 fixed 조인트로 부착 (그래프 파서에서 부모에 병합되는 표기).

        top_z = 부모 프레임 기준 몸통 상단 높이. 머리 중심은 그 위 1.1*반지름.
        """
        head = LinkSpec("head", [GeomSpec(GeomType.SPHERE, (head_r, 0, 0))], mass=0.5 + 30 * head_r ** 3)
        spec.links.append(head)
        spec.joints.append(JointSpec(
            "neck", "fixed", parent, "head",
            origin_xyz=(0, 0, top_z + head_r * 1.1),
        ))

    # ---------- 다리 ----------

    def _add_legs(self, spec, rng, hip_sep, ph, l1, l2, leg_r, ankle_h,
                  foot_shape, gamma, knee, density):
        """양쪽 다리 조립. (발 치수, 보정된 발목 높이, 고관절 연쇄 낙차)를 반환.

        관절 체인: 골반 바닥 -[고관절 3연쇄 (yaw z / roll x / pitch y, 순서 랜덤)]-
        대퇴 -[knee y]- 정강이 -[ankle_pitch y]- 블록 -[ankle_roll x]- 발.
        기립 각도: hip_pitch=-gamma, knee=+knee, ankle_pitch=gamma-knee
        -> y축 회전 합 0 = 발바닥 수평 보장. 기립에서 yaw/roll=0이므로
        축 순서는 기립 기하에 영향이 없다. 무릎 방향은 인체형 고정.
        """
        blk = max(leg_r * 1.1, 0.025)

        # 연쇄 낙차: 블록 반지름 + 여유 -> 거리 2 링크(블록-대퇴 등)의 접촉 방지
        off = blk + 0.004

        # 발목 블록(반지름 blk)이 지면을 뚫지 않도록 발목 높이 하한 보정
        ankle_h = max(ankle_h, blk + 0.018)
        foot_dims = self._sample_foot(rng, hip_sep, ankle_h)

        # 고관절 축 순서 랜덤화 (로봇당 1개, 표기 다양성 축)
        base_chain = [("hip_yaw", (0, 0, 1)), ("hip_roll", (1, 0, 0)), ("hip_pitch", (0, 1, 0))]
        order = [int(i) for i in rng.permutation(3)]
        chain = [base_chain[i] for i in order]
        spec.params["hip_axis_order"] = "_".join(c[0].split("_")[1] for c in chain)

        for sy, side in ((1, "l"), (-1, "r")):
            # 고관절 3연쇄: 앞 2개는 작은 구체 블록 링크, 마지막의 자식이 대퇴
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
                # 기립 각도는 피치 축만 -gamma (yaw/roll은 0)
                spec.standing_pose[f"{side}_{jname}"] = -gamma if jname == "hip_pitch" else 0.0
                parent, origin = child, (0, 0, -off)

            # 무릎: 대퇴 하단에서 y축 revolute
            shin = self._limb(f"{side}_shin", l2, leg_r, density)
            spec.links.append(shin)
            spec.joints.append(JointSpec(
                f"{side}_knee", "revolute", f"{side}_thigh", f"{side}_shin",
                origin_xyz=(0, 0, -l1), axis=(0, 1, 0),
            ))
            spec.standing_pose[f"{side}_knee"] = knee

            # 발목 pitch -> roll -> 발 (pitch = gamma - knee로 발바닥 수평 유지)
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
        """발 치수 [길이, 폭, 두께] 샘플.

        폭 < 고관절 간격 - 3cm (좌우 발 간섭 방지, plan 클램프),
        두께는 발목 높이의 80% 이하 (발등이 발목 블록을 넘지 않게).
        """
        lf = self._u(rng, "foot_length")
        wf = min(self._u(rng, "foot_width"), hip_sep - 0.03)
        hf = min(rng.uniform(0.02, 0.05), ankle_h * 0.8)
        return [lf, wf, hf]

    def _foot_geoms(self, foot_shape, foot_dims, ankle_h):
        """발 형상 목록 생성 (발 링크 원점 = 발목 관절).

        발바닥이 발목 아래 ankle_h에 오도록 형상 중심 z = -ankle_h + 두께/2.
        앞뒤 분할(dual)은 링크 1개에 충돌 박스 2개 (plan 표기 명시).
        """
        lf, wf, hf = foot_dims
        z = -ankle_h + hf / 2
        if foot_shape == "box":
            return [GeomSpec(GeomType.BOX, (lf, wf, hf), origin_xyz=(lf * 0.2, 0, z))]
        if foot_shape == "disc":
            r = max(lf, wf) / 2
            return [GeomSpec(GeomType.CYLINDER, (r, hf, 0), origin_xyz=(lf * 0.1, 0, z))]

        # dual: 뒤꿈치 + 앞꿈치 2박스 분할
        return [
            GeomSpec(GeomType.BOX, (lf * 0.45, wf, hf), origin_xyz=(-lf * 0.2, 0, z)),
            GeomSpec(GeomType.BOX, (lf * 0.45, wf * 0.9, hf), origin_xyz=(lf * 0.35, 0, z)),
        ]

    # ---------- 팔 ----------

    def _add_arms(self, spec, rng, torso_parent, tw, th, arm_dof, density):
        """양쪽 팔 조립 (_ARM_CHAIN 앞에서 arm_dof개, 3=어깨2+팔꿈치 ... 7=+손목3).

        어깨는 몸통 옆면 상부, y 오프셋 = 몸통 반폭 + 1.5*팔 반지름 (몸통 간섭 방지).
        기립 자세: shoulder_roll 바깥 벌림(sy*out_roll) + 팔꿈치 앞 굽힘(-elbow_bend)
        으로 몸통과의 충돌을 피한다. 세그먼트: 상완(어깨 연쇄 뒤) / 전완(팔꿈치 뒤)
        / 손 구체(손목 연쇄 끝). 관절 사이는 작은 구체 블록으로 연결.
        """
        la, lf = self._u(rng, "upper_arm_length"), self._u(rng, "forearm_length")
        arm_r = float(np.clip((la + lf) * rng.uniform(0.05, 0.09), 0.015, 0.05))
        blk = max(arm_r * 1.1, 0.02)

        # 연쇄 낙차: 블록 접촉 방지 여유 (다리와 동일 원칙)
        off = blk + 0.004

        # dof 3: 어깨 pitch/roll + 팔꿈치 (yaw는 4부터) -> 체인 구성
        chain = list(_ARM_CHAIN[:arm_dof])
        if arm_dof == 3:
            chain = [_ARM_CHAIN[0], _ARM_CHAIN[1], _ARM_CHAIN[3]]

        # merged면 어깨 z가 base 프레임 기준이므로 몸통 형상 오프셋을 더한다
        z_sh = th * rng.uniform(0.75, 0.95) if torso_parent == "torso" \
            else spec.links[0].geoms[-1].origin_xyz[2] + th * rng.uniform(0.25, 0.45)
        out_roll = rng.uniform(0.05, 0.2)
        elbow_bend = rng.uniform(0.05, 0.35)
        for sy, side in ((1, "l"), (-1, "r")):
            parent = torso_parent
            origin = (0, sy * (tw / 2 + arm_r * 1.5), z_sh)

            # 세그먼트 경계(elbow 뒤 = 전완, 마지막 손목 뒤 = 손) 판단하며 체인 조립
            for i, (jname, axis) in enumerate(chain):
                is_elbow = jname == "elbow"
                is_last = i == len(chain) - 1

                # 자식 링크 결정: elbow -> 전완, 마지막 손목 -> 손, 어깨 마지막 -> 상완
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

                # 기립 각도: roll 바깥 벌림, 팔꿈치 앞 굽힘, 나머지 0
                if jname == "shoulder_roll":
                    spec.standing_pose[f"{side}_{jname}"] = sy * out_roll
                elif is_elbow:
                    spec.standing_pose[f"{side}_{jname}"] = -elbow_bend
                else:
                    spec.standing_pose[f"{side}_{jname}"] = 0.0

                # 다음 조인트 원점: 세그먼트 링크면 그 길이만큼, 블록이면 낙차만큼
                if child.endswith("upper_arm"):
                    origin = (0, 0, -la)
                elif child.endswith("forearm"):
                    origin = (0, 0, -lf)
                else:
                    origin = (0, 0, -off)
                parent = child
        spec.params.update({"upper_arm_length": la, "forearm_length": lf})

    def _limb(self, name, length, radius, density):
        """-z 방향 실린더 사지 링크 생성 (원점 = 상단 관절, 중심 = -length/2)."""
        link = LinkSpec(name, [GeomSpec(GeomType.CYLINDER, (radius, length, 0),
                                        origin_xyz=(0, 0, -length / 2))])
        link.mass = link.geoms[0].volume * density
        return link

    # ---------- 한계 설정 ----------

    def _set_limits(self, spec, rng, l1, l2, gamma, knee, foot_dims, stance):
        """관절 가동 범위·토크·속도 한계를 역할 단위 샘플 + 좌우 미러로 설정.

        다리 토크 = 단일 지지(F = m g — 보행 스윙 국면은 한 다리가 전 체중 지탱)
        정적 요구 x 여유율 (configs/urdf.yaml gait 규약, multileg와 동일 원칙):
        모멘트 팔 = max(기립 자세 수평 팔, arm_min_frac x 관절 아래 도달 길이).
        발목은 발 지렛대(CoP 이동 범위 = 발 길이/폭의 절반)가 물리적 팔.
        다리 속도 하한 = vel_margin x 스윙 피크 요구 (pi x v_max / 다리 전장,
        v_max = froude_biped x sqrt(g x 기립고)).
        팔·waist는 비로코모션 (롤아웃 홈 포즈 PD 홀드)이라 종전 스케일 계수
        규칙 유지 — 단 좌우는 미러 (역할 단위 1회 샘플).
        roll/yaw 축 관절의 가동 범위는 우측에서 마진을 교환한다 (xz평면 반사).
        """
        g = self._cfg["gait"]
        m = spec.total_mass()
        weight = m * 9.81
        leg_len = l1 + l2

        # 기립 자세 수평 팔: 정강이 기울기 |sin(knee - gamma)| (발은 고관절 바로 아래),
        # 발목은 발 접촉면의 CoP 이동 반범위
        tilt = abs(math.sin(knee - gamma))
        arms = {"hip_yaw": 0.0, "hip_roll": 0.0, "hip_pitch": 0.0,
                "knee": l2 * tilt, "ankle_pitch": foot_dims[0] / 2, "ankle_roll": foot_dims[1] / 2}
        reach = {"hip_yaw": leg_len, "hip_roll": leg_len, "hip_pitch": leg_len, "knee": l2}

        # 스윙 피크 속도 요구 (multileg와 동일 근사, biped froude)
        v_max = g["froude_biped"] * math.sqrt(9.81 * stance)
        w_min = g["vel_margin"] * math.pi * v_max / leg_len
        w_hi = max(g["vel_hi_biped"], g["vel_span"] * w_min)

        # 비로코모션(팔) 스케일 계수 (종전 규칙 유지 — 기준 토크만 체중 x 다리 전장)
        arm_scale = {"shoulder": 0.25, "elbow": 0.15, "wrist": 0.08}

        # 역할(이름에서 좌우 접두사 제거) 단위로 1회 샘플 -> 좌우 동일 값 (미러 규약)
        drawn: dict[str, dict] = {}
        stance_tau = spec.params.setdefault("stance_torque", {})
        for j in spec.actuated_joints():
            if j.name == "waist":
                continue
            role = re.sub(r"^[lr]_", "", j.name)
            if role not in drawn:
                if role in arms:
                    # 다리: 스탠스 요구 앵커
                    arm = max(arms[role], g["arm_min_frac"] * reach.get(role, 0.0))
                    effort = weight * arm * rng.uniform(*g["stance_margin"])
                    velocity = rng.uniform(w_min, w_hi)
                    tau_req = weight * arm
                else:
                    # 팔: 종전 스케일 규칙 (스탠스 요구 없음 — tau_req 미기록)
                    k = next((v for key, v in arm_scale.items() if key in role), 0.1)
                    effort = weight * leg_len * k * rng.uniform(0.5, 2.5)
                    velocity = rng.uniform(4.0, 12.0)
                    tau_req = None
                drawn[role] = {"lo": rng.uniform(0.45, 1.3), "up": rng.uniform(0.45, 1.3),
                               "effort": effort, "velocity": velocity, "tau_req": tau_req}
            d = drawn[role]
            # 다리 관절별 스탠스 정적 요구 [Nm] — 하위 단계(RL 게인 앵커)가 사용
            if d["tau_req"] is not None:
                stance_tau[j.name] = d["tau_req"]

            # roll(x)/yaw(z) 축은 xz평면 미러가 상·하한을 교환한다 (pitch는 그대로)
            lo, up = d["lo"], d["up"]
            if j.name.startswith("r_") and (j.axis[0] != 0 or j.axis[2] != 0):
                lo, up = up, lo
            center = spec.standing_pose.get(j.name, 0.0)
            j.lower = float(np.clip(center - lo, -2.9, 2.9))
            j.upper = float(np.clip(center + up, -2.9, 2.9))
            j.effort = d["effort"]
            j.velocity = d["velocity"]

        # waist 토크·속도는 별도 샘플 (가동 범위는 _add_torso에서 설정)
        if any(j.name == "waist" for j in spec.joints):
            waist = next(j for j in spec.joints if j.name == "waist")
            waist.effort = weight * leg_len * rng.uniform(0.5, 1.5)
            waist.velocity = rng.uniform(3.0, 8.0)

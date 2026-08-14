"""wheeled 휴머노이드 (모바일 매니퓰레이터) 생성기. (예: PR2, Fetch, Tiago)

기본 구조: wheeled 베이스(diff/skid/omni 재사용) + 휴머노이드 상체(몸통 + 팔 + 머리).
구조 축: 베이스 타입(balancing 서브타입은 상속 제외 — 전복 검사와 양립 불가),
몸통 리프트(고정 / prismatic — Fetch torso lift), 팔 1-2개 x 2-7 DoF, 머리 유무.
물리 제약: 전복 안정성 비율(특수 검사), 리프트 최대·팔 전방 자세 추가 검사,
베이스 상속 검사(하중 비율)는 상체 포함 전체 무게중심 기준으로 재평가
(특수 검사가 전체 질량중심을 쓰므로 자동 충족).
롤아웃 명목 자세: 리프트 최하단, 팔 홈 포즈 PD 홀드 (plan GT 정의 고정).
"""
from __future__ import annotations

import math

import numpy as np

from ..core.base import BaseGenerator, GeomSpec, GeomType, JointSpec, LinkSpec, RobotSpec
from .diff import DiffGenerator
from .omni import OmniGenerator
from .skid import SkidGenerator

# 팔 관절 체인 (dof 3-7은 앞에서 자르고, 3은 pitch/roll/elbow, 2는 pitch/elbow 구성)
_ARM_CHAIN = [
    ("shoulder_pitch", (0, 1, 0)), ("shoulder_roll", (1, 0, 0)), ("shoulder_yaw", (0, 0, 1)),
    ("elbow", (0, 1, 0)), ("wrist_pitch", (0, 1, 0)), ("wrist_roll", (1, 0, 0)),
    ("wrist_yaw", (0, 0, 1)),
]


class WheeledHumanoidGenerator(BaseGenerator):
    """wheeled 베이스 위에 상체를 얹는 생성기 (베이스 생성 로직 재사용)."""

    FAMILY = "wheeled_humanoid"
    FORMS = ("wheeled_humanoid",)

    def __init__(self, cfg: dict):
        """베이스 생성기 3종을 함께 보관한다 (같은 cfg 공유)."""
        super().__init__(cfg)
        self._bases = {"diff": DiffGenerator(cfg), "skid": SkidGenerator(cfg),
                       "omni": OmniGenerator(cfg)}

    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:
        """베이스 샘플 -> 상체 부착 -> 한계 재설정 순서로 조립한다."""
        # 베이스 타입 선택 후 해당 생성 로직 재사용 (diff는 balancing 제외)
        base_type = str(rng.choice(self._cfg[self.FAMILY]["base_types"]))
        if base_type == "diff":
            spec = self._bases["diff"].sample("diff", rng, allow_balancing=False)
        else:
            spec = self._bases[base_type].sample(base_type, rng)
        base_tag = spec.control_tag

        # 정체성 재지정 (베이스 파라미터·특수 검사·추가 자세는 그대로 상속)
        spec.name = "wheeled_humanoid"
        spec.family = self.FAMILY
        spec.form = form
        spec.control_tag = f"wheeled_humanoid_{base_tag}"

        # 상체 조립: 몸통(리프트) -> 팔 -> 머리
        geo_top = spec.params["body_height"] / 2
        torso, th = self._add_torso(spec, rng, geo_top)
        self._add_arms(spec, rng, torso, th)
        if rng.random() < 0.7:
            self._add_head(spec, rng, torso, th)

        # 전복 안정성 특수 검사 + 상체 반영 구동 한계 재설정
        spec.special["overturn"] = {}
        self._bases[base_type]._set_drive_limits(spec, rng, spec.params["wheel_radius"])
        self._clamp_mass_ratio(spec)
        spec.params.update({
            "base_type": base_type, "total_mass": spec.total_mass(),
            "has_upper_body": True,
        })
        return spec

    # ---------- 상체 몸통 ----------

    def _add_torso(self, spec, rng, base_top: float) -> tuple[str, float]:
        """상체 몸통을 베이스 상면에 부착. (몸통 링크 이름, 몸통 높이) 반환.

        장착은 리프트(prismatic z) 또는 fixed (파서 병합되는 표기).
        리프트 스트로크 최대 인출 자세를 추가 검사로 등록한다.
        """
        W, L = spec.params["body_width"], spec.params["body_length"]
        tw = W * rng.uniform(0.5, 0.9)
        td = L * rng.uniform(0.2, 0.5)
        th = self._u(rng, "torso_height")
        torso = LinkSpec("torso", [GeomSpec(GeomType.BOX, (td, tw, th),
                                            origin_xyz=(0, 0, th / 2))])
        torso.mass = torso.geoms[0].volume * self._u(rng, "body_density")
        spec.links.append(torso)

        # 장착 위치는 전후 지터, 리프트 여부는 구조 축
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
            # 명목 자세 = 최하단 (plan GT 정의), 최대 인출은 추가 검사
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

    # ---------- 팔 ----------

    def _add_arms(self, spec, rng, torso: str, th: float):
        """팔 1-2개를 몸통에 부착 (팔당 2-7 DoF).

        2개면 몸통 좌우, 1개면 전면 중앙(Fetch식). 기립 자세는 팔을 아래로
        늘어뜨리되 팔꿈치를 앞으로 굽혀 몸통·베이스와 간섭을 피하고,
        말단이 베이스 상면 아래로 내려가지 않도록 도달 길이를 클램프한다.
        팔 전방 뻗기 자세를 추가 검사로 등록한다 (plan 물리 제약).
        """
        n_arms = int(rng.choice([1, 2]))
        arm_dof = int(rng.integers(2, 8))
        tw = spec.params["torso"][1]
        td = spec.params["torso"][0]

        # 체인 구성: 2 = pitch+elbow, 3 = pitch/roll/elbow, 4-7 = 목록 앞에서 자름
        if arm_dof == 2:
            chain = [_ARM_CHAIN[0], _ARM_CHAIN[3]]
        elif arm_dof == 3:
            chain = [_ARM_CHAIN[0], _ARM_CHAIN[1], _ARM_CHAIN[3]]
        else:
            chain = list(_ARM_CHAIN[:arm_dof])

        # 팔 길이: 어깨 높이 아래로 뻗어도 베이스 상면(z=0)에 닿지 않게 클램프
        z_sh = th * rng.uniform(0.75, 0.9)
        la, lf = self._u(rng, "upper_arm_length"), self._u(rng, "forearm_length")
        reach_max = z_sh - 0.03
        if la + lf > reach_max:
            k = reach_max / (la + lf)
            la, lf = max(la * k, 0.05), max(lf * k, 0.05)
        arm_r = float(np.clip((la + lf) * rng.uniform(0.06, 0.1), 0.015, 0.05))
        blk = max(arm_r * 1.1, 0.02)
        off = blk + 0.004

        # 팔 부착 위치: 2개 = 좌우 옆면, 1개 = 전면 중앙
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

                # 자식 링크: elbow -> 전완, elbow 직전 -> 상완, 마지막 손목 -> 손, 그 외 블록
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

                # 기립 각도: roll 바깥 벌림(중앙 팔은 0), 팔꿈치 앞 굽힘, 나머지 0
                if jname == "shoulder_roll":
                    spec.standing_pose[f"{side}_{jname}"] = sy * out_roll
                elif is_elbow:
                    spec.standing_pose[f"{side}_{jname}"] = -elbow_bend
                else:
                    spec.standing_pose[f"{side}_{jname}"] = 0.0

                # 다음 원점: 세그먼트는 길이만큼, 블록은 낙차만큼
                if child.endswith("upper_arm"):
                    origin = (0, 0, -la)
                elif child.endswith("forearm"):
                    origin = (0, 0, -lf)
                else:
                    origin = (0, 0, -off)
                parent = child

            # 팔 전방 뻗기 자세 추가 검사 (전복·간섭 확인용, plan 물리 제약)
            spec.check_poses.append({f"{side}_shoulder_pitch": -1.2})

        # 관절 한계·토크: 팔 질량 모멘트 기준 여유율
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
        """머리 구체를 몸통 상단에 fixed로 부착 (파서 병합되는 표기)."""
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
        """-z 방향 실린더 사지 링크 (질량은 호출부에서 밀도 곱으로 채움)."""
        link = LinkSpec(name, [GeomSpec(GeomType.CYLINDER, (radius, length, 0),
                                        origin_xyz=(0, 0, -length / 2))])
        return link

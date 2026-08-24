"""다족보행 로봇 생성기. (예: 4족 = Go2·ANYmal·Spot, 6족 = 헥사포드)

기본 구조: 몸통 + 다리 N개 (N ∈ {4, 6}), 다리 = 고관절 2자유도 + 무릎 (+ 발목), 발끝 구.
구조 축: 다리 개수(quad/hex), 장착(포유류형 mammal / 파충류형 sprawling),
무릎 방향(elbow/knee, 다리 열별 독립), 세그먼트 수(2/3절),
고관절 축 순서(mammal: roll-pitch / pitch-roll, sprawling: yaw-pitch — 실물 6족 coxa 구조),
다리 열 위치 지터(열 단위 — 좌우는 항상 미러).
물리 제약: 기립 도달 가능성(고관절-발 유클리드 거리 기준), 기립고 부족 시 다리 자동 보정.

좌우 대칭 규약: 배치·치수·관절 한계는 열(row) 단위로 1회 샘플하고 좌우에 미러로
적용한다 (실로봇·레퍼런스 GenLoco/X-Nav/GenBot-1K 전부 대칭 — 관절 단위 독립
샘플은 대칭 걸음 해가 없는 개체를 만들었던 결함). roll/yaw 축 관절의 가동 범위는
미러 시 상·하한 마진을 교환한다 (xz평면 반사 대칭).

토크·속도 한계 규약 (configs/urdf.yaml gait 섹션):
- 토크 하한 = 스탠스(절반 다리 지지: quad 트롯 2 / hex 트라이포드 3) 정적 요구
  x 여유율. 모멘트 팔 = max(기립 자세 수평 팔, arm_min_frac x 관절 아래 도달 길이)
  — 보폭 자세에서 발이 관절 아래를 벗어나는 만큼의 하한 확보
- 속도 하한 = vel_margin x 스윙 피크 요구 (pi x v_max / 다리 전장, duty 0.5
  사인 스윙 근사. v_max = froude x sqrt(g x 기립고) — rl.yaml 명령 상한 규칙과 정합)

기하 규약: base_link 원점 = 몸통 중심, 고관절 마운트는 몸통 옆면 중간 높이(z=0).
세그먼트 링크는 자기 프레임 -z로 뻗고 (sprawling 대퇴는 +x), 회전 축 +y의 양의
각도는 오른손 법칙으로 다리 끝을 -x(뒤)로 보낸다.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..core.base import BaseGenerator, GeomSpec, GeomType, JointSpec, LinkSpec, RobotSpec

# 관절 이름에서 (열, 좌우, 역할)을 분리하는 패턴: leg{row}{l|r}_{role}
_LEG_JOINT = re.compile(r"^leg(\d+)([lr])_(.+)$")


class MultilegGenerator(BaseGenerator):
    """다리 개수·장착 방식·세그먼트 구성을 랜덤화하는 다족보행 생성기."""

    FAMILY = "multileg"
    FORMS = ("quad", "hex")

    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:
        """form(quad/hex)에 해당하는 다족보행 스펙 하나를 샘플링한다.

        샘플링 순서: 몸통 -> 세그먼트 길이(몸통 비례) -> 기립 자세 해석
        -> 다리 조립(열 단위 샘플, 좌우 미러) -> 관절 한계. 기립 자세를 먼저
        풀어 다리가 짧으면 (몸통이 지면에 닿으면) 그 단계에서 보정한다.
        """
        n_legs = 4 if form == "quad" else 6
        n_rows = n_legs // 2

        # 몸통 치수와 구조 축 샘플
        length = self._u(rng, "body_length")
        width = min(self._u(rng, "body_width"), length)
        height = self._u(rng, "body_height")
        mount = str(rng.choice(["mammal", "sprawl"]))
        n_seg = int(rng.choice([2, 3]))

        # 세그먼트 길이는 몸통 비례 범위에서 각자 독립 샘플 (고정 비율 금지)
        seg_lens = [length * self._u(rng, "seg_length_factor") for _ in range(n_seg)]
        leg_r = float(np.clip(sum(seg_lens) * rng.uniform(0.05, 0.1), 0.012, 0.05))
        foot_r = leg_r * rng.uniform(1.2, 1.8)

        spec = RobotSpec(name=f"multileg_{form}", family=self.FAMILY, form=form, control_tag=form)
        body = LinkSpec("base_link", [GeomSpec(GeomType.BOX, (length, width, height))])
        body.mass = body.geoms[0].volume * self._u(rng, "body_density")
        spec.links.append(body)

        # 기립 자세 계산 (장착 방식별) + 필요시 다리 길이 보정
        limb_density = self._u(rng, "limb_density")
        pose, seg_lens, stance = self._solve_standing(rng, mount, n_seg, seg_lens, height, foot_r)

        # 무릎 방향은 다리 열별 독립, 고관절 축 순서는 로봇당 1개
        knee_dirs = [int(rng.choice([1, -1])) for _ in range(n_rows)]
        axis_order = str(rng.choice(["roll_pitch", "pitch_roll"])) if mount == "mammal" else "yaw_pitch"

        # 다리 조립: 배치·마운트 오프셋은 열 단위 1회 샘플 -> 좌우 미러 적용
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

    # ---------- 기립 자세 ----------

    def _solve_standing(self, rng, mount, n_seg, seg_lens, height, foot_r):
        """기립 자세 각도를 계산한다. 몸통 바닥이 뜰 수 없으면 다리를 늘린다.

        도달 가능성 원칙 (plan): 고관절에서 명목 발 위치까지의 유클리드 거리가
        다리 도달 범위 안이어야 한다 — mammal은 발이 고관절 바로 아래라 수직
        거리 h로 충분하고, sprawling은 수평 벌림을 포함한 구성으로 직접 만든다.

        반환: (pose, 보정된 seg_lens, stance = 몸통 중심의 지면 높이).
        pose = {"pitch","knee","ankle"} (mammal) / {"femur","beta"} (sprawl)
        """
        if mount == "mammal":
            # 2링크 IK 대상: l1 = 대퇴, l2 = 정강이. 3절은 마지막 세그먼트(l3)를
            # 수직으로 세우는 지행(digitigrade) 지그재그 — 발목 0도 일직선 기립은
            # 무릎-발목 열이 좌굴 특이점이라 시뮬에서 홀드 불안정 실측 (mammal
            # 3절만 전도·요동, sprawl·2절·go2는 안정 — 홀드 진단 대조)
            l1 = seg_lens[0]
            l2 = seg_lens[1] if len(seg_lens) == 3 else sum(seg_lens[1:])
            l3 = seg_lens[2] if len(seg_lens) == 3 else 0.0

            # 도달 조건 |l1-l2| < h2 < l1+l2 (h2 = 고관절-발목 수직 낙차).
            # 하한 = 몸통 바닥이 뜨는 높이 - 수직 l3, 상한 0.93(l1+l2)은 특이 자세 마진
            h_min = max(height / 2 - foot_r + 0.05 - l3, 1.05 * abs(l1 - l2))
            if h_min > 0.93 * (l1 + l2):
                scale = h_min / (0.93 * (l1 + l2)) * 1.15
                seg_lens = [s * scale for s in seg_lens]
                l1, l2, l3 = l1 * scale, l2 * scale, l3 * scale
            h2 = float(np.clip((l1 + l2) * rng.uniform(0.6, 0.88), h_min, 0.93 * (l1 + l2)))

            # 법코사인: h2^2 = l1^2 + l2^2 + 2 l1 l2 cos(psi)
            # (psi = 무릎 굽힘각, 0 = 곧게 폄, 무릎 내각 = pi - psi)
            psi = math.acos(np.clip((h2 * h2 - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1, 1))

            # gamma = 수직선-대퇴 사이 각 (표준 2R IK 해).
            # 고관절 피치 -gamma, 무릎 +psi를 주면 발목이 고관절 바로 아래에 온다
            gamma = math.atan2(l2 * math.sin(psi), l1 + l2 * math.cos(psi))

            # 발목 = gamma - psi: 조립이 관절각에 knee_dir을 곱하므로 아래 누적
            # 피치 = knee_dir*(psi-gamma) + knee_dir*(gamma-psi) = 0 -> 마지막
            # 세그먼트 수직 (지그재그 성립, knee_dir과 무관). IK 항등식
            # l1 sin(gamma) = l2 sin(psi-gamma)로 발목은 고관절 바로 아래
            ankle = gamma - psi if l3 > 0 else 0.0

            # 고관절이 몸통 중심 높이(z=0) -> 몸통 중심 높이 = h2 + l3 + 발 구체 반지름
            stance = h2 + l3 + foot_r
            return {"pitch": gamma, "knee": psi, "ankle": ankle}, seg_lens, stance

        # sprawling: 대퇴는 바깥으로 femur만큼 내려가고 마지막 세그먼트는 수직
        femur = rng.uniform(0.25, 0.7)
        beta = rng.uniform(0.15, 0.6) if n_seg == 3 else 0.0

        # 고관절 아래 낙차가 몸통 반높이보다 작으면 지면에 닿으므로
        # 원위 세그먼트를 늘려 보정 (대퇴는 유지해 발 벌림 폭 보존)
        drop = self._sprawl_drop(seg_lens, femur, beta, foot_r)
        need = height / 2 + 0.05
        if drop < need:
            scale = need / max(drop, 1e-3) * 1.1
            seg_lens = [seg_lens[0]] + [s * scale for s in seg_lens[1:]]
            drop = self._sprawl_drop(seg_lens, femur, beta, foot_r)

        # 고관절이 몸통 중심 높이(z=0) -> 몸통 중심 높이 = drop
        stance = drop
        return {"femur": femur, "beta": beta}, seg_lens, stance

    def _sprawl_drop(self, seg_lens, femur, beta, foot_r):
        """파충류형 기립에서 고관절 기준 발바닥까지 수직 낙차 [m].

        대퇴 낙차 l1*sin(femur) + (3절이면 중간 세그 수직 성분 l2*cos(beta))
        + 수직인 마지막 세그먼트 + 발 구체 반지름.
        """
        if len(seg_lens) == 2:
            return seg_lens[0] * math.sin(femur) + seg_lens[1] + foot_r
        return seg_lens[0] * math.sin(femur) + seg_lens[1] * math.cos(beta) + seg_lens[2] + foot_r

    def _row_positions(self, n_rows, length):
        """다리 열의 공칭 전후 위치 [m]. 몸통 길이의 80-84% 범위에 등간격."""
        if n_rows == 2:
            return [0.4 * length, -0.4 * length]
        return [0.42 * length, 0.0, -0.42 * length]

    # ---------- 포유류형 다리 ----------

    def _add_leg_mammal(self, spec, leg, x, sy, width, seg_lens, leg_r, foot_r,
                        density, knee_dir, axis_order, pose, d_ab):
        """포유류형 다리 조립: 힙 블록(2연쇄 고관절) + 수직 하강 세그먼트 체인.

        관절 구성: 몸통 -[roll|pitch]- 힙 블록 -[나머지 고관절]- 대퇴 -[knee]-
        (중간) -[ankle]- 말단. 기립 각도 부호: 고관절 피치 = -knee_dir*gamma,
        무릎 = +knee_dir*psi (knee_dir = 열별 elbow/knee 방향, IK 해는 부호 대칭).
        d_ab = 어브덕션 오프셋 (열 단위 샘플 -> 좌우 공유, sample()에서 전달).
        """
        blk = max(leg_r * 1.4, 0.02)
        hip = LinkSpec(f"{leg}_hip", [GeomSpec(GeomType.BOX, (blk, d_ab + blk, blk),
                                               origin_xyz=(0, sy * d_ab / 2, 0))])
        hip.mass = hip.geoms[0].volume * density
        spec.links.append(hip)

        axes = {"roll": (1, 0, 0), "pitch": (0, 1, 0)}
        order = axis_order.split("_")

        # 첫 관절: 몸통 옆면 -> 힙 블록 (기립 각도는 피치 축에만 부여)
        spec.joints.append(JointSpec(
            f"{leg}_hip_{order[0]}", "revolute", "base_link", f"{leg}_hip",
            origin_xyz=(x, sy * width / 2, 0), axis=axes[order[0]],
        ))
        spec.standing_pose[f"{leg}_hip_{order[0]}"] = \
            0.0 if order[0] == "roll" else -knee_dir * pose["pitch"]

        # 둘째 관절: 힙 블록 바깥 끝 -> 대퇴
        upper = self._seg_link(f"{leg}_upper", seg_lens[0], leg_r, density, None)
        spec.links.append(upper)
        spec.joints.append(JointSpec(
            f"{leg}_hip_{order[1]}", "revolute", f"{leg}_hip", f"{leg}_upper",
            origin_xyz=(0, sy * d_ab, 0), axis=axes[order[1]],
        ))
        spec.standing_pose[f"{leg}_hip_{order[1]}"] = \
            0.0 if order[1] == "roll" else -knee_dir * pose["pitch"]

        # 무릎(-발목) 체인: 각 세그먼트 하단(-z)에서 y축 revolute로 연결
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

        # 말단 세그먼트(발 구체 포함)가 접촉 링크
        spec.contact_links.append(parent)

    # ---------- 파충류형 다리 ----------

    def _add_leg_sprawl(self, spec, leg, x, sy, width, seg_lens, leg_r, foot_r,
                        density, knee_dir, pose, yaw_jit, d_cox):
        """파충류형 다리 조립: 코사(z yaw) + 수평 대퇴(+x) + 수직 하강 체인.

        마운트 yaw = sy*(pi/2 + yaw_jit) -> 다리 로컬 +x가 몸통 바깥을 향하고
        좌우가 xz평면 미러가 된다 (yaw_jit·d_cox는 열 단위 샘플, sample()에서 전달).
        회전 합성(모두 y축)은 누적되므로 마지막 세그먼트를 수직으로
        만드는 조건은 "y축 각도 합 = 0":
        - 2절: knee = -femur
        - 3절: knee = knee_dir*beta - femur, ankle = -knee_dir*beta
        고관절 축 순서 = yaw-pitch (실물 6족의 coxa yaw 구조, plan M4 반영).
        """
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

        # 대퇴: 바깥쪽 +x 실린더 (rpy로 실린더 축 z -> x 회전).
        # femur 피치(+y) 양의 각도는 R_y(θ)(1,0,0) = (cosθ, 0, -sinθ) -> 아래로 내려감
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

        # 무릎(-발목): "y축 각도 합 = 0" 조건으로 기립 각 구성 (docstring)
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

            # 무릎은 대퇴 끝(+x 방향), 발목은 세그먼트 하단(-z 방향)
            origin = (seg_lens[0], 0, 0) if i == 1 else (0, 0, -seg_lens[i - 1])
            spec.joints.append(JointSpec(
                jname, "revolute", parent, child, origin_xyz=origin, axis=(0, 1, 0),
            ))
            spec.standing_pose[jname] = angles[i - 1]
            parent = child

        spec.contact_links.append(parent)

    def _seg_link(self, name, seg_len, leg_r, density, foot_r):
        """-z 방향으로 뻗는 세그먼트 링크 생성. foot_r가 있으면 끝에 발 구체를 붙인다.

        링크 원점 = 상단 관절 위치, 실린더 중심 = (0,0,-len/2), 발 구체 = (0,0,-len).
        """
        geoms = [GeomSpec(GeomType.CYLINDER, (leg_r, seg_len, 0), origin_xyz=(0, 0, -seg_len / 2))]
        if foot_r is not None:
            geoms.append(GeomSpec(GeomType.SPHERE, (foot_r, 0, 0), origin_xyz=(0, 0, -seg_len)))
        link = LinkSpec(name, geoms)
        link.mass = sum(g.volume for g in geoms) * density
        return link

    # ---------- 한계 설정 ----------

    def _stance_arms(self, mount, axis_order, pose, seg_lens, d_ab):
        """역할별 스탠스 모멘트 팔 [m] (기립 자세, 수직 GRF가 발에 걸릴 때).

        mammal: 발이 둘째 고관절 바로 아래 -> 고관절 피치 팔 0, 첫 관절이 roll이면
        팔 = 어브덕션 오프셋 d_ab (발이 roll 축에서 d_ab만큼 옆). 정강이는
        수직에서 |gamma - psi| 기울어짐 -> 무릎 팔 = 정강이 x sin|gamma - psi|,
        3절의 마지막 세그먼트는 수직(지행 기립) -> 발목 팔 0.
        sprawl: 대퇴가 수평 성분 l1 cos(femur)를 만들고 (3절은 중간 세그 sin(beta)
        추가), 마지막 세그먼트는 수직 -> 발목 팔 0. 수직 GRF는 yaw 토크 0.
        반환 key는 관절 역할명 (hip_roll 등 — 첫/둘째 관절 구분 포함).
        """
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
        """관절 가동 범위·토크·속도 한계를 열 단위 샘플 + 좌우 미러로 설정.

        토크 = 스탠스 하중 x 모멘트 팔 x 여유율 (모듈 docstring 규약):
        F = m g / (n_legs/2) — 절반 다리 지지 (트롯/트라이포드), 팔은 기립 자세
        수평 팔과 하한(arm_min_frac x 관절 아래 도달 길이) 중 큰 값.
        속도 = U(스윙 피크 요구 x vel_margin, max(vel_hi, vel_span x 하한)).
        가동 범위: 기립 각도 중심 마진 — roll/yaw 축은 우측에서 마진 교환 (미러).
        """
        g = self._cfg["gait"]
        m = spec.total_mass()
        support = m * 9.81 / (n_legs // 2)
        leg_len = sum(seg_lens)

        # 역할별 관절 아래 도달 길이 (팔 하한용): 고관절 = 다리 전장
        reach = {"knee": sum(seg_lens[1:]), "ankle": sum(seg_lens[2:]) or 0.0}

        # 스윙 피크 속도 요구: v_max = froude x sqrt(g h), omega = pi v / 다리 전장
        v_max = g["froude"] * math.sqrt(9.81 * stance)
        w_min = g["vel_margin"] * math.pi * v_max / leg_len
        w_hi = max(g["vel_hi"], g["vel_span"] * w_min)

        # 열 x 역할 단위로 1회 샘플해 좌우가 같은 값을 쓴다 (미러 규약)
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
            # 관절별 스탠스 정적 요구 [Nm] — 하위 단계(RL 게인 앵커)가 사용
            stance_tau[j.name] = d["tau_req"]

            # roll(x)/yaw(z) 축은 xz평면 미러가 상·하한을 교환한다 (pitch는 그대로)
            lo, up = d["lo"], d["up"]
            if side == "r" and (j.axis[0] != 0 or j.axis[2] != 0):
                lo, up = up, lo
            center = spec.standing_pose.get(j.name, 0.0)
            j.lower = float(np.clip(center - lo, -2.9, 2.9))
            j.upper = float(np.clip(center + up, -2.9, 2.9))
            j.effort = d["effort"]
            j.velocity = d["velocity"]

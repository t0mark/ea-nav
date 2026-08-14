"""diff (차동 구동) 생성기. (예: TurtleBot3, 룸바, Pioneer 3-DX)

기본 구조: 몸통 + 좌우 구동 휠 2개 (continuous, 회전축 공유).
구조 축: 몸통 형상(원통/박스/원통+박스 적층), 구동축 앞뒤 위치(±35%L),
캐스터 종류(없음=2륜 balancing / 볼 / 스위블), 캐스터 배치(반대편 1 / 양쪽 각 1 /
반대편 2 벌림), 휠 노출(밖/몸통 아래 숨김).
물리 제약: 구동륜 하중 비율(특수 검사), balancing 전용 4검사 — 무게중심을
구동축 조건에 맞춰 구성적으로 샘플링해 기각률을 낮춘다 (plan 권장).
"""
from __future__ import annotations

import math

import numpy as np

from ..core.base import GeomSpec, GeomType, JointSpec, LinkSpec, RobotSpec
from .wheeled_base import WheeledBase


class DiffGenerator(WheeledBase):
    """차동 구동 로봇 생성기 (2륜 balancing 서브타입 포함)."""

    FAMILY = "diff"
    FORMS = ("diff",)

    def sample(self, form: str, rng: np.random.Generator,
               allow_balancing: bool = True) -> RobotSpec:
        """diff 스펙 하나를 샘플링한다.

        allow_balancing=False면 캐스터 없음(2륜) 서브타입을 뽑지 않는다
        (wheeled 휴머노이드가 베이스로 재사용할 때 — 전복 검사와 양립 불가).
        """
        spec = RobotSpec(name="diff", family=self.FAMILY, form=form, control_tag="diff")

        # 구조 축: 캐스터 종류 (없음 = balancing)
        kinds = list(self._cfg[self.FAMILY]["caster_kinds"])
        if not allow_balancing and "none" in kinds:
            kinds.remove("none")
        caster_kind = str(rng.choice(kinds))
        balancing = caster_kind == "none"

        # 몸통 치수 -> 바퀴 -> 지상고 순서 (몸통 비례로 물리 제약 내장)
        dims = self._sample_body_dims(rng)
        radius = min(self._u(rng, "wheel_radius"), 0.5 * dims["length"])
        radius = max(radius, 0.015)
        wheel_w = radius * self._u(rng, "wheel_width_factor")

        # balancing은 역진자 성립을 위해 몸통을 키워 무게중심이 축 위로 오게 한다
        if balancing:
            dims["height"] = max(dims["height"], 1.6 * radius)
            clearance = max(rng.uniform(0.3, 0.8) * radius, 0.02)
        elif caster_kind == "swivel":
            # 스위블 캐스터가 들어갈 지상고 확보
            clearance = max(radius * rng.uniform(0.7, 1.2), 0.03)
        else:
            clearance = max(radius * rng.uniform(0.4, 1.2), 0.02)
        geo = self._build_base(spec, rng, dims, clearance)
        geo["wheel_density"] = self._u(rng, "wheel_density")

        # 구동축 앞뒤 위치: balancing은 중앙 부근, 그 외 ±35%L (plan)
        L = geo["length"]
        axle_x = L * (rng.uniform(-0.1, 0.1) if balancing else rng.uniform(-0.35, 0.35))

        # 휠 노출 방식 + 구동 휠 좌우 1쌍
        exposed = bool(rng.random() < 0.5) if not balancing else True
        track_half = self._track_half(rng, geo, wheel_w, exposed)
        axle_z = radius - geo["body_z"]
        for sy in (1, -1):
            self._add_wheel(spec, rng, geo, f"wheel_{'l' if sy > 0 else 'r'}",
                            (axle_x, sy * track_half, axle_z), drive=True,
                            radius=radius, width=wheel_w)

        # 캐스터 배치와 무게중심 (서브타입별 구성적 샘플링)
        if balancing:
            self._setup_balancing(spec, rng, geo, axle_x, radius)
        else:
            caster_xs = self._add_casters(spec, rng, geo, caster_kind, axle_x, radius)
            self._setup_com_load(spec, rng, geo, axle_x, caster_xs)

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

    # ---------- 서브타입 구성 ----------

    def _setup_balancing(self, spec, rng, geo, axle_x, radius):
        """2륜 balancing 구성: 무게중심을 축 위·축선 상에 구성적으로 배치.

        역진자 성립 조건 com_z > 축 높이(radius)를 만족하도록 z 오프셋 하한을
        역산하고, x는 축선 위 소량 지터만 준다 (특수 검사 com_line_tol 이내).
        """
        # com_z(월드) = body_z + dz > radius + 여유 -> dz 하한 역산
        dz_min = max(-0.1 * geo["height"], radius + 0.02 - geo["body_z"])
        dz = rng.uniform(dz_min, 0.45 * geo["height"]) if dz_min < 0.45 * geo["height"] else dz_min
        dx = axle_x + rng.uniform(-0.015, 0.015)
        self._offset_com(spec, (dx, rng.uniform(-0.02, 0.02) * geo["width"], dz))

        spec.control_tag = "diff_balancing"
        spec.special["balancing"] = {"axle_x": axle_x, "axle_z": radius}
        spec.params["balancing"] = True

    def _add_casters(self, spec, rng, geo, kind, axle_x, radius) -> list[float]:
        """캐스터를 배치 축(반대편 1 / 양쪽 각 1 / 반대편 2 벌림)에 따라 추가.

        반환 = 캐스터 x 위치 목록 (하중 비율 검사용). 양쪽 배치는 구동축이
        중앙 부근일 때만 허용한다 (plan: "축이 중앙일 때").
        """
        placements = ["opp1", "opp2"]
        if abs(axle_x) < 0.12 * geo["length"]:
            placements.append("both")
        placement = str(rng.choice(placements))

        # 반대편 방향: 축이 중앙이면 무작위, 아니면 축 반대쪽
        s = -math.copysign(1.0, axle_x) if abs(axle_x) > 1e-6 else float(rng.choice([1, -1]))
        L = geo["length"]
        xs = []
        if placement == "both":
            # 양쪽 각 1개: 축 기준 대칭 거리
            d = L * rng.uniform(0.3, 0.42)
            spots = [(axle_x + d, 0.0), (axle_x - d, 0.0)]
        elif placement == "opp1":
            cx = float(np.clip(axle_x + s * L * rng.uniform(0.4, 0.8), -0.46 * L, 0.46 * L))
            spots = [(cx, 0.0)]
        else:
            # 반대편 2개 좌우 벌림
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

        # 스위블은 선회각에 따라 간섭이 없는지 추가 자세로 검사
        if kind == "swivel":
            swivels = [j.name for j in spec.joints if j.name.startswith("swivel_")]
            spec.check_poses += [{n: ang for n in swivels} for ang in (math.pi / 2, math.pi)]
        spec.params["caster_placement"] = placement
        return xs

    def _setup_com_load(self, spec, rng, geo, axle_x, caster_xs):
        """무게중심을 하중 비율 조건에 맞게 구성적으로 배치 + 특수 검사 등록.

        1차원 지렛대: 구동축 하중 비율 = (캐스터x - com_x) / (캐스터x - 축x).
        com_x를 축-캐스터 사이에서 비율이 min_share 이상이 되는 구간에 직접
        샘플해 기각을 예방하고, 검증은 특수 검사(load_share)가 이중 확인한다.
        양쪽 배치(both)는 무게중심을 축 부근에 둬 자연히 만족시킨다.
        """
        if len(caster_xs) == 2 and (caster_xs[0] - axle_x) * (caster_xs[1] - axle_x) < 0:
            # 양쪽 배치: 축 부근 소량 지터만
            dx = axle_x + rng.uniform(-0.05, 0.05) * geo["length"]
        else:
            # 한쪽 배치: t = (com-축)/(캐스터-축) <= 1-min_share 가 되도록 t 직접 샘플
            other_x = float(np.mean(caster_xs))
            t = rng.uniform(0.05, 1.0 - self._cfg["validation"]["load_share_min"] - 0.05)
            dx = axle_x + (other_x - axle_x) * t
            spec.special["load_share"] = {"drive_x": axle_x, "other_x": other_x}
        dy = rng.uniform(-0.04, 0.04) * geo["width"]
        dz = rng.uniform(-0.15, 0.15) * geo["height"]
        self._offset_com(spec, (dx, dy, dz))

    # ---------- 캐스터 부품 ----------

    def _add_ball_caster(self, spec, rng, geo, name, cx, cy, radius):
        """볼 캐스터: fixed 조인트 구 1개 (plan: 구 1개로 모델링).

        실물 볼 캐스터의 전방향 구름은 시뮬 접촉 마찰을 낮춰 근사한다
        (마찰 상수는 전 로봇 공통 — plan 제어기 섹션). 구 중심의 지면 높이 =
        반지름 r_s, 몸통 바닥과 뜨면 연결 스토크를 넣는다.
        """
        r_s = max(min(radius * rng.uniform(0.3, 0.7), geo["clearance"] * 0.95), 0.012)
        geoms = [GeomSpec(GeomType.SPHERE, (r_s, 0, 0))]

        # 구 상단(2*r_s)과 몸통 바닥(clearance) 사이 틈에 스토크 삽입
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
        """스위블 캐스터: 수직 선회(continuous z) + 휠 회전(continuous y) 2관절 체인.

        트레일 오프셋(선회축-접지점 수평 거리)이 있어야 진행 방향으로 자기
        정렬되므로 하한을 둔다 (plan 파라미터). 캐스터 휠 지름(2*r_c)은
        지상고보다 작게 샘플해 몸통 관통을 방지한다.
        """
        r_c = geo["clearance"] * rng.uniform(0.25, 0.45)
        drop = geo["clearance"] - r_c

        # 마운트: 몸통 바닥에서 축 높이까지 내려오는 수직 실린더
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

        # 트레일: 자기 정렬 하한 0.3*r_c (plan: 0이면 임의 방향 잠김)
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

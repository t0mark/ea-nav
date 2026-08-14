"""ackermann (자동차식 조향) 생성기. (예: RC카, F1TENTH, AgileX Hunter)

기본 구조: 몸통 + 뒤 축 구동 휠 2개 + 앞 축 조향 너클(수직 revolute) + 앞 휠 2개.
실차의 타이로드(폐루프)는 URDF가 트리만 허용해 표현 불가 -> 좌우 너클을 독립
revolute로 두고 조향각 배분·차동은 제어기 소프트웨어가 담당한다 (plan 명기).
구조 축: 몸통 형상(박스/적층), 구동 방식(후륜/전륜/4륜 — 수동 휠은 effort 0),
킹핀 오프셋 유무.
물리 제약: 조향 극한 간섭(앞 트랙 확장 + 극한각 자세 검사), 최소 회전 반경 상한,
구동축 하중 비율(특수 검사 — 무게중심을 구성적으로 배치).
"""
from __future__ import annotations

import math

import numpy as np

from ..core.base import GeomSpec, GeomType, JointSpec, LinkSpec, RobotSpec
from .wheeled_base import WheeledBase


class AckermannGenerator(WheeledBase):
    """아커만 조향 로봇 생성기."""

    FAMILY = "ackermann"
    FORMS = ("ackermann",)

    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:
        """ackermann 스펙 하나를 샘플링한다."""
        spec = RobotSpec(name="ackermann", family=self.FAMILY, form=form, control_tag="ackermann")

        # 몸통 -> 휠베이스 -> 바퀴 순서
        dims = self._sample_body_dims(rng)
        wheelbase = dims["length"] * self._u(rng, "wheelbase_factor")
        radius = min(self._u(rng, "wheel_radius"), 0.5 * dims["length"], 0.45 * wheelbase)
        radius = max(radius, 0.015)
        wheel_w = radius * self._u(rng, "wheel_width_factor")
        clearance = max(radius * rng.uniform(0.3, 1.2), 0.02)
        geo = self._build_base(spec, rng, dims, clearance)
        geo["wheel_density"] = self._u(rng, "wheel_density")

        # 조향 범위: 최소 회전 반경 R = wb/tan(delta) <= factor x 몸통 길이 보장
        # (샘플이 부족하면 하한을 역산해 끌어올림)
        steer = rng.uniform(0.3, 0.7)
        steer = min(max(steer, math.atan(wheelbase / (self._cfg[self.FAMILY]["turn_radius_factor"]
                                                      * geo["length"]))), 0.75)

        # 구동 방식·킹핀 오프셋 (구조 축)
        drive_mode = str(rng.choice(["rear", "front", "all"]))
        kingpin = wheel_w * rng.uniform(0.6, 1.2) if rng.random() < 0.5 else 0.0

        # 뒤 축: 기본 트랙
        track_r = geo["width"] / 2 + rng.uniform(0.005, 0.03) + wheel_w / 2
        axle_z = radius - geo["body_z"]
        for sy in (1, -1):
            self._add_wheel(spec, rng, geo, f"wheel_rear_{'l' if sy > 0 else 'r'}",
                            (-wheelbase / 2, sy * track_r, axle_z),
                            drive=drive_mode in ("rear", "all"), radius=radius, width=wheel_w)

        # 앞 축: 조향 침범량 D = (ko - w/2)(1-cos d) + r sin d 만큼 트랙 확장
        # (조향해도 바퀴 안쪽 면이 몸통에 닿지 않는 조건 — 극한각 자세 검사가 이중 확인)
        encroach = ((kingpin - wheel_w / 2) * (1 - math.cos(steer))
                    + radius * math.sin(steer))
        track_f = geo["width"] / 2 + rng.uniform(0.005, 0.03) + wheel_w / 2 + max(encroach, 0) + 0.008
        for sy in (1, -1):
            side = "l" if sy > 0 else "r"

            # 너클: 축 높이의 작은 수직 실린더, 조향축(z revolute)의 자식
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

            # 앞바퀴: 너클에서 킹핀 오프셋만큼 바깥, 축 높이는 너클과 동일(z=0)
            self._add_wheel(spec, rng, geo, f"wheel_front_{side}",
                            (0.0, sy * kingpin, 0.0),
                            drive=drive_mode in ("front", "all"),
                            radius=radius, width=wheel_w, parent=f"knuckle_{side}")

        # 무게중심: 구동축 하중 비율을 구성적으로 만족 (1차원 지렛대)
        com_dx = self._setup_com_load(spec, rng, geo, wheelbase, drive_mode)

        # 조향 극한 두 방향을 추가 검사 자세로 등록
        spec.check_poses += [{"steer_l": s * steer, "steer_r": s * steer} for s in (1, -1)]

        self._set_drive_limits(spec, rng, radius)

        # 조향 유지 토크: 전축 하중 x 접촉 팔(킹핀 + 접지면 반폭) 기준 여유 샘플
        # (plan: 장애물 접촉 시 조향 유지력이 통과 가능성에 영향 -> 질량과 연동)
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
        """무게중심 x를 구동축 하중 비율 >= min_share가 되게 직접 샘플. dx 반환.

        share = (비구동축x - com_x) / (비구동축x - 구동축x) 이므로 share를
        [min_share+여유, 0.9]에서 뽑아 com_x를 역산한다. 4륜 구동은 양 축이
        모두 구동이라 비율 검사가 없고 축 사이 지터만 준다.
        반환값(dx)은 조향 토크의 전축 하중 계산에 쓰인다.
        """
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

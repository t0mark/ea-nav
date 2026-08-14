"""omni (홀로노믹 휠 베이스) 생성기.

하위 타입 2종 (구조 축):
- 매커넘 4륜: 45도 수동 롤러, 표준 X 배치 고정 (예: KUKA youBot, 공장 AGV)
- 옴니휠 3-4륜: 90도 롤러 휠을 방사형 배치 (예: 로봇축구 베이스)

매커넘 배치 규칙 (plan 명기 — 뷰 의존 표기 대신 부호 규칙):
바퀴 i의 롤러 축 y성분 부호 s_i = -sign(x_i * y_i).
유도: 바퀴가 지면에 낼 수 있는 힘 방향은 롤러 축의 지면 투영 (1, s_i)/√2 이고
yaw 모멘트 팔 = x_i*s_i - y_i = -sign(y_i)(|x_i|+|y_i|) -> 모든 바퀴의 모멘트 팔
크기가 (반휠베이스+반트랙)로 최대가 된다. 반전 배치는 팔이 (L-W)라
정사각형 풋프린트에서 yaw 권한이 0이 되므로 금지 (검증 리뷰 C1).

롤러는 수동 continuous 조인트로 명시 (fixed면 파서 병합 때 skid와 구별 불가).
제로샷 한계(실로봇 매커넘 URDF에는 롤러가 없음)는 plan에 감수로 문서화됨.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.base import RobotSpec
from .wheeled_base import WheeledBase


class OmniGenerator(WheeledBase):
    """매커넘·옴니휠 홀로노믹 베이스 생성기."""

    FAMILY = "omni"
    FORMS = ("omni",)

    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:
        """omni 스펙 하나를 샘플링한다 (하위 타입 분기)."""
        spec = RobotSpec(name="omni", family=self.FAMILY, form=form, control_tag="omni")
        subtype = str(rng.choice(self._cfg[self.FAMILY]["subtypes"]))
        if subtype == "mecanum4":
            self._build_mecanum(spec, rng)
        else:
            self._build_omniwheel(spec, rng, k=3 if subtype == "omni3" else 4)
        spec.params.update({
            "subtype": subtype, "total_mass": spec.total_mass(),
            "has_wheels": True, "has_legs": False,
        })
        return spec

    # ---------- 매커넘 4륜 ----------

    def _build_mecanum(self, spec, rng):
        """매커넘 4륜 조립: 표준 X 배치 + 커버리지 연동 롤러."""
        dims = self._sample_body_dims(rng)
        wheelbase = dims["length"] * self._u(rng, "wheelbase_factor")
        radius = min(self._u(rng, "wheel_radius"), 0.5 * dims["length"], 0.45 * wheelbase)
        radius = max(radius, 0.02)
        wheel_w = radius * self._u(rng, "wheel_width_factor")
        clearance = max(radius * rng.uniform(0.3, 1.0), 0.02)
        geo = self._build_base(spec, rng, dims, clearance)
        geo["wheel_density"] = self._u(rng, "wheel_density")

        # 롤러 치수: 커버리지(인접 롤러 사이 호를 접선 투영 l*cos45가 덮음) 역산
        n_roller = int(rng.integers(10, 15))
        roller_r = radius * rng.uniform(0.2, 0.3)
        hub_r = radius - roller_r
        l_need = 2.3 * hub_r * math.sin(math.pi / n_roller) / math.cos(math.pi / 4)
        roller_l = float(np.clip(l_need, 2.0 * roller_r, 3.0 * roller_r))

        # 트랙: 롤러가 바퀴 중심면에서 안쪽으로 (l/2+r_r)*cos45 돌출 -> 그만큼 확장
        protrusion = (roller_l / 2) * math.sin(math.pi / 4) + roller_r * math.cos(math.pi / 4)
        track = (geo["width"] / 2 + rng.uniform(0.005, 0.03) + wheel_w / 2
                 + max(0.0, protrusion - wheel_w / 2) + 0.008)

        # 표준 X 배치: s_i = -sign(x_i*y_i) -> yaw 모멘트 팔 = |x|+|y| (모듈 docstring)
        axle_z = radius - geo["body_z"]
        for key, (sx, sy) in dict(fl=(1, 1), fr=(1, -1), rl=(-1, 1), rr=(-1, -1)).items():
            wname = f"wheel_{key}"
            self._add_wheel(spec, rng, geo, wname,
                            (sx * wheelbase / 2, sy * track, axle_z),
                            drive=True, radius=hub_r, width=wheel_w)

            # 롤러 부착: 위상 랜덤, 틸트 부호 = 배치 규칙
            tilt = -float(sx * sy) * math.pi / 4
            phase = rng.uniform(0, 2 * math.pi)
            for k in range(n_roller):
                theta = 2 * math.pi * k / n_roller + phase
                self._add_roller(spec, geo, wname, k, theta, tilt, hub_r, roller_r, roller_l)

        # 접지는 허브가 아니라 롤러가 담당
        spec.contact_links = [l.name for l in spec.links if "_roller_" in l.name]

        # 무게중심 지터 (다각형 검사가 확인)
        self._offset_com(spec, (rng.uniform(-0.2, 0.2) * wheelbase / 2,
                                rng.uniform(-0.04, 0.04) * geo["width"],
                                rng.uniform(-0.15, 0.15) * geo["height"]))
        self._set_drive_limits(spec, rng, radius)
        self._clamp_mass_ratio(spec)
        spec.params.update({
            "wheelbase": wheelbase, "track_width": 2 * track, "wheel_width": wheel_w,
            "n_rollers_per_wheel": n_roller, "roller_radius": roller_r,
            "roller_length": roller_l, "mecanum_pattern": "X_standard",
            "n_wheels": 4, "est_step_height": radius * 0.6,
        })

    # ---------- 옴니휠 3-4륜 ----------

    def _build_omniwheel(self, spec, rng, k: int):
        """옴니휠 방사 배치 조립: 바퀴 회전축이 반경 방향, 롤러는 접선(틸트 0).

        바퀴 자식 프레임 +y(회전축)를 조인트 yaw = phi - 90도로 반경 방향에
        맞춘다. 접지력은 각 바퀴의 접선 방향이라 3륜 이상 방사 배치는 항상
        완전한 평면 이동·회전 권한을 가진다.
        """
        # 옴니휠 베이스는 원통 몸통이 일반적 (4륜은 박스도 허용)
        dims = self._sample_body_dims(rng)
        if k == 3:
            dims["shape"] = "cylinder"
            dims["length"] = dims["width"]
        radius = max(min(self._u(rng, "wheel_radius"), 0.35 * dims["width"]), 0.02)
        wheel_w = radius * self._u(rng, "wheel_width_factor")
        clearance = max(radius * rng.uniform(0.3, 1.0), 0.02)
        geo = self._build_base(spec, rng, dims, clearance)
        geo["wheel_density"] = self._u(rng, "wheel_density")

        # 롤러 (틸트 0 = 접선): 커버리지 투영에 cos45 인자가 없음
        n_roller = int(rng.integers(12, 17))
        roller_r = radius * rng.uniform(0.2, 0.3)
        hub_r = radius - roller_r
        roller_l = float(np.clip(2.3 * hub_r * math.sin(math.pi / n_roller),
                                 1.6 * roller_r, 3.0 * roller_r))

        # 배치 원: 롤러가 반경 방향으로 r_r만큼 돌출 -> 몸통 간극에 반영
        gap = max(rng.uniform(0.005, 0.03), roller_r - wheel_w / 2 + 0.008)
        ring = geo["width"] / 2 + gap + wheel_w / 2
        axle_z = radius - geo["body_z"]

        # 바퀴 k개를 등간격 각도로 방사 배치
        phase0 = rng.uniform(0, 2 * math.pi / k)
        for i in range(k):
            phi = 2 * math.pi * i / k + phase0
            wname = f"wheel_{i}"
            self._add_wheel(spec, rng, geo, wname,
                            (ring * math.cos(phi), ring * math.sin(phi), axle_z),
                            drive=True, radius=hub_r, width=wheel_w,
                            joint_yaw=phi - math.pi / 2)
            phase = rng.uniform(0, 2 * math.pi)
            for j in range(n_roller):
                theta = 2 * math.pi * j / n_roller + phase
                self._add_roller(spec, geo, wname, j, theta, 0.0, hub_r, roller_r, roller_l)

        spec.contact_links = [l.name for l in spec.links if "_roller_" in l.name]

        # 무게중심 지터 (배치 원 안쪽)
        self._offset_com(spec, (rng.uniform(-0.15, 0.15) * ring,
                                rng.uniform(-0.15, 0.15) * ring,
                                rng.uniform(-0.15, 0.15) * geo["height"]))
        self._set_drive_limits(spec, rng, radius)
        self._clamp_mass_ratio(spec)
        spec.params.update({
            "ring_radius": ring, "wheel_width": wheel_w,
            "n_rollers_per_wheel": n_roller, "roller_radius": roller_r,
            "roller_length": roller_l,
            "n_wheels": k, "est_step_height": radius * 0.5,
        })

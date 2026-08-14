"""skid steer (스키드 조향) 생성기. (예: Husky, Jackal)

기본 구조: 몸통 + 좌우 대칭 고정축 구동 휠 4-6개 (전부 continuous, 조향 없음).
좌우 속도차로 도는 탱크 조향 방식.
구조 축: 휠 개수(4/6), 몸통 형상(박스/박스+상판), 중간 축 지터(6륜), 휠 노출.
물리 제약: 인접 휠 간섭 클램프(휠 지름 < 최소 축 간격), 휠베이스:트랙 비율 상한
(비율 과대 시 제자리 회전의 옆미끄럼 저항 과대 — 실차 설계 관례 약 1.5).
"""
from __future__ import annotations

import numpy as np

from ..core.base import RobotSpec
from .wheeled_base import WheeledBase


class SkidGenerator(WheeledBase):
    """스키드 조향 로봇 생성기."""

    FAMILY = "skid"
    FORMS = ("skid",)

    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:
        """skid 스펙 하나를 샘플링한다 (4륜 또는 6륜)."""
        spec = RobotSpec(name="skid", family=self.FAMILY, form=form, control_tag="skid")

        # 몸통 -> 휠베이스 -> 바퀴 순서 (축간격이 바퀴 지름 클램프의 기준)
        dims = self._sample_body_dims(rng)
        n_axle = int(rng.choice([2, 3]))
        wheelbase = dims["length"] * self._u(rng, "wheelbase_factor")

        # 6륜 최소 축 간격 = wb/2 - 중간축 지터(0.08wb) = 0.42wb -> 반지름 < 0.21wb
        gap_cap = 0.19 if n_axle == 3 else 0.45
        radius = min(self._u(rng, "wheel_radius"), 0.5 * dims["length"], gap_cap * wheelbase)
        radius = max(radius, 0.015)
        wheel_w = radius * self._u(rng, "wheel_width_factor")

        clearance = max(radius * rng.uniform(0.3, 1.2), 0.02)
        geo = self._build_base(spec, rng, dims, clearance)
        geo["wheel_density"] = self._u(rng, "wheel_density")

        # 트랙 결정 후 휠베이스:트랙 비율 상한 클램프 (초과분은 휠베이스 축소)
        exposed = bool(rng.random() < 0.5)
        track_half = self._track_half(rng, geo, wheel_w, exposed)
        ratio_max = self._cfg[self.FAMILY]["wb_track_ratio_max"]
        wheelbase = min(wheelbase, ratio_max * 2 * track_half)
        radius = min(radius, gap_cap * wheelbase)

        # 축 위치: 앞뒤 축은 휠베이스 양 끝, 중간 축(6륜)은 중앙 지터
        xs = [wheelbase / 2, -wheelbase / 2]
        if n_axle == 3:
            xs.insert(1, rng.uniform(-0.08, 0.08) * wheelbase)
        axle_z = radius - geo["body_z"]
        for i, x in enumerate(xs):
            for sy in (1, -1):
                self._add_wheel(spec, rng, geo, f"wheel_{i}_{'l' if sy > 0 else 'r'}",
                                (x, sy * track_half, axle_z), drive=True,
                                radius=radius, width=wheel_w)

        # 무게중심 오프셋: 축들이 감싸는 범위 안쪽 지터 (다각형 검사가 이중 확인)
        self._offset_com(spec, (rng.uniform(-0.2, 0.2) * wheelbase / 2,
                                rng.uniform(-0.04, 0.04) * geo["width"],
                                rng.uniform(-0.15, 0.15) * geo["height"]))

        self._set_drive_limits(spec, rng, radius)
        self._clamp_mass_ratio(spec)
        spec.params.update({
            "n_wheels": 2 * n_axle, "wheelbase": wheelbase, "track_width": 2 * track_half,
            "wheel_width": wheel_w, "wheel_exposed": exposed,
            "wb_track_ratio": wheelbase / (2 * track_half),
            "total_mass": spec.total_mass(),
            "has_wheels": True, "has_legs": False,
            "est_step_height": radius * 0.6,
        })
        return spec

"""wheeled 계열 생성기의 공통 부품 (diff / skid / ackermann / omni / wheeled 휴머노이드가 상속·재사용).

기하 규약: base_link 원점 = 몸통 전체 높이의 중심. 지면 기준 몸통 중심 높이
body_z = clearance + height/2 이므로, 지면 높이 h에 붙는 부품의 몸통 프레임
z 오프셋은 (h - body_z)로 계산한다. 바퀴 링크의 자식 프레임은 회전축 = +y로
통일하고 (실린더를 x축 기준 90도 회전), 방사형 배치(옴니휠)는 조인트 rpy의
yaw로 자식 y축을 반경 방향에 맞춘다.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from ..core.base import BaseGenerator, GeomSpec, GeomType, JointSpec, LinkSpec, RobotSpec


class WheeledBase(BaseGenerator):
    """wheeled 공통 헬퍼 모음 (추상 — sample()은 서브클래스가 구현)."""

    # ---------- 몸통 ----------

    def _sample_body_dims(self, rng: np.random.Generator) -> dict:
        """몸통 치수와 형상 축을 샘플한다 (geoms는 아직 만들지 않음).

        반환 {shape, length, width, height}. 원통 몸통은 지름 = width = length
        (풋프린트가 원이므로 두 값을 같게 둔다).
        """
        shape = str(rng.choice(self._cfg[self.FAMILY]["body_shapes"]))
        length = self._u(rng, "body_length")
        width = min(self._u(rng, "body_width"), length * 1.2)
        height = self._u(rng, "body_height")

        # 원통은 풋프린트가 원 -> 길이·폭을 지름으로 통일
        if shape.startswith("cylinder"):
            length = width
        return {"shape": shape, "length": length, "width": width, "height": height}

    def _build_base(self, spec: RobotSpec, rng: np.random.Generator,
                    dims: dict, clearance: float) -> dict:
        """base_link를 만들고 배치 계산에 쓸 geo dict를 반환한다.

        형상 축: box(단일 박스) / cylinder(수직 원통) / stack(아래 본체 + 위 상판 박스)
        / cylinder_stack(아래 원통 + 위 박스). geo.top_z = 몸통 상면의 링크 프레임 z.
        """
        L, W, H = dims["length"], dims["width"], dims["height"]
        shape = dims["shape"]
        geoms: list[GeomSpec] = []

        # 적층 형상은 아래 본체(h1) + 위 박스(h2)로 분할, 위 박스는 풋프린트 축소
        if shape in ("stack", "cylinder_stack"):
            h1 = H * rng.uniform(0.5, 0.75)
            h2 = H - h1
            top = GeomSpec(GeomType.BOX,
                           (L * rng.uniform(0.4, 0.85), W * rng.uniform(0.4, 0.85), h2),
                           origin_xyz=(rng.uniform(-0.15, 0.15) * L, 0, -H / 2 + h1 + h2 / 2))
            if shape == "stack":
                geoms.append(GeomSpec(GeomType.BOX, (L, W, h1), origin_xyz=(0, 0, -H / 2 + h1 / 2)))
            else:
                geoms.append(GeomSpec(GeomType.CYLINDER, (W / 2, h1, 0),
                                      origin_xyz=(0, 0, -H / 2 + h1 / 2)))
            geoms.append(top)
        elif shape == "cylinder":
            geoms.append(GeomSpec(GeomType.CYLINDER, (W / 2, H, 0)))
        else:
            geoms.append(GeomSpec(GeomType.BOX, (L, W, H)))

        # 질량 = 부피 x 유효 밀도 (샘플링 순서 내장 원칙)
        body = LinkSpec("base_link", geoms)
        body.mass = sum(g.volume for g in geoms) * self._u(rng, "body_density")
        spec.links.append(body)

        geo = {"length": L, "width": W, "height": H, "shape": shape,
               "clearance": clearance, "body_z": clearance + H / 2, "top_z": H / 2}
        spec.params.update({"body_shape": shape, "body_length": L, "body_width": W,
                            "body_height": H, "ground_clearance": clearance})
        return geo

    def _offset_com(self, spec: RobotSpec, offset: tuple[float, float, float]):
        """base_link 질량중심을 자연값 + offset으로 명시한다 (무게중심 위치 랜덤화).

        내부 밸러스트(배터리 등) 치우침을 표현하며, 관성은 명시 com 기준으로
        write_urdf가 재계산한다.
        """
        from ..core.base import compute_com
        base = spec.links[0]
        natural = compute_com(base)
        base.com_xyz = tuple(float(natural[i] + offset[i]) for i in range(3))
        spec.params["com_offset"] = [float(v) for v in offset]

    # ---------- 바퀴·롤러 ----------

    def _add_wheel(self, spec: RobotSpec, rng: np.random.Generator, geo: dict, name: str,
                   xyz: tuple[float, float, float], drive: bool, radius: float, width: float,
                   parent: str = "base_link", joint_yaw: float = 0.0):
        """바퀴 링크 + continuous 조인트 추가.

        xyz = 부모 프레임 기준 조인트 위치 (z는 호출자가 축 높이로 계산).
        joint_yaw = 조인트 rpy의 yaw — 자식 +y(회전축)를 원하는 방향으로 돌린다
        (옴니휠 방사 배치용, 0이면 좌우 축). drive 여부는 조인트 이름 접두사로
        구분하고, 구동 토크는 _set_drive_limits에서 총 질량 확정 후 일괄 설정한다.
        """
        # 바퀴 링크 프레임: 실린더를 x축 기준 90도 돌려 회전축을 +y로 맞춘다
        link = LinkSpec(name, [GeomSpec(GeomType.CYLINDER, (radius, width, 0),
                                        origin_rpy=(math.pi / 2, 0, 0))])
        link.mass = link.geoms[0].volume * geo["wheel_density"]
        spec.links.append(link)

        jname = f"drive_{name}" if drive else f"spin_{name}"
        spec.joints.append(JointSpec(
            jname, "continuous", parent, name,
            origin_xyz=xyz, origin_rpy=(0, 0, joint_yaw), axis=(0, 1, 0),
            effort=0.0, velocity=rng.uniform(10.0, 30.0),
        ))
        spec.contact_links.append(name)
        spec.standing_pose[jname] = 0.0

    def _add_roller(self, spec: RobotSpec, geo: dict, wheel_name: str, idx: int,
                    theta: float, tilt: float, hub_r: float, roller_r: float, roller_l: float):
        """수동 롤러 링크 + continuous 조인트 추가 (매커넘 45도 / 옴니휠 0도).

        바퀴 자식 프레임(회전축 +y) 기준 기하:
        - 림 위 위치 p = hub_r * (cos(theta), 0, sin(theta))  (xz 평면이 림 단면)
        - 림 접선 t = (-sin(theta), 0, cos(theta))
        - 롤러 축 = cos(tilt) * t + sin(tilt) * y_hat
          (tilt = +-45도 -> 매커넘, 부호가 대각 방향. tilt = 0 -> 옴니휠 접선 롤러)
        조인트 rpy는 자식(롤러) 프레임 z축이 롤러 축과 일치하도록 잡아,
        롤러 실린더를 회전 없이 쓰고 조인트 축을 (0,0,1)로 통일한다.
        """
        # 림 위 위치와 롤러 축 (위 docstring의 식)
        pos = (hub_r * math.cos(theta), 0.0, hub_r * math.sin(theta))
        tangent = np.array([-math.sin(theta), 0.0, math.cos(theta)])
        axis = tangent * math.cos(tilt) + np.array([0.0, math.sin(tilt), 0.0])
        axis /= np.linalg.norm(axis)

        # 자식 z축 -> 롤러 축 회전을 rpy로 (align_vectors: b를 a로 보내는 회전)
        rot, _ = Rotation.align_vectors(axis[None, :], np.array([[0.0, 0.0, 1.0]]))
        rpy = tuple(rot.as_euler("xyz"))

        name = f"{wheel_name}_roller_{idx}"
        link = LinkSpec(name, [GeomSpec(GeomType.CYLINDER, (roller_r, roller_l, 0))])
        link.mass = link.geoms[0].volume * geo["wheel_density"]
        spec.links.append(link)
        spec.joints.append(JointSpec(
            f"passive_{name}", "continuous", wheel_name, name,
            origin_xyz=pos, origin_rpy=rpy, axis=(0, 0, 1),
            effort=0.0, velocity=50.0,
        ))
        spec.standing_pose[f"passive_{name}"] = 0.0

    def _track_half(self, rng: np.random.Generator, geo: dict, wheel_w: float,
                    exposed: bool) -> float:
        """트랙 절반(바퀴 중심 y)을 노출 방식에 따라 계산한다.

        노출: 몸통 반폭 + 간극 + 바퀴 반폭 (몸통 밖).
        은닉: 바퀴가 몸통 풋프린트 안쪽 (몸통 아래에 숨음 — 하우징 구멍 없음,
        바퀴-몸통 겹침은 부모-자식 허용 쌍이라 검사에 걸리지 않는다).
        """
        if exposed:
            return geo["width"] / 2 + rng.uniform(0.005, 0.03) + wheel_w / 2
        # 은닉: 바퀴 바깥면이 몸통 안쪽에 오도록 상한 클램프
        upper = geo["width"] / 2 * 0.95 - wheel_w / 2
        return max(min(geo["width"] / 2 * rng.uniform(0.5, 0.85), upper), wheel_w * 0.6)

    # ---------- 한계 ----------

    def _set_drive_limits(self, spec: RobotSpec, rng: np.random.Generator, radius: float):
        """구동 조인트(drive_*)의 토크·속도 한계를 총 질량 기준으로 설정.

        토크: 바퀴당 접지 하중 x 반지름 (m g r / 구동 바퀴 수)이 평지 구동에
        필요한 스케일 -> 0.6-2.5배 여유 샘플로 등판각이 물리적 범위에 오게 한다.
        속도: 목표 선속도 v_max / 반지름 = 바퀴 각속도 [rad/s].
        상체 추가(wheeled 휴머노이드) 후 재호출해 총 질량 변화를 반영할 수 있다.
        """
        m = spec.total_mass()
        drives = [j for j in spec.joints if j.name.startswith("drive_")]
        v_max = rng.uniform(0.5, 3.0)
        for j in drives:
            j.effort = m * 9.81 * radius / max(len(drives), 1) * rng.uniform(0.6, 2.5)
            j.velocity = v_max / radius
        spec.params["max_lin_vel"] = v_max
        spec.params["wheel_radius"] = radius

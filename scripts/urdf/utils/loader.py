"""URDF를 기립 자세로 놓은 기하 모델 로더.

정적 검사(validate)와 렌더링(render)이 공용으로 쓰는 계층: yourdfpy로 FK를 풀고,
링크별 충돌 프리미티브를 월드 좌표 trimesh로 변환해 들고 있는다.
월드 좌표계 = base_link 프레임 (지면 정렬은 사용하는 쪽에서 접촉 최저점으로 처리).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy


@dataclass
class PosedLink:
    """자세가 반영된 링크 하나: 월드 좌표 메시들 + 질량·질량중심."""

    name: str
    # 월드(base_link) 좌표로 변환된 trimesh 목록 (프리미티브 충돌 형상만)
    meshes: list = field(default_factory=list)
    mass: float = 0.0
    com_world: np.ndarray | None = None


class PosedModel:
    """URDF + 자세(조인트 값) -> 링크별 월드 기하·관성 정보.

    로드 시점에 FK를 한 번 풀어 결과를 보관한다 (자세 변경 불가, 필요하면 재로드).
    """

    def __init__(self, urdf_path: str | Path, pose: dict | None = None):
        """urdf_path를 로드하고 pose(조인트 이름 -> 값 [rad|m])를 적용한다.

        pose에 없는 actuated 조인트는 0으로 채운다. 메시 파일은 읽지 않는다
        (프리미티브만 변환하므로 실로봇 mesh URDF도 로드 자체는 가능).
        """
        self._urdf = yourdfpy.URDF.load(
            str(urdf_path), build_scene_graph=True, load_meshes=False,
            build_collision_scene_graph=False, load_collision_meshes=False,
        )

        # 자세 적용: 빠진 조인트는 0으로 채워 전체 cfg를 만든다
        pose = pose or {}
        cfg = {j: pose.get(j, 0.0) for j in self._urdf.actuated_joint_names}
        if cfg:
            self._urdf.update_cfg(cfg)
        self._links = self._build_links()

    @property
    def links(self) -> list[PosedLink]:
        """자세가 반영된 링크 목록 (URDF 정의 순서)."""
        return self._links

    @property
    def base_link(self) -> str:
        """루트 링크 이름."""
        return self._urdf.base_link

    @property
    def joints(self):
        """yourdfpy 조인트 객체 목록 (type, axis, limit, mimic 등 원본 필드 접근용)."""
        return list(self._urdf.robot.joints)

    def link_transform(self, name: str) -> np.ndarray:
        """base_link 기준 링크 프레임의 4x4 동차변환."""
        return self._urdf.get_transform(name, self._urdf.base_link)

    def adjacency(self, depth: int = 1) -> set[frozenset]:
        """그래프 거리 depth 이하로 연결된 링크 쌍 (셀프 충돌 제외 목록용).

        조인트를 무방향 간선으로 보고 각 링크에서 BFS를 depth 단계 돌려
        도달한 쌍을 모은다. depth=1 = 부모-자식만.
        """
        # 무방향 인접 리스트 구성
        edges = [(j.parent, j.child) for j in self._urdf.robot.joints]
        neighbors: dict[str, set] = {}
        for p, c in edges:
            neighbors.setdefault(p, set()).add(c)
            neighbors.setdefault(c, set()).add(p)

        # 각 링크에서 depth 단계 BFS로 도달 쌍 수집
        pairs: set[frozenset] = set()
        for start in neighbors:
            frontier, seen = {start}, {start}
            for _ in range(depth):
                frontier = {n for f in frontier for n in neighbors.get(f, ())} - seen
                seen |= frontier
                pairs |= {frozenset((start, n)) for n in frontier}
        return pairs

    def sibling_pairs_nonbase(self) -> set[frozenset]:
        """base_link가 아닌 같은 부모를 공유하는, 가동 조인트 자식끼리의 링크 쌍.

        매커넘·옴니휠 롤러처럼 한 부모(바퀴)에 밀집 배치돼 서로 닿는 것이
        정상인 부품을 셀프 충돌에서 허용하기 위한 목록. 다음은 포함하지 않는다:
        - base_link의 자식들 (바퀴·다리·지지): 서로 떨어져 있어야 정상
        - fixed 조인트 자식 (머리·센서): 위치가 고정이라 겹치면 실제 관통
        """
        by_parent: dict[str, list] = {}
        for j in self._urdf.robot.joints:
            if j.type == "fixed":
                continue
            by_parent.setdefault(j.parent, []).append(j.child)

        # 부모별 자식 조합 쌍 생성 (base_link 부모는 건너뜀)
        pairs: set[frozenset] = set()
        for parent, children in by_parent.items():
            if parent == self._urdf.base_link:
                continue
            for i in range(len(children)):
                for k in range(i + 1, len(children)):
                    pairs.add(frozenset((children[i], children[k])))
        return pairs

    def _build_links(self) -> list[PosedLink]:
        """FK 결과로 링크별 월드 메시·질량중심을 만든다.

        충돌 형상마다 (링크 변환 x 형상 origin)을 적용해 월드 좌표 trimesh를
        만들고, inertial이 있으면 질량중심도 월드로 변환해 보관한다.
        """
        links = []
        for link in self._urdf.robot.links:
            world = self.link_transform(link.name)
            posed = PosedLink(link.name)

            # 프리미티브 충돌 형상 -> 월드 trimesh (메시 형상은 건너뜀)
            for col in link.collisions:
                mesh = self._primitive_mesh(col.geometry)
                if mesh is None:
                    continue
                origin = col.origin if col.origin is not None else np.eye(4)
                mesh.apply_transform(world @ origin)
                posed.meshes.append(mesh)

            # 관성 정보가 있으면 질량중심을 월드 좌표로 변환
            if link.inertial is not None and link.inertial.mass:
                posed.mass = float(link.inertial.mass)
                io = link.inertial.origin if link.inertial.origin is not None else np.eye(4)
                posed.com_world = (world @ io)[:3, 3]
            links.append(posed)
        return links

    def _primitive_mesh(self, geometry):
        """URDF 프리미티브 -> trimesh. 메시 형상·퇴화 형상이면 None.

        메시 형상은 정적 검사 대상이 아니다 (실로봇 풀은 파서 통과 검사만 수행).
        치수가 0 이하인 퇴화 프리미티브는 건너뛴다 — trimesh가 반지름 0 구에서
        NaN 정점을 만들기 때문 (실로봇 atlas_drc의 발에 반지름 0 구 표기가 존재).
        """
        if geometry.box is not None:
            size = np.asarray(geometry.box.size, dtype=float)
            if (size <= 0).any():
                return None
            return trimesh.creation.box(extents=size)
        if geometry.cylinder is not None:
            if geometry.cylinder.radius <= 0 or geometry.cylinder.length <= 0:
                return None
            return trimesh.creation.cylinder(
                radius=float(geometry.cylinder.radius),
                height=float(geometry.cylinder.length), sections=24,
            )
        if geometry.sphere is not None:
            if geometry.sphere.radius <= 0:
                return None
            return trimesh.creation.icosphere(subdivisions=2, radius=float(geometry.sphere.radius))
        return None

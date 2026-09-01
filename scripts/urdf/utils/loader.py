from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy

@dataclass
class PosedLink:

    name: str

    meshes: list = field(default_factory=list)
    mass: float = 0.0
    com_world: np.ndarray | None = None

class PosedModel:

    def __init__(self, urdf_path: str | Path, pose: dict | None = None):

        self._urdf = yourdfpy.URDF.load(
            str(urdf_path), build_scene_graph=True, load_meshes=False,
            build_collision_scene_graph=False, load_collision_meshes=False,
        )

        pose = pose or {}
        cfg = {j: pose.get(j, 0.0) for j in self._urdf.actuated_joint_names}
        if cfg:
            self._urdf.update_cfg(cfg)
        self._links = self._build_links()

    @property
    def links(self) -> list[PosedLink]:

        return self._links

    @property
    def base_link(self) -> str:

        return self._urdf.base_link

    @property
    def joints(self):

        return list(self._urdf.robot.joints)

    def link_transform(self, name: str) -> np.ndarray:

        return self._urdf.get_transform(name, self._urdf.base_link)

    def adjacency(self, depth: int = 1) -> set[frozenset]:

        edges = [(j.parent, j.child) for j in self._urdf.robot.joints]
        neighbors: dict[str, set] = {}
        for p, c in edges:
            neighbors.setdefault(p, set()).add(c)
            neighbors.setdefault(c, set()).add(p)

        pairs: set[frozenset] = set()
        for start in neighbors:
            frontier, seen = {start}, {start}
            for _ in range(depth):
                frontier = {n for f in frontier for n in neighbors.get(f, ())} - seen
                seen |= frontier
                pairs |= {frozenset((start, n)) for n in frontier}
        return pairs

    def sibling_pairs_nonbase(self) -> set[frozenset]:

        by_parent: dict[str, list] = {}
        for j in self._urdf.robot.joints:
            if j.type == "fixed":
                continue
            by_parent.setdefault(j.parent, []).append(j.child)

        pairs: set[frozenset] = set()
        for parent, children in by_parent.items():
            if parent == self._urdf.base_link:
                continue
            for i in range(len(children)):
                for k in range(i + 1, len(children)):
                    pairs.add(frozenset((children[i], children[k])))
        return pairs

    def _build_links(self) -> list[PosedLink]:

        links = []
        for link in self._urdf.robot.links:
            world = self.link_transform(link.name)
            posed = PosedLink(link.name)

            for col in link.collisions:
                mesh = self._primitive_mesh(col.geometry)
                if mesh is None:
                    continue
                origin = col.origin if col.origin is not None else np.eye(4)
                mesh.apply_transform(world @ origin)
                posed.meshes.append(mesh)

            if link.inertial is not None and link.inertial.mass:
                posed.mass = float(link.inertial.mass)
                io = link.inertial.origin if link.inertial.origin is not None else np.eye(4)
                posed.com_world = (world @ io)[:3, 3]
            links.append(posed)
        return links

    def _primitive_mesh(self, geometry):

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

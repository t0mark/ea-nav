from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

class GeomType(str, Enum):

    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"

@dataclass
class GeomSpec:

    gtype: GeomType
    size: tuple[float, float, float]
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def volume(self) -> float:

        if self.gtype == GeomType.BOX:
            return self.size[0] * self.size[1] * self.size[2]
        if self.gtype == GeomType.CYLINDER:
            return math.pi * self.size[0] ** 2 * self.size[1]
        return 4.0 / 3.0 * math.pi * self.size[0] ** 3

@dataclass
class LinkSpec:

    name: str
    geoms: list[GeomSpec] = field(default_factory=list)
    mass: float = 0.0

    com_xyz: tuple[float, float, float] | None = None

    inertia: tuple[float, float, float, float, float, float] | None = None

@dataclass
class JointSpec:

    name: str
    jtype: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0

@dataclass
class RobotSpec:

    name: str
    family: str
    form: str
    links: list[LinkSpec] = field(default_factory=list)
    joints: list[JointSpec] = field(default_factory=list)

    params: dict = field(default_factory=dict)

    standing_pose: dict = field(default_factory=dict)

    contact_links: list[str] = field(default_factory=list)

    control_tag: str = ""

    check_poses: list[dict] = field(default_factory=list)

    special: dict = field(default_factory=dict)

    def link(self, name: str) -> LinkSpec:

        for l in self.links:
            if l.name == name:
                return l
        raise KeyError(name)

    def total_mass(self) -> float:

        return sum(l.mass for l in self.links)

    def actuated_joints(self) -> list[JointSpec]:

        return [j for j in self.joints if j.jtype in ("revolute", "continuous", "prismatic")]

def compute_com(link: LinkSpec) -> np.ndarray:

    if link.com_xyz is not None:
        return np.asarray(link.com_xyz, dtype=float)

    vols = np.array([g.volume for g in link.geoms])
    centers = np.array([g.origin_xyz for g in link.geoms])
    return (vols[:, None] * centers).sum(axis=0) / vols.sum()

def compute_inertia(link: LinkSpec) -> np.ndarray:

    if link.inertia is not None:
        ixx, iyy, izz, ixy, ixz, iyz = link.inertia
        return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

    com = compute_com(link)
    vols = np.array([g.volume for g in link.geoms])
    masses = link.mass * vols / vols.sum()

    total = np.zeros((3, 3))
    for g, m in zip(link.geoms, masses):

        local = _geom_inertia(g, m)
        rot = Rotation.from_euler("xyz", g.origin_rpy).as_matrix()
        rotated = rot @ local @ rot.T

        d = np.asarray(g.origin_xyz) - com
        shift = m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        total += rotated + shift
    return total

def _geom_inertia(g: GeomSpec, m: float) -> np.ndarray:

    if g.gtype == GeomType.BOX:
        x, y, z = g.size
        diag = [m / 12.0 * (y * y + z * z), m / 12.0 * (x * x + z * z), m / 12.0 * (x * x + y * y)]
    elif g.gtype == GeomType.CYLINDER:
        r, h = g.size[0], g.size[1]
        lat = m / 12.0 * (3 * r * r + h * h)
        diag = [lat, lat, m * r * r / 2.0]
    else:
        v = 2.0 / 5.0 * m * g.size[0] ** 2
        diag = [v, v, v]
    return np.diag(diag)

def write_urdf(spec: RobotSpec, path: str | Path) -> Path:

    root = ET.Element("robot", name=spec.name)
    for link in spec.links:
        root.append(_link_elem(link))
    for joint in spec.joints:
        root.append(_joint_elem(joint))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="unicode", xml_declaration=True)
    return path

def _link_elem(link: LinkSpec) -> ET.Element:

    elem = ET.Element("link", name=link.name)

    if link.geoms and link.mass > 0:
        elem.append(_inertial_elem(link))

    for g in link.geoms:
        for tag in ("visual", "collision"):
            sub = ET.SubElement(elem, tag)
            _origin_elem(sub, g.origin_xyz, g.origin_rpy)
            geom = ET.SubElement(sub, "geometry")
            _shape_elem(geom, g)
    return elem

def _inertial_elem(link: LinkSpec) -> ET.Element:

    inertial = ET.Element("inertial")
    com = compute_com(link)
    _origin_elem(inertial, tuple(com), (0, 0, 0))
    ET.SubElement(inertial, "mass", value=_fmt(link.mass))
    tensor = compute_inertia(link)
    ET.SubElement(
        inertial, "inertia",
        ixx=_fmt(tensor[0, 0]), iyy=_fmt(tensor[1, 1]), izz=_fmt(tensor[2, 2]),
        ixy=_fmt(tensor[0, 1]), ixz=_fmt(tensor[0, 2]), iyz=_fmt(tensor[1, 2]),
    )
    return inertial

def _joint_elem(joint: JointSpec) -> ET.Element:

    elem = ET.Element("joint", name=joint.name, type=joint.jtype)
    _origin_elem(elem, joint.origin_xyz, joint.origin_rpy)
    ET.SubElement(elem, "parent", link=joint.parent)
    ET.SubElement(elem, "child", link=joint.child)

    if joint.jtype != "fixed":
        ET.SubElement(elem, "axis", xyz=_vec(joint.axis))
        limit = ET.SubElement(
            elem, "limit", effort=_fmt(joint.effort), velocity=_fmt(joint.velocity),
        )
        if joint.jtype != "continuous":
            limit.set("lower", _fmt(joint.lower))
            limit.set("upper", _fmt(joint.upper))
    return elem

def _shape_elem(parent: ET.Element, g: GeomSpec):

    if g.gtype == GeomType.BOX:
        ET.SubElement(parent, "box", size=_vec(g.size))
    elif g.gtype == GeomType.CYLINDER:
        ET.SubElement(parent, "cylinder", radius=_fmt(g.size[0]), length=_fmt(g.size[1]))
    else:
        ET.SubElement(parent, "sphere", radius=_fmt(g.size[0]))

def _origin_elem(parent: ET.Element, xyz, rpy):

    ET.SubElement(parent, "origin", xyz=_vec(xyz), rpy=_vec(rpy))

def _fmt(x: float) -> str:

    return f"{float(x):.6g}"

def _vec(vec) -> str:

    return " ".join(_fmt(x) for x in vec)

class BaseGenerator(ABC):

    FAMILY: str = ""
    FORMS: tuple[str, ...] = ()

    def __init__(self, cfg: dict):

        self._cfg = cfg

    @abstractmethod
    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:
        pass

    def _u(self, rng: np.random.Generator, key: str, scale: float = 1.0) -> float:

        lo, hi = self._cfg[self.FAMILY][key]
        return float(rng.uniform(lo, hi)) * scale

    def _clamp_mass_ratio(self, spec: RobotSpec, max_ratio: float = 500.0):

        floor = max(l.mass for l in spec.links) / max_ratio
        for link in spec.links:
            if 0 < link.mass < floor:
                link.mass = floor

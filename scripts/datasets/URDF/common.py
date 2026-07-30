"""랜덤 URDF 생성 공통 유틸.

좌표 규약: x 전방, y 좌측, z 상방. 단위 m / kg / rad.
- rest pose(모든 관절 각도 0) = 스탠딩 자세. 템플릿이 joint origin rpy에 자세를 미리 반영한다.
- 질량·관성은 primitive 해석식 + 유효 밀도로만 계산한다(수기 관성 입력 금지).
- 관절 effort는 하류 질량 스케일 법칙 tau = C * m_down * g * L 로 일괄 산출(finalize).

링크 이름 prefix = GNN 노드 타입:
  base | torso | head | neck | arm | hand | leg | foot | ball(관절 연결구) |
  wheel | caster | steer | mast | sensor_rgb | sensor_depth | sensor_lidar | sensor_imu
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from xml.dom import minidom

G = 9.81
TORQUE_C = 2.5          # effort 스케일 계수
REVOLUTE_VEL = 12.0     # rad/s
CONTINUOUS_VEL = 40.0   # rad/s (바퀴)

NODE_TYPES = [
    "base", "torso", "head", "neck", "arm", "hand", "leg", "foot", "ball",
    "wheel", "caster", "steer", "mast",
    "sensor_rgb", "sensor_depth", "sensor_lidar", "sensor_imu",
]

# 타입별 시각화 색 (r,g,b,a)
TYPE_COLOR = {
    "base": (0.45, 0.52, 0.60, 1), "torso": (0.45, 0.52, 0.60, 1),
    "head": (0.75, 0.72, 0.65, 1), "neck": (0.6, 0.6, 0.6, 1),
    "arm": (0.65, 0.68, 0.72, 1), "hand": (0.5, 0.5, 0.55, 1),
    "leg": (0.65, 0.68, 0.72, 1), "foot": (0.30, 0.30, 0.33, 1),
    "ball": (0.55, 0.45, 0.35, 1),
    "wheel": (0.15, 0.15, 0.15, 1), "caster": (0.25, 0.25, 0.25, 1),
    "steer": (0.5, 0.4, 0.3, 1), "mast": (0.55, 0.55, 0.6, 1),
    "sensor_rgb": (0.85, 0.2, 0.2, 1), "sensor_depth": (0.9, 0.55, 0.1, 1),
    "sensor_lidar": (0.2, 0.4, 0.85, 1), "sensor_imu": (0.2, 0.7, 0.3, 1),
}


def node_type(link_name: str) -> str:
    for t in sorted(NODE_TYPES, key=len, reverse=True):
        if link_name.startswith(t):
            return t
    raise ValueError(f"알 수 없는 링크 타입: {link_name}")


@dataclass
class Geom:
    kind: str            # box | cylinder | sphere
    size: tuple          # box:(sx,sy,sz)  cylinder:(r,l)  sphere:(r,)
    rpy: tuple = (0.0, 0.0, 0.0)  # 링크 프레임 내 기하 회전(바퀴 등)

    def volume(self):
        if self.kind == "box":
            sx, sy, sz = self.size
            return sx * sy * sz
        if self.kind == "cylinder":
            r, l = self.size
            return math.pi * r * r * l
        r = self.size[0]
        return 4.0 / 3.0 * math.pi * r ** 3

    def inertia_diag(self, m):
        """기하 프레임 기준 주축 관성(대각)."""
        if self.kind == "box":
            sx, sy, sz = self.size
            return (m / 12 * (sy ** 2 + sz ** 2),
                    m / 12 * (sx ** 2 + sz ** 2),
                    m / 12 * (sx ** 2 + sy ** 2))
        if self.kind == "cylinder":
            r, l = self.size
            ixx = m / 12 * (3 * r ** 2 + l ** 2)
            return (ixx, ixx, m / 2 * r ** 2)
        r = self.size[0]
        i = 2.0 / 5.0 * m * r ** 2
        return (i, i, i)

    def char_len(self):
        """토크 스케일용 특성 길이."""
        if self.kind == "box":
            return max(self.size)
        if self.kind == "cylinder":
            return max(self.size[1], self.size[0] * 2)
        return self.size[0] * 2


@dataclass
class Link:
    name: str
    geom: Geom
    density: float
    origin: tuple = (0.0, 0.0, 0.0)  # 링크 프레임 내 기하 중심
    connector: bool = False          # 관절 연결구(ball) 여부 — 충돌 검사 인접 확장에 사용

    @property
    def mass(self):
        return self.geom.volume() * self.density

    @property
    def inertia(self):
        return self.geom.inertia_diag(self.mass)


@dataclass
class Joint:
    name: str
    jtype: str           # revolute | continuous | fixed
    parent: str
    child: str
    xyz: tuple = (0.0, 0.0, 0.0)
    rpy: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (0.0, 0.0, 1.0)
    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0


@dataclass
class Robot:
    name: str
    links: list = field(default_factory=list)
    joints: list = field(default_factory=list)

    def add_link(self, name, geom, density, origin=(0, 0, 0), connector=False):
        self.links.append(Link(name, geom, density, origin, connector))
        return name

    def add_joint(self, jtype, parent, child, xyz, rpy=(0, 0, 0),
                  axis=(0, 0, 1), lower=0.0, upper=0.0):
        self.joints.append(Joint(f"j_{child}", jtype, parent, child,
                                 tuple(xyz), tuple(rpy), tuple(axis), lower, upper))

    def link(self, name):
        return next(l for l in self.links if l.name == name)

    def children_of(self, name):
        return [j.child for j in self.joints if j.parent == name]

    def subtree_mass(self, name):
        m = self.link(name).mass
        for c in self.children_of(name):
            m += self.subtree_mass(c)
        return m

    def finalize(self):
        """effort/velocity를 하류 질량 스케일 법칙으로 일괄 산출."""
        for j in self.joints:
            if j.jtype == "fixed":
                continue
            m_down = self.subtree_mass(j.child)
            L = max(self.link(j.child).geom.char_len(), 0.05)
            j.effort = max(2.0, TORQUE_C * m_down * G * L)
            j.velocity = CONTINUOUS_VEL if j.jtype == "continuous" else REVOLUTE_VEL

    def shrink_limits(self, factor):
        """자기충돌 시 revolute 리밋을 0 방향으로 축소."""
        for j in self.joints:
            if j.jtype == "revolute":
                j.lower *= factor
                j.upper *= factor

    def total_mass(self):
        return sum(l.mass for l in self.links)

    # ---------------- URDF 직렬화 ----------------

    def _geom_xml(self, parent_el, link):
        g = link.geom
        for tag in ("visual", "collision"):
            el = ET.SubElement(parent_el, tag)
            ET.SubElement(el, "origin",
                          xyz=_v(link.origin), rpy=_v(g.rpy))
            geo = ET.SubElement(el, "geometry")
            if g.kind == "box":
                ET.SubElement(geo, "box", size=_v(g.size))
            elif g.kind == "cylinder":
                ET.SubElement(geo, "cylinder",
                              radius=str(g.size[0]), length=str(g.size[1]))
            else:
                ET.SubElement(geo, "sphere", radius=str(g.size[0]))
            if tag == "visual":
                mat = ET.SubElement(el, "material",
                                    name=f"c_{node_type(link.name)}")
                ET.SubElement(mat, "color",
                              rgba=_v(TYPE_COLOR[node_type(link.name)]))

    def to_urdf(self) -> str:
        root = ET.Element("robot", name=self.name)
        for link in self.links:
            le = ET.SubElement(root, "link", name=link.name)
            inertial = ET.SubElement(le, "inertial")
            ET.SubElement(inertial, "origin",
                          xyz=_v(link.origin), rpy=_v(link.geom.rpy))
            ET.SubElement(inertial, "mass", value=str(link.mass))
            ixx, iyy, izz = link.inertia
            ET.SubElement(inertial, "inertia", ixx=str(ixx), iyy=str(iyy),
                          izz=str(izz), ixy="0", ixz="0", iyz="0")
            self._geom_xml(le, link)
        for j in self.joints:
            je = ET.SubElement(root, "joint", name=j.name, type=j.jtype)
            ET.SubElement(je, "parent", link=j.parent)
            ET.SubElement(je, "child", link=j.child)
            ET.SubElement(je, "origin", xyz=_v(j.xyz), rpy=_v(j.rpy))
            if j.jtype != "fixed":
                ET.SubElement(je, "axis", xyz=_v(j.axis))
            if j.jtype == "revolute":
                ET.SubElement(je, "limit", lower=str(j.lower), upper=str(j.upper),
                              effort=str(j.effort), velocity=str(j.velocity))
            elif j.jtype == "continuous":
                ET.SubElement(je, "limit", effort=str(j.effort),
                              velocity=str(j.velocity))
        return minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

    def node_summary(self):
        return [{"name": l.name, "type": node_type(l.name),
                 "geom": l.geom.kind, "size": list(l.geom.size),
                 "mass": round(l.mass, 4)} for l in self.links]


def _v(vec):
    return " ".join(f"{x:.6g}" for x in vec)


# ---------------- 센서 ----------------

SENSOR_GEOM = {
    "sensor_rgb": Geom("box", (0.022, 0.06, 0.022)),
    "sensor_depth": Geom("box", (0.024, 0.09, 0.026)),
    "sensor_lidar": Geom("cylinder", (0.036, 0.04)),
    "sensor_imu": Geom("box", (0.016, 0.016, 0.008)),
}
SENSOR_DENSITY = 600.0


def attach_sensor(robot: Robot, stype: str, parent: str, xyz, rpy=(0, 0, 0), idx=0):
    """센서 = 소형 link + fixed joint. GNN에서는 이름 prefix로 노드 타입 구분."""
    name = f"{stype}_{idx}"
    robot.add_link(name, Geom(SENSOR_GEOM[stype].kind, SENSOR_GEOM[stype].size),
                   SENSOR_DENSITY)
    robot.add_joint("fixed", parent, name, xyz, rpy)
    return name

"""로봇 생성의 공통 기반.

- 스펙 자료구조: GeomSpec / LinkSpec / JointSpec / RobotSpec (URDF 이전의 중간 표현)
- 관성 계산 함수: compute_com / compute_inertia (프리미티브 조합, 평행축 정리)
- URDF 쓰기 함수: write_urdf (스펙 -> XML 파일)
- 생성기 추상 클래스: BaseGenerator (platform/ 생성기들이 상속)

좌표계 규약: 모든 생성기는 base_link 원점 = 몸통 기하 중심, x 전방, y 좌측, z 상방(중력 반대).
길이 단위 m, 질량 kg, 각도 rad. 표기 랜덤화는 계획상 인코더 입력측 증강(4단계)이므로
이 단계의 산출물은 정본 URDF뿐이다.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


# ---------- 스펙 자료구조 ----------

class GeomType(str, Enum):
    """URDF가 지원하는 충돌 프리미티브 종류."""

    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"


@dataclass
class GeomSpec:
    """단일 프리미티브 형상.

    size 해석: box=(x,y,z 변 길이), cylinder=(radius, length, 0), sphere=(radius, 0, 0).
    실린더의 축은 로컬 z이며, 다른 방향이 필요하면 origin_rpy로 돌린다.
    origin은 링크 프레임 기준 형상 중심의 위치·자세.
    """

    gtype: GeomType
    size: tuple[float, float, float]
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def volume(self) -> float:
        """형상 부피 [m^3]. 링크 내 질량 배분(부피 비례)에 쓰인다."""
        if self.gtype == GeomType.BOX:
            return self.size[0] * self.size[1] * self.size[2]
        if self.gtype == GeomType.CYLINDER:
            return math.pi * self.size[0] ** 2 * self.size[1]
        return 4.0 / 3.0 * math.pi * self.size[0] ** 3


@dataclass
class LinkSpec:
    """링크 하나. 질량은 링크 전체 값.

    com_xyz/inertia가 None이면 write_urdf 시점에 compute_* 함수가
    형상 부피 비례 배분으로 계산한다. com_xyz 명시값은 무게중심 위치
    랜덤화(밸러스트 치우침)에 쓰인다.
    """

    name: str
    geoms: list[GeomSpec] = field(default_factory=list)
    mass: float = 0.0
    # None = 계산값 사용, 값 존재 = 명시값 그대로 기록
    com_xyz: tuple[float, float, float] | None = None
    # 관성텐서 성분 (ixx, iyy, izz, ixy, ixz, iyz), 링크 질량중심 기준
    inertia: tuple[float, float, float, float, float, float] | None = None


@dataclass
class JointSpec:
    """조인트 하나.

    jtype: revolute | continuous | prismatic | fixed.
    origin은 부모 링크 프레임 기준 자식 링크 프레임의 위치·자세,
    axis는 자식 프레임 기준 회전(이동)축. lower/upper는 revolute/prismatic만 사용.
    effort=0인 가동 조인트는 수동(무동력)을 의미한다.
    """

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
    """로봇 전체 스펙. links[0]은 base_link여야 한다."""

    name: str
    family: str
    form: str
    links: list[LinkSpec] = field(default_factory=list)
    joints: list[JointSpec] = field(default_factory=list)
    # 샘플링된 원시 파라미터 (보조 손실 라벨·커버리지 분석용 속성 딕셔너리)
    params: dict = field(default_factory=dict)
    # 정적 검사·시뮬 스폰에 쓰는 기립 자세 (actuated 조인트 이름 -> 값 [rad|m])
    standing_pose: dict = field(default_factory=dict)
    # 지면 접촉 링크 이름 (바퀴·롤러·발)
    contact_links: list[str] = field(default_factory=list)
    # 제어기 매핑용 구동 타입 태그 (diff, diff_balancing, skid, omni, quad, ... )
    control_tag: str = ""
    # 기립 외에 셀프 충돌을 추가 검사할 자세 목록 (조향 극한, 리프트 최대 등).
    # 각 원소는 standing_pose에 덮어쓸 부분 자세 dict
    check_poses: list[dict] = field(default_factory=list)
    # 플랫폼 특수 검사 파라미터 (validate.py의 special 핸들러가 해석)
    # 예: {"load_share": {...}, "balancing": {...}, "overturn": {...}}
    special: dict = field(default_factory=dict)

    def link(self, name: str) -> LinkSpec:
        """이름으로 링크를 찾는다. 없으면 KeyError."""
        for l in self.links:
            if l.name == name:
                return l
        raise KeyError(name)

    def total_mass(self) -> float:
        """전체 질량 [kg]. 토크 한계 샘플링의 기준값으로 쓰인다."""
        return sum(l.mass for l in self.links)

    def actuated_joints(self) -> list[JointSpec]:
        """fixed를 제외한, 자세 값을 갖는 조인트 목록."""
        return [j for j in self.joints if j.jtype in ("revolute", "continuous", "prismatic")]


# ---------- 관성 계산 ----------

def compute_com(link: LinkSpec) -> np.ndarray:
    """링크 질량중심 [m, 링크 프레임].

    명시값(com_xyz)이 있으면 그대로 반환. 없으면 균일 밀도 가정하에
    형상 중심들의 부피 가중 평균으로 계산한다.
    """
    if link.com_xyz is not None:
        return np.asarray(link.com_xyz, dtype=float)

    # 균일 밀도 가정 -> 질량 배분이 부피 비례이므로 부피 가중 평균이 질량중심
    vols = np.array([g.volume for g in link.geoms])
    centers = np.array([g.origin_xyz for g in link.geoms])
    return (vols[:, None] * centers).sum(axis=0) / vols.sum()


def compute_inertia(link: LinkSpec) -> np.ndarray:
    """링크 질량중심 기준 3x3 관성텐서 [kg m^2, 링크 프레임 방향].

    명시값(inertia)이 있으면 그대로 행렬로 복원한다. 없으면:
    링크 질량을 형상 부피 비례로 나누고, 각 형상의 표준 관성을
    (1) 형상 자세만큼 회전(I' = R I R^T),
    (2) 평행축 정리(I += m(|d|^2 E - d d^T), d = 형상중심 - 질량중심)
    로 링크 질량중심 기준으로 합산한다.
    """
    if link.inertia is not None:
        ixx, iyy, izz, ixy, ixz, iyz = link.inertia
        return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

    # 링크 질량을 형상 부피 비례로 배분
    com = compute_com(link)
    vols = np.array([g.volume for g in link.geoms])
    masses = link.mass * vols / vols.sum()

    total = np.zeros((3, 3))
    for g, m in zip(link.geoms, masses):
        # 형상 자세 회전: 관성텐서는 2계 텐서이므로 I' = R I R^T
        local = _geom_inertia(g, m)
        rot = Rotation.from_euler("xyz", g.origin_rpy).as_matrix()
        rotated = rot @ local @ rot.T

        # 평행축 정리: 형상 중심 -> 링크 질량중심으로 기준점 이동
        d = np.asarray(g.origin_xyz) - com
        shift = m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        total += rotated + shift
    return total


def _geom_inertia(g: GeomSpec, m: float) -> np.ndarray:
    """프리미티브 중심·자체 프레임 기준 관성텐서 (표준 강체 공식).

    box: I = m/12 * diag(y^2+z^2, x^2+z^2, x^2+y^2)
    cylinder(축=z): Ixx=Iyy=m(3r^2+h^2)/12, Izz=mr^2/2
    sphere: I = 2/5 m r^2 (등방)
    """
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


# ---------- URDF 쓰기 ----------

def write_urdf(spec: RobotSpec, path: str | Path) -> Path:
    """스펙을 URDF 파일로 저장하고 경로를 반환한다.

    관성·질량중심은 명시값이 없으면 compute_*로 계산해 기록한다.
    """
    # 링크 -> 조인트 순서로 직렬화 (URDF는 순서 무관하지만 가독성을 위해 고정)
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
    """<link> 요소 생성. visual과 collision은 같은 프리미티브를 사용한다."""
    elem = ET.Element("link", name=link.name)

    # 형상·질량이 없는 더미 링크는 inertial을 생략 (URDF 허용)
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
    """<inertial> 요소 생성. origin = 질량중심, 축은 링크 프레임과 평행."""
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
    """<joint> 요소 생성."""
    elem = ET.Element("joint", name=joint.name, type=joint.jtype)
    _origin_elem(elem, joint.origin_xyz, joint.origin_rpy)
    ET.SubElement(elem, "parent", link=joint.parent)
    ET.SubElement(elem, "child", link=joint.child)

    # fixed는 축·한계가 없고, continuous는 가동 범위(lower/upper)를 쓰지 않는다
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
    """<geometry> 하위의 프리미티브 요소 생성."""
    if g.gtype == GeomType.BOX:
        ET.SubElement(parent, "box", size=_vec(g.size))
    elif g.gtype == GeomType.CYLINDER:
        ET.SubElement(parent, "cylinder", radius=_fmt(g.size[0]), length=_fmt(g.size[1]))
    else:
        ET.SubElement(parent, "sphere", radius=_fmt(g.size[0]))


def _origin_elem(parent: ET.Element, xyz, rpy):
    """<origin xyz rpy> 요소 생성."""
    ET.SubElement(parent, "origin", xyz=_vec(xyz), rpy=_vec(rpy))


def _fmt(x: float) -> str:
    """실수 -> URDF 속성 문자열 (유효숫자 6자리)."""
    return f"{float(x):.6g}"


def _vec(vec) -> str:
    """벡터 -> 공백 구분 문자열."""
    return " ".join(_fmt(x) for x in vec)


# ---------- 생성기 추상 클래스 ----------

class BaseGenerator(ABC):
    """템플릿 기반 2층 샘플링 생성기의 공통 골격.

    구조 축(이산)과 파라미터(연속)는 각 서브클래스의 sample()에서 뽑는다.
    cfg는 전체 설정 dict이며, 범위 헬퍼(_u)는 자기 FAMILY 섹션에서 읽는다
    (wheeled 휴머노이드처럼 다른 생성기를 재사용하는 경우가 있어 전체를 보관).
    FAMILY = 설정 섹션 이름, FORMS = 이 생성기가 만드는 form 태그 목록.
    """

    FAMILY: str = ""
    FORMS: tuple[str, ...] = ()

    def __init__(self, cfg: dict):
        """cfg: configs/urdf.yaml 전체 dict."""
        self._cfg = cfg

    @abstractmethod
    def sample(self, form: str, rng: np.random.Generator) -> RobotSpec:
        """form에 해당하는 로봇 스펙 하나를 샘플링한다."""

    def _u(self, rng: np.random.Generator, key: str, scale: float = 1.0) -> float:
        """자기 FAMILY 섹션의 [lo, hi] 범위에서 균등 샘플. scale = 비례 샘플링 배수."""
        lo, hi = self._cfg[self.FAMILY][key]
        return float(rng.uniform(lo, hi)) * scale

    def _clamp_mass_ratio(self, spec: RobotSpec, max_ratio: float = 500.0):
        """로봇 내 링크 질량비를 max_ratio 이하로 클램프.

        관절로 연결된 링크의 질량비가 1e3-1e4를 넘으면 PhysX solver가
        불안정해지므로 (플랜: 상한 1000), 최대 링크 질량 기준 하한으로
        소형 링크(롤러·너클·블록)를 끌어올린다. 기본값 500은 상한 대비
        보수적 여유. 각 생성기 sample() 마지막에 호출.
        """
        floor = max(l.mass for l in spec.links) / max_ratio
        for link in spec.links:
            if 0 < link.mass < floor:
                link.mass = floor

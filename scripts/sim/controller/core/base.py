from __future__ import annotations

import json
import logging
import math
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    # 타입 힌트 전용 — isaaclab 의존성을 base.py 런타임에 끌어들이지 않기 위해 지연 임포트
    from .scan_terrain import TerrainScan

@dataclass
class UrdfJoint:

    name: str
    jtype: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray

@dataclass
class UrdfLink:

    name: str
    mass: float
    com: np.ndarray

    inertia: np.ndarray

@dataclass
class UrdfModel:

    links: dict[str, UrdfLink]
    joints: dict[str, UrdfJoint]

    parent_joint: dict[str, str]

def _floats(text: str | None, default: str) -> np.ndarray:

    return np.array([float(v) for v in (text or default).split()])

def parse_urdf(urdf_path: Path) -> UrdfModel:

    root = ET.parse(urdf_path).getroot()
    links, joints, parent_joint = {}, {}, {}
    for l in root.findall("link"):
        inertial = l.find("inertial")
        if inertial is None:

            links[l.get("name")] = UrdfLink(l.get("name"), 0.0, np.zeros(3), np.zeros(6))
            continue
        origin = inertial.find("origin")
        inertia = inertial.find("inertia")

        if origin is not None and any(abs(v) > 1e-9
                                      for v in _floats(origin.get("rpy"), "0 0 0")):
            logging.getLogger(__name__).warning(
                "%s: inertial origin rpy != 0 — 관성 회전 미반영 (생성 규약 위반)",
                l.get("name"))
        links[l.get("name")] = UrdfLink(
            name=l.get("name"),
            mass=float(inertial.find("mass").get("value")),
            com=_floats(origin.get("xyz") if origin is not None else None, "0 0 0"),
            inertia=np.array([float(inertia.get(k, 0.0))
                              for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]),
        )
    for j in root.findall("joint"):
        origin = j.find("origin")
        axis = j.find("axis")
        spec = UrdfJoint(
            name=j.get("name"), jtype=j.get("type"),
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            xyz=_floats(origin.get("xyz") if origin is not None else None, "0 0 0"),
            rpy=_floats(origin.get("rpy") if origin is not None else None, "0 0 0"),
            axis=_floats(axis.get("xyz") if axis is not None else None, "1 0 0"),
        )
        joints[spec.name] = spec
        parent_joint[spec.child] = spec.name
    return UrdfModel(links, joints, parent_joint)

def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:

    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p),        math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])

def zero_pose_frame(model: UrdfModel, link: str) -> tuple[np.ndarray, np.ndarray]:

    R, p = np.eye(3), np.zeros(3)

    while link in model.parent_joint:
        j = model.joints[model.parent_joint[link]]
        Rj = _rpy_matrix(j.rpy)
        R = Rj @ R
        p = Rj @ p + j.xyz
        link = j.parent
    return R, p

@dataclass
class WheelFrame:

    joint: str
    link: str
    pos: np.ndarray
    axis: np.ndarray

@dataclass
class RobotCtrlParams:

    name: str
    control_tag: str
    base_tag: str
    wheel_radius: float
    max_lin_vel: float

    wheels: list[WheelFrame] = field(default_factory=list)
    wheel_vel_limit: dict[str, float] = field(default_factory=dict)
    wheel_effort_limit: dict[str, float] = field(default_factory=dict)

    steer_joints: list[str] = field(default_factory=list)
    steer_range: float = 0.0
    steer_vel_limit: float = 0.0
    wheelbase: float = 0.0
    steer_y: dict[str, float] = field(default_factory=dict)

    mecanum_sign: dict[str, float] = field(default_factory=dict)

    holonomic: bool = False

    tip_accel: float = 0.0

def _base_tag(control_tag: str) -> str:

    if control_tag.startswith("wheeled_humanoid_"):
        return control_tag[len("wheeled_humanoid_"):]
    return control_tag

def extract_ctrl_params(urdf_path: Path, usd_dir: Path) -> RobotCtrlParams:

    usd_dir = Path(usd_dir)
    with open(usd_dir / "meta.json") as f:
        meta = json.load(f)
    with open(usd_dir / "joints.json") as f:
        jinfo = json.load(f)
    model = parse_urdf(urdf_path)

    params = meta["params"]
    tag = meta["control_tag"]

    out = RobotCtrlParams(
        name=meta["name"], control_tag=tag, base_tag=_base_tag(tag),
        wheel_radius=float(params.get("wheel_radius", 0.0)),
        max_lin_vel=float(params.get("max_lin_vel", 0.0)),
        holonomic=_base_tag(tag) == "omni" and params.get("subtype") != "mecanum4",
    )

    metrics = meta["metrics"]
    supports = [params[k] for k in ("track_width", "track_front", "track_rear",
                                    "wheelbase", "ring_radius") if k in params]
    com_h = float(metrics["base_height"]) + float(metrics["com"][2])
    if supports and com_h > 1e-3:
        out.tip_accel = 9.81 * 0.5 * min(map(float, supports)) / com_h

    by_name = {j["name"]: j for j in jinfo["joints"]}
    for jname in jinfo["groups"]["velocity"]:

        j = model.joints[jname]
        R, p = zero_pose_frame(model, j.child)
        out.wheels.append(WheelFrame(joint=jname, link=j.child, pos=p,
                                     axis=R[:, 1].copy()))
        out.wheel_vel_limit[jname] = by_name[jname]["velocity"]
        out.wheel_effort_limit[jname] = by_name[jname]["effort"]
        if out.base_tag == "omni" and params.get("subtype") == "mecanum4":

            out.mecanum_sign[jname] = -math.copysign(1.0, p[0] * p[1])

    if out.base_tag == "ackermann":
        out.steer_joints = [n for n in jinfo["groups"]["position"]
                            if n.startswith("steer_")]
        out.steer_range = float(params["steer_range"])
        out.steer_vel_limit = min(by_name[n]["velocity"] for n in out.steer_joints)
        out.wheelbase = float(params["wheelbase"])
        for n in out.steer_joints:
            out.steer_y[n] = float(model.joints[n].xyz[1])
    return out

@dataclass
class ControlObs:

    pos_xy: torch.Tensor
    yaw: torch.Tensor
    pitch: torch.Tensor
    vel_b: torch.Tensor
    ang_b: torch.Tensor
    joint_pos: torch.Tensor | None = None
    joint_vel: torch.Tensor | None = None
    gravity_b: torch.Tensor | None = None
    height_scan: torch.Tensor | None = None
    terrain_scan: "TerrainScan | None" = None

    @classmethod
    def from_articulation(cls, art) -> "ControlObs":

        q = art.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        return cls(pos_xy=art.data.root_pos_w[:, :2], yaw=yaw, pitch=pitch,
                   vel_b=art.data.root_lin_vel_b, ang_b=art.data.root_ang_vel_b,
                   joint_pos=art.data.joint_pos, joint_vel=art.data.joint_vel,
                   gravity_b=art.data.projected_gravity_b)

@dataclass
class JointTargets:

    pos: torch.Tensor | None
    vel: torch.Tensor | None
    effort: torch.Tensor | None
    cmd: torch.Tensor | None = None

class BaseController(ABC):

    def __init__(self, params: RobotCtrlParams, joint_names: list[str],
                 default_pose: torch.Tensor, num_envs: int, device: str):

        self._params = params
        self._joint_index = {n: i for i, n in enumerate(joint_names)}
        self._default_pose = default_pose
        self._num_envs = num_envs
        self._device = device

    @property
    def params(self) -> RobotCtrlParams:

        return self._params

    def _index_of(self, names: list[str]) -> torch.Tensor:

        return torch.tensor([self._joint_index[n] for n in names],
                            dtype=torch.long, device=self._device)

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None):
        pass

    @abstractmethod
    def compute(self, obs: ControlObs, goal_xy: torch.Tensor) -> JointTargets:
        pass

LEGGED_TAGS = ("quad", "hex", "humanoid")

def make_controller(urdf_path: Path, usd_dir: Path, cfg: dict,
                    joint_names: list[str], default_pose: torch.Tensor,
                    num_envs: int, device: str, physics_dt: float, *,
                    policy_dir: Path | None = None) -> BaseController:

    from ..wheeled.controller import WheeledRobotController

    params = extract_ctrl_params(urdf_path, usd_dir)
    if params.base_tag in ("diff", "skid", "ackermann", "omni"):
        return WheeledRobotController(params, joint_names, default_pose,
                                      num_envs, device, cfg, physics_dt)
    if params.base_tag in LEGGED_TAGS:
        from ..legged.controller import LeggedRobotController

        if policy_dir is None:
            raise ValueError(f"legged form({params.base_tag})은 policy_dir가 필요하다"
                             " — train으로 정책을 먼저 학습할 것")
        return LeggedRobotController(params, joint_names, default_pose,
                                     num_envs, device, cfg, physics_dt,
                                     urdf_path, usd_dir, Path(policy_dir))
    raise ValueError(f"지원하지 않는 control_tag: {params.control_tag}")

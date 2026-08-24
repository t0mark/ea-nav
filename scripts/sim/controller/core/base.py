"""제어기 공통 기반: 인터페이스·관측/목표 자료구조·로봇 물성 추출·팩토리.

Isaac 비의존 (torch·numpy·표준 라이브러리만) — 앱 기동 전에도 임포트할 수
있고, wheeled·legged 제어기가 공유한다.

좌표·부호 규약 (1단계 생성기·Isaac Lab과 정합):
- 몸체 좌표계: +x 전진, +y 좌측, +z 상방 (URDF base_link와 동일)
- yaw: 월드 z축 반시계 양수. pitch: 몸체 +y축 회전 양수 = 앞으로 숙임
  (Isaac 쿼터니언 (w,x,y,z)에서 ZYX 오일러로 추출)
- 바퀴 조인트 회전축 = 자식 프레임 +y (생성 규약) -> 양의 각속도 = 전진 굴림

로봇별 물성의 출처:
- meta.json  : control_tag·params (트랙 폭·조향 범위 등 대표 치수)
- joints.json: 가동 조인트의 드라이브 그룹·토크/속도 한계
- robot.urdf : 조인트 원점·축(바퀴 배치 기하), 링크 질량·관성 (LQR 모델용)
  — meta에 없는 기하·관성은 정본 URDF에서 직접 읽는다
"""
from __future__ import annotations

import json
import logging
import math
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# ---------- URDF 기하·관성 파싱 ----------

@dataclass
class UrdfJoint:
    """URDF 조인트 1개의 기하 (부모 프레임 기준 원점·축)."""

    name: str
    jtype: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


@dataclass
class UrdfLink:
    """URDF 링크 1개의 관성 (링크 프레임 기준 질량중심·관성 텐서)."""

    name: str
    mass: float
    com: np.ndarray
    # 관성 텐서 성분 (질량중심 기준, ixx iyy izz ixy ixz iyz)
    inertia: np.ndarray


@dataclass
class UrdfModel:
    """정본 URDF의 기하·관성 요약 (제어 파라미터 추출용)."""

    links: dict[str, UrdfLink]
    joints: dict[str, UrdfJoint]
    # 자식 링크 -> 그 링크를 매다는 조인트 (트리 상행용)
    parent_joint: dict[str, str]


def _floats(text: str | None, default: str) -> np.ndarray:
    """공백 구분 실수 문자열을 배열로 파싱한다 (미기재는 default 사용)."""
    return np.array([float(v) for v in (text or default).split()])


def parse_urdf(urdf_path: Path) -> UrdfModel:
    """robot.urdf에서 링크 관성과 조인트 기하를 읽는다.

    단위는 URDF 규약 (m, kg, rad). 관성 origin의 rpy는 생성기가 항상 0으로
    쓰므로 무시한다 (1단계 write_urdf 규약).
    """
    root = ET.parse(urdf_path).getroot()
    links, joints, parent_joint = {}, {}, {}
    for l in root.findall("link"):
        inertial = l.find("inertial")
        if inertial is None:
            # 관성 없는 링크(정본 생성 URDF에는 없음)는 질량 0으로 둔다
            links[l.get("name")] = UrdfLink(l.get("name"), 0.0, np.zeros(3), np.zeros(6))
            continue
        origin = inertial.find("origin")
        inertia = inertial.find("inertia")
        # 관성 origin rpy는 0 규약 (1단계 write_urdf). 위반하면 관성 텐서
        # 회전이 누락된 채 합성되므로 조용히 넘기지 않고 경고를 남긴다
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
    """URDF rpy(고정축 x-y-z 순 적용)를 회전행렬 R = Rz Ry Rx로 만든다."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), \
        math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def zero_pose_frame(model: UrdfModel, link: str) -> tuple[np.ndarray, np.ndarray]:
    """조인트 변위 0 자세에서 링크 프레임의 base_link 기준 (R, p)를 구한다.

    가동 조인트 변위를 0으로 두므로 조인트 origin 변환만 트리 상행 누적하면
    된다 (기립 자세 스폰 시 바퀴·조향의 기준 배치와 일치).
    """
    R, p = np.eye(3), np.zeros(3)
    # base_link까지 부모 방향으로 (R, p)를 왼쪽 합성한다
    while link in model.parent_joint:
        j = model.joints[model.parent_joint[link]]
        Rj = _rpy_matrix(j.rpy)
        R = Rj @ R
        p = Rj @ p + j.xyz
        link = j.parent
    return R, p


# ---------- 로봇별 제어 물성 ----------

@dataclass
class WheelFrame:
    """구동 바퀴 1개의 base_link 기준 배치 (조인트 변위 0 자세).

    pos = 바퀴 중심 위치 (m), axis = 회전축 단위 벡터 (자식 +y의 월드 방향).
    굴림 전진 방향 t = axis x z_hat (회전축이 수평이라는 생성 규약 전제).
    """

    joint: str
    pos: np.ndarray
    axis: np.ndarray


@dataclass
class RobotCtrlParams:
    """제어기가 쓰는 로봇 1대의 물성 요약.

    base_tag = 베이스 구동 타입 (diff/skid/ackermann/omni/diff_balancing —
    wheeled_humanoid는 접두사를 벗겨 베이스 타입으로 환원).
    한계값 단위: 바퀴 속도 rad/s, 토크 Nm, 선속도 m/s, 조향각 rad.
    """

    name: str
    control_tag: str
    base_tag: str
    wheel_radius: float
    max_lin_vel: float
    # 구동 바퀴 배치와 조인트별 (속도, 토크) 한계 (joints.json 순서와 무관)
    wheels: list[WheelFrame] = field(default_factory=list)
    wheel_vel_limit: dict[str, float] = field(default_factory=dict)
    wheel_effort_limit: dict[str, float] = field(default_factory=dict)
    # ackermann 전용: 조향 조인트 이름(l/r)·가동 범위·속도 한계·휠베이스·
    # 조향축 y좌표
    steer_joints: list[str] = field(default_factory=list)
    steer_range: float = 0.0
    steer_vel_limit: float = 0.0
    wheelbase: float = 0.0
    steer_y: dict[str, float] = field(default_factory=dict)
    # omni 매커넘 전용: 조인트별 롤러 축 y부호 s_i = -sign(x_i*y_i)
    mecanum_sign: dict[str, float] = field(default_factory=dict)
    # 홀로노믹 여부 (명령 차원 결정: True -> (vx, vy, w))
    holonomic: bool = False
    # 전도 한계 가속도 (m/s^2): g x 지지 다각형 반폭 / 무게중심 높이 —
    # 상체 무거운 개체의 가감속·선회 경계 산출용 (0 = 정보 없음)
    tip_accel: float = 0.0


def _base_tag(control_tag: str) -> str:
    """control_tag에서 베이스 구동 타입을 얻는다 (wheeled_humanoid 접두사 제거)."""
    if control_tag.startswith("wheeled_humanoid_"):
        return control_tag[len("wheeled_humanoid_"):]
    return control_tag


def extract_ctrl_params(urdf_path: Path, usd_dir: Path) -> RobotCtrlParams:
    """로봇 산출물(robot.urdf + usd 폴더의 meta·joints)에서 제어 물성을 뽑는다.

    구동 바퀴(drive_* velocity 그룹)의 배치는 URDF 조인트 변위 0 자세 FK로
    구한다 — 매커넘 부호·옴니 방사각·ackermann 좌우 y가 전부 여기서 나온다.
    """
    usd_dir = Path(usd_dir)
    with open(usd_dir / "meta.json") as f:
        meta = json.load(f)
    with open(usd_dir / "joints.json") as f:
        jinfo = json.load(f)
    model = parse_urdf(urdf_path)

    params = meta["params"]
    tag = meta["control_tag"]
    # 홀로노믹 명령은 omni 중 방사 배치(omni3/4)만. mecanum4는 Isaac 실측에서
    # 45도 롤러 접촉이 횡이동을 만들지 못해(물리 dt·마찰·솔버 반복 무관)
    # 비홀로노믹으로 강등한다 — 전진+차동 yaw만 사용 (code.md 실측 기록)
    # legged form은 바퀴 물성이 없으므로 0으로 두고 (명령 경계는 정책
    # 번들의 학습 명령 범위가 결정), 이하 바퀴·조향 추출 루프는 자연히 빈다
    out = RobotCtrlParams(
        name=meta["name"], control_tag=tag, base_tag=_base_tag(tag),
        wheel_radius=float(params.get("wheel_radius", 0.0)),
        max_lin_vel=float(params.get("max_lin_vel", 0.0)),
        holonomic=_base_tag(tag) == "omni" and params.get("subtype") != "mecanum4",
    )

    # 전도 한계 가속도: 준정적 전도 조건 a > g x (지지 반폭 / 질량중심 높이).
    # 지지 반폭은 트랙 계열 치수의 최솟값(보수), 질량중심 높이는 기립
    # 자세의 지면 기준 (base 원점 높이 + base 프레임 com z)
    metrics = meta["metrics"]
    supports = [params[k] for k in ("track_width", "track_front", "track_rear",
                                    "wheelbase", "ring_radius") if k in params]
    com_h = float(metrics["base_height"]) + float(metrics["com"][2])
    if supports and com_h > 1e-3:
        out.tip_accel = 9.81 * 0.5 * min(map(float, supports)) / com_h

    by_name = {j["name"]: j for j in jinfo["joints"]}
    for jname in jinfo["groups"]["velocity"]:
        # 구동 바퀴 배치: 자식 링크 프레임의 위치·+y축 (회전축 규약)
        j = model.joints[jname]
        R, p = zero_pose_frame(model, j.child)
        out.wheels.append(WheelFrame(joint=jname, pos=p, axis=R[:, 1].copy()))
        out.wheel_vel_limit[jname] = by_name[jname]["velocity"]
        out.wheel_effort_limit[jname] = by_name[jname]["effort"]
        if out.base_tag == "omni" and params.get("subtype") == "mecanum4":
            # 매커넘 롤러 축 y부호 규칙 s = -sign(x*y) (1단계 배치 규약 재현)
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


# ---------- 관측·조인트 목표 자료구조 ----------

@dataclass
class ControlObs:
    """제어기 입력 관측 (전부 (N,) 또는 (N,k) 배치 텐서, N = env 수).

    pos_xy = 월드 평면 위치 (m), yaw/pitch = rad, vel_b = 몸체 좌표 선속도
    (m/s), ang_b = 몸체 좌표 각속도 (rad/s). pitch 양수 = 앞으로 숙임.
    joint_pos/joint_vel = 전 DoF 관절 상태 (N,D) (rad, rad/s — articulation
    조인트 순서), gravity_b = 몸체 좌표 중력 방향 단위 벡터 (N,3) (직립 =
    (0,0,-1)). 셋은 legged RL 정책 관측용 — wheeled 제어기는 쓰지 않는다.
    height_scan = 지형 높이 스캔 (N,R) (low_rl 정규화 규약: 0 = 기립 높이의
    평지) — 지형 롤아웃(3단계)에서 RayCaster 값으로 채우고, 평지 씬은 None
    (제어기가 0으로 대체 — "평지" 의미와 정확히 일치).
    """

    pos_xy: torch.Tensor
    yaw: torch.Tensor
    pitch: torch.Tensor
    vel_b: torch.Tensor
    ang_b: torch.Tensor
    joint_pos: torch.Tensor | None = None
    joint_vel: torch.Tensor | None = None
    gravity_b: torch.Tensor | None = None
    height_scan: torch.Tensor | None = None

    @classmethod
    def from_articulation(cls, art) -> "ControlObs":
        """Isaac Lab Articulation 데이터에서 관측을 조립한다 (덕 타이핑).

        쿼터니언 (w,x,y,z) -> ZYX 오일러: yaw = atan2(2(wz+xy), 1-2(y^2+z^2)),
        pitch = asin(2(wy-zx)). 순수 +y 회전 q=(cos a/2,0,sin a/2,0)에서
        pitch = a 가 나오므로 양수 = 앞으로 숙임 규약과 일치한다.
        """
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
    """1 제어 스텝의 조인트 목표 (전 DoF (N,D) 텐서, 없는 항목은 None).

    SimEnvironment.step(joint_pos_target, joint_vel_target, joint_effort_target)
    에 그대로 전달한다. 제어기가 소유하지 않는 position 조인트(팔·리프트 등)는
    기본 기립 자세 값을 유지한다.

    cmd = 이번 스텝의 몸체 속도 명령 (N,3) = (vx, vy, w), 모델 수준 (미끄럼
    보상 배율 이전). 3단계 GT의 WVN식 안정성 점수(명령 대비 실제 속도 추종
    오차) 계산용으로 노출한다.
    """

    pos: torch.Tensor | None
    vel: torch.Tensor | None
    effort: torch.Tensor | None
    cmd: torch.Tensor | None = None


# ---------- 제어기 인터페이스·팩토리 ----------

class BaseController(ABC):
    """제어기 공통 계약: reset() 후 매 제어 스텝 compute(관측, 목표) 호출.

    articulation 조인트 순서(joint_names)를 기준으로 전 DoF 목표 텐서를
    구성한다 — 조인트 이름 -> 인덱스 매핑을 여기서 한 번만 만든다.
    """

    def __init__(self, params: RobotCtrlParams, joint_names: list[str],
                 default_pose: torch.Tensor, num_envs: int, device: str):
        """공통 상태 구성.

        joint_names = articulation 조인트 순서, default_pose = (N,D) 기립 자세
        (비소유 position 조인트의 유지 목표로 사용).
        """
        self._params = params
        self._joint_index = {n: i for i, n in enumerate(joint_names)}
        self._default_pose = default_pose
        self._num_envs = num_envs
        self._device = device

    @property
    def params(self) -> RobotCtrlParams:
        """로봇 물성 요약 (진입점의 보고·로그용)."""
        return self._params

    def _index_of(self, names: list[str]) -> torch.Tensor:
        """조인트 이름 목록 -> articulation 인덱스 텐서."""
        return torch.tensor([self._joint_index[n] for n in names],
                            dtype=torch.long, device=self._device)

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None):
        """내부 상태(추종 기준선·적응 배율 등)를 초기화한다.

        env_ids = 초기화할 env 인덱스 (None = 전체). 3단계 롤아웃처럼 env마다
        에피소드가 다른 시점에 끝나는 사용 형태에서, 부분 리셋이 없으면 이전
        에피소드의 상태가 다음 시도로 새어 GT 점수가 이력에 오염된다.
        """

    @abstractmethod
    def compute(self, obs: ControlObs, goal_xy: torch.Tensor) -> JointTargets:
        """관측과 목표점(월드 (N,2))으로 이번 제어 스텝의 조인트 목표를 만든다."""


# legged form 태그 (RL 정책 제어 — form당 정책 1개)
LEGGED_TAGS = ("quad", "hex", "humanoid")


def make_controller(urdf_path: Path, usd_dir: Path, cfg: dict,
                    joint_names: list[str], default_pose: torch.Tensor,
                    num_envs: int, device: str, physics_dt: float, *,
                    on_terrain: bool,
                    policy_dir: Path | None = None) -> BaseController:
    """control_tag로 제어기 구현을 골라 생성한다 (wheeled + legged 전 form).

    cfg = configs/controller.yaml 전체, physics_dt = 시뮬 물리 스텝 (s,
    제어 주기·슬루 산정용). on_terrain = 지형 롤아웃 여부 (기본값 없는 필수
    인자 — 지형에서는 wheeled 적응형 yaw 보상이 장애물 걸림을 선회 저항으로
    오인해 GT를 오염시키므로 자동 차단한다. config 문서 의존으로는 누락
    위험이 있어 호출측이 명시하게 강제). policy_dir = legged form의 정책
    번들 폴더 ({form}/policy.pt + bundle.json — legged에만 필수).
    전 form 결정적(pure pursuit + 평균 액션 추론)이라 같은 초기 상태·목표면
    같은 주행이 나온다 (GT 재현성).
    """
    # 순환 임포트 방지 — 구현 모듈은 호출 시점에 로드한다
    from ..wheeled.controller import BalancingRobotController, WheeledRobotController

    params = extract_ctrl_params(urdf_path, usd_dir)
    if params.base_tag == "diff_balancing":
        return BalancingRobotController(params, joint_names, default_pose,
                                        num_envs, device, cfg, physics_dt,
                                        urdf_path)
    if params.base_tag in ("diff", "skid", "ackermann", "omni"):
        return WheeledRobotController(params, joint_names, default_pose,
                                      num_envs, device, cfg, physics_dt,
                                      on_terrain=on_terrain)
    if params.base_tag in LEGGED_TAGS:
        from ..legged.controller import LeggedRobotController

        if policy_dir is None:
            raise ValueError(f"legged form({params.base_tag})은 policy_dir가 필요하다"
                             " — train으로 정책을 먼저 학습할 것")
        return LeggedRobotController(params, joint_names, default_pose,
                                     num_envs, device, cfg, physics_dt,
                                     urdf_path, usd_dir, Path(policy_dir))
    raise ValueError(f"지원하지 않는 control_tag: {params.control_tag}")

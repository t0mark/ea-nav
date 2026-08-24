"""URDF -> USD 변환·검증과 로봇 스폰 스펙 구성 (상태 없는 모듈 함수).

Isaac 앱이 기동된 뒤에만 임포트할 수 있다 (tools/utils/sim.py launch_app 참고).

역할 — 로봇을 씬에 올리기까지의 에셋 준비 전부:
1. URDF 가동 조인트 파싱·드라이브 분류 (변환 검증과 스폰이 같은 규칙을 공유)
2. URDF -> USD 변환 (게인 중립 원칙: 게인은 USD에 굽지 않고 스폰 시점에 결정)
3. 변환된 USD 정합성 검사 (가동 조인트 보존·게인 0·articulation root)
4. 스폰 스펙(ArticulationCfg) 구성 — environment.SimEnvironment가 배치에 사용

드라이브 분류 규칙 (configs/sim.yaml drive 섹션과 짝):
- passive  : effort 한계 <= 0 (수동 휠·롤러·스위블 캐스터) -> 게인 0으로 무력화
- velocity : continuous 이면서 effort > 0 (구동 바퀴) -> 속도 드라이브 (감쇠만)
- position : 나머지 revolute/prismatic (다리·팔·조향·리프트) -> 자세 유지 PD

Isaac Sim 5.1 임포터 실측 특성 (검사 기준에 반영):
- merge_fixed_joints=True여도 질량·충돌이 있는 fixed 링크는 병합되지 않고
  PhysicsFixedJoint로 남는다 (강체 부착과 물리 동일 -> 병합/유지 둘 다 허용)
- URDF effort 0은 드라이브 maxForce 무제한으로 임포트된다
  -> 수동성은 드라이브 게인 0으로 보장하고, 검사도 게인 기준으로 한다

경로 규약: {robots_root}/{form}/{이름}/robot.urdf -> {usd_root}/{form}/{이름}/
산출 폴더에는 robot.usd와 함께 meta.json 복사본·joints.json을 남겨
하위 단계(스폰·제어기·롤아웃)가 URDF 재파싱 없이 동작하게 한다.
"""
from __future__ import annotations

import json
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from pxr import Usd, UsdPhysics

logger = logging.getLogger(__name__)

# URDF에서 자유도를 만드는 조인트 타입 (fixed·mimic 등은 제외)
_MOVABLE_TYPES = ("revolute", "continuous", "prismatic")


def _limit_value(limit_elem, key: str) -> float:
    """joint/limit 요소에서 수치 속성을 읽는다. 태그·속성 미기재는 0.0(=수동)."""
    if limit_elem is None:
        return 0.0
    return float(limit_elem.get(key, 0.0))


def parse_joints(urdf_path: Path) -> list[dict]:
    """URDF의 가동 조인트 목록을 파싱한다.

    반환: [{name, type, effort, velocity, lower, upper}].
    단위는 URDF 규약 그대로 (rad·m, Nm·N, rad/s·m/s).
    """
    root = ET.parse(urdf_path).getroot()
    joints = []
    for j in root.findall("joint"):
        jtype = j.get("type")
        if jtype not in _MOVABLE_TYPES:
            continue
        limit = j.find("limit")
        joints.append({
            "name": j.get("name"),
            "type": jtype,
            "effort": _limit_value(limit, "effort"),
            "velocity": _limit_value(limit, "velocity"),
            "lower": _limit_value(limit, "lower"),
            "upper": _limit_value(limit, "upper"),
        })
    return joints


def classify_drive_groups(joints: list[dict]) -> dict[str, list[str]]:
    """가동 조인트를 드라이브 그룹(passive/velocity/position)으로 분류한다.

    입력은 parse_joints 결과, 반환은 {그룹명: 조인트 이름 목록} (모듈 규칙 참고).
    """
    groups = {"passive": [], "velocity": [], "position": []}
    for j in joints:
        if j["effort"] <= 0:
            groups["passive"].append(j["name"])
        elif j["type"] == "continuous":
            groups["velocity"].append(j["name"])
        else:
            groups["position"].append(j["name"])
    return groups


def compute_drive_gains(joints: list[dict], drive_cfg: dict) -> dict[str, dict[str, float]]:
    """드라이브 그룹별 규칙으로 조인트별 (강성, 감쇠)를 계산한다.

    규칙 (configs/sim.yaml drive — 제어기 단계 전까지의 자세 유지 기본값):
    - position : k = pos_stiffness_scale x effort, d = k x pos_damping_ratio
      (자세 오차 1/scale rad에서 토크가 포화하는 수준의 유지 PD)
    - velocity : k = 0, d = vel_damping_scale x effort / velocity 한계
      (전 속도 오차에서 토크 포화 — 속도 드라이브는 감쇠가 이득 역할)
    - passive  : k = d = 0 (게인 0 = 드라이브 무력화로 수동성 보장)
    반환: {"stiffness": {이름: k}, "damping": {이름: d}}.
    """
    groups = classify_drive_groups(joints)
    by_name = {j["name"]: j for j in joints}
    stiffness, damping = {}, {}
    for name in groups["passive"]:
        stiffness[name] = 0.0
        damping[name] = 0.0
    for name in groups["velocity"]:
        j = by_name[name]
        stiffness[name] = 0.0
        damping[name] = drive_cfg["vel_damping_scale"] * j["effort"] / max(j["velocity"], 1e-6)
    for name in groups["position"]:
        j = by_name[name]
        k = drive_cfg["pos_stiffness_scale"] * j["effort"]
        stiffness[name] = k
        damping[name] = k * drive_cfg["pos_damping_ratio"]
    return {"stiffness": stiffness, "damping": damping}


def convert_robot(robot_dir: Path, out_dir: Path, conv_cfg: dict) -> Path:
    """로봇 폴더 1개(robot.urdf + meta.json)를 USD 산출 폴더로 변환한다.

    게인 중립 원칙: joint_drive target "none" + 강성·감쇠 0으로 변환해
    스폰 시점(compute_drive_gains)에만 게인이 부여되게 한다.
    셀프충돌도 같은 원칙 — USD에는 굽지 않고(여기서는 False) 스폰 시점에
    articulation_props로 활성한다 (environment.spawn_robot self_collision).
    산출: out_dir/robot.usd + meta.json 복사본 + joints.json({joints, groups}).
    반환: 생성된 robot.usd 경로.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = UrdfConverterCfg(
        asset_path=str(robot_dir / "robot.urdf"),
        usd_dir=str(out_dir),
        usd_file_name="robot.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        fix_base=False,
        merge_fixed_joints=conv_cfg["merge_fixed_joints"],
        link_density=conv_cfg["link_density"],
        collider_type=conv_cfg["collider_type"],
        replace_cylinders_with_capsules=conv_cfg["replace_cylinders_with_capsules"],
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
    converter = UrdfConverter(cfg)

    # 하위 단계가 URDF 없이 동작하도록 스폰에 필요한 정보를 동봉한다
    # (base_link = 루트 링크 이름 — 로봇마다 다르다: 생성기 base_link / go2 base 등.
    #  높이 스캐너 장착·종료 판정 등 베이스 참조가 이름 하드코딩으로 깨지는 것 방지)
    shutil.copy2(robot_dir / "meta.json", out_dir / "meta.json")
    joints = parse_joints(robot_dir / "robot.urdf")
    root = ET.parse(robot_dir / "robot.urdf").getroot()
    children = {j.find("child").get("link") for j in root.findall("joint")}
    base_link = next(l.get("name") for l in root.findall("link") if l.get("name") not in children)
    with open(out_dir / "joints.json", "w") as f:
        json.dump({"base_link": base_link, "joints": joints,
                   "groups": classify_drive_groups(joints)}, f, indent=1)
    return Path(converter.usd_path)


def inspect_usd(usd_dir: Path, urdf_path: Path) -> dict:
    """변환된 USD를 열어 원본 URDF와 구조를 대조한다.

    검사 항목:
    - 가동 조인트 보존: URDF 가동 조인트 이름이 USD 물리 조인트에 전부 존재
    - 게인 중립: 모든 드라이브의 강성·감쇠 0 (수동성·중립 게인 보장)
    - articulation root 존재
    참고 기록(판정 미반영): 강체·fixed 조인트 수, extra_joints(URDF에 없는
    가동 조인트 — 임포터가 조인트를 만들어내는 이상 상황 관찰용).
    반환: {"pass": bool, ...세부 항목}.
    """
    stage = Usd.Stage.Open(str(Path(usd_dir) / "robot.usd"))
    urdf_joints = {j["name"] for j in parse_joints(urdf_path)}

    usd_movable, fixed_count, rigid_count = set(), 0, 0
    nonzero_gains, has_root = [], False
    for prim in stage.Traverse():
        # 물리 조인트 프림 집계 (revolute/prismatic = 가동, fixed = 병합 잔존)
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            usd_movable.add(prim.GetName())
        elif prim.IsA(UsdPhysics.FixedJoint):
            fixed_count += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_count += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            has_root = True
        # 게인 중립 검사 — 변환기가 남긴 드라이브의 강성·감쇠가 모두 0이어야 함
        for drive_kind in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, drive_kind)
            if not drive:
                continue
            k = drive.GetStiffnessAttr().Get() or 0.0
            d = drive.GetDampingAttr().Get() or 0.0
            if k != 0.0 or d != 0.0:
                nonzero_gains.append(prim.GetName())

    missing = sorted(urdf_joints - usd_movable)
    extra = sorted(usd_movable - urdf_joints)
    return {
        "pass": not missing and not nonzero_gains and has_root,
        "missing_joints": missing,
        "extra_joints": extra,
        "nonzero_gain_joints": nonzero_gains,
        "articulation_root": has_root,
        "movable_joints": len(usd_movable),
        "fixed_joints_kept": fixed_count,
        "rigid_bodies": rigid_count,
    }


def _dcmotor_actuators(joints: list[dict], gains: dict,
                       armature: float) -> dict:
    """조인트를 토크 한계값별 그룹으로 묶어 DCMotor 액추에이터를 만든다.

    DCMotor = 속도-토크 포화 모델 (스톡 Unitree 계열 표준): 관절 속도가
    한계에 가까울수록 가용 토크가 줄어 고속 디더링의 제동력이 소멸한다 —
    implicit PD(무포화)는 이 순응 동역학이 없어 전도가 구조적으로 희소
    (기립 고착 병리의 동역학 축. code.md 스톡 클론 판정). saturation_effort
    가 스칼라 필드라 같은 토크 한계끼리 그룹을 나눈다 (URDF effort 사용).
    """
    by_effort: dict[float, list[str]] = {}
    for j in joints:
        by_effort.setdefault(round(float(j["effort"]), 3), []).append(j["name"])
    by_name = {j["name"]: j for j in joints}
    actuators = {}
    for i, (effort, names) in enumerate(sorted(by_effort.items())):
        actuators[f"dc{i}"] = DCMotorCfg(
            joint_names_expr=[f"^{n}$" for n in names],
            stiffness={n: gains["stiffness"][n] for n in names},
            damping={n: gains["damping"][n] for n in names},
            effort_limit=effort,
            saturation_effort=effort,
            velocity_limit={n: max(float(by_name[n]["velocity"]), 1e-3)
                            for n in names},
            armature=armature,
        )
    return actuators


def make_articulation_cfg(usd_dir: Path, drive_cfg: dict, prim_path: str,
                          spawn_margin: float,
                          gain_overrides: dict | None = None,
                          actuator_model: str = "implicit") \
        -> tuple[ArticulationCfg, dict]:
    """USD 산출 폴더에서 스폰 스펙(ArticulationCfg)과 meta를 구성한다.

    - 초기 상태: meta의 기립 자세(standing_pose) + 스폰 높이(base_height + 여유 낙하)
    - 액추에이터: 게인은 drive 규칙으로 조인트별 결정. actuator_model
      "implicit" = PhysX PD (wheeled·홀드 기본), "dcmotor" = 속도-토크
      포화 모델 (legged 학습·평가 — _dcmotor_actuators 주석)
    - gain_overrides: {조인트 이름: (강성, 감쇠)} — 규칙 결과의 개별 덮어쓰기
      (balancing 바퀴를 effort 직접 제어로 쓰기 위한 게인 0 등)
    - spawn=None: 프림 배치는 environment 쪽(클로너)이 수행하고 여기서는 상태만 정의
    반환: (ArticulationCfg, meta dict).
    """
    usd_dir = Path(usd_dir)
    with open(usd_dir / "meta.json") as f:
        meta = json.load(f)
    with open(usd_dir / "joints.json") as f:
        joints = json.load(f)["joints"]

    gains = compute_drive_gains(joints, drive_cfg)
    for name, (stiffness, damping) in (gain_overrides or {}).items():
        gains["stiffness"][name] = stiffness
        gains["damping"][name] = damping
    if actuator_model == "dcmotor":
        actuators = _dcmotor_actuators(joints, gains, drive_cfg["armature"])
    else:
        actuators = {
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=gains["stiffness"],
                damping=gains["damping"],
                armature=drive_cfg["armature"],
            ),
        }
    cfg = ArticulationCfg(
        prim_path=prim_path,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, meta["metrics"]["base_height"] + spawn_margin),
            joint_pos=dict(meta["standing_pose"]),
        ),
        actuators=actuators,
    )
    return cfg, meta


def iter_robot_dirs(root: Path) -> list[Path]:
    """{root}/{form}/{이름}/ 2단 구조의 로봇 폴더를 정렬 순서로 나열한다."""
    dirs = []
    for form_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        dirs += sorted(p for p in form_dir.iterdir() if p.is_dir())
    return dirs

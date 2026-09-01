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

_MOVABLE_TYPES = ("revolute", "continuous", "prismatic")

def _limit_value(limit_elem, key: str) -> float:

    if limit_elem is None:
        return 0.0
    return float(limit_elem.get(key, 0.0))

def parse_joints(urdf_path: Path) -> list[dict]:

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

    stage = Usd.Stage.Open(str(Path(usd_dir) / "robot.usd"))
    urdf_joints = {j["name"] for j in parse_joints(urdf_path)}

    usd_movable, fixed_count, rigid_count = set(), 0, 0
    nonzero_gains, has_root = [], False
    for prim in stage.Traverse():

        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            usd_movable.add(prim.GetName())
        elif prim.IsA(UsdPhysics.FixedJoint):
            fixed_count += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_count += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            has_root = True

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
        "pass": not missing and not extra and not nonzero_gains and has_root,
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
                          actuator_model: str = "implicit")        -> tuple[ArticulationCfg, dict]:

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

def standard_friction_links(contact_links: list[str]) -> list[str]:

    return [n for n in contact_links
            if not n.startswith("ball_") and "_roller_" not in n]

def iter_robot_dirs(root: Path) -> list[Path]:

    dirs = []
    for form_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        dirs += sorted(p for p in form_dir.iterdir() if p.is_dir())
    return dirs

def convert_batch(robot_dirs: list[Path], robots_root: Path, usd_root: Path,
                  cfg: dict) -> dict:

    result = {}
    logger.info("변환 시작: %d대 (%s -> %s)", len(robot_dirs), robots_root, usd_root)
    for i, robot_dir in enumerate(robot_dirs):
        rel = str(robot_dir.relative_to(robots_root))
        try:
            convert_robot(robot_dir, usd_root / rel, cfg["converter"])
            result[rel] = {"ok": True}
        except Exception as e:
            result[rel] = {"ok": False, "error": str(e)}
        logger.info("[%d/%d] 변환 %s %s", i + 1, len(robot_dirs), rel,
                    "OK" if result[rel]["ok"] else "FAIL")
    return result

def validate_batch(robot_dirs: list[Path], robots_root: Path, usd_root: Path,
                   convert_report: dict) -> dict:

    result = {}
    for robot_dir in robot_dirs:
        rel = str(robot_dir.relative_to(robots_root))
        if not convert_report[rel]["ok"]:
            continue
        try:
            r = inspect_usd(usd_root / rel, robot_dir / "robot.urdf")
        except Exception as e:
            r = {"pass": False, "error": str(e)}
            logger.error("검증 예외 %s: %s", rel, e)
        result[rel] = r
        if "error" not in r:
            logger.info("검증 %s %s (가동 %d, 추가 %d, fixed 유지 %d, 강체 %d)", rel,
                        "OK" if r["pass"] else "FAIL", r["movable_joints"],
                        len(r["extra_joints"]), r["fixed_joints_kept"],
                        r["rigid_bodies"])
    return result

def spawn_with_standard_friction(env, usd_dir: Path, meta: dict, drive_cfg: dict,
                                 contact_cfg: dict, friction_links: list[str] | None = None,
                                 **kwargs):

    links = friction_links if friction_links is not None        else standard_friction_links(meta["contact_links"])
    return env.spawn_robot(usd_dir, drive_cfg, contact_cfg=contact_cfg,
                           friction_links=links,
                           friction_links_mu=float(contact_cfg["foot_friction"]),
                           **kwargs)

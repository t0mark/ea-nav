from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from .loader import PosedModel

def validate_set(robots_root, cfg: dict) -> tuple[int, int]:

    log = logging.getLogger("urdf.validate")
    n_ok = n_fail = 0
    for meta_path in sorted(Path(robots_root).glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        report = validate_static(meta_path.parent / "robot.urdf",
                                 meta["standing_pose"], meta["contact_links"], cfg,
                                 check_poses=None, special=meta.get("special_checks"))
        if report["passed"]:
            n_ok += 1
        else:
            n_fail += 1
            fails = [k for k, v in report["checks"].items() if not v]
            log.warning("%s 실패: %s", meta["name"], fails)
    log.info("재검증 결과: 통과 %d / 실패 %d", n_ok, n_fail)
    return n_ok, n_fail

def validate_static(urdf_path, standing_pose: dict, contact_links: list[str], cfg: dict,
                    check_poses: list[dict] | None = None, special: dict | None = None) -> dict:

    model = PosedModel(urdf_path, standing_pose)
    special = special or {}
    checks, metrics = {}, {}

    z0 = _contact_floor(model, contact_links)
    metrics["base_height"] = float(-z0)

    com, total_mass = _total_com(model)
    metrics["com"] = [float(v) for v in com]
    metrics["total_mass"] = total_mass

    checks["clearance"] = _check_clearance(model, contact_links, z0, cfg, metrics)
    checks["self_collision"] = _check_self_collision(model, metrics)
    checks["coplanar_contact"] = _check_coplanar(model, contact_links, z0, cfg, metrics)
    checks["gravity_torque"] = _check_torque(model, cfg, metrics)
    checks["stance_torque"] = _check_stance_torque(model, contact_links, total_mass, cfg, metrics)
    _measure_bbox(model, z0, metrics)

    checks["com_support"] = _check_com_polygon(model, com, contact_links, z0, cfg, metrics)

    if "slope_static" in special:
        checks["slope_static"] = _check_slope_static(model, com, contact_links, z0, cfg, metrics)

    if "wheeled_geometry" in special:
        checks["wheeled_geometry"] = _check_wheeled_geometry(special["wheeled_geometry"], cfg, metrics)

    if "load_share" in special:
        checks["load_share"] = _check_load_share(com, special["load_share"], cfg, metrics)

    if "overturn" in special and "com_margin" in metrics:
        checks["overturn"] = _check_overturn(com, z0, cfg, metrics)

    if check_poses:
        checks["extra_poses"] = _check_extra_poses(
            urdf_path, standing_pose, check_poses, contact_links, cfg, special, metrics)

    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}

def _contact_floor(model: PosedModel, contact_links) -> float:

    zs = [m.vertices[:, 2].min() for l in model.links if l.name in contact_links for m in l.meshes]
    if not zs:
        raise ValueError("contact link geometry not found")
    return float(min(zs))

def _total_com(model: PosedModel) -> tuple[np.ndarray, float]:

    masses = np.array([l.mass for l in model.links if l.com_world is not None])
    coms = np.array([l.com_world for l in model.links if l.com_world is not None])
    com = (masses[:, None] * coms).sum(axis=0) / masses.sum()
    return com, float(masses.sum())

def _check_clearance(model, contact_links, z0, cfg, metrics) -> bool:

    zs = [m.vertices[:, 2].min() for l in model.links
          if l.name not in contact_links and l.meshes for m in l.meshes]
    clearance = float(min(zs) - z0) if zs else float("inf")
    metrics["clearance"] = clearance
    return clearance >= cfg["clearance_min"]

def _check_self_collision(model, metrics, key: str = "self_collision_pairs") -> bool:

    manager = trimesh.collision.CollisionManager()
    for link in model.links:
        if link.meshes:
            manager.add_object(link.name, trimesh.util.concatenate(link.meshes))
    _, pairs = manager.in_collision_internal(return_names=True)

    allowed = model.adjacency(depth=1) | model.sibling_pairs_nonbase()
    bad = [tuple(sorted(p)) for p in pairs if frozenset(p) not in allowed]
    metrics[key] = bad
    return len(bad) == 0

def _check_coplanar(model, contact_links, z0, cfg, metrics) -> bool:

    lows: dict[str, float] = {}
    for l in model.links:
        if l.name not in contact_links or not l.meshes:
            continue
        unit = l.name.split("_roller_")[0] if "_roller_" in l.name else l.name
        z = min(m.vertices[:, 2].min() for m in l.meshes)
        lows[unit] = min(lows.get(unit, float("inf")), z)

    spread = float(max(lows.values()) - min(lows.values())) if lows else 0.0
    metrics["contact_spread"] = spread
    return spread <= cfg["contact_coplanar_tol"]

def _check_com_polygon(model, com, contact_links, z0, cfg, metrics) -> bool:

    pts = _support_points(model, contact_links, z0, cfg)

    try:
        hull = ConvexHull(pts)
    except Exception:
        metrics["com_margin"] = -1.0
        return False

    margin = float(-(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]).max())
    metrics["com_margin"] = margin
    return margin >= cfg["com_margin_min"]

def _support_points(model, contact_links, z0, cfg) -> np.ndarray:

    """접지 링크의 바닥 근처 vertex를 모아 support polygon 후보점을 만든다."""

    pts = []
    for l in model.links:

        if l.name not in contact_links:
            continue

        for m in l.meshes:

            v = m.vertices
            band = v[v[:, 2] < z0 + cfg["contact_band"], :2]
            if len(band):
                pts.append(band)
    if not pts:
        return np.zeros((0, 2))
    return np.vstack(pts)

def _support_margin(pts: np.ndarray, point_xy: np.ndarray) -> float:

    """ConvexHull 반공간 식으로 점에서 support polygon 경계까지의 signed margin을 계산한다."""

    hull = ConvexHull(pts)
    return float(-(hull.equations[:, :2] @ point_xy + hull.equations[:, 2]).max())

def _check_slope_static(model, com, contact_links, z0, cfg, metrics) -> bool:

    """경사면을 중력 방향 변화로 근사해 네 방향 준정적 안정성을 검사한다."""

    pts = _support_points(model, contact_links, z0, cfg)
    theta = float(cfg.get("slope_static_angle", 0.0))
    height = float(com[2] - z0)
    directions = [np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
                  np.array([0.0, 1.0]), np.array([0.0, -1.0])]

    margins = []
    try:
        for direction in directions:

            # 기울어진 중력선과 접지 평면의 교점을 support polygon 안에서 검사한다.
            projected = com[:2] + height * math.tan(theta) * direction
            margins.append(_support_margin(pts, projected))
    except Exception:
        metrics["slope_margin_min"] = -1.0
        return False

    metrics["slope_margin_min"] = float(min(margins))
    return metrics["slope_margin_min"] >= float(cfg.get("slope_margin_min", 0.0))

def _check_torque(model, cfg, metrics) -> bool:

    children: dict[str, list] = {}
    for j in model.joints:
        children.setdefault(j.parent, []).append(j.child)
    gravity = np.array([0.0, 0.0, -9.81])

    worst = float("inf")
    worst_joint = ""
    ok = True
    for j in model.joints:

        if j.type in (None, "fixed") or j.limit is None or not j.limit.effort:
            continue

        subtree = _collect_subtree(j.child, children)
        T = model.link_transform(j.child)
        p = T[:3, 3]
        axis = T[:3, :3] @ np.asarray(j.axis if j.axis is not None else [0, 0, 1], dtype=float)

        load_vec = np.zeros(3)
        for l in model.links:
            if l.name in subtree and l.com_world is not None:
                if j.type == "prismatic":
                    load_vec += l.mass * gravity
                else:
                    load_vec += np.cross(l.com_world - p, l.mass * gravity)
        load = abs(float(load_vec @ axis))

        ratio = j.limit.effort / max(load, 1e-9)
        if ratio < worst:
            worst, worst_joint = ratio, j.name
        if load * cfg["torque_margin"] > j.limit.effort:
            ok = False
    metrics["torque_margin_min"] = None if worst == float("inf") else worst
    metrics["torque_worst_joint"] = worst_joint
    return ok

def _check_stance_torque(model, contact_links, total_mass, cfg, metrics) -> bool:

    parent_joint = {j.child: j for j in model.joints}
    legs = []
    for name in contact_links:
        chain, node, is_leg = [], name, False
        while node in parent_joint:
            j = parent_joint[node]
            if j.type == "continuous":
                is_leg = False
                break
            if j.type == "revolute":
                chain.append(j)
                is_leg = True
            node = j.parent
        if is_leg and chain:
            legs.append((name, chain))
    if not legs:
        return True

    force = total_mass * 9.81 / max(len(legs) // 2, 1)
    f_vec = np.array([0.0, 0.0, force])
    worst, worst_joint, ok = float("inf"), "", True
    for name, chain in legs:

        verts = np.vstack([m.vertices for l in model.links if l.name == name for m in l.meshes])
        band = verts[verts[:, 2] < verts[:, 2].min() + cfg["contact_band"]]
        p_c = band.mean(axis=0)
        for j in chain:

            T = model.link_transform(j.child)
            p = T[:3, 3]
            axis = T[:3, :3] @ np.asarray(j.axis if j.axis is not None else [0, 0, 1], dtype=float)
            tau = abs(float(np.cross(p_c - p, f_vec) @ axis))
            limit = j.limit.effort if j.limit is not None and j.limit.effort else 0.0
            ratio = limit / max(tau, 1e-9)
            if ratio < worst:
                worst, worst_joint = ratio, j.name
            if tau * cfg["stance_torque_margin"] > limit:
                ok = False
    metrics["stance_margin_min"] = None if worst == float("inf") else worst
    metrics["stance_worst_joint"] = worst_joint
    return ok

def _collect_subtree(root, children) -> set:

    out, stack = {root}, [root]
    while stack:
        for c in children.get(stack.pop(), ()):
            out.add(c)
            stack.append(c)
    return out

def _measure_bbox(model, z0, metrics):

    vs = np.vstack([m.vertices for l in model.links for m in l.meshes])
    lo, hi = vs.min(axis=0), vs.max(axis=0)
    metrics["overall_length"] = float(hi[0] - lo[0])
    metrics["overall_width"] = float(hi[1] - lo[1])
    metrics["overall_height"] = float(hi[2] - z0)

def _check_load_share(com, params, cfg, metrics) -> bool:

    span = params["other_x"] - params["drive_x"]
    share = float((params["other_x"] - com[0]) / span) if abs(span) > 1e-6 else 1.0
    metrics["drive_load_share"] = share
    return cfg["load_share_min"] <= share <= float(cfg.get("load_share_max", 1.0))

def _check_overturn(com, z0, cfg, metrics) -> bool:

    ratio = float((com[2] - z0) / max(metrics["com_margin"], 1e-6))
    tip_accel = 9.81 / max(ratio, 1e-6)
    metrics["overturn_ratio"] = ratio
    metrics["tip_accel"] = tip_accel
    return ratio <= cfg["overturn_max"] and tip_accel >= float(cfg.get("tip_accel_min", 0.0))

def _check_wheeled_geometry(params, cfg, metrics) -> bool:

    """wheeled 로봇의 타입별 형상 비율이 제어 가능한 범위에 있는지 검사한다."""

    rules = cfg.get("wheeled", {})
    base_type = str(params.get("base_type", ""))
    length = float(params.get("body_length", 0.0))
    width = float(params.get("body_width", 0.0))
    height = float(params.get("body_height", 0.0))
    radius = float(params.get("wheel_radius", 0.0))
    clearance = float(params.get("ground_clearance", 0.0))
    wheelbase = float(params.get("wheelbase", 0.0))
    track_values = [float(params[k]) for k in ("track_width", "track_front", "track_rear")
                    if k in params and float(params[k]) > 0.0]
    track = min(track_values) if track_values else 0.0

    checks = {
        "track_body_width": _ratio(track, width) >= float(rules.get("track_body_width_min", 0.0)),
        "height_track": _ratio(height, track) <= float(rules.get("body_height_track_max", 1.5)),
        "wheel_radius_body_height": _ratio(radius, height) >= float(rules.get("wheel_radius_body_height_min", 0.0)),
        "clearance_wheel_radius": (
            float(rules.get("clearance_wheel_radius_min", 0.0))
            <= _ratio(clearance, radius)
            <= float(rules.get("clearance_wheel_radius_max", float("inf")))),
    }

    if base_type == "diff":
        checks["diff_wheelbase_body_length"] = (
            _ratio(wheelbase, length) >= float(rules.get("diff_wheelbase_body_length_min", 0.0)))
    elif base_type == "skid":
        checks["skid_wheelbase_body_length"] = (
            _ratio(wheelbase, length) >= float(rules.get("skid_wheelbase_body_length_min", 0.0)))
        checks["skid_wheelbase_track"] = (
            _ratio(wheelbase, track) <= float(rules.get("skid_wheelbase_track_max", float("inf"))))
    elif base_type == "ackermann":
        checks["ackermann_wheelbase_body_length"] = (
            _ratio(wheelbase, length) >= float(rules.get("ackermann_wheelbase_body_length_min", 0.0)))
        checks["ackermann_turn_radius_body_length"] = (
            _ratio(float(params.get("min_turn_radius", 0.0)), length)
            <= float(rules.get("ackermann_turn_radius_body_length_max", float("inf"))))
    elif base_type == "omni":
        if wheelbase > 0.0:
            checks["omni_wheelbase_body_length"] = (
                _ratio(wheelbase, length) >= float(rules.get("omni_wheelbase_body_length_min", 0.0)))
        if "ring_radius" in params:
            checks["omni_ring_body_width"] = (
                _ratio(float(params["ring_radius"]), width) >= float(rules.get("omni_ring_body_width_min", 0.0)))
        checks["omni_roller_count"] = (
            int(params.get("n_rollers_per_wheel", 0)) >= int(rules.get("omni_roller_count_min", 0)))

    metrics["wheeled_geometry"] = {k: bool(v) for k, v in checks.items()}
    metrics["track_body_width_ratio"] = _ratio(track, width)
    metrics["body_height_track_ratio"] = _ratio(height, track)
    metrics["wheel_radius_body_height_ratio"] = _ratio(radius, height)
    metrics["clearance_wheel_radius_ratio"] = _ratio(clearance, radius)
    return all(checks.values())

def _ratio(num: float, den: float) -> float:

    """0 또는 누락된 치수 때문에 비율 검사가 통과하지 않도록 안전 비율을 만든다."""

    if den <= 1e-9:
        return float("inf")
    return num / den

def _check_extra_poses(urdf_path, standing_pose, check_poses, contact_links,
                       cfg, special, metrics) -> bool:

    ok = True
    bad_all = []
    for i, pose in enumerate(check_poses):
        posed = PosedModel(urdf_path, {**standing_pose, **pose})
        sub: dict = {}

        if not _check_self_collision(posed, sub, key="pairs"):
            ok = False
            bad_all.append({"pose_index": i, "pairs": sub["pairs"]})

        z0p = _contact_floor(posed, contact_links)
        com_p, _ = _total_com(posed)
        if not _check_com_polygon(posed, com_p, contact_links, z0p, cfg, sub):
            ok = False
            bad_all.append({"pose_index": i, "com_margin": sub["com_margin"]})
        elif "overturn" in special and not _check_overturn(com_p, z0p, cfg, sub):
            ok = False
            bad_all.append({"pose_index": i, "overturn": sub["overturn_ratio"]})
    metrics["extra_pose_fails"] = bad_all
    return ok

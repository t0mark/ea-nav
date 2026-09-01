from __future__ import annotations

import json
import logging
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

    pts = []
    for l in model.links:
        if l.name not in contact_links:
            continue
        for m in l.meshes:
            v = m.vertices
            pts.append(v[v[:, 2] < z0 + cfg["contact_band"], :2])
    pts = np.vstack([p for p in pts if len(p)])

    try:
        hull = ConvexHull(pts)
    except Exception:
        metrics["com_margin"] = -1.0
        return False

    margin = float(-(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]).max())
    metrics["com_margin"] = margin
    return margin >= cfg["com_margin_min"]

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
    return share >= cfg["load_share_min"]

def _check_overturn(com, z0, cfg, metrics) -> bool:

    ratio = float((com[2] - z0) / max(metrics["com_margin"], 1e-6))
    metrics["overturn_ratio"] = ratio
    return ratio <= cfg["overturn_max"]

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

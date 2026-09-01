from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_MOVABLE = {"revolute", "continuous", "prismatic"}

def parse_graph(urdf_path: str | Path, meta_path: str | Path | None = None,
                summarize_rollers: bool = True,
                merge_fixed: bool = True) -> dict:

    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    raw_links = {elem.get("name"): _read_link(elem) for elem in root.findall("link")}
    raw_joints = [_read_joint(elem) for elem in root.findall("joint")]
    meta = json.loads(Path(meta_path).read_text()) if meta_path is not None else {}

    parent = _disjoint(raw_links)
    if merge_fixed:
        for joint in raw_joints:
            if joint["type"] == "fixed" or joint["mimic"]:
                parent[_find(parent, joint["child"])] = _find(parent, joint["parent"])

    nodes = {}
    for name, link in raw_links.items():
        rep = _find(parent, name)
        node = nodes.setdefault(rep, _empty_node(rep))
        _merge_link(node, link, merged=(name != rep))

    if summarize_rollers:
        _fold_rollers(nodes, raw_links, raw_joints, parent)

    edges = []
    removed = set()
    for node in nodes.values():
        removed.update(node.pop("_removed_links", []))
    for joint in raw_joints:
        p = _find(parent, joint["parent"])
        c = _find(parent, joint["child"])
        if p == c or c in removed or joint["child"] in removed:
            continue
        if joint["type"] not in _MOVABLE:
            continue
        edges.append({
            "name": joint["name"],
            "parent": p,
            "child": c,
            "type": joint["type"],
            "axis": joint["axis"],
            "origin_xyz": joint["origin_xyz"],
            "origin_rpy": joint["origin_rpy"],
            "limit": joint["limit"],
        })

    graph_nodes = [node for name, node in sorted(nodes.items())
                   if name not in removed]
    base = _base_link(raw_links, raw_joints)
    for node in graph_nodes:
        node["is_base_link"] = node["name"] == _find(parent, base)

    return {
        "name": root.get("name"),
        "source_urdf": str(urdf_path),
        "form": meta.get("form"),
        "control_tag": meta.get("control_tag"),
        "nodes": graph_nodes,
        "edges": sorted(edges, key=lambda e: (e["parent"], e["child"], e["name"])),
        "stats": {
            "raw_links": len(raw_links),
            "raw_movable_joints": sum(j["type"] in _MOVABLE for j in raw_joints),
            "graph_nodes": len(graph_nodes),
            "graph_edges": len(edges),
            "roller_links_folded": sum(
                n["roller_summary"]["count"] for n in graph_nodes),
            "fixed_links_merged": sum(
                len(n["merged_fixed_children"]) for n in graph_nodes),
        },
    }

def _disjoint(raw_links: dict) -> dict[str, str]:

    return {name: name for name in raw_links}

def _find(parent: dict[str, str], name: str) -> str:

    while parent[name] != name:
        parent[name] = parent[parent[name]]
        name = parent[name]
    return name

def _base_link(raw_links: dict, joints: list[dict]) -> str:

    children = {joint["child"] for joint in joints}
    return next(name for name in raw_links if name not in children)

def _empty_node(name: str) -> dict:

    return {
        "name": name,
        "mass": 0.0,
        "collision_geoms": [],
        "merged_fixed_children": [],
        "roller_summary": {
            "count": 0,
            "radius_mean": 0.0,
            "length_mean": 0.0,
            "tilt_abs_mean": 0.0,
        },
        "_removed_links": [],
    }

def _merge_link(node: dict, link: dict, merged: bool):

    node["mass"] += link["mass"]
    node["collision_geoms"].extend(link["collision_geoms"])
    if merged:
        node["merged_fixed_children"].append(link["name"])

def _fold_rollers(nodes: dict, raw_links: dict, joints: list[dict],
                  parent: dict[str, str]):

    by_parent: dict[str, list[tuple[dict, dict]]] = {}
    for joint in joints:
        child = joint["child"]
        if "_roller_" not in child:
            continue
        wheel = _find(parent, joint["parent"])
        by_parent.setdefault(wheel, []).append((joint, raw_links[child]))

    for wheel, rollers in by_parent.items():
        if wheel not in nodes:
            continue
        radii, lengths, tilts = [], [], []
        for joint, link in rollers:
            cyl = next((g for g in link["collision_geoms"]
                        if g["type"] == "cylinder"), None)
            if cyl is not None:
                radii.append(cyl["size"][0])
                lengths.append(cyl["size"][1])
            tilts.append(_axis_tilt_abs(joint["origin_rpy"]))
            nodes[wheel]["_removed_links"].append(link["name"])
        nodes[wheel]["roller_summary"] = {
            "count": len(rollers),
            "radius_mean": _mean(radii),
            "length_mean": _mean(lengths),
            "tilt_abs_mean": _mean(tilts),
        }

def _axis_tilt_abs(rpy: list[float]) -> float:

    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rot = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    z_axis = rot[:, 2]
    return float(math.asin(min(1.0, abs(z_axis[1]))))

def _mean(values: list[float]) -> float:

    return float(sum(values) / len(values)) if values else 0.0

def _read_link(elem: ET.Element) -> dict:

    inertial = elem.find("inertial")
    mass = 0.0
    if inertial is not None and inertial.find("mass") is not None:
        mass = float(inertial.find("mass").get("value", 0.0))
    return {
        "name": elem.get("name"),
        "mass": mass,
        "collision_geoms": [_read_collision(c) for c in elem.findall("collision")
                            if _read_collision(c) is not None],
    }

def _read_collision(elem: ET.Element) -> dict | None:

    origin = elem.find("origin")
    geom = elem.find("geometry")
    if geom is None:
        return None
    if geom.find("box") is not None:
        return {"type": "box",
                "size": _float_list(geom.find("box").get("size"), 3),
                "origin_xyz": _float_list(origin.get("xyz") if origin is not None else None, 3),
                "origin_rpy": _float_list(origin.get("rpy") if origin is not None else None, 3)}
    if geom.find("cylinder") is not None:
        cyl = geom.find("cylinder")
        return {"type": "cylinder",
                "size": [float(cyl.get("radius")), float(cyl.get("length")), 0.0],
                "origin_xyz": _float_list(origin.get("xyz") if origin is not None else None, 3),
                "origin_rpy": _float_list(origin.get("rpy") if origin is not None else None, 3)}
    if geom.find("sphere") is not None:
        return {"type": "sphere",
                "size": [float(geom.find("sphere").get("radius")), 0.0, 0.0],
                "origin_xyz": _float_list(origin.get("xyz") if origin is not None else None, 3),
                "origin_rpy": _float_list(origin.get("rpy") if origin is not None else None, 3)}
    return None

def _read_joint(elem: ET.Element) -> dict:

    origin = elem.find("origin")
    axis = elem.find("axis")
    limit = elem.find("limit")
    return {
        "name": elem.get("name"),
        "type": elem.get("type"),
        "parent": elem.find("parent").get("link"),
        "child": elem.find("child").get("link"),
        "mimic": elem.find("mimic") is not None,
        "axis": _float_list(axis.get("xyz") if axis is not None else None, 3,
                            default="0 0 1"),
        "origin_xyz": _float_list(origin.get("xyz") if origin is not None else None, 3),
        "origin_rpy": _float_list(origin.get("rpy") if origin is not None else None, 3),
        "limit": {
            "lower": float(limit.get("lower", 0.0)) if limit is not None else 0.0,
            "upper": float(limit.get("upper", 0.0)) if limit is not None else 0.0,
            "effort": float(limit.get("effort", 0.0)) if limit is not None else 0.0,
            "velocity": float(limit.get("velocity", 0.0)) if limit is not None else 0.0,
        },
    }

def _float_list(text: str | None, n: int, default: str = "0 0 0") -> list[float]:

    vals = [float(v) for v in (text or default).split()]
    return (vals + [0.0] * n)[:n]

from __future__ import annotations

import importlib
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import trimesh

from .utils.loader import PosedModel

def index_mesh_files(pool_root: str | Path) -> dict:

    idx: dict = {}
    for p in Path(pool_root).rglob("*"):
        if p.suffix.lower() in (".stl", ".dae", ".obj") and p.name.lower() not in idx:
            idx[p.name.lower()] = p
    return idx

def mesh_local_aabb(path: Path, scale) -> tuple[np.ndarray, np.ndarray] | None:

    try:
        mesh = trimesh.load(str(path), force="mesh")
        lo, hi = mesh.bounds
    except Exception:
        return None

    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        return None

    s = np.ones(3) if scale is None else np.asarray(scale, dtype=float).reshape(-1)
    if s.size == 1:
        s = np.repeat(s, 3)
    return lo * s, hi * s

class RealRobotCollector:

    _RD_ROBOTS = {
        "multileg": [
            "a1_description", "aliengo_description", "anymal_b_description",
            "anymal_c_description", "anymal_d_description", "b1_description",
            "b2_description", "go1_description", "go2_description", "hyq_description",
            "laikago_description", "mini_cheetah_description", "solo_description",
        ],
        "humanoid": [
            "g1_description", "h1_description", "atlas_drc_description",
            "atlas_v4_description", "talos_description", "jvrc_description",
            "icub_description", "valkyrie_description", "jaxon_description",
            "sigmaban_description", "berkeley_humanoid_description",
            "draco3_description", "ergocub_description",
        ],
        "wheeled": [
            "upkie_description", "stretch_description",
            "fetch_description", "pr2_description", "tiago_description",
        ],
    }

    _REPO_ROBOTS = [
        {"name": "turtlebot3_burger", "family": "wheeled",
         "repo": "https://github.com/ROBOTIS-GIT/turtlebot3.git", "branch": "humble",
         "entry": "turtlebot3_description/urdf/turtlebot3_burger.urdf"},
        {"name": "turtlebot3_waffle", "family": "wheeled",
         "repo": "https://github.com/ROBOTIS-GIT/turtlebot3.git", "branch": "humble",
         "entry": "turtlebot3_description/urdf/turtlebot3_waffle.urdf"},
        {"name": "husky", "family": "wheeled",
         "repo": "https://github.com/husky/husky.git", "branch": "noetic-devel",
         "entry": "husky_description/urdf/husky.urdf.xacro"},
        {"name": "jackal", "family": "wheeled",
         "repo": "https://github.com/jackal/jackal.git", "branch": "noetic-devel",
         "entry": "jackal_description/urdf/jackal.urdf.xacro"},
        {"name": "dingo", "family": "wheeled",
         "repo": "https://github.com/dingo-cpr/dingo.git", "branch": "melodic-devel",
         "entry": "dingo_description/urdf/dingo-d.urdf.xacro"},
        {"name": "spot", "family": "multileg",
         "repo": "https://github.com/clearpathrobotics/spot_ros.git", "branch": "master",
         "entry": "spot_description/urdf/spot.urdf.xacro"},

        {"name": "ridgeback", "family": "wheeled",
         "repo": "https://github.com/ridgeback/ridgeback.git", "branch": "melodic-devel",
         "entry": "ridgeback_description/urdf/ridgeback.urdf.xacro"},
        {"name": "youbot", "family": "wheeled",
         "repo": "https://github.com/youbot/youbot_description.git", "branch": "indigo-devel",
         "entry": "robots/youbot.urdf.xacro"},

        {"name": "phantomx", "family": "multileg",
         "repo": "https://github.com/HumaRobotics/phantomx_description.git", "branch": "master",
         "entry": "urdf/phantomx.urdf"},

        {"name": "racecar", "family": "wheeled",
         "repo": "https://github.com/f1tenth/f1tenth_simulator.git", "branch": "master",
         "entry": "racecar.xacro"},
    ]

    _MASS_HINTS = {"multileg/spot": 32.5}

    def __init__(self, out_root: str | Path):

        self._out = Path(out_root)
        self._log = logging.getLogger("urdf.real_pool")

    def collect(self) -> dict:

        results = {"ok": [], "failed": []}
        for family, names in self._RD_ROBOTS.items():
            for name in names:
                self._try(results, family, name.replace("_description", ""),
                          lambda: self._collect_rd(family, name))
        for item in self._REPO_ROBOTS:
            self._try(results, item["family"], item["name"],
                      lambda item=item: self._collect_repo(item))

        self._repair_inertials()

        self._log.info("수집 완료: 성공 %d / 실패 %d", len(results["ok"]), len(results["failed"]))
        with open(self._out / "collection_log.json", "w") as f:
            json.dump(results, f, indent=1)
        return results

    def _repair_inertials(self):

        import xml.etree.ElementTree as ET
        idx = index_mesh_files(self._out)
        for key, total in self._MASS_HINTS.items():
            urdf = self._out / key / "robot.urdf"
            if not urdf.exists():
                continue
            tree = ET.parse(urdf)
            root = tree.getroot()

            if sum(float(m.get("value", 0)) for m in root.iter("mass")) > 1e-6:
                continue

            boxes = {}
            for link in root.iter("link"):
                lo_all = hi_all = None
                for col in link.findall("collision"):
                    mesh = col.find("./geometry/mesh")
                    hit = idx.get(Path(mesh.get("filename", "")).name.lower()) if mesh is not None else None
                    if hit is None:
                        continue
                    scale = mesh.get("scale")
                    ab = mesh_local_aabb(hit, [float(s) for s in scale.split()] if scale else None)
                    if ab is None:
                        continue
                    origin = col.find("origin")
                    off = np.array([float(v) for v in origin.get("xyz").split()])                        if origin is not None and origin.get("xyz") else np.zeros(3)
                    lo, hi = ab[0] + off, ab[1] + off
                    lo_all = lo if lo_all is None else np.minimum(lo_all, lo)
                    hi_all = hi if hi_all is None else np.maximum(hi_all, hi)
                if lo_all is not None:
                    boxes[link.get("name")] = (lo_all, hi_all)

            vol = {n: float(np.prod(hi - lo)) for n, (lo, hi) in boxes.items()}
            vsum = sum(vol.values())
            if vsum <= 0:
                continue
            for link in root.iter("link"):
                n = link.get("name")
                if n not in boxes:
                    continue
                lo, hi = boxes[n]
                m = total * vol[n] / vsum
                c = (lo + hi) / 2
                d = hi - lo

                ixx = m / 12 * (d[1] ** 2 + d[2] ** 2)
                iyy = m / 12 * (d[0] ** 2 + d[2] ** 2)
                izz = m / 12 * (d[0] ** 2 + d[1] ** 2)
                inertial = ET.Element("inertial")
                ET.SubElement(inertial, "origin", xyz=f"{c[0]:.6g} {c[1]:.6g} {c[2]:.6g}", rpy="0 0 0")
                ET.SubElement(inertial, "mass", value=f"{m:.6g}")
                ET.SubElement(inertial, "inertia", ixx=f"{ixx:.6g}", iyy=f"{iyy:.6g}",
                              izz=f"{izz:.6g}", ixy="0", ixz="0", iyz="0")
                link.insert(0, inertial)
            ET.indent(tree, space="  ")
            tree.write(urdf, encoding="unicode", xml_declaration=True)
            self._log.info("관성 보수: %s (총질량 %.1fkg, 링크 %d개)", key, total, len(boxes))

    def _try(self, results, family, name, fn):

        try:
            fn()
            results["ok"].append(f"{family}/{name}")
            self._log.info("수집 성공: %s/%s", family, name)
        except Exception as e:
            results["failed"].append({"robot": f"{family}/{name}", "error": str(e)[:200]})
            self._log.warning("수집 실패: %s/%s (%s)", family, name, str(e)[:120])

    def _collect_rd(self, family: str, module_name: str):

        mod = importlib.import_module(f"robot_descriptions.{module_name}")
        src = Path(mod.URDF_PATH)
        dst = self._out / family / module_name.replace("_description", "") / "robot.urdf"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    def _collect_repo(self, item: dict):

        repo_dir = self._out / "_repos" / Path(item["repo"]).stem
        if not repo_dir.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "-b", item["branch"], item["repo"], str(repo_dir)],
                check=True, capture_output=True, timeout=300,
            )

        entry = repo_dir / item["entry"]
        dst = self._out / item["family"] / item["name"] / "robot.urdf"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if entry.suffix == ".xacro":
            self._process_xacro(repo_dir, entry, dst)
        else:
            shutil.copy(entry, dst)

    def _process_xacro(self, repo_dir: Path, entry: Path, dst: Path):

        pkg_map = {}
        for pkg_xml in repo_dir.rglob("package.xml"):
            m = re.search(r"<name>\s*([\w-]+)\s*</name>", pkg_xml.read_text())
            if m:
                pkg_map[m.group(1)] = str(pkg_xml.parent)

        for f in repo_dir.rglob("*"):
            if f.suffix not in (".xacro", ".urdf"):
                continue
            text = f.read_text()
            for pkg, path in pkg_map.items():
                text = text.replace(f"$(find {pkg})", path)

            text = re.sub(r"\bM_PI\b", "pi", text)
            text, n_drop = re.subn(
                r"<xacro:include\b[^>]*?\$\(find[^>]*?/>", "", text, flags=re.S)
            if n_drop:
                self._log.info("외부 패키지 include %d개 제거: %s", n_drop, f.name)
            f.write_text(text)

        import xacro
        doc = xacro.process_file(str(entry))
        dst.write_text(doc.toprettyxml(indent="  "))

class RealPoolReport:

    def __init__(self, pool_root: str | Path, gen_root: str | Path | None = None):

        self._pool = Path(pool_root)
        self._gen = Path(gen_root) if gen_root else None
        self._log = logging.getLogger("urdf.real_pool")

        self._mesh_idx: dict | None = None

    def _mesh_index(self) -> dict:

        if self._mesh_idx is None:
            self._mesh_idx = index_mesh_files(self._pool)
        return self._mesh_idx

    def run(self) -> dict:

        rows, parse_fail = [], []
        for urdf in sorted(self._pool.glob("*/*/robot.urdf")):
            family, name = urdf.parent.parent.name, urdf.parent.name

            if family.startswith("_"):
                continue

            try:
                rows.append(self._analyze(family, name, urdf))
            except Exception as e:
                parse_fail.append({"robot": f"{family}/{name}", "error": str(e)[:200]})
                self._log.warning("파서 실패: %s/%s (%s)", family, name, str(e)[:120])

        report = {
            "n_parsed": len(rows), "n_parse_fail": len(parse_fail),
            "parse_fail": parse_fail, "robots": rows,
            "notation_stats": self._notation_stats(rows),
        }
        if self._gen is not None:
            report["coverage"] = self._coverage(rows)
        out = self._pool / "pool_report.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
        self._log.info("파서 통과 %d / 실패 %d -> %s", len(rows), len(parse_fail), out)
        return report

    def _analyze(self, family: str, name: str, urdf_path: Path) -> dict:

        model = PosedModel(urdf_path, pose={})

        geom_hist = {"box": 0, "cylinder": 0, "sphere": 0, "mesh": 0}
        n_diag = n_inertial = 0
        for link in model._urdf.robot.links:
            for col in link.collisions:
                g = col.geometry
                key = "box" if g.box is not None else "cylinder" if g.cylinder is not None                    else "sphere" if g.sphere is not None else "mesh"
                geom_hist[key] += 1
            if link.inertial is not None and link.inertial.inertia is not None:
                n_inertial += 1
                inertia = np.asarray(link.inertial.inertia)
                if np.allclose(inertia - np.diag(np.diag(inertia)), 0):
                    n_diag += 1

        joints = model._urdf.robot.joints
        mass = sum(l.mass for l in model.links)
        pts = [m.vertices for l in model.links for m in l.meshes]
        for link in model._urdf.robot.links:
            world = model.link_transform(link.name)
            for col in link.collisions:
                g = col.geometry
                if g.mesh is None:
                    continue
                hit = self._mesh_index().get(Path(g.mesh.filename).name.lower())
                ab = mesh_local_aabb(hit, getattr(g.mesh, "scale", None)) if hit else None
                if ab is None:
                    continue

                lo, hi = ab
                corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                                    for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                origin = col.origin if col.origin is not None else np.eye(4)
                T = world @ origin
                pts.append((T[:3, :3] @ corners.T).T + T[:3, 3])
        dims = None
        if pts:
            v = np.vstack(pts)
            dims = [float(x) for x in (v.max(axis=0) - v.min(axis=0))]

        n_mimic = sum(1 for j in joints if getattr(j, "mimic", None) is not None)
        return {
            "family": family, "name": name,
            "n_links": len(model.links), "n_joints": len(joints),
            "n_actuated": sum(1 for j in joints if j.type not in (None, "fixed")) - n_mimic,
            "n_mimic": n_mimic,
            "total_mass": float(mass), "collision_geoms": geom_hist,
            "diag_inertia_frac": (n_diag / n_inertial) if n_inertial else None,
            "bbox": dims,
        }

    def _notation_stats(self, rows: list) -> dict:

        total = {"box": 0, "cylinder": 0, "sphere": 0, "mesh": 0}
        diag = [r["diag_inertia_frac"] for r in rows if r["diag_inertia_frac"] is not None]
        for r in rows:
            for k, v in r["collision_geoms"].items():
                total[k] += v
        return {"collision_geom_total": total,
                "diag_inertia_frac_mean": float(np.mean(diag)) if diag else None}

    def _coverage(self, rows: list) -> dict:

        gen_mass, gen_len = [], []
        for meta_path in self._gen.glob("*/*/meta.json"):
            m = json.loads(meta_path.read_text())
            gen_mass.append(m["metrics"]["total_mass"])
            gen_len.append(m["metrics"]["overall_length"])
        if not gen_mass:
            return {"error": "generated set not found"}

        lo_m, hi_m = min(gen_mass), max(gen_mass)
        lo_l, hi_l = min(gen_len), max(gen_len)
        out_of = [r["name"] for r in rows
                  if (r["total_mass"] > 0 and not (lo_m <= r["total_mass"] <= hi_m))
                  or (r["bbox"] and not (lo_l <= max(r["bbox"][:2]) <= hi_l))]
        return {"gen_mass_range": [lo_m, hi_m], "gen_length_range": [lo_l, hi_l],
                "out_of_range": out_of}

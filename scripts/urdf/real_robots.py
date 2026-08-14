"""실로봇 URDF 풀 수집·검사.

용도 (plan/urdf.md):
1. 파서 검증: 전부 우리 로더(PosedModel)에 통과시켜 못 먹는 표기를 조기 발견
2. 표기 스타일 추출: 충돌 형상 근사 방식·관성 표기 관례 통계
3. 커버리지 검증: 실로봇 파라미터가 생성 분포 안에 드는지 확인 (제로샷 평가셋, 학습 미사용)

수집 경로: robot_descriptions 패키지(대부분의 다족·휴머노이드) + 저장소 클론/xacro(wheeled).
plan 보강 목록 반영: ackermann(F1TENTH 계열)·omni(Ridgeback·youBot)·6족(PhantomX) 추가.
메시 파일은 복사하지 않고 URDF만 풀에 저장한다 (로더가 메시를 읽지 않으므로 충분).
메시 원본은 _cache/(robot_descriptions)·_repos/(클론)에 남아 제로샷 sim 평가 때 쓴다.
"""
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
    """풀 아래(_cache/_repos 포함) 메시 파일 basename(소문자) -> 경로 색인.

    URDF의 package://·상대 경로 메시 참조를 풀 내부 실제 파일로 잇는
    근사 해석 (basename 유일성 가정, 중복 시 첫 파일 우선).
    """
    idx: dict = {}
    for p in Path(pool_root).rglob("*"):
        if p.suffix.lower() in (".stl", ".dae", ".obj") and p.name.lower() not in idx:
            idx[p.name.lower()] = p
    return idx


def mesh_local_aabb(path: Path, scale) -> tuple[np.ndarray, np.ndarray] | None:
    """메시 파일의 로컬 AABB (min, max) [m]. 로드 실패 시 None.

    URDF <mesh scale>을 정점 스케일로 반영한다 (스칼라/3벡터 모두 허용).
    """
    try:
        mesh = trimesh.load(str(path), force="mesh")
        lo, hi = mesh.bounds
    except Exception:
        return None

    # 일부 메시는 로드가 되어도 비유한 bounds를 반환한다 (atlas_drc 사례)
    # -> NaN이 bbox 합산에 전파되지 않도록 실패로 취급
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        return None

    # scale 정규화: None -> 1, 스칼라 -> 3벡터
    s = np.ones(3) if scale is None else np.asarray(scale, dtype=float).reshape(-1)
    if s.size == 1:
        s = np.repeat(s, 3)
    return lo * s, hi * s


class RealRobotCollector:
    """공개 실로봇 URDF를 풀 디렉토리({out}/{family}/{robot}/robot.urdf)로 수집."""

    # robot_descriptions 모듈 이름 (없는 이름은 수집 단계에서 건너뛰고 기록만 남김)
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

    # 저장소 직접 클론 대상 (xacro 포함). plan 보강: ackermann/omni/6족
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
        # plan 보강: omni (매커넘/옴니 베이스)
        {"name": "ridgeback", "family": "wheeled",
         "repo": "https://github.com/ridgeback/ridgeback.git", "branch": "melodic-devel",
         "entry": "ridgeback_description/urdf/ridgeback.urdf.xacro"},
        {"name": "youbot", "family": "wheeled",
         "repo": "https://github.com/youbot/youbot_description.git", "branch": "indigo-devel",
         "entry": "robots/youbot.urdf.xacro"},
        # plan 보강: 6족
        {"name": "phantomx", "family": "multileg",
         "repo": "https://github.com/HumaRobotics/phantomx_description.git", "branch": "master",
         "entry": "urdf/phantomx.urdf"},
        # plan 보강: ackermann (F1TENTH 계열 racecar)
        {"name": "racecar", "family": "wheeled",
         "repo": "https://github.com/f1tenth/f1tenth_simulator.git", "branch": "master",
         "entry": "racecar.xacro"},
    ]

    # 원본에 관성이 없는 로봇의 총질량 힌트 [kg] (plan 데이터 품질 보수. spot = 공식 스펙)
    _MASS_HINTS = {"multileg/spot": 32.5}

    def __init__(self, out_root: str | Path):
        """out_root = 풀 루트 디렉토리 (기본 /data/EA-Trav/urdf/real_robots)."""
        self._out = Path(out_root)
        self._log = logging.getLogger("urdf.real_pool")

    def collect(self) -> dict:
        """전체 수집을 수행하고 {"ok": [...], "failed": [...]}를 반환·저장한다.

        로봇 하나의 실패가 전체를 멈추지 않도록 개별 try로 감싼다.
        수집 후 관성 없는 URDF를 보수하고, 결과는 {out}/collection_log.json에 기록.
        """
        results = {"ok": [], "failed": []}
        for family, names in self._RD_ROBOTS.items():
            for name in names:
                self._try(results, family, name.replace("_description", ""),
                          lambda: self._collect_rd(family, name))
        for item in self._REPO_ROBOTS:
            self._try(results, item["family"], item["name"],
                      lambda item=item: self._collect_repo(item))

        # 데이터 품질 보수: 관성 없는 URDF에 근사 관성 주입 (plan 명시 항목)
        self._repair_inertials()

        self._log.info("수집 완료: 성공 %d / 실패 %d", len(results["ok"]), len(results["failed"]))
        with open(self._out / "collection_log.json", "w") as f:
            json.dump(results, f, indent=1)
        return results

    def _repair_inertials(self):
        """관성이 전혀 없는 URDF에 근사 관성을 보수한다 (예: spot).

        방법: 링크별 충돌 메시의 AABB 부피 비례로 힌트 총질량을 배분하고,
        AABB 중심을 질량중심, AABB 박스 공식을 관성으로 기록한다.
        제로샷 평가·커버리지 집계용 근사값이며 이미 관성이 있으면 건드리지 않는다.
        """
        import xml.etree.ElementTree as ET
        idx = index_mesh_files(self._out)
        for key, total in self._MASS_HINTS.items():
            urdf = self._out / key / "robot.urdf"
            if not urdf.exists():
                continue
            tree = ET.parse(urdf)
            root = tree.getroot()

            # 이미 관성이 있으면(질량 합 > 0) 원본 존중
            if sum(float(m.get("value", 0)) for m in root.iter("mass")) > 1e-6:
                continue

            # 링크별 충돌 메시 AABB 수집 (collision origin의 평행 이동만 반영)
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
                    off = np.array([float(v) for v in origin.get("xyz").split()]) \
                        if origin is not None and origin.get("xyz") else np.zeros(3)
                    lo, hi = ab[0] + off, ab[1] + off
                    lo_all = lo if lo_all is None else np.minimum(lo_all, lo)
                    hi_all = hi if hi_all is None else np.maximum(hi_all, hi)
                if lo_all is not None:
                    boxes[link.get("name")] = (lo_all, hi_all)

            # AABB 부피 비례로 질량 배분 후 <inertial> 삽입
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

                # AABB 박스 관성 근사: I = m/12 * (변 제곱 합)
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
        """수집 함수 fn을 실행하고 성공/실패를 results에 기록한다 (실패는 격리)."""
        try:
            fn()
            results["ok"].append(f"{family}/{name}")
            self._log.info("수집 성공: %s/%s", family, name)
        except Exception as e:
            results["failed"].append({"robot": f"{family}/{name}", "error": str(e)[:200]})
            self._log.warning("수집 실패: %s/%s (%s)", family, name, str(e)[:120])

    def _collect_rd(self, family: str, module_name: str):
        """robot_descriptions 모듈을 import해 (필요 시 자동 다운로드) URDF만 복사한다.

        다운로드 캐시는 환경변수 ROBOT_DESCRIPTIONS_CACHE 위치에 저장된다.
        """
        mod = importlib.import_module(f"robot_descriptions.{module_name}")
        src = Path(mod.URDF_PATH)
        dst = self._out / family / module_name.replace("_description", "") / "robot.urdf"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    def _collect_repo(self, item: dict):
        """GitHub 저장소를 얕은 클론하고 entry 파일(.urdf 또는 .xacro)을 풀로 가져온다.

        이미 클론돼 있으면 재사용한다 (재실행 시 네트워크 절약).
        """
        repo_dir = self._out / "_repos" / Path(item["repo"]).stem
        if not repo_dir.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "-b", item["branch"], item["repo"], str(repo_dir)],
                check=True, capture_output=True, timeout=300,
            )

        # xacro는 변환이 필요하고, 순수 URDF는 그대로 복사
        entry = repo_dir / item["entry"]
        dst = self._out / item["family"] / item["name"] / "robot.urdf"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if entry.suffix == ".xacro":
            self._process_xacro(repo_dir, entry, dst)
        else:
            shutil.copy(entry, dst)

    def _process_xacro(self, repo_dir: Path, entry: Path, dst: Path):
        """ROS 없이 xacro를 URDF로 변환해 dst에 저장한다.

        문제와 해법:
        - $(find pkg)는 ROS 패키지 검색이 필요 -> 저장소 내 package.xml을 훑어
          패키지 -> 경로 맵을 만들고 텍스트로 선치환한다.
        - include가 저장소 내부 원본 경로로 풀리므로 클론 전체를 제자리 치환한다.
        - 저장소에 없는 외부 센서 패키지 include는 제거한다 (액세서리 매크로용이라
          기본 env에서는 호출되지 않음).
        - pip xacro는 모듈 실행(-m)이 안 되므로 API(process_file)로 처리한다.
        """
        # 패키지 -> 경로 맵: 저장소 내 package.xml의 <name>을 수집
        pkg_map = {}
        for pkg_xml in repo_dir.rglob("package.xml"):
            m = re.search(r"<name>\s*([\w-]+)\s*</name>", pkg_xml.read_text())
            if m:
                pkg_map[m.group(1)] = str(pkg_xml.parent)

        # 클론 전체 제자리 치환 + 외부 패키지 include 제거
        for f in repo_dir.rglob("*"):
            if f.suffix not in (".xacro", ".urdf"):
                continue
            text = f.read_text()
            for pkg, path in pkg_map.items():
                text = text.replace(f"$(find {pkg})", path)

            # 구식 xacro 상수 표기 호환: ROS 배포판 xacro는 M_PI를 내장했지만
            # pip xacro는 pi만 지원한다 (youbot 등 indigo 시절 파일 대응)
            text = re.sub(r"\bM_PI\b", "pi", text)
            text, n_drop = re.subn(
                r"<xacro:include\b[^>]*?\$\(find[^>]*?/>", "", text, flags=re.S)
            if n_drop:
                self._log.info("외부 패키지 include %d개 제거: %s", n_drop, f.name)
            f.write_text(text)

        # 변환 실행 (API 호출)
        import xacro
        doc = xacro.process_file(str(entry))
        dst.write_text(doc.toprettyxml(indent="  "))


class RealPoolReport:
    """풀 전체를 로더에 통과시켜 파서 검증 + 표기 스타일·커버리지 통계를 만든다."""

    def __init__(self, pool_root: str | Path, gen_root: str | Path | None = None):
        """pool_root = 실로봇 풀 루트, gen_root = 생성 셋 루트 (커버리지 비교용, 선택)."""
        self._pool = Path(pool_root)
        self._gen = Path(gen_root) if gen_root else None
        self._log = logging.getLogger("urdf.real_pool")
        # 메시 파일 색인은 비싸므로 첫 사용 때 한 번만 구축
        self._mesh_idx: dict | None = None

    def _mesh_index(self) -> dict:
        """풀 내 메시 파일 색인 (지연 구축)."""
        if self._mesh_idx is None:
            self._mesh_idx = index_mesh_files(self._pool)
        return self._mesh_idx

    def run(self) -> dict:
        """풀의 모든 robot.urdf를 분석하고 pool_report.json으로 저장·반환한다."""
        rows, parse_fail = [], []
        for urdf in sorted(self._pool.glob("*/*/robot.urdf")):
            family, name = urdf.parent.parent.name, urdf.parent.name

            # _cache, _repos 등 내부 작업 폴더는 풀이 아니므로 제외
            if family.startswith("_"):
                continue

            # 파서 실패도 결과의 일부 (못 먹는 표기 조기 발견이 목적)
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
        """로봇 하나를 로더에 통과시켜 구조·표기 통계 행(dict)을 만든다.

        수집 항목: 링크·조인트·구동 조인트 수(mimic 제외한 독립 DoF), mimic 수,
        총 질량, 충돌 형상 타입 히스토그램, 관성 대각 전용 비율,
        bbox(프리미티브 정점 + 메시 파일 AABB 결합 — mesh 전용 로봇도 산출).
        """
        model = PosedModel(urdf_path, pose={})

        # 충돌 형상 타입 히스토그램 + 관성 표기(대각 전용) 비율
        geom_hist = {"box": 0, "cylinder": 0, "sphere": 0, "mesh": 0}
        n_diag = n_inertial = 0
        for link in model._urdf.robot.links:
            for col in link.collisions:
                g = col.geometry
                key = "box" if g.box is not None else "cylinder" if g.cylinder is not None \
                    else "sphere" if g.sphere is not None else "mesh"
                geom_hist[key] += 1
            if link.inertial is not None and link.inertial.inertia is not None:
                n_inertial += 1
                inertia = np.asarray(link.inertial.inertia)
                if np.allclose(inertia - np.diag(np.diag(inertia)), 0):
                    n_diag += 1

        # bbox: 프리미티브 정점 + 메시 AABB 8꼭짓점(파일 로드) 결합
        # (mesh 전용 로봇도 커버리지 전장 비교가 성립하도록 — plan 데이터 품질 보수)
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

                # 로컬 AABB 8꼭짓점을 월드로 변환해 합산 (회전 반영)
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

        # mimic 조인트는 master 값에 종속이므로 독립 DoF에서 제외하고 따로 센다
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
        """풀 전체 표기 통계 (입력측 표기 랜덤화 설계 근거).

        충돌 형상 타입 총합과 관성 대각 전용 비율 평균을 집계한다.
        """
        total = {"box": 0, "cylinder": 0, "sphere": 0, "mesh": 0}
        diag = [r["diag_inertia_frac"] for r in rows if r["diag_inertia_frac"] is not None]
        for r in rows:
            for k, v in r["collision_geoms"].items():
                total[k] += v
        return {"collision_geom_total": total,
                "diag_inertia_frac_mean": float(np.mean(diag)) if diag else None}

    def _coverage(self, rows: list) -> dict:
        """생성 셋의 질량·전장 분포 범위와 실로봇 값을 비교한다.

        범위 밖 로봇 이름을 out_of_range로 반환 -> 생성 config 범위 확장 근거.
        bbox는 메시 AABB를 포함하므로 mesh 전용 로봇도 전장 비교에 들어간다.
        """
        # 생성 셋 meta에서 질량·전장 분포 수집
        gen_mass, gen_len = [], []
        for meta_path in self._gen.glob("*/*/meta.json"):
            m = json.loads(meta_path.read_text())
            gen_mass.append(m["metrics"]["total_mass"])
            gen_len.append(m["metrics"]["overall_length"])
        if not gen_mass:
            return {"error": "generated set not found"}

        # 분포 범위 밖 실로봇 검출
        lo_m, hi_m = min(gen_mass), max(gen_mass)
        lo_l, hi_l = min(gen_len), max(gen_len)
        out_of = [r["name"] for r in rows
                  if (r["total_mass"] > 0 and not (lo_m <= r["total_mass"] <= hi_m))
                  or (r["bbox"] and not (lo_l <= max(r["bbox"][:2]) <= hi_l))]
        return {"gen_mass_range": [lo_m, hi_m], "gen_length_range": [lo_l, hi_l],
                "out_of_range": out_of}

"""정적 검사 (시뮬 없이 하는 물리 타당성 필터).

공통 검사 (plan/urdf.md "물리적 제약 (모든 플랫폼 공통 검증)"):
- 지상고: 접촉 링크(바퀴·롤러·발) 외의 형상이 지면에서 충분히 떠 있는가
- 셀프 충돌: 기립 자세에서 인접하지 않은 링크끼리 겹치는가 (+ 추가 자세)
- 무게중심: 접촉점 지지 다각형 안에 여유를 갖고 들어가는가 (balancing은 특수 검사로 대체)
- 중력 토크: 기립 자세의 정적 중력 토크가 조인트 토크 한계 이내인가
- 접지 공면: 모든 접지 유닛(바퀴/발, 롤러는 바퀴 단위로 묶음)의 최저점이 같은 평면인가

플랫폼 특수 검사 (spec.special로 선언, 생성기가 파라미터를 채움):
- load_share: 구동축 정적 하중 비율 >= 임계값 (diff·ackermann)
- balancing: 2륜 밸런싱 성립 조건 (축선 위 무게중심, 축 위 무게중심 높이, 복원 토크 여유)
- overturn: 전복 안정성 비율 (무게중심 높이 / 지지 다각형 여유) <= 상한 (wheeled 휴머노이드)

동적 검사(평지 기립·전진)는 2단계 제어기 이후 별도 단계에서 수행한다.
지면 규약: 로더의 월드 좌표는 base_link 프레임이므로, 접촉 링크 형상의
최저점 z0를 지면으로 정의하고 모든 높이를 z0 기준으로 계산한다.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from .loader import PosedModel


def validate_set(robots_root, cfg: dict) -> tuple[int, int]:
    """저장된 생성 셋 전체를 디스크에서 다시 읽어 재검증한다. (통과, 실패) 반환.

    meta.json의 기립 자세·접촉 링크·특수 검사 선언을 그대로 사용한다
    (추가 자세 검사는 생성 시점 전용이라 재검증에서는 생략).
    """
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
    """URDF를 기립 자세로 놓고 공통 검사 + 특수 검사 + 추가 자세 검사를 수행한다.

    cfg = configs/urdf.yaml의 validation 섹션.
    반환: {"passed": 전체 통과 여부, "checks": 항목별 통과 여부,
           "metrics": 수치 (스폰 높이, 지상고, 무게중심 여유, bbox 등)}.
    """
    model = PosedModel(urdf_path, standing_pose)
    special = special or {}
    checks, metrics = {}, {}

    # 지면 정의: 접촉 링크 최저점 (base_height = 스폰 시 base_link 원점 높이)
    z0 = _contact_floor(model, contact_links)
    metrics["base_height"] = float(-z0)

    # 전체 질량중심은 여러 검사가 공유하므로 한 번만 계산
    com, total_mass = _total_com(model)
    metrics["com"] = [float(v) for v in com]
    metrics["total_mass"] = total_mass

    checks["clearance"] = _check_clearance(model, contact_links, z0, cfg, metrics)
    checks["self_collision"] = _check_self_collision(model, metrics)
    checks["coplanar_contact"] = _check_coplanar(model, contact_links, z0, cfg, metrics)
    checks["gravity_torque"] = _check_torque(model, cfg, metrics)
    _measure_bbox(model, z0, metrics)

    # 무게중심-지지 다각형: balancing은 지지가 선분이라 특수 검사로 대체
    if "balancing" in special:
        checks["balancing"] = _check_balancing(model, com, z0, special["balancing"], cfg, metrics)
    else:
        checks["com_support"] = _check_com_polygon(model, com, contact_links, z0, cfg, metrics)

    # 구동축 하중 비율 (diff / ackermann / wheeled 휴머노이드 상속)
    if "load_share" in special:
        checks["load_share"] = _check_load_share(com, special["load_share"], cfg, metrics)

    # 전복 안정성 비율 (com_support가 계산한 여유를 사용하므로 그 뒤에)
    if "overturn" in special and "com_margin" in metrics:
        checks["overturn"] = _check_overturn(com, z0, cfg, metrics)

    # 추가 자세: 기립 검사만으로 놓치는 가동 자세의 충돌·무게중심 (조향 극한·리프트 최대 등)
    if check_poses:
        checks["extra_poses"] = _check_extra_poses(
            urdf_path, standing_pose, check_poses, contact_links, cfg, special, metrics)

    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}


# ---------- 공통 검사 ----------

def _contact_floor(model: PosedModel, contact_links) -> float:
    """접촉 링크 형상들의 최저 z = 지면 높이. 접촉 형상이 없으면 스펙 오류."""
    zs = [m.vertices[:, 2].min() for l in model.links if l.name in contact_links for m in l.meshes]
    if not zs:
        raise ValueError("contact link geometry not found")
    return float(min(zs))


def _total_com(model: PosedModel) -> tuple[np.ndarray, float]:
    """전체 질량중심 [base 프레임]과 총 질량 (링크 질량 가중 평균)."""
    masses = np.array([l.mass for l in model.links if l.com_world is not None])
    coms = np.array([l.com_world for l in model.links if l.com_world is not None])
    com = (masses[:, None] * coms).sum(axis=0) / masses.sum()
    return com, float(masses.sum())


def _check_clearance(model, contact_links, z0, cfg, metrics) -> bool:
    """지상고 검사: 비접촉 링크의 최저점이 지면에서 clearance_min 이상 떠 있는가."""
    zs = [m.vertices[:, 2].min() for l in model.links
          if l.name not in contact_links and l.meshes for m in l.meshes]
    clearance = float(min(zs) - z0) if zs else float("inf")
    metrics["clearance"] = clearance
    return clearance >= cfg["clearance_min"]


def _check_self_collision(model, metrics, key: str = "self_collision_pairs") -> bool:
    """셀프 충돌 검사: 허용 쌍을 제외하고 링크 간 접촉이 없는가.

    링크별 메시를 합쳐 fcl CollisionManager로 전수 검사한다.
    허용 쌍은 최소로 잡는다: 그래프 거리 1(부모-자식, 관절 연결부는 의도적 겹침)
    + base 외 링크의 가동 형제 쌍(롤러처럼 밀집 배치가 정상인 부품).
    """
    # 링크당 하나의 충돌 객체로 등록 (형상 여러 개는 합침)
    manager = trimesh.collision.CollisionManager()
    for link in model.links:
        if link.meshes:
            manager.add_object(link.name, trimesh.util.concatenate(link.meshes))
    _, pairs = manager.in_collision_internal(return_names=True)

    # 허용 목록에 없는 접촉 쌍만 위반으로 집계
    allowed = model.adjacency(depth=1) | model.sibling_pairs_nonbase()
    bad = [tuple(sorted(p)) for p in pairs if frozenset(p) not in allowed]
    metrics[key] = bad
    return len(bad) == 0


def _check_coplanar(model, contact_links, z0, cfg, metrics) -> bool:
    """접지 공면 검사: 접지 유닛별 최저점의 산포가 허용 범위 이내인가.

    유닛 = 접촉 링크 1개. 단 롤러(이름에 "_roller_")는 바퀴 단위로 묶는다
    (한 바퀴의 여러 롤러 중 최저점 하나만 실제 접지점이므로).
    명목 자세·명목 롤러 위상 기준 검사 (plan 공통 검증 항목).
    """
    # 유닛별 최저점 수집 (롤러 -> 소속 바퀴 이름으로 그룹)
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
    """무게중심 검사: 질량중심의 수평 투영이 지지 다각형 안에 여유를 갖고 있는가.

    지지 다각형 = 지면 밴드(contact_band) 안 접촉 정점들의 xy 볼록 껍질.
    여유(margin) = ConvexHull.equations(nx x + ny y + c <= 0 이 내부)의
    최대 위반값 부호 반전 = 경계까지 최소 거리 [m].
    """
    # 접촉점: 접촉 링크 정점 중 지면 밴드 안의 xy 좌표
    pts = []
    for l in model.links:
        if l.name not in contact_links:
            continue
        for m in l.meshes:
            v = m.vertices
            pts.append(v[v[:, 2] < z0 + cfg["contact_band"], :2])
    pts = np.vstack([p for p in pts if len(p)])

    # 접촉점이 일직선 등 퇴화 배치면 볼록 껍질 실패 -> 지지 불가로 판정
    try:
        hull = ConvexHull(pts)
    except Exception:
        metrics["com_margin"] = -1.0
        return False

    # 경계 여유 = -(최대 위반값): 모든 반평면을 margin 이상 안쪽에서 만족해야 통과
    margin = float(-(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]).max())
    metrics["com_margin"] = margin
    return margin >= cfg["com_margin_min"]


def _check_torque(model, cfg, metrics) -> bool:
    """중력 토크 검사: 각 구동 조인트의 정적 중력 토크 x 여유율 <= 토크 한계.

    조인트 j의 정적 토크 = axis_w . sum_i (c_i - p_j) x (m_i g)
    (i = j의 서브트리 링크, c_i = 월드 질량중심, p_j = 조인트 원점,
    axis_w = 월드 좌표 조인트 축). 기립 자세 유지에 필요한 최소 토크다.
    prismatic은 축 방향 힘 f = axis_w . sum m_i g 로 같은 원리를 적용한다.
    """
    # 자식 목록 구성 (서브트리 순회용)
    children: dict[str, list] = {}
    for j in model.joints:
        children.setdefault(j.parent, []).append(j.child)
    gravity = np.array([0.0, 0.0, -9.81])

    # 조인트별 여유율(한계/부하)을 계산하며 최소값 추적
    worst = float("inf")
    worst_joint = ""
    ok = True
    for j in model.joints:
        # 수동(effort=0)·고정 조인트는 토크 요구가 없으므로 건너뜀
        if j.type in (None, "fixed") or j.limit is None or not j.limit.effort:
            continue

        # 조인트 원점·축을 월드 좌표로 변환 (URDF axis는 자식 프레임 기준)
        subtree = _collect_subtree(j.child, children)
        T = model.link_transform(j.child)
        p = T[:3, 3]
        axis = T[:3, :3] @ np.asarray(j.axis if j.axis is not None else [0, 0, 1], dtype=float)

        # 서브트리 중력 부하: revolute는 모멘트, prismatic은 축 방향 힘
        load_vec = np.zeros(3)
        for l in model.links:
            if l.name in subtree and l.com_world is not None:
                if j.type == "prismatic":
                    load_vec += l.mass * gravity
                else:
                    load_vec += np.cross(l.com_world - p, l.mass * gravity)
        load = abs(float(load_vec @ axis))

        # 여유율 갱신 및 한계 초과 판정
        ratio = j.limit.effort / max(load, 1e-9)
        if ratio < worst:
            worst, worst_joint = ratio, j.name
        if load * cfg["torque_margin"] > j.limit.effort:
            ok = False
    metrics["torque_margin_min"] = None if worst == float("inf") else worst
    metrics["torque_worst_joint"] = worst_joint
    return ok


def _collect_subtree(root, children) -> set:
    """root 링크와 그 모든 자손 링크 이름 집합 (DFS)."""
    out, stack = {root}, [root]
    while stack:
        for c in children.get(stack.pop(), ()):
            out.add(c)
            stack.append(c)
    return out


def _measure_bbox(model, z0, metrics):
    """전체 형상의 축 정렬 bbox 측정 (보조 손실 라벨용 전장·전폭·전고).

    전고는 지면(z0) 기준이라 z 최소값 대신 z0에서 잰다.
    """
    vs = np.vstack([m.vertices for l in model.links for m in l.meshes])
    lo, hi = vs.min(axis=0), vs.max(axis=0)
    metrics["overall_length"] = float(hi[0] - lo[0])
    metrics["overall_width"] = float(hi[1] - lo[1])
    metrics["overall_height"] = float(hi[2] - z0)


# ---------- 플랫폼 특수 검사 ----------

def _check_balancing(model, com, z0, params, cfg, metrics) -> bool:
    """2륜 밸런싱 성립 검사 (plan diff 섹션의 4개 조건 중 정적 3개).

    params = {"axle_x": 축의 base 프레임 x, "axle_z": 축 높이(=바퀴 반지름)}.
    - 무게중심 수직 투영이 휠 축선(y방향 직선) 위 ±com_line_tol 안에 오는가
    - 무게중심이 축보다 min_com_above 이상 위에 있는가 (역진자 성립)
    - 복원 토크 여유: m g l sin(theta_max) x margin <= 구동 토크 합
      (l = 무게중심-축 거리, theta_max = 회복을 보장할 최대 기울기)
    직립 지상고는 공통 clearance 검사가 담당한다.
    """
    bcfg = cfg["balancing"]
    line_ok = abs(float(com[0]) - params["axle_x"]) <= bcfg["com_line_tol"]

    # 역진자 조건: 무게중심이 회전축(바퀴 축)보다 위.
    # com은 base 프레임 값이므로 접촉 최저점 z0를 빼서 지면 기준으로 통일한 뒤
    # 축 높이(axle_z = 바퀴 반지름, 지면 기준)와 비교한다
    lever = (float(com[2]) - z0) - params["axle_z"]
    height_ok = lever >= bcfg["min_com_above"]

    # 복원 토크: 기울기 theta_max에서 중력 모멘트를 구동 토크 합이 이겨야 함
    _, total_mass = _total_com(model)
    effort_sum = sum(j.limit.effort for j in model.joints
                     if j.name.startswith("drive_") and j.limit is not None)
    need = total_mass * 9.81 * max(lever, 0.0) * np.sin(bcfg["theta_max"])
    torque_ok = effort_sum >= need * bcfg["torque_margin"]

    metrics["balancing"] = {"com_line_err": abs(float(com[0]) - params["axle_x"]),
                            "com_lever": lever, "recovery_need": float(need),
                            "effort_sum": float(effort_sum)}
    return bool(line_ok and height_ok and torque_ok)


def _check_load_share(com, params, cfg, metrics) -> bool:
    """구동축 정적 하중 비율 검사 (1차원 지렛대 근사).

    params = {"drive_x": 구동축 x, "other_x": 반대 지지(캐스터/비구동축) x}.
    share = (other_x - com_x) / (other_x - drive_x): 무게중심이 구동축에
    가까울수록 1에 가까움. min_share 미만이면 접지력 부족으로 필터링.
    """
    span = params["other_x"] - params["drive_x"]
    share = float((params["other_x"] - com[0]) / span) if abs(span) > 1e-6 else 1.0
    metrics["drive_load_share"] = share
    return share >= cfg["load_share_min"]


def _check_overturn(com, z0, cfg, metrics) -> bool:
    """전복 안정성 비율 검사 (wheeled 휴머노이드).

    비율 = 무게중심 높이(지면 기준) / 지지 다각형 경계 여유(com_margin).
    키가 크고 받침 여유가 좁을수록 커지며, overturn_max를 넘으면 필터링.
    """
    ratio = float((com[2] - z0) / max(metrics["com_margin"], 1e-6))
    metrics["overturn_ratio"] = ratio
    return ratio <= cfg["overturn_max"]


def _check_extra_poses(urdf_path, standing_pose, check_poses, contact_links,
                       cfg, special, metrics) -> bool:
    """추가 자세들에서 셀프 충돌 + 무게중심 검사를 재수행한다.

    팔 전방·리프트 최대 같은 자세는 무게중심이 실제로 이동하므로
    com-다각형(overturn 선언 시 전복 비율까지)을 자세별로 다시 본다
    (plan wheeled 휴머노이드 물리 제약). balancing은 다각형이 없어
    셀프 충돌만 검사한다. 각 자세는 기립 자세 위에 부분 덮어쓰기.
    """
    ok = True
    bad_all = []
    for i, pose in enumerate(check_poses):
        posed = PosedModel(urdf_path, {**standing_pose, **pose})
        sub: dict = {}

        # 셀프 충돌은 모든 추가 자세에서 공통
        if not _check_self_collision(posed, sub, key="pairs"):
            ok = False
            bad_all.append({"pose_index": i, "pairs": sub["pairs"]})

        # 무게중심 이동 반영 검사 (balancing 제외 — 다각형이 선분)
        if "balancing" not in special:
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

"""생성 URDF 정적 검증 필터 (pybullet DIRECT).

검사 항목(플랜 3단):
  parse         yourdfpy 로드 왕복
  inertia       질량>0, 관성 대각>0 + 삼각 부등식 (해석식이라 사실상 항상 통과 — 회귀 방지용)
  ground_gap    스탠딩 자세에서 최저점이 지면(0) ±2.5cm
  dims          스탠딩 AABB가 목표 (w,l,h) 대비 max(10%, 3.5cm) 이내
  self_col_rest rest pose 자기충돌 (인접쌍 + connector(ball) 경유 2차쌍 제외)
  self_col_conf 관절 랜덤 K컨피그 자기충돌 → 호출부에서 리밋 축소 후 1회 재시도
  support       CoM 수평 투영이 지지 다각형(접지 링크 hull, 1.5cm 침식) 내부
"""

import math
import os
import random
import tempfile

import numpy as np
import pybullet as p
from shapely.geometry import MultiPoint, Point

_CID = None
PEN_TOL = -1e-3          # 이보다 깊은 침투만 충돌로 판정
GROUND_TOL = 0.025
SUPPORT_MARGIN = 0.015
CONTACT_Z = 0.01


def _client():
    global _CID
    if _CID is None:
        _CID = p.connect(p.DIRECT)
    return _CID


def _excluded_pairs(robot, name2idx):
    """자기충돌 검사 제외쌍: 인접(부모-자식) + connector 링크 경유 2차 인접."""
    adj = set()
    parent_of = {}
    for j in robot.joints:
        a, b = name2idx[j.parent], name2idx[j.child]
        adj.add(frozenset((a, b)))
        parent_of[j.child] = j.parent
    conn = {l.name for l in robot.links if l.connector}
    for j in robot.joints:
        if j.parent in conn:
            gp = parent_of[j.parent]
            adj.add(frozenset((name2idx[gp], name2idx[j.child])))
    return adj


def _self_collision(body, pairs):
    for a, b in pairs:
        pts = p.getClosestPoints(body, body, distance=0.0,
                                 linkIndexA=a, linkIndexB=b)
        for pt in pts:
            if pt[8] < PEN_TOL:
                return (a, b, pt[8])
    return None


def _union_aabb(body, n_links):
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    per_link = {}
    for i in range(-1, n_links):
        a, b = p.getAABB(body, i)
        per_link[i] = (a, b)
        lo = np.minimum(lo, a)
        hi = np.maximum(hi, b)
    return lo, hi, per_link


def validate(robot, meta, target, k_configs=20, seed=0):
    """returns (ok, reasons(list), measured(dict))"""
    reasons, measured = [], {}

    # inertia 물리성
    for link in robot.links:
        m = link.mass
        ixx, iyy, izz = link.inertia
        if m <= 1e-6 or min(ixx, iyy, izz) <= 0 or \
           ixx + iyy < izz * 0.999 or iyy + izz < ixx * 0.999 or izz + ixx < iyy * 0.999:
            return False, ["inertia"], measured

    urdf = robot.to_urdf()
    fd, path = tempfile.mkstemp(suffix=".urdf")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(urdf)

        try:
            import yourdfpy
            yourdfpy.URDF.load(path)
        except Exception:
            return False, ["parse"], measured

        cid = _client()
        p.resetSimulation(physicsClientId=cid)
        try:
            body = p.loadURDF(path, basePosition=(0, 0, meta["base_z"]),
                              flags=p.URDF_USE_INERTIA_FROM_FILE |
                              p.URDF_MAINTAIN_LINK_ORDER)
        except Exception:
            return False, ["parse"], measured

        nj = p.getNumJoints(body)
        name2idx = {p.getBodyInfo(body)[0].decode(): -1}
        movable = []
        for i in range(nj):
            info = p.getJointInfo(body, i)
            name2idx[info[12].decode()] = i
            if info[2] == p.JOINT_REVOLUTE and info[8] < info[9]:
                movable.append((i, info[8], info[9]))

        excluded = _excluded_pairs(robot, name2idx)
        idxs = list(range(-1, nj))
        pairs = [(a, b) for i, a in enumerate(idxs) for b in idxs[i + 1:]
                 if frozenset((a, b)) not in excluded]

        # 스탠딩 AABB → ground / dims
        lo, hi, per_link = _union_aabb(body, nj)
        ext = hi - lo
        measured = {"w": round(float(ext[1]), 4), "l": round(float(ext[0]), 4),
                    "h": round(float(ext[2]), 4), "zmin": round(float(lo[2]), 4)}
        if abs(lo[2]) > GROUND_TOL:
            reasons.append("ground_gap")
        tw, tl, th = target
        for name, m_, t_ in (("w", ext[1], tw), ("l", ext[0], tl), ("h", ext[2], th)):
            if abs(m_ - t_) > max(0.10 * t_, 0.035):
                reasons.append(f"dims_{name}")

        # 자기충돌: rest + 랜덤 컨피그
        idx2name = {v: k for k, v in name2idx.items()}
        hit = _self_collision(body, pairs)
        if hit:
            reasons.append("self_col_rest")
            measured["col_pair"] = [idx2name[hit[0]], idx2name[hit[1]],
                                    round(hit[2], 4)]
        else:
            rng = random.Random(seed)
            for _ in range(k_configs):
                for i, lo_j, hi_j in movable:
                    p.resetJointState(body, i, rng.uniform(lo_j, hi_j))
                if _self_collision(body, pairs):
                    reasons.append("self_col_conf")
                    break
            for i, _, _ in movable:
                p.resetJointState(body, i, 0.0)

        # 지지 다각형 vs CoM
        contact_pts = []
        for i, (a, b) in per_link.items():
            if a[2] < CONTACT_Z:
                contact_pts += [(a[0], a[1]), (a[0], b[1]),
                                (b[0], a[1]), (b[0], b[1])]
        if len(contact_pts) < 3:
            reasons.append("support")
        else:
            masses, coms = [], []
            m0 = p.getDynamicsInfo(body, -1)[0]
            masses.append(m0)
            coms.append(p.getBasePositionAndOrientation(body)[0])
            for i in range(nj):
                masses.append(p.getDynamicsInfo(body, i)[0])
                coms.append(p.getLinkState(body, i)[0])
            com = np.average(np.array(coms), axis=0, weights=masses)
            hull = MultiPoint(contact_pts).convex_hull
            if not hull.buffer(-SUPPORT_MARGIN).contains(Point(com[0], com[1])):
                reasons.append("support")
            measured["com_xy"] = [round(float(com[0]), 4), round(float(com[1]), 4)]

        return len(reasons) == 0, reasons, measured
    finally:
        os.unlink(path)

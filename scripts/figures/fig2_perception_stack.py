"""Fig 2 지각 스택 — 한 프레임의 RGB·depth·히트맵·후보, 그리고 기억 그래프.

한 에피소드의 같은 시각 t에서 다섯 장을 뽑는다. 앞 네 장은 같은 프레임이라
나란히 두면 입력에서 후보까지의 흐름이 그대로 읽힌다.
  fig2_1_rgb          관측 RGB
  fig2_2_depth        같은 프레임 depth (planar z-depth)
  fig2_3_heat         RGB 위에 제안기 히트맵
  fig2_4_candidates   히트맵 + 고스트 4개 + 주행 노드 1개
  fig2_5_graph        기억(토폴로지) 그래프 — 배경 없이 그래프만

후보는 학습된 제안기를 돌린 결과가 아니라 **GT 히트맵에 문서화된 후보 규약을
그대로 적용**한 것이다: 임계 통과 셀 → 셀 중심 depth로 리프트 → 바닥·5m 검사
→ 월드 0.5m 억제 → 상위 K=5. 저장된 체크포인트는 현행 모델과 비호환이라
(재학습 대기) 추론 결과를 쓰면 그림이 현재 구조를 대변하지 못한다.

주행 노드는 그 다섯 중 다음 홉 위치(pose[t+1])에 가장 가까운 후보다 —
전역 선택기 학습이 정답을 고르는 규칙과 같다.
"""

import argparse
import glob
import json
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = "/data/EVLN_dataset"
MANSION = "/data/MansionWorld/mansionworld"
OUT_DIR = "/workspace/research/check/figure/fig2"
CACHE = os.path.join(OUT_DIR, "_cache")

# 투영 규약 — 데이터 생성부(common.project)와 같은 값이어야 한다.
RES, FX = 256, 128.0
HEAT_HW = 32
HEAT_IGNORE = 2
GROUND_TOL = 0.12
MAX_DIST = 5.0
NMS_R = 0.5
TOP_K = 5

C_POS = "#22C55E"
C_IGN = "#94A3B8"
C_GHOST = "#F59E0B"
C_DRIVE = "#2563EB"
C_INK = "#0F172A"
C_MUTE = "#64748B"
FLOOR_COLORS = ("#2563EB", "#0EA5E9", "#22C55E", "#F59E0B", "#EF4444", "#A855F7")
C_MEM = "#F59E0B"
C_GHOST_G = "#94A3B8"
C_EDGE = "#64748B"
C_STAIR = "#7C3AED"
C_ELEV = "#0D9488"

# 같은 위치로 볼 노드 병합 반경. GraphMap의 node_radius와 같은 값이라
# 로봇별 기억을 합칠 때 중복 노드가 생기지 않는다.
MERGE_R = 1.0


def lift(u, v, depth, pose):
    """히트맵 셀 (u, v) → 월드 좌표. common.project의 역변환이다.

    셀 중심 렌더 픽셀은 u*8+4 (중앙값 풀링이 아니라 중심 표본 — GT와 같은 규약).
    depth는 planar z-depth라 광선 길이가 아니라 전방 성분 그대로다.
    """
    k = RES // HEAT_HW
    px, py = u * k + k // 2, v * k + k // 2
    fwd = float(depth[py, px])
    if not np.isfinite(fwd) or fwd < 0.2:
        return None
    _, x, z, yaw, cam_y = pose
    right = (px - RES / 2) * fwd / FX
    rel_y = -(py - RES / 2) * fwd / FX
    ry = math.radians(yaw)
    wx = x + fwd * math.sin(ry) + right * math.cos(ry)
    wz = z + fwd * math.cos(ry) - right * math.sin(ry)
    return wx, cam_y + rel_y, wz, fwd


def candidates(heat, depth, pose, top_k=TOP_K):
    """문서화된 후보 규약대로 히트맵에서 후보를 뽑는다.

    바닥 검사는 리프트한 점의 높이가 0 근처인지 본다 — 벽·가구 위 점이 후보로
    올라오면 주행 불가 지점을 목적지로 삼게 된다.
    """
    # 테두리 셀은 뺀다 — 마커가 화면 밖으로 잘리고, 경계 1픽셀은 GT에서도
    # 라벨을 무시하는 구간이라 후보로 쓰기에 근거가 약하다.
    cells = [(v, u) for v in range(2, HEAT_HW - 2)
             for u in range(2, HEAT_HW - 2) if heat[v, u] == 1]
    hits = []
    for v, u in cells:
        p = lift(u, v, depth, pose)
        if p is None:
            continue
        wx, wy, wz, fwd = p
        if abs(wy) > GROUND_TOL or fwd > MAX_DIST:
            continue
        hits.append({"u": u, "v": v, "x": wx, "z": wz, "d": fwd})
    if not hits:
        return []
    # 최원점 표집으로 K개를 고른다. 거리 오름차순으로 뽑으면 후보가 발밑
    # 한 뭉치에 몰려 "자유공간 전반의 후보"라는 성격이 안 보인다. 학습된
    # 제안기는 점수 내림차순으로 뽑지만 GT 히트맵은 0/1이라 점수가 없다.
    keep = [min(hits, key=lambda h: h["d"])]
    while len(keep) < top_k:
        far, best = None, -1.0
        for h in hits:
            d = min(math.hypot(h["x"] - k["x"], h["z"] - k["z"]) for k in keep)
            if d >= NMS_R and d > best:
                far, best = h, d
        if far is None:
            break
        keep.append(far)
    return keep


def pick_frame(pattern, want=TOP_K, limit=200):
    """후보가 정확히 K개 나오고 서로 충분히 벌어진 프레임을 고른다."""
    best = None
    for f in sorted(glob.glob(pattern))[:limit]:
        z = np.load(f)
        pose, heat, dep, sp = z["pose"], z["heat"], z["depth"], z["special"]
        for t in range(len(heat) - 1):
            if sp[t] != 0 or (heat[t] == 1).sum() < 60:
                continue
            cand = candidates(heat[t], dep[t].astype(np.float32), pose[t])
            if len(cand) < want:
                continue
            nxt = pose[t + 1]
            dist = [math.hypot(c["x"] - nxt[1], c["z"] - nxt[2]) for c in cand]
            # 화면에서도 퍼져 있어야 그림이 읽힌다. 월드에서만 벌어지고 화면
            # 아래 한 줄에 몰리는 프레임이 흔하다(도달 가능 영역이 발밑 바닥).
            di = int(np.argmin(dist))
            dc = cand[di]
            # 주행 노드가 화면 맨 아래 줄에 찍히면 범례·테두리에 묻힌다.
            # 다음 홉이 가까운 프레임에서 늘 그렇게 되므로 아예 걸러낸다.
            if not (4 <= dc["v"] <= 27 and 3 <= dc["u"] <= 28):
                continue
            rows = max(c["v"] for c in cand) - min(c["v"] for c in cand)
            cols = max(c["u"] for c in cand) - min(c["u"] for c in cand)
            spread = min(math.hypot(a["x"] - b["x"], a["z"] - b["z"])
                         for i, a in enumerate(cand) for b in cand[i + 1:])
            score = spread + 0.12 * (rows + cols) - 0.6 * min(dist)
            if best is None or score > best[0]:
                best = (score, f, t, cand, di)
    return best


def hires_frame(building, pose, size):
    """저장된 포즈 그대로 같은 시점을 고해상도로 다시 렌더한다.

    데이터셋 프레임은 모델 입력 크기인 256²라 그림에 쓰기엔 거칠다. 포즈
    (층·x·z·yaw·cam_y)가 그대로 있으므로 같은 화각으로 다시 찍으면 된다.
    화각은 데이터 생성 규약과 같아야 한다 — RES/2 = FX·tan(HFOV/2)에서
    HFOV=90°이고 정사각이라 수직 화각도 90°다. 결과는 캐시한다(렌더가 비쌈).
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "datasets", "MANSION"))
    import cv2
    import common

    fl, x, z, yaw, cam_y = pose
    key = f"{building}_f{int(fl)}_{x:.3f}_{z:.3f}_{yaw:.2f}_{size}"
    png = os.path.join(CACHE, key + ".png")
    npy = os.path.join(CACHE, key + ".npy")
    if os.path.exists(png) and os.path.exists(npy):
        img = cv2.cvtColor(cv2.imread(png), cv2.COLOR_BGR2RGB)
        return img, np.load(npy)

    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(f"{MANSION}/{building}#0",
                           f"floor_{int(fl)}.json")) as f:
        scene = json.load(f)
    ctrl = common.launch_controller(width=size, height=size, render=True)
    try:
        common.load_floor_scene(ctrl, scene)
        ev = ctrl.step(action="AddThirdPartyCamera",
                       position=dict(x=float(x), y=float(cam_y), z=float(z)),
                       rotation=dict(x=0.0, y=float(yaw), z=0.0),
                       fieldOfView=90.0)
        if not ev.metadata["lastActionSuccess"]:
            raise RuntimeError(ev.metadata.get("errorMessage"))
        rgb = np.array(ev.third_party_camera_frames[0])
        dep = np.array(ev.third_party_depth_frames[0], dtype=np.float32)
    finally:
        ctrl.stop()
    cv2.imwrite(png, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(npy, dep)
    print(f"고해상도 재렌더 {size}²: {png}")
    return rgb, dep


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fp = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print(f"저장: {fp}")


# 표시용 화면비. 데이터셋 렌더는 정사각(HFOV=VFOV=90°)이지만 그림에는
# 일반적인 비율이 낫다. 재렌더 대신 **위쪽을 잘라** 만든다 — 가로 화각을
# 그대로 두므로 히트맵 셀 대응이 어긋나지 않고, 잘리는 곳은 천장이라
# 도달 가능 영역(바닥)은 하나도 잃지 않는다.
ASPECT = 16.0 / 9.0


def crop_top(img):
    """정사각 프레임을 ASPECT 비율로 만들되 아래쪽을 남긴다."""
    h, w = img.shape[:2]
    keep = int(round(w / ASPECT))
    return img[max(h - keep, 0):], max(h - keep, 0)


def bare_axes(w, h, size=5.6):
    fig = plt.figure(figsize=(size, size * h / w), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return fig, ax


def heat_layers(heat, res):
    """히트맵을 렌더 해상도로 올린 (양성, 무시) 알파 맵.

    최근접 확대라 32² 격자가 그대로 보인다 — 제안기가 픽셀이 아니라 셀 단위로
    후보를 낸다는 사실이 그림에 드러나야 한다.
    """
    k = res // HEAT_HW
    big = np.kron(heat, np.ones((k, k), dtype=heat.dtype))
    return big == 1, big == HEAT_IGNORE


def draw_rgb(rgb):
    img, _ = crop_top(rgb)
    fig, ax = bare_axes(img.shape[1], img.shape[0])
    ax.imshow(img)
    save(fig, "fig2_1_rgb")


def draw_depth(depth):
    """depth는 planar z-depth다. 유한 구간만 정규화해 원경 클리핑에 색이
    다 먹히지 않게 한다."""
    d, _ = crop_top(depth.astype(np.float32))
    fig, ax = bare_axes(d.shape[1], d.shape[0])
    # 원경은 20m에서 잘려 있다. 그 값까지 색 범위에 넣으면 실내 기하가
    # 한 색으로 뭉개지므로 클리핑 값을 뺀 뒤 정규화한다.
    d[~np.isfinite(d)] = 0.0
    clip = float(d.max())
    finite = d[(d > 0) & (d < clip - 0.05)]
    hi = float(np.percentile(finite, 98)) if len(finite) else clip
    ax.imshow(np.clip(d, 0, hi), cmap="magma_r", vmin=0, vmax=hi)
    save(fig, "fig2_2_depth")
    return hi


def overlay(ax, masks, shape):
    for mask, color, alpha in masks:
        layer = np.zeros((*shape, 4), np.float32)
        layer[..., :3] = matplotlib.colors.to_rgb(color)
        layer[..., 3] = mask * alpha
        ax.imshow(layer)


def draw_heat(rgb, heat):
    img, off = crop_top(rgb)
    res = rgb.shape[0]
    fig, ax = bare_axes(img.shape[1], img.shape[0])
    ax.imshow(img)
    pos, ign = heat_layers(heat, res)
    overlay(ax, ((ign[off:], C_IGN, 0.35), (pos[off:], C_POS, 0.45)),
            img.shape[:2])
    save(fig, "fig2_3_heat")


def draw_candidates(rgb, heat, cand, drive_i):
    """히트맵 위에 고스트와 주행 노드. 셀 중심에 찍어야 후보가 셀 단위라는
    규약과 어긋나지 않는다."""
    img, off = crop_top(rgb)
    res = rgb.shape[0]
    fig, ax = bare_axes(img.shape[1], img.shape[0])
    ax.imshow(img)
    pos, ign = heat_layers(heat, res)
    overlay(ax, ((ign[off:], C_IGN, 0.28), (pos[off:], C_POS, 0.32)),
            img.shape[:2])
    # 마커는 그래프(fig2_5)와 같은 규약 — 고스트는 회색 테두리 빈 원, 선택은
    # 파란 테두리 빈 원. 두 그림을 나란히 두면 같은 후보임이 색으로 이어진다.
    # 크기는 해상도와 무관하다(그림의 물리 크기가 같으므로 포인트 단위 고정).
    # 실사 배경 위에서는 얇은 흰 링을 깔아야 테두리가 묻히지 않는다.
    k = res // HEAT_HW
    for i, c in enumerate(cand):
        px, py = c["u"] * k + k // 2, c["v"] * k + k // 2 - off
        drive = i == drive_i
        ax.scatter([px], [py], s=230 if drive else 195, facecolor="none",
                   edgecolor="white", linewidth=3.6, zorder=5)
        ax.scatter([px], [py], s=230 if drive else 195, facecolor="white",
                   edgecolor=C_DRIVE if drive else C_GHOST_G,
                   linewidth=2.4 if drive else 2.2, zorder=6)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="white", mec=C_GHOST_G, mew=2.0,
               ms=8, label="ghost node"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=C_DRIVE, mew=2.2,
               ms=8.5, label="selected node")],
        loc="upper left", fontsize=9, framealpha=0.92, borderpad=0.45)
    save(fig, "fig2_4_candidates")


def draw_graph(topo, robot):
    """기억 그래프만 그린다. 층을 세로로 쌓아 계단·엘베 전환 엣지가 층 사이를
    잇는 것이 보이게 한다 — 평면에 겹쳐 그리면 다층 구조가 사라진다."""
    nodes = {n["id"]: n for n in topo["nodes"]}
    floors = sorted({n["floor"] for n in nodes.values()})
    xs = [n["x"] for n in nodes.values()]
    zs = [n["z"] for n in nodes.values()]
    span_x, span_z = max(xs) - min(xs), max(zs) - min(zs)
    # 층 간격이 층 자체 크기보다 커야 겹치지 않는다. 0.62배로는 F2 노드가
    # F3 영역까지 내려와 층 구분이 사라졌다.
    dy = 1.15 * span_z

    def pos(n):
        # 층마다 위로 올리고 살짝 밀어 아이소메트릭처럼 보이게 한다
        i = floors.index(n["floor"])
        return n["x"] + 0.10 * span_x * i, -n["z"] - dy * i

    fig = plt.figure(figsize=(7.2, 8.4), dpi=300)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_aspect("equal")
    ax.axis("off")

    trans = 0
    for e in topo["edges"]:
        a, b = nodes.get(e["a"]), nodes.get(e["b"])
        if a is None or b is None:
            continue
        (x0, y0), (x1, y1) = pos(a), pos(b)
        if a["floor"] != b["floor"]:
            ax.plot([x0, x1], [y0, y1], color=C_INK, lw=1.9,
                    ls=(0, (3.0, 2.0)), zorder=3)
            trans += 1
        else:
            ax.plot([x0, x1], [y0, y1], color=C_MUTE, lw=1.2, alpha=0.75,
                    zorder=2)
    for n in nodes.values():
        x, y = pos(n)
        col = FLOOR_COLORS[floors.index(n["floor"]) % len(FLOOR_COLORS)]
        ax.scatter([x], [y], s=46, facecolor=col, edgecolor="white",
                   linewidth=1.0, zorder=4)
    for i, f in enumerate(floors):
        ys = [pos(n)[1] for n in nodes.values() if n["floor"] == f]
        ax.text(min(xs) + 0.10 * span_x * i - 0.10 * span_x,
                sum(ys) / len(ys), f"F{f}", fontsize=15, fontweight="bold",
                color=FLOOR_COLORS[i % len(FLOOR_COLORS)], ha="right",
                va="center", zorder=5)
    ax.legend(handles=[
        Line2D([], [], color=C_MUTE, lw=1.6, label="walk edge"),
        Line2D([], [], color=C_INK, lw=1.9, ls=(0, (3.0, 2.0)),
               label="floor transition (stairs / elevator)")],
        loc="center left", fontsize=10.5, framealpha=0.92)
    save(fig, "fig2_5_graph")
    print(f"   그래프: 노드 {len(nodes)} 엣지 {len(topo['edges'])} "
          f"(층 전환 {trans}) 로봇 {robot}")


def union_floor(building, floor):
    """로봇별 토포맵을 한 층에서 합친 기억 그래프.

    한 로봇의 기억에는 계단·엘베 중 하나만 남는다 — 탐색이 비용이 싼 수단만
    쓰기 때문이다(legged는 계단 3엣지, wheeled는 엘베 3엣지). 실제 배포에서
    쓰는 기억은 mansion_adapter.UnionMemory, 즉 로봇별 토포맵의 합집합이므로
    그림도 그것을 그린다. 병합 반경은 GraphMap의 node_radius와 같다.
    """
    nodes, edges, links = [], [], []
    for fp in sorted(glob.glob(f"{DATA}/topomap/{building}_*.json")):
        if "elevonly" in fp:
            continue
        with open(fp) as fh:
            topo = json.load(fh)
        allf = {n["id"]: n["floor"] for n in topo["nodes"]}
        local = {}
        for n in topo["nodes"]:
            if n["floor"] != floor:
                continue
            hit = next((i for i, m in enumerate(nodes)
                        if math.hypot(m["x"] - n["x"], m["z"] - n["z"])
                        < MERGE_R), None)
            if hit is None:
                nodes.append({"x": n["x"], "z": n["z"], "kind": n["kind"]})
                hit = len(nodes) - 1
            local[n["id"]] = hit
        for e in topo["edges"]:
            a, b = local.get(e["a"]), local.get(e["b"])
            if a is None and b is None:
                continue
            if a is None or b is None:
                # 층을 넘는 엣지는 한쪽 끝만 이 층에 있다. 지우면 전환
                # 지점이 평범한 노드로 보이므로 나가는 방향만 남긴다.
                here = a if a is not None else b
                other = allf.get(e["b"] if a is not None else e["a"])
                if other is not None:
                    key = (here, e["mode"], int(other))
                    if key not in links:
                        links.append(key)
                continue
            if a == b:
                continue
            key = (min(a, b), max(a, b), e["mode"])
            if key not in edges:
                edges.append(key)
    return nodes, edges, links


def room_center(building, floor, rtype):
    """층 평면에서 방 중심을 읽는다. 계단·엘베 표시를 노드 kind가 아니라
    실제 방 위치로 붙이기 위한 것이다."""
    fp = f"{MANSION}/{building}#0/floor_{int(floor)}.json"
    with open(fp) as fh:
        scene = json.load(fh)
    for r in scene["rooms"]:
        if r["roomType"] == rtype:
            v = np.array(r["vertices"], float)
            return float(v[:, 0].mean()), float(v[:, 1].mean())
    return None


def shortest_path(nodes, edges, src, dst):
    """홉 수 기준 최단 경로. 그래프가 작아 BFS로 충분하다."""
    adj = {}
    for a, b, _ in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    prev, seen = {src: None}, [src]
    while seen:
        cur = seen.pop(0)
        if cur == dst:
            out = []
            while cur is not None:
                out.append(cur)
                cur = prev[cur]
            return out[::-1]
        for nb in adj.get(cur, []):
            if nb not in prev:
                prev[nb] = cur
                seen.append(nb)
    return []


def skeleton(nodes, edges, pose, elev_xz, n_mem):
    """현재 노드 주변의 기억 노드 뭉치와 엘베로 나가는 노드를 고른다.

    경로를 따라 뽑으면 기억 노드가 한 줄로 늘어서 "이미 아는 구역"이라는
    성격이 안 보인다. 현재 노드에서 홉 수가 가까운 순으로 뽑아 뭉치를 만든다.
    """
    here = min(range(len(nodes)),
               key=lambda i: math.hypot(nodes[i]["x"] - pose[1],
                                        nodes[i]["z"] - pose[2]))
    adj = {}
    for a, b, _ in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    order, seen = [], {here}
    queue = [here]
    while queue and len(order) < n_mem:
        cur = queue.pop(0)
        for nb in sorted(adj.get(cur, []),
                         key=lambda i: math.hypot(nodes[i]["x"] - pose[1],
                                                  nodes[i]["z"] - pose[2])):
            if nb in seen:
                continue
            seen.add(nb)
            order.append(nb)
            queue.append(nb)
            if len(order) >= n_mem:
                break
    mids = order[:n_mem]
    kept = {here, *mids}
    kept_edges = [(a, b) for a, b, _ in edges if a in kept and b in kept]
    gate = min(kept, key=lambda i: math.hypot(nodes[i]["x"] - elev_xz[0],
                                              nodes[i]["z"] - elev_xz[1]))
    return here, mids, kept_edges, gate


# 기억 노드 뭉치의 배치(도식). 실제 월드 좌표를 그대로 쓰면 노드가 별의
# 좌우로 흩어져 "이미 아는 구역 → 현재 위치 → 앞의 후보"라는 읽는 순서가
# 만들어지지 않는다. Fig 2는 지도가 아니라 자료구조 그림이므로 배치를 정한다.
CLUSTER_C = (-3.3, 0.05)
CLUSTER_OFF = ((-1.05, -0.70), (0.10, -1.20), (1.05, -0.45))
ELEV_AT = (-3.5, -2.75)
FAN_DEG = 52.0
FAN_R = (1.75, 3.05)


def layout(n_mem, cand, drive_i, pose):
    """도식 좌표를 만든다.

    기억 노드는 왼쪽 뭉치, 현재 노드는 그 오른쪽 끝, 후보는 더 오른쪽으로
    부채꼴이다 — 주행이 왼쪽에서 오른쪽으로 읽힌다. 후보의 좌우 순서와 상대
    거리는 실제 값에서 가져오므로(방위각을 부채꼴에 사상) fig2_4의 화면 배치와
    대응이 유지된다.
    """
    mem = [(CLUSTER_C[0] + dx, CLUSTER_C[1] + dy)
           for dx, dy in CLUSTER_OFF[:n_mem]]
    here = (0.0, 0.0)
    bear = [math.atan2(-(c["z"] - pose[2]), c["x"] - pose[1]) for c in cand]
    mid = math.atan2(sum(math.sin(b) for b in bear),
                     sum(math.cos(b) for b in bear))
    dev = [(b - mid + math.pi) % (2 * math.pi) - math.pi for b in bear]
    lim = max(abs(d) for d in dev) or 1.0
    dist = [c["d"] for c in cand]
    lo, hi = min(dist), max(dist)
    pts = []
    for i, c in enumerate(cand):
        ang = math.radians(FAN_DEG) * dev[i] / lim
        r = FAN_R[0] + (FAN_R[1] - FAN_R[0]) * (
            (c["d"] - lo) / (hi - lo) if hi > lo else 0.5)
        pts.append((here[0] + r * math.cos(ang), here[1] + r * math.sin(ang)))
    return mem, here, pts, ELEV_AT


def cluster_edges(mem):
    """뭉치 안을 최소 신장 트리로 잇는다.

    최근접 2개씩 잇는 방식은 노드가 셋일 때 삼각형이 되어 먼 대각선이 생긴다.
    MST는 개수와 무관하게 연결되면서 불필요한 긴 엣지를 만들지 않는다.
    """
    if len(mem) < 2:
        return []
    inside, out = {0}, []
    while len(inside) < len(mem):
        best = min(((i, j) for i in inside
                    for j in range(len(mem)) if j not in inside),
                   key=lambda e: math.hypot(mem[e[0]][0] - mem[e[1]][0],
                                            mem[e[0]][1] - mem[e[1]][1]))
        out.append((min(best), max(best)))
        inside.add(best[1])
    return out


def draw_floor_graph(building, floor, cand, drive_i, pose, n_mem):
    """기억 그래프 뼈대 + 이번 프레임의 고스트·주행 노드 (도식 배치).

    로봇 포즈는 반드시 기억 노드 위에 있다 — 홉은 노드에서 노드로 일어나므로
    현재 위치는 그래프 밖의 자유점이 아니다. 그 자리에는 기억 노드 마커 대신
    별을 찍는다(두 마커가 겹치면 둘 다 안 읽힌다). 엘베는 기억 노드가 아니라
    자기 마커를 쓴다 — 전환 지점은 목적지가 아니라 수단이다.
    """
    mem, here, cpts, elev = layout(n_mem, cand, drive_i, pose)

    fig = plt.figure(figsize=(6.8, 4.2), dpi=300)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_aspect("equal")
    ax.axis("off")

    for i, j in cluster_edges(mem):
        ax.plot([mem[i][0], mem[j][0]], [mem[i][1], mem[j][1]],
                color=C_EDGE, lw=1.6, alpha=0.8, zorder=2)
    right = max(range(len(mem)), key=lambda i: mem[i][0])
    ax.plot([mem[right][0], here[0]], [mem[right][1], here[1]],
            color=C_EDGE, lw=1.6, alpha=0.8, zorder=2)
    low = min(range(len(mem)), key=lambda i: mem[i][1])
    ax.plot([mem[low][0], elev[0]], [mem[low][1], elev[1]],
            color=C_EDGE, lw=1.6, alpha=0.8, zorder=2)

    for i, (px, py) in enumerate(cpts):
        col = C_DRIVE if i == drive_i else C_GHOST_G
        ax.plot([here[0], px], [here[1], py], color=col, lw=1.6,
                ls=(0, (2.4, 2.2)), alpha=0.95, zorder=3)

    for x, y in mem:
        ax.scatter([x], [y], s=185, facecolor=C_MEM, edgecolor="white",
                   linewidth=1.8, zorder=5)
    ax.scatter([elev[0]], [elev[1]], s=300, marker="s", facecolor=C_ELEV,
               edgecolor="white", linewidth=1.8, zorder=6)
    for i, (px, py) in enumerate(cpts):
        drive = i == drive_i
        ax.scatter([px], [py], s=200 if drive else 160, facecolor="white",
                   edgecolor=C_DRIVE if drive else C_GHOST_G,
                   linewidth=2.4 if drive else 2.2, zorder=7)
    ax.scatter([here[0]], [here[1]], s=340, marker="*", facecolor=C_INK,
               edgecolor="white", linewidth=1.4, zorder=8)

    ax.legend(handles=[
        Line2D([], [], marker="*", ls="", mfc=C_INK, mec="white", ms=13,
               label="current node"),
        Line2D([], [], marker="o", ls="", mfc=C_MEM, mec="white", ms=9,
               label="memory node"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=C_GHOST_G, mew=1.9,
               ms=9, label="ghost node"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=C_DRIVE, mew=2.2,
               ms=10, label="selected node"),
        Line2D([], [], marker="^", ls="", mfc=C_STAIR, mec="white", ms=10,
               label="stairs"),
        Line2D([], [], marker="s", ls="", mfc=C_ELEV, mec="white", ms=9,
               label="elevator")],
        loc="upper left", fontsize=8.5, framealpha=0.94, ncol=3,
        handletextpad=0.4, columnspacing=0.9, borderpad=0.45)
    ax.margins(x=0.08, y=0.22)
    save(fig, "fig2_5_graph")
    print(f"   도식 그래프: 기억 {len(mem)} + 현재 1 + 후보 {len(cpts)} "
          f"+ 엘베 1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="public_hotel_dormitory_4f_300_fp001")
    ap.add_argument("--episode", default=None, help="지정 시 자동 선택 생략")
    ap.add_argument("--t", type=int, default=None)
    ap.add_argument("--robot", default="humanoid_humanoid_159",
                    help="프레임을 고를 로봇 (카메라 높이가 로봇마다 다름)")
    ap.add_argument("--mem-nodes", type=int, default=3,
                    help="그래프에 남길 기억 노드 수")
    ap.add_argument("--all-floors", action="store_true",
                    help="층별 적층 그래프(구판)를 대신 그린다")
    ap.add_argument("--res", type=int, default=1280,
                    help="표시용 재렌더 해상도 (256이면 데이터셋 원본 사용)")
    args = ap.parse_args()

    # 카메라 높이는 로봇마다 다르다(실측 0.38~1.81 m). 낮으면 바닥이 화면을
    # 채워 방 맥락이 사라지므로 그림에는 높은 쪽을 쓴다.
    pat = (f"{DATA}/episodes/{args.building}_ep*_{args.robot}.npz"
           if args.robot else f"{DATA}/episodes/{args.building}_ep*.npz")
    if args.episode:
        f, t = args.episode, args.t
        z = np.load(f)
        cand = candidates(z["heat"][t], z["depth"][t].astype(np.float32),
                          z["pose"][t])
        nxt = z["pose"][t + 1]
        drive_i = int(np.argmin([math.hypot(c["x"] - nxt[1], c["z"] - nxt[2])
                                 for c in cand]))
    else:
        got = pick_frame(pat)
        if got is None:
            raise SystemExit("조건을 만족하는 프레임 없음")
        _, f, t, cand, drive_i = got
    z = np.load(f)
    print(f"프레임: {os.path.basename(f)} t={t}")
    print(f"  후보 {len(cand)}개 / 주행 노드 = #{drive_i}")
    for i, c in enumerate(cand):
        tag = "주행" if i == drive_i else "고스트"
        print(f"    [{i}] {tag} 셀=({c['u']},{c['v']}) "
              f"월드=({c['x']:.2f}, {c['z']:.2f}) 거리={c['d']:.2f}m")

    heat = z["heat"][t]
    if args.res > RES:
        rgb, dep = hires_frame(args.building, z["pose"][t], args.res)
    else:
        rgb, dep = z["rgb"][t], z["depth"][t].astype(np.float32)
    print(f"  표시 해상도 {rgb.shape[0]}²")
    draw_rgb(rgb)
    hi = draw_depth(dep)
    print(f"  depth 표시 범위 0 ~ {hi:.2f} m")
    draw_heat(rgb, heat)
    draw_candidates(rgb, heat, cand, drive_i)

    if args.all_floors:
        robot = os.path.basename(f).split("_ep")[1].split("_", 1)[1][:-4]
        tp = [p for p in sorted(glob.glob(f"{DATA}/topomap/"
                                          f"{args.building}_*.json"))
              if "elevonly" not in p]
        hit = [p for p in tp if robot and robot in p] or tp
        with open(hit[0]) as fh:
            draw_graph(json.load(fh), robot)
    else:
        draw_floor_graph(args.building, z["pose"][t][0], cand, drive_i,
                         z["pose"][t], args.mem_nodes)


if __name__ == "__main__":
    main()

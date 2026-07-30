"""Fig 1 합성 — 렌더·경로 캐시를 읽어 최종 그림을 그린다.

실사 렌더 위에 두 로봇의 실제 계획 경로를 얹은 한 장짜리 그림이다. 두 로봇은
같은 지시와 같은 출발점을 받았고 다른 것은 몸(URDF)뿐이므로, 갈라지는 두 선
자체가 embodiment가 경로를 바꾼다는 주장이 된다.

경로선은 흰 테두리를 깔고 그 위에 색선을 얹는다. 실사 배경은 가구 색이 제각각
이라 단색 선만으로는 구간에 따라 묻힌다.
"""

import json
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon

INSTRUCTION = "Go to the kitchen on the third floor"

C_ELEV = "#2563EB"
C_STAIR = "#EA580C"
C_SHARED = "#475569"
C_INK = "#0F172A"
C_PAPER = "#FFFFFF"
C_MUTE = "#64748B"

# 화면에 담을 월드 영역. 계단·엘베(z0~2)에서 출발점 리셉션(z8.5)까지의 복도와
# 그 양옆 벽이 들어가는 상자다. 렌더 깊이가 20m에서 잘려 하늘을 깊이로는 못
# 가려내므로, 이 상자를 투영한 사각형으로 잘라 하늘을 화면 밖으로 보낸다.
ROI_X = (4.6, 10.6)
ROI_Z = (0.5, 8.5)
ROI_Y = (0.0, 3.1)

# 계단·엘베 방 중심. 라벨 지시선의 끝점으로 쓴다.
ROOM_MARK = {"STAIRS": (6.0, 1.6, C_STAIR), "ELEVATOR": (8.0, 1.6, C_ELEV)}

# 계단실·엘베는 z 0~2를 차지한다. 로봇은 이 선 바깥(복도 쪽)에 세운다.
ROOM_FRONT_Z = 2.5

# 크롭 가로:세로 하한. 내용만 감싸면 정사각에 가까워져 본문 폭에서 작아진다.
TARGET_ASPECT = 1.52

# 로봇은 공개 description 패키지의 메시 그대로 렌더한다. 색을 입히지 않는 대신
# 발밑에 경로 색 원반을 깔아 어느 선이 어느 로봇의 것인지 잇는다.
# yaw는 문을 향해 선 것처럼 보이는 각도다.
ROBOT_MULT = 1.8
ROBOT_ART = {
    "turtlebot3_waffle": {"color": C_ELEV},
    "unitree_go2": {"color": C_STAIR,
                    "stance": {"thigh": 0.72, "calf": -1.45}},
}


def load_cache(cache):
    import cv2

    img = cv2.imread(os.path.join(cache, "f1_bev.png"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dep = np.load(os.path.join(cache, "f1_bev_depth.npy"))
    with open(os.path.join(cache, "f1_bev_cam.json")) as f:
        cam = json.load(f)
    with open(os.path.join(cache, "paths.json")) as f:
        paths = json.load(f)
    return img, dep, cam, paths


def make_projector(cam):
    """fig1_system_overview의 투영식과 같은 규약. 순환 임포트를 피하려고
    카메라 기저를 여기서 다시 만든다."""
    pos = np.array(cam["pos"], float)
    d = np.array(cam["target"], float) - pos
    yaw = math.atan2(d[0], d[2])
    pitch = math.atan2(-d[1], math.hypot(d[0], d[2]))
    cy, sy, cp, sp = (math.cos(yaw), math.sin(yaw),
                      math.cos(pitch), math.sin(pitch))
    right = np.array([cy, 0.0, -sy])
    up = np.array([sy * sp, cp, cy * sp])
    fwd = np.array([sy * cp, -sp, cy * cp])
    f = (cam["height"] / 2.0) / math.tan(math.radians(cam["fov"]) / 2.0)

    def w2p(x, z, y=0.06):
        rel = np.array([x, y, z], float) - pos
        zc = float(rel @ fwd)
        if zc <= 1e-3:
            return None
        return (cam["width"] / 2.0 + f * float(rel @ right) / zc,
                cam["height"] / 2.0 - f * float(rel @ up) / zc)

    def depth_of(x, z, y=0.06):
        return float((np.array([x, y, z], float) - pos) @ fwd)

    w2p.depth_of = depth_of
    return w2p


def split_runs(w2p, poly, dep, tol=0.25):
    """경로를 (구간, 보임여부) 목록으로 쪼갠다.

    오버레이는 깊이를 모르므로 그냥 그리면 벽이나 가구 앞을 지나간다. 렌더
    깊이가 그 점까지의 거리보다 뚜렷하게 가까우면 그 점은 무언가 뒤에 있다.
    가려진 구간을 버리면 이미 주행한 길이 실제보다 짧아 보이므로, 버리지 않고
    가는 점선으로 남긴다 — 도면에서 은선을 다루는 방식과 같다.
    """
    h, w = dep.shape
    out, cur, cur_vis = [], [], None
    for x, z in poly:
        uv = w2p(x, z)
        if uv is None:
            continue
        u, v = int(round(uv[0])), int(round(uv[1]))
        if not (0 <= u < w and 0 <= v < h):
            continue
        vis = float(dep[v, u]) > w2p.depth_of(x, z) - tol
        if cur_vis is None or vis == cur_vis:
            cur.append([uv[0], uv[1]])
            cur_vis = vis
        else:
            out.append((np.array(cur, float), cur_vis))
            cur, cur_vis = [cur[-1], [uv[0], uv[1]]], vis
    if len(cur) > 1:
        out.append((np.array(cur, float), cur_vis))
    return out


def floor_polyline(rec, floor=1):
    """한 로봇의 지정 층 구간을 이어붙인 월드 폴리라인."""
    pts = []
    for seg in rec["segments"]:
        if seg["floor"] != floor:
            continue
        for p in seg["pts"]:
            if not pts or abs(p[0] - pts[-1][0]) + abs(p[1] - pts[-1][1]) > 1e-6:
                pts.append([float(p[0]), float(p[1])])
    return np.array(pts, float)


def split_shared(a, b, tol=0.05):
    """두 폴리라인의 공통 앞부분과 각자의 분기 뒷부분으로 나눈다."""
    n = 0
    while n < min(len(a), len(b)) and np.hypot(*(a[n] - b[n])) < tol:
        n += 1
    return a[:n], a[max(n - 1, 0):], b[max(n - 1, 0):]


def to_px(w2p, poly):
    out = [w2p(x, z) for x, z in poly]
    return np.array([p for p in out if p is not None], float)


def draw_path(ax, px, color, lw=5.0, z=6, visible=True):
    """보이는 구간은 굵은 실선(흰 테두리), 가려진 구간은 가는 점선."""
    if len(px) < 2:
        return
    if visible:
        ax.plot(px[:, 0], px[:, 1], color="white", lw=lw + 2.4,
                solid_capstyle="round", solid_joinstyle="round", zorder=z)
        ax.plot(px[:, 0], px[:, 1], color=color, lw=lw,
                solid_capstyle="round", solid_joinstyle="round", zorder=z + 1)
    else:
        ax.plot(px[:, 0], px[:, 1], color="white", lw=lw * 0.52,
                alpha=0.55, zorder=z)
        ax.plot(px[:, 0], px[:, 1], color=color, lw=lw * 0.34,
                ls=(0, (2.6, 2.2)), alpha=0.95, zorder=z + 1)


def place_robot(ax, robot_id, at_px, scene_ppm, pts_per_px, yaw, color,
                stance=None, pitch=35.0, mult=1.0):
    """메시 URDF에서 렌더한 로봇을 지정 화면 위치에 얹는다.

    pybullet 렌더가 담은 실제 높이(픽셀÷ppm)를 그 지점의 씬 배율로 환산해
    크기를 정하므로 로봇이 장면과 같은 축척으로 놓인다. mult는 논문 판형에서
    읽히도록 실치수보다 키우는 배율이다 — Go2가 0.4 m라 실축이면 너무 작다.
    발밑 원반은 로봇이 바닥에 붙어 보이게 하면서 경로 색과 로봇을 잇는다.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    import urdf_render as ur

    img, ppm = ur.render(robot_id, pitch_deg=pitch, yaw_deg=yaw, stance=stance)
    img = ur.outlined(img, width=3)
    h_data = (img.shape[0] / ppm) * scene_ppm * mult
    ax.add_patch(Ellipse(at_px, 0.95 * scene_ppm, 0.40 * scene_ppm,
                         fc=color, ec="white", lw=1.6, alpha=0.55,
                         zorder=13))
    ab = AnnotationBbox(
        OffsetImage(img, zoom=h_data * pts_per_px / img.shape[0],
                    interpolation="bilinear"),
        at_px, frameon=False, pad=0.0, box_alignment=(0.5, 0.10),
        zorder=14, annotation_clip=False)
    ax.add_artist(ab)


def label_chip(ax, x, y, text, color, fs=13, pad=0.42):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color="white", zorder=15,
            bbox=dict(boxstyle="round,pad=%f" % pad, fc=color, ec="white",
                      lw=1.8))


def bubble_path(x, y, w, h, r, tx, tw, tl):
    """말풍선 외곽을 꼬리까지 한 붓으로 잇는 점열.

    사각형과 삼각형을 따로 그리면 경계선이 남아 둘이 분리돼 보인다. 모서리는
    작은 호로 근사한다. 축의 y가 아래로 향하므로 h·tl은 아래쪽이 양수다.
    """
    def arc(cx, cy, a0, a1, n=8):
        return [(cx + r * math.cos(a), cy + r * math.sin(a))
                for a in np.linspace(a0, a1, n)]

    pts = arc(x + r, y + r, math.pi, 1.5 * math.pi)
    pts += arc(x + w - r, y + r, 1.5 * math.pi, 2 * math.pi)
    pts += arc(x + w - r, y + h - r, 0.0, 0.5 * math.pi)
    pts += [(x + tx + tw, y + h), (x + tx + tw * 0.15, y + h + tl),
            (x + tx, y + h)]
    pts += arc(x + r, y + h - r, 0.5 * math.pi, math.pi)
    return pts


def speech_bubble(ax, text, box, pts_per_px):
    """왼쪽에서 건네는 지시 말풍선.

    긴 지시선으로 특정 지점을 찌르면 장면을 가로질러 어수선해진다. 화면 왼편에
    붙이고 아래로 짧은 꼬리만 내어 '왼쪽에서 말한다'는 것만 나타낸다.
    글자 크기는 상자 폭에서 역산해 상자를 넘치지 않게 한다.
    """
    x0, x1, y0, y1 = box
    w, h = x1 - x0, y1 - y0
    bw = 0.545 * w
    fs = min(0.92 * bw * pts_per_px / (0.485 * len(text)), 26.0)
    bh = 2.35 * fs / pts_per_px
    bx, by = x0 + 0.030 * w, y0 + 0.048 * h
    pts = bubble_path(bx, by, bw, bh, 0.30 * bh,
                      0.085 * bw, 0.075 * bw, 0.62 * bh)
    ax.add_patch(Polygon(pts, closed=True, fc=C_PAPER, ec=C_INK, lw=2.0,
                         joinstyle="round", zorder=17))
    ax.text(bx + bw / 2, by + bh / 2, text, ha="center", va="center",
            fontsize=fs, fontstyle="italic", color=C_INK, zorder=18)


def legend_box(ax, box):
    """로봇 두 대를 한 줄씩 적는 범례. 이름 자체를 경로 색으로 칠하므로 색
    견본 선은 없앴고, 계단 통행 가부만 덧붙여 분기의 원인이 읽히게 한다."""
    x0, x1, y0, y1 = box
    w, h = x1 - x0, y1 - y0
    bx, by = x0 + 0.018 * w, y1 - 0.112 * h
    bw, bh = 0.225 * w, 0.092 * h
    ax.add_patch(FancyBboxPatch(
        (bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=%f" % (0.008 * w),
        fc="white", ec=C_INK, lw=1.4, alpha=0.94, zorder=18))
    rows = ((0.30, C_ELEV, "TurtleBot 3", "wheeled  ·  stairs \u2717"),
            (0.70, C_STAIR, "Unitree Go2", "quadruped  ·  stairs \u2713"))
    for fy, col, name, note in rows:
        cy = by + fy * bh
        ax.text(bx + 0.050 * bw, cy, name, fontsize=9.0, fontweight="bold",
                color=col, va="center", ha="left", zorder=19)
        ax.text(bx + 0.950 * bw, cy, note, fontsize=7.8, color=C_MUTE,
                va="center", ha="right", zorder=19)


def roi_box(w2p, cam, anchors, margin=0.13, extend_down=0.55):
    """그림이 말하려는 것들만 감싸는 화면 사각형.

    상자는 **성긴 기준점**(두 문·두 로봇 자리·분기점)으로만 잡는다. 조밀한
    경로점을 넣으면 카메라 코앞 구간이 표본을 지배해 화면이 그쪽으로 확대되고,
    최소·최대를 쓰면 반대로 화면 밖 수천 픽셀까지 벌어진다(실측 v 최대 12085,
    이미지 높이 1400). 이미 주행한 복도는 아래로만 늘려 담는다.
    """
    arr = np.array([p for p in anchors if p is not None], float)
    x0, x1 = arr[:, 0].min(), arr[:, 0].max()
    y0, y1 = arr[:, 1].min(), arr[:, 1].max()
    y1 += extend_down * (y1 - y0)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    if (x1 - x0) / (y1 - y0) < TARGET_ASPECT:
        # 양쪽으로 고르게 넓힌다. 예전에는 오른쪽이 파란 타일벽이라 왼쪽으로
        # 치우쳐 넓혔는데, 벽 재질을 평탄화한 뒤로는 그럴 이유가 없고 오히려
        # 왼쪽 소파 구역이 화면의 절반을 먹었다.
        need = TARGET_ASPECT * (y1 - y0) - (x1 - x0)
        x0, x1 = x0 - need * 0.42, x1 + need * 0.58
    return (max(x0, 0.0), min(x1, cam["width"] - 1),
            max(y0, 0.0), min(y1, cam["height"] - 1))


def stand_point(poly, z_front=ROOM_FRONT_Z):
    """경로에서 계단·엘베 방을 벗어나 복도에 남는 마지막 점.

    경로의 마지막 점은 방 안의 전환 앵커다. 거기 세우면 로봇이 문턱을 넘어
    방 안에 들어가 있거나 문짝에 겹친다. 거리로 되짚으면 방 깊이가 제각각이라
    또 안쪽에 걸리므로, 방 경계(z=2) 바깥이라는 조건으로 잡는다.
    """
    p = np.array(poly, float)
    for i in range(len(p) - 1, -1, -1):
        if p[i][1] >= z_front:
            return p[i]
    return p[-1]


def facing_yaw(poly, cam_yaw, at):
    """로봇이 진행 방향(=문 쪽)을 보도록 하는 pybullet 카메라 yaw.

    실측한 규약: pybullet yaw 270이 카메라 정반대(등이 보임), 90이 정면,
    0이 화면 오른쪽이다. 그러므로 월드 진행 방향과 카메라 방위의 차이를
    270에 더하면 된다. 고정값으로 두면 카메라를 옮길 때마다 어긋난다.
    """
    p = np.array(poly, float)
    i = int(np.argmin(np.hypot(p[:, 0] - at[0], p[:, 1] - at[1])))
    j = min(i + 8, len(p) - 1)
    d = p[j] - p[i]
    if np.hypot(*d) < 1e-6:
        d = p[-1] - p[i]
    head = math.degrees(math.atan2(d[0], d[1]))
    return (270.0 + head - cam_yaw) % 360.0


def px_per_m(w2p, x, z):
    """그 지점에서 월드 1m가 몇 화소인지. 원근이라 위치마다 다르므로 아이콘·
    말풍선 크기를 여기에 맞춰야 장면과 어울린다."""
    a, b = w2p(x, z), w2p(x + 1.0, z)
    return float(np.hypot(b[0] - a[0], b[1] - a[1])) if a and b else 40.0


def run(cache, out_dir):
    img, dep, cam, paths = load_cache(cache)
    w2p = make_projector(cam)
    tb = paths["robots"]["turtlebot3_waffle"]
    go = paths["robots"]["unitree_go2"]
    a, b = floor_polyline(tb), floor_polyline(go)
    shared, tail_tb, tail_go = split_shared(a, b)

    # 축 상자를 크롭 비율에 정확히 맞춰 배치한다. gridspec에 맡기면 imshow가
    # 화면비를 지키느라 상자 안에 여백을 남기고, 그 여백 위에 얹힌 주석이
    # 이미지 밖으로 밀려난다.
    # 화면 기준점 = 두 문(바닥~라벨 높이), 로봇이 설 자리, 경로 분기점
    anchors = [w2p(rx, rz, ry) for rx, rz, _ in ROOM_MARK.values()
               for ry in (0.0, 2.75)]
    anchors += [w2p(*stand_point(p_)) for p_ in (tail_tb, tail_go) if len(p_)]
    if len(shared):
        anchors.append(w2p(*shared[-1]))
    box = roi_box(w2p, cam, anchors)
    x0, x1, y0, y1 = box
    h_in, pad = 7.6, 0.006
    ax_h = h_in * (1 - 2 * pad)
    total = ax_h * (x1 - x0) / (y1 - y0)
    fig = plt.figure(figsize=(total, h_in), dpi=200)
    ax = fig.add_axes([0.0, pad, 1.0, 1 - 2 * pad])
    ax.imshow(img)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")

    for poly, col, lw in ((shared, C_SHARED, 6.0), (tail_tb, C_ELEV, 6.0),
                          (tail_go, C_STAIR, 6.0)):
        for run_px, vis in split_runs(w2p, poly, dep):
            draw_path(ax, run_px, col, lw=lw, visible=vis)

    for name, (rx, rz, col) in ROOM_MARK.items():
        p_ = w2p(rx, rz, 2.35)
        if p_:
            label_chip(ax, p_[0], p_[1], name, col)

    # 로봇은 각자 고른 전환 지점(엘베 문 앞·계단 입구)에 세운다. 경로선은 층
    # 평면 위의 2D라 계단을 오르는 모습까지는 담지 못하므로, 오르기 직전
    # 자리에 두어 선이 끝나는 곳과 로봇이 선 곳을 일치시킨다.
    pts_per_px = (total * 72.0) / (x1 - x0)
    for rid, poly in (("turtlebot3_waffle", tail_tb),
                      ("unitree_go2", tail_go)):
        spec = ROBOT_ART[rid]
        if not len(poly):
            continue
        wx, wz = stand_point(poly)
        at = w2p(wx, wz)
        if at is None:
            continue
        place_robot(ax, rid, at, px_per_m(w2p, wx, wz), pts_per_px,
                    yaw=facing_yaw(poly, cam["yaw"], (wx, wz)),
                    color=spec["color"],
                    stance=spec.get("stance"), pitch=cam["pitch"],
                    mult=ROBOT_MULT)

    speech_bubble(ax, INSTRUCTION, box, pts_per_px)
    legend_box(ax, box)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fp = os.path.join(out_dir, f"fig1_system_overview.{ext}")
        fig.savefig(fp, dpi=300, facecolor="white")
        print(f"저장: {fp}")
    plt.close(fig)

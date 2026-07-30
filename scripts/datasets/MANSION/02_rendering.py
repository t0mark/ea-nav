"""렌더링 파일럿 (Phase 2): 게이트 스냅샷 + 에피소드 RGB-D 시퀀스.

렌더 사양(확정): 전방 RGB-D 단일 뷰, HFOV 90°, 256²로 렌더(팩킹 시 RGB만
224²로 축소), 카메라 = URDF sensor_rgb 실높이(cam_h), GPU 렌더
(launch_controller(render=True), 서드파티 캠 1개를 Update로 이동).

① 게이트 스냅샷: 게이트당 거리 {1,2,3m} × 각도 {−30°,0°,+30°} = 9포즈,
   대표 로봇 3대(소/중/대)의 cam_h로 각각 렌더 → 로봇별 통과 라벨 테두리
   (초록=통과/빨강=불가)를 입힌 카드. "같은 게이트, 다른 관측·다른 라벨" 검수.
② 에피소드 시퀀스: legged 층 이동 에피소드를 0.5m 간격 프레임으로 렌더
   (계단 등반은 climb (iz,ix,y) 폴리라인 + cam_h 합성 높이), 다음 waypoint
   (1m 전방 경로점)를 핀홀 투영해 마커 표시 → 프레임 그리드 카드.

실행:
  docker exec airlab_hw_mansion bash -c 'cd /workspace/research/scripts/datasets/MANSION \
    && python 02_rendering.py --building "/data/MansionWorld/mansionworld/<빌딩>#0" \
    --out /workspace/research/check/MANSION/02_rendering'
"""

import argparse
import json
import math
import os

import cv2
import numpy as np

import common

RES = common.RES              # 렌더 해상도 (depth 규격, RGB는 팩킹 시 224로)
FX = common.FX                # HFOV 90° 핀홀 초점거리(px)
SNAP_DISTS = (1.0, 2.0, 3.0)
SNAP_ANGS = (-30.0, 0.0, 30.0)
FRAME_STEP = 1.5              # 에피소드 프레임 간격 (m) — 사용자 확정
WAYPOINT_AHEAD = 1.0          # waypoint GT = 경로 1m 전방 지점

# 렌더 헬퍼는 03_topomap과 공유 — common으로 이관 (별칭 유지)
yaw_deg = common.yaw_deg
update_cam = common.update_cam
project = common.project


def approach_dir(geom, g):
    """게이트 접근 방향 (자유 공간이 넓은 쪽)."""
    tiny = geom.passable(common.BELIEF_W, 0.4)
    best, bestn = (1.0, 0.0), -1
    for a in range(0, 360, 30):
        dx, dz = math.sin(math.radians(a)), math.cos(math.radians(a))
        n = 0
        for d in np.arange(0.5, 3.0, 0.25):
            iz, ix = geom.to_idx(g["x"] + dx * d, g["z"] + dz * d)
            if 0 <= iz < geom.nz and 0 <= ix < geom.nx and tiny[iz, ix]:
                n += 1
        if n > bestn:
            bestn, best = n, (dx, dz)
    return best


def pick_gates(geom, robots, n):
    """갈림 게이트 우선으로 n개 (문·협착 섞어서)."""
    gates = common.floor_gates(geom)
    scored = []
    for g in gates:
        labels = [min(common.soft_label(g["width"] - r["w_eff"]),
                      common.soft_label(g["height"] - r["h"])) for r in robots]
        split = (max(labels) > 0.9) and (min(labels) < 0.1)
        scored.append((split, g))
    scored.sort(key=lambda t: (not t[0], t[1]["width"]))
    out, seen_types = [], []
    for split, g in scored:
        if len(out) >= n:
            break
        out.append(g)
        seen_types.append(g["type"])
    return out


def snapshot_cards(c, geom, gates, reps, out_dir, building_dir):
    os.makedirs(out_dir, exist_ok=True)
    for gi, g in enumerate(gates):
        dx, dz = approach_dir(geom, g)
        base_yaw = yaw_deg(-dx, -dz)  # 게이트를 바라보는 방향
        rows = []
        for r in reps:
            lab = min(common.soft_label(g["width"] - r["w_eff"]),
                      common.soft_label(g["height"] - r["h"]))
            color = (0, 200, 0) if lab >= 0.5 else (0, 0, 255)
            tiles = []
            for d in SNAP_DISTS:
                for a in SNAP_ANGS:
                    ra = math.radians(a)
                    ax = dx * math.cos(ra) - dz * math.sin(ra)
                    az = dx * math.sin(ra) + dz * math.cos(ra)
                    cx, cz = g["x"] + ax * d, g["z"] + az * d
                    cyaw = yaw_deg(g["x"] - cx, g["z"] - cz)
                    rgb, _ = update_cam(c, cx, r["cam_h"], cz, cyaw)
                    uv = project((g["x"], min(g["height"], 1.2) / 2, g["z"]),
                                 (cx, r["cam_h"], cz), cyaw)
                    if uv:  # 판정 대상 게이트(틈) 위치를 타일에 표시
                        cv2.circle(rgb, uv, 14, (0, 165, 255), 2)
                    cv2.rectangle(rgb, (0, 0), (RES - 1, RES - 1), color, 4)
                    cv2.putText(rgb, f"d{d:.0f} a{a:+.0f}", (6, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                    tiles.append(rgb)
            row = np.concatenate(tiles, axis=1)
            bar = np.zeros((26, row.shape[1], 3), dtype=np.uint8)
            cv2.putText(bar, f"{r['id']} cam_h={r['cam_h']:.2f} w_eff="
                        f"{r['w_eff']:.2f} h={r['h']:.2f} label={lab:.2f}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            rows += [bar, row]
        card = np.concatenate(rows, axis=0)
        head = np.zeros((28, card.shape[1], 3), dtype=np.uint8)
        cv2.putText(head, f"gate {gi}: {g['type']} F{g['floor']} "
                    f"w={g['width']:.2f}m h={g['height']:.2f}m "
                    f"({g['x']:.1f},{g['z']:.1f}) | passability tuple: "
                    "orange circle=target gap, row=robot at own cam height, "
                    "border green=pass red=fail", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        # 탑뷰 크롭: 게이트 ±4m, 게이트 원 + 카메라 9포즈 점
        timg, w2p = common.sim_view(geom, building_dir)
        cams = []
        for d in SNAP_DISTS:
            for a in SNAP_ANGS:
                ra = math.radians(a)
                cams.append((g["x"] + (dx * math.cos(ra) - dz * math.sin(ra)) * d,
                             g["z"] + (dx * math.sin(ra) + dz * math.cos(ra)) * d))
        gu, gv = w2p(g["x"], g["z"])
        cv2.circle(timg, (gu, gv), 10, (0, 165, 255), 3)
        for cx_, cz_ in cams:
            cv2.circle(timg, w2p(cx_, cz_), 5, (255, 255, 255), -1)
        u0, v0 = w2p(g["x"] - 4, g["z"] + 4)
        u1, v1 = w2p(g["x"] + 4, g["z"] - 4)
        H, W = timg.shape[:2]
        crop = timg[max(0, v0):min(H, v1), max(0, u0):min(W, u1)]
        full = np.concatenate([head, card], axis=0)
        sc = full.shape[0] / crop.shape[0]
        crop = cv2.resize(crop, None, fx=sc, fy=sc,
                          interpolation=cv2.INTER_AREA)
        full = np.concatenate([crop, full], axis=1)
        path = os.path.join(out_dir, common.out_name(
            building_dir, f"gate{gi}_{g['type']}.png"))
        cv2.imwrite(path, full)
        print(f"게이트 카드: {path}")


def topview_montage(floors, robot, res, building_dir, frame_poses, height,
                    goal=None):
    """에피소드 탑뷰 몽타주: 경로·전환·판정 + 프레임 촬영 위치(흰 점).

    실패 에피소드의 "어디가 막혔나"를 표시(사용자 요청): 로봇이 도달 가능한
    영역을 초록 틴트로 칠하고(목표가 그 밖이면 한눈에 보임), 판정된 게이트를
    빨간 원 + "폭 < 로봇 폭" 수치로 표시.
    """
    fls = sorted({fl for fl, _ in res.get("segments", [])})
    if goal is not None and goal[0] not in fls:
        fls.append(goal[0])
    start_cell = (res["segments"][0][1][0] if res.get("segments") else None)
    start_fl = res["segments"][0][0] if res.get("segments") else None
    panels = []
    for no in fls:
        geom = floors[no]
        img, w2p = common.sim_view(geom, building_dir)
        # 도달 가능 영역 틴트 (시작 층에서만, 시작 성분)
        if start_cell is not None and no == start_fl:
            wb = round(robot["w_eff"] / 0.05) * 0.05
            hb = round(robot["h"] / 0.1) * 0.1
            ok = geom.passable(wb, hb).astype(np.uint8)
            nlab, lab = cv2.connectedComponents(ok)
            comp = (lab == lab[start_cell]) & (lab[start_cell] > 0)
            uu, vv = np.meshgrid(np.arange(geom.nx), np.arange(geom.nz))
            pts = np.array([w2p(*geom.to_world(z, x))
                            for z, x in zip(vv[comp][::9], uu[comp][::9])])
            H, W = img.shape[:2]
            mask = np.zeros((H, W), np.uint8)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W)
                      & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
            mask[pts[:, 1], pts[:, 0]] = 1
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
            img[mask > 0] = (img[mask > 0] * 0.6
                             + np.array((60, 200, 60)) * 0.4).astype(np.uint8)
        # 판정 게이트: 빨간 원 + 수치
        for (gfl, g), (vfl, vd) in zip(res.get("verdict_gates", []),
                                       res.get("verdicts", [])):
            if gfl != no:
                continue
            u, v = w2p(g["x"], g["z"])
            cv2.circle(img, (u, v), 14, (0, 0, 255), 3)
            cv2.putText(img, f"{g['width']:.2f}m<{robot['w_eff']:.2f}m",
                        (u + 16, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            # 판정 지점 → 게이트 점선 + 거리: 남은 간격이 "더는 물리적으로
            # 접근 불가"한 구간임을 카드에서 읽히게(사용자 지적)
            vx, vz = geom.to_world(*vd)
            vu, vv = w2p(vx, vz)
            dist_m = math.hypot(g["x"] - vx, g["z"] - vz)
            n_dash = max(2, int(math.hypot(u - vu, v - vv) / 12))
            for t0 in np.linspace(0, 1, n_dash * 2)[::2]:
                p1 = (int(vu + (u - vu) * t0), int(vv + (v - vv) * t0))
                p2 = (int(vu + (u - vu) * (t0 + 0.5 / n_dash)),
                      int(vv + (v - vv) * (t0 + 0.5 / n_dash)))
                cv2.line(img, p1, p2, (0, 0, 255), 2)
            cv2.putText(img, f"stop {dist_m:.1f}m before (closest "
                        "reachable)", (vu + 12, vv + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if goal is not None and goal[0] == no:
            u, v = w2p(*geom.to_world(*goal[1:]))
            cv2.drawMarker(img, (u, v), (255, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 26, 3)
            cv2.putText(img, "GOAL", (u + 14, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if start_cell is not None and no == start_fl:
            u, v = w2p(*geom.to_world(*start_cell))
            cv2.putText(img, "START", (u + 14, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.drawMarker(img, (u, v), (255, 255, 255), cv2.MARKER_STAR,
                           22, 2)
        for fl, cells in res["segments"]:
            if fl != no:
                continue
            sm = common.natural_path(geom, robot, cells)
            pts = common.cells_to_px(geom, sm, w2p)
            cv2.polylines(img, [np.array(pts, np.int32)], False,
                          (200, 60, 200), 3, cv2.LINE_AA)
        common.draw_transitions(img, geom, w2p, res.get("transitions", []),
                                (200, 60, 200))
        common.draw_climb(img, geom, w2p, res.get("transitions", []),
                          (200, 60, 200))
        for vfl, vd in res.get("verdicts", []):
            if vfl == no:
                u, v = w2p(*geom.to_world(*vd))
                cv2.drawMarker(img, (u, v), (0, 255, 255),
                               cv2.MARKER_DIAMOND, 20, 3)
        for fl, x, z in frame_poses:
            if fl != no:
                continue
            u, v = w2p(x, z)
            cv2.circle(img, (u, v), 6, (255, 255, 255), -1)
            cv2.circle(img, (u, v), 6, (0, 0, 0), 2)
        cv2.putText(img, f"F{no}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        panels.append(img)
    if not panels:
        return None
    mont = np.concatenate(panels, axis=1)
    sc = height / mont.shape[0]
    return cv2.resize(mont, None, fx=sc, fy=sc,
                      interpolation=cv2.INTER_AREA)


def episode_frames(c, floors, robot, res, out_png, building_dir,
                   allow_fail=False, goal=None):
    """expert 결과(res)를 프레임으로 렌더 → 스트립 카드 저장.

    부분 궤적(실패 에피소드)·판정 지점(빨간 테두리 VERDICT 프레임)·엘리베이터
    대기중(WAITING 정지 프레임 3장)을 지원한다 — 케이스 전수 렌더 검증용.
    """
    if not res.get("segments"):
        print("세그먼트 없음:", res.get("reason"))
        return []
    verd = {(fl, tuple(v) if not isinstance(v, tuple) else v)
            for fl, v in res.get("verdicts", [])}
    vgate = {(fl, tuple(v)): g for (fl, v), (_, g)
             in zip(res.get("verdicts", []), res.get("verdict_gates", []))}
    climbs = {min(t["floor"], t["to"]): t["climb"]
              for t in res.get("transitions", []) if t["mode"] == "stairs"}
    elev_waits = {(t["floor"], tuple(t["from_cell"])): t.get("from_door")
                  for t in res.get("transitions", [])
                  if t["mode"] == "elevator" and t.get("from_cell")}
    # 내부 대기(궤적 레벨): 진입 세그먼트 끝(카 중심)에서 hold — 사용자 지시
    elev_holds = {(t["floor"], tuple(t["wait_cell_in"])):
                  (t.get("from_door"), t.get("wait_steps", 3))
                  for t in res.get("transitions", [])
                  if t["mode"] == "elevator" and t.get("wait_cell_in")}
    # 탑승/진출 세그먼트: 문을 열고(도어 오브젝트 비활성) 통과해야 함 —
    # 닫힌 문을 그대로 지나가는 프레임이 생김(사용자 지적)
    entry_ends = {(t["floor"], tuple(t["wait_cell_in"]))
                  for t in res.get("transitions", [])
                  if t["mode"] == "elevator" and t.get("wait_cell_in")}
    exit_starts = {(t["to"], tuple(t["wait_cell_out"]))
                   for t in res.get("transitions", [])
                   if t["mode"] == "elevator" and t.get("wait_cell_out")}

    def elev_door_ids():
        ev = c.step(action="Pass")
        return [o["objectId"] for o in ev.metadata["objects"]
                if o["objectId"].startswith("elevator_doors")]

    def set_doors(open_):
        for oid in elev_door_ids():
            c.step(action="DisableObject" if open_ else "EnableObject",
                   objectId=oid)
    frames, frame_poses, special_idx = [], [], []
    # 시선 look-ahead — 제안기 GT가 히트맵으로 바뀌어 "보이는 단일 지점까지
    # 당기기(walk-back)"는 폐기. 이 거리는 카메라가 향할 방향 결정에만 쓴다.
    wp_far = FRAME_STEP
    cur_fl = None
    for fl, cells in res["segments"]:
        geom = floors[fl]
        if fl != cur_fl:
            common.load_floor_scene(c, geom.scene)
            c.step(action="AddThirdPartyCamera",
                   position=dict(x=0, y=1, z=0),
                   rotation=dict(x=0, y=0, z=0), fieldOfView=90)
            cur_fl = fl
        sm = common.natural_path(geom, robot, cells)
        boarding = (fl, tuple(sm[-1])) in entry_ends
        exiting = (fl, tuple(sm[0])) in exit_starts
        if boarding or exiting:
            set_doors(open_=True)  # 통과 구간 동안 문 열림
        pts = [geom.to_world(iz, ix) for iz, ix in sm]
        # 실패 세그먼트: 정지점 너머 belief 잔여(게이트 틈→goal 방향)를
        # look-ahead 전용 폴리라인으로 연장. 카메라 위치는 실행 경로에서
        # 멈추되, 시선·heading GT는 로봇이 주행 중이라 믿었던 경로를 계속
        # 향한다 — 실행 경로 끝에서 클램프하면 마지막 프레임들의 GT가 goal
        # 지향성을 잃고 판정 프레임과 시선이 불연속(사용자 지적).
        la_pts = list(pts)
        bt = res.get("belief_tail")
        if bt and bt[0] == fl and bt[1] and cells is res["segments"][-1][1]:
            acc, prev = 0.0, pts[-1]
            for cz, cx in bt[1][::8]:
                w = geom.to_world(cz, cx)
                acc += math.hypot(w[0] - prev[0], w[1] - prev[1])
                prev = w
                la_pts.append(w)
                if acc > 4.0:
                    break
        # 등반 구간 여부: 셀이 계단 존 안이면 climb 높이 보간 사용
        climb = climbs.get(fl)
        cl_pts = ([(geom.to_world(cz, cx), y) for cz, cx, y in climb]
                  if climb else [])
        # 폴리라인 호길이 보간으로 FRAME_STEP 간격 등분 샘플 — DP 꼭짓점
        # 단위 샘플은 직선 구간에서 수 m에 한 장이 됨(사용자 지적: 노드 간격)
        seglens = [math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(pts[:-1], pts[1:])]
        total_len = sum(seglens)
        la_seglens = [math.hypot(b[0] - a[0], b[1] - a[1])
                      for a, b in zip(la_pts[:-1], la_pts[1:])]
        la_total = sum(la_seglens)

        def arc_at(tq):
            """look-ahead 폴리라인의 호길이 tq 지점 (wp/heading/판정 시선 공용)."""
            kq, kq_acc = 0, 0.0
            while kq < len(la_seglens) - 1 and kq_acc + la_seglens[kq] < tq:
                kq_acc += la_seglens[kq]
                kq += 1
            fq = (tq - kq_acc) / max(la_seglens[kq], 1e-9)
            return (la_pts[kq][0] + (la_pts[kq + 1][0] - la_pts[kq][0]) * fq,
                    la_pts[kq][1] + (la_pts[kq + 1][1] - la_pts[kq][1]) * fq)
        samples, si, s_acc = [], 0, 0.0
        t = 0.0
        while t <= total_len and seglens:
            while si < len(seglens) - 1 and s_acc + seglens[si] < t:
                s_acc += seglens[si]
                si += 1
            f = (t - s_acc) / max(seglens[si], 1e-9)
            a, b = pts[si], pts[si + 1]
            samples.append(((a[0] + (b[0] - a[0]) * f,
                             a[1] + (b[1] - a[1]) * f),
                            yaw_deg(b[0] - a[0], b[1] - a[1]), t))
            t += FRAME_STEP
        # 프레임 시선을 먼저 전부 구한다 — 도착 heading GT가 "다음 프레임의
        # 실제 heading"이라 현재 프레임에서 다음 값을 참조해야 한다
        def view_yaw(pos, seg_yaw, s_along):
            """시선 = 전방 waypoint 방향. 구간 접선은 랜딩 급회전에서 벽면
            응시 프레임을 만든다(사용자 버그 리포트)."""
            w = arc_at(min(s_along + wp_far, la_total))
            return (yaw_deg(w[0] - pos[0], w[1] - pos[1])
                    if math.hypot(w[0] - pos[0], w[1] - pos[1]) > 0.15
                    else seg_yaw)
        yaws = [view_yaw(p_, sy_, sa_) for p_, sy_, sa_ in samples]
        for fi_, ((x0, z0), seg_yaw, s_along) in enumerate(samples):
            # waypoint GT = belief 경로 호길이 wp_far 전방 지점
            tw = min(s_along + wp_far, la_total)
            wp = arc_at(tw)
            yaw = yaws[fi_]
            # 도착 heading = 다음 프레임의 실제 heading (plan 확정: 선택기가
            # 예측하고 GT는 pose[t+1] 실측). 마지막 프레임은 없음
            wp_head = yaws[fi_ + 1] if fi_ + 1 < len(yaws) else None
            # 높이: 계단 존이면 가장 가까운 climb 점의 y를 더함 — 단 천장
            # 아래로 클램프(등반 상단에서 카메라가 지붕 위로 나가는 버그)
            h_off = 0.0
            iz, ix = geom.to_idx(x0, z0)
            if cl_pts and geom.stair_mask[min(iz, geom.nz - 1),
                                          min(ix, geom.nx - 1)]:
                h_off = min(cl_pts, key=lambda p: (p[0][0] - x0) ** 2
                            + (p[0][1] - z0) ** 2)[1]
            cam_y = min(h_off + robot["cam_h"], geom.ceil_h - 0.15)
            rgb, depth = update_cam(c, x0, cam_y, z0, yaw)
            # 제안기 GT = 도달 가능 자유 공간의 픽셀 래스터 (히트맵)
            reach = common.reach_mask(geom, robot, x0, z0)
            heat = common.heat_gt(geom, reach, x0, z0, yaw, cam_y, depth)
            big = cv2.resize(heat, (RES, RES),
                             interpolation=cv2.INTER_NEAREST)
            rgb[big == 1] = (0.45 * rgb[big == 1]
                             + 0.55 * np.array([60, 220, 60])).astype(np.uint8)
            rgb[big == common.HEAT_IGNORE] = (
                0.6 * rgb[big == common.HEAT_IGNORE]
                + 0.4 * np.array([200, 200, 200])).astype(np.uint8)
            # 전역 선택 정답이 고스트면 그 픽셀에 십자 + 도착 heading 화살표
            uv = common.heat_pixel(x0, z0, yaw, cam_y, wp[0], wp[1])
            if uv is not None and heat[uv[1], uv[0]] == 1:
                k = RES // common.HEAT_HW
                cu, cvv = int((uv[0] + 0.5) * k), int((uv[1] + 0.5) * k)
                cv2.drawMarker(rgb, (cu, cvv), (255, 255, 255),
                               cv2.MARKER_CROSS, 18, 2)
                if wp_head is not None:
                    common.draw_heading(rgb, x0, z0, yaw, cam_y, wp[0], wp[1],
                                        wp_head, color=(60, 170, 255))
            valid = heat != common.HEAT_IGNORE
            pos = (heat[valid] == 1).mean() if valid.any() else 0.0
            cv2.putText(rgb, f"pos={pos:.0%}", (6, RES - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if boarding or exiting:
                special_idx.append(len(frames))
                cv2.putText(rgb, "BOARDING (doors open)" if boarding
                            else "EXITING (doors open)", (6, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 255), 2)
            cv2.putText(rgb, f"F{fl}" + (" climb" if h_off > 0 else ""),
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            dep_vis = cv2.applyColorMap(
                np.clip(depth / 10.0 * 255, 0, 255).astype(np.uint8),
                cv2.COLORMAP_TURBO)
            frames.append(np.concatenate([rgb, dep_vis], axis=0))
            frame_poses.append((fl, x0, z0))
        if boarding or exiting:
            set_doors(open_=False)  # 통과 완료 — 문 닫힘 (내부 대기는 닫힌 문)
        # 세그먼트 끝: 판정 지점(보고 포기)이면 VERDICT 프레임
        end_cell = sm[-1]
        if (fl, tuple(end_cell)) in verd or any(
                (fl, tuple(v)) in verd for v in sm[-3:]):
            ex, ez = geom.to_world(*end_cell)
            g = next((vgate[(fl, tuple(v))] for v in sm[-3:]
                      if (fl, tuple(v)) in vgate), None)
            # 판정 시선도 주행 프레임과 같은 규칙(belief 경로 1m look-ahead) —
            # 연장 폴리라인이 게이트 틈을 지나므로 자연히 게이트를 정면으로
            # 본다. 특례로 게이트만 바라보게 하면 직전 프레임과 시선이
            # 불연속(사용자 지적). 잔여가 없을 때만 게이트 중심 폴백.
            vp = arc_at(min(total_len + wp_far, la_total))
            if math.hypot(vp[0] - ex, vp[1] - ez) > 0.3:
                vyaw = yaw_deg(vp[0] - ex, vp[1] - ez)
            else:
                vyaw = yaw_deg(g["x"] - ex, g["z"] - ez) if g else yaw
            rgb, depth = update_cam(c, ex, robot["cam_h"], ez, vyaw)
            cv2.rectangle(rgb, (0, 0), (RES - 1, RES - 1), (0, 0, 255), 6)
            special_idx.append(len(frames))
            cv2.putText(rgb, "VERDICT: impossible", (6, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if g:  # 판정된 게이트를 1인칭 프레임에 투영 표시
                uv = project((g["x"], min(g["height"], 1.0) / 2, g["z"]),
                             (ex, robot["cam_h"], ez), vyaw)
                if uv:
                    cv2.circle(rgb, uv, 16, (0, 0, 255), 3)
                cv2.putText(rgb, f"gate {g['width']:.2f}m < "
                            f"{robot['w_eff']:.2f}m", (6, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            dep_vis = cv2.applyColorMap(
                np.clip(depth / 10.0 * 255, 0, 255).astype(np.uint8),
                cv2.COLORMAP_TURBO)
            frames.append(np.concatenate([rgb, dep_vis], axis=0))
            frame_poses.append((fl, ex, ez))
        # 엘리베이터 탑승 지점이면 WAITING 정지 프레임 3장 (문을 바라봄)
        wd = elev_waits.get((fl, tuple(end_cell)))
        if wd:
            # 대기 카메라는 문에서 1.3m 물러나 문을 바라봄 — 앵커 셀에서 찍으면
            # 문짝이 화면을 가득 채워 엘리베이터인지 식별 불가(사용자 지적)
            ax, az = geom.to_world(*end_cell)
            dxw, dzw = ax - wd[0], az - wd[1]
            L = max(math.hypot(dxw, dzw), 1e-6)
            ex, ez = wd[0] + dxw / L * 1.3, wd[1] + dzw / L * 1.3
            wyaw = yaw_deg(wd[0] - ex, wd[1] - ez)
            for _ in range(3):
                rgb, depth = update_cam(c, ex, robot["cam_h"], ez, wyaw)
                special_idx.append(len(frames))
                cv2.putText(rgb, "WAITING (elevator)", (6, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 255), 2)
                dep_vis = cv2.applyColorMap(
                    np.clip(depth / 10.0 * 255, 0, 255).astype(np.uint8),
                    cv2.COLORMAP_TURBO)
                frames.append(np.concatenate([rgb, dep_vis], axis=0))
                frame_poses.append((fl, ex, ez))
            # 내부 대기 hold(궤적 메타 기반): 진입 세그먼트가 카 중심에서
            # 끝나면 wait_steps 프레임을 문 방향으로 정지 촬영
            hold = elev_holds.get((fl, tuple(end_cell)))
            if hold and hold[0]:
                hx, hz = geom.to_world(*end_cell)
                hyaw = yaw_deg(hold[0][0] - hx, hold[0][1] - hz)
                for _ in range(hold[1]):
                    rgb, depth = update_cam(c, hx, robot["cam_h"], hz, hyaw)
                    special_idx.append(len(frames))
                    cv2.putText(rgb, "WAITING (inside elevator)", (6, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (60, 200, 255), 2)
                    dep_vis = cv2.applyColorMap(
                        np.clip(depth / 10.0 * 255, 0, 255).astype(np.uint8),
                        cv2.COLORMAP_TURBO)
                    frames.append(np.concatenate([rgb, dep_vis], axis=0))
                    frame_poses.append((fl, hx, hz))
    if frames:
        # 상태 프레임(WAITING/내부/VERDICT)은 균등 선별에서 빠질 수 있어
        # 강제 포함(실제 발생 — 카드에서 내부 대기가 안 보임)
        idxs = {int(i) for i in
                np.linspace(0, len(frames) - 1, min(12, len(frames)))}
        idxs |= set(special_idx)
        sel = [frames[i] for i in sorted(idxs)][:16]
        strip = np.concatenate(sel, axis=1)
        top = topview_montage(floors, robot, res, building_dir, frame_poses,
                              strip.shape[0], goal=goal)
        card = (np.concatenate([top, strip], axis=1)
                if top is not None else strip)
        cv2.imwrite(out_png, card)
        print(f"프레임 {len(frames)}장 → {out_png}")
    return frames




def _near_passable(geom, ok, x, z, r_m=1.0):
    iz, ix = geom.to_idx(x, z)
    r = int(r_m / common.GRID)
    z0, z1 = max(0, iz - r), min(geom.nz, iz + r + 1)
    x0, x1 = max(0, ix - r), min(geom.nx, ix + r + 1)
    sub = ok[z0:z1, x0:x1]
    if not sub.any():
        return None
    zz, xx = np.where(sub)
    k = int(np.argmin((zz + z0 - iz) ** 2 + (xx + x0 - ix) ** 2))
    return (int(zz[k] + z0), int(xx[k] + x0))


def case_elevator(c, floors, nav, robots, out_dir, building_dir):
    """케이스: wheeled 엘리베이터 (탑승 대기중 → 카 내부 → 하차)."""
    f0, f1 = sorted(floors)[0], sorted(floors)[1]
    wheeled = max((r for r in robots if not r["stairs_ok"]),
                  key=lambda r: r["cam_h"])
    ok0 = floors[f0].passable(wheeled["w_eff"], wheeled["h"])
    ok1 = floors[f1].passable(wheeled["w_eff"], wheeled["h"])
    s0 = _near_passable(floors[f0], ok0, 14.0, 8.0, 3.0)
    g1 = _near_passable(floors[f1], ok1, 13.0, 12.0, 3.0)
    res = nav.plan(wheeled, (f0, *s0), (f1, *g1))
    print(f"[elevator] {wheeled['id']} 성공={res.get('success')} "
          f"전환={[t['mode'] for t in res.get('transitions', [])]}")
    episode_frames(c, floors, wheeled, res,
                   os.path.join(out_dir, common.out_name(
                       building_dir, "case_elevator.png")), building_dir)


def _snap_free(geom, ok, x, z, r_m=1.0):
    """(x,z) 주변 r_m 내 최근접 자유 셀 (없으면 None)."""
    iz, ix = geom.to_idx(x, z)
    r = int(r_m / common.GRID)
    z0, z1 = max(0, iz - r), min(geom.nz, iz + r + 1)
    x0, x1 = max(0, ix - r), min(geom.nx, ix + r + 1)
    sub = ok[z0:z1, x0:x1]
    if not sub.any():
        return None
    zz, xx = np.where(sub)
    k = int(np.argmin((zz + z0 - iz) ** 2 + (xx + x0 - ix) ** 2))
    return (int(zz[k] + z0), int(xx[k] + x0))


def case_abandon(c, floors, nav, robots, out_dir, building_dir):
    """관측 후 포기 — 좁은 게이트 표적 구성.

    랜덤 (시작,목표) 표집은 케이스가 드문 빌딩(주택)에서 25분+ 공회전
    (실측) — 게이트 폭 오름차순으로 "열린 쪽 6m 시작 → 게이트 뒤편 2m
    목표"를 직접 구성하고, 그 게이트를 못 지나는 로봇(폭 내림차순)로
    주행-판정을 유도한다.
    """
    small = min(robots, key=lambda r: r["w_eff"])
    hit = f0 = None
    for fl in sorted(floors):  # 좁은 입구 방이 상층에 있는 빌딩(주택) 대응
        geom = floors[fl]
        okm = geom.passable(small["w_eff"], small["h"])
        for g in sorted(common.floor_gates(geom),
                        key=lambda gg: gg["width"]):
            if g["width"] < 0.25:
                continue  # 슬릿급 — 뒤편 자유 셀 없음
            adx, adz = approach_dir(geom, g)
            g0 = None
            for d in (2.0, 1.5, 2.5):
                g0 = _snap_free(geom, okm, g["x"] - adx * d,
                                g["z"] - adz * d)
                if g0:
                    break
            if not g0:
                continue
            for cand in sorted(robots, key=lambda r: -r["w_eff"]):
                if cand["w_eff"] <= g["width"]:
                    break  # 이하는 통과 가능 — 판정 불가
                # 시작 셀은 **후보 로봇이 설 수 있는 곳**이어야 함 — 최소
                # 로봇 기준 스냅이면 광폭 로봇이 사전 실패(무판정, 실측)
                okc = geom.passable(cand["w_eff"], cand["h"])
                s0 = None
                for d in (6.0, 4.0, 3.0):
                    s0 = _snap_free(geom, okc, g["x"] + adx * d,
                                    g["z"] + adz * d, r_m=2.0)
                    if s0:
                        break
                if not s0:
                    continue
                res = nav.plan(cand, (fl, *s0), (fl, *g0))
                if not res.get("success") and res.get("verdicts"):
                    hit = (cand, res, s0, g0)
                    f0 = fl
                    break
            if hit:
                break
        if hit:
            break
    if hit is None:
        print("[abandon] 관측 후 포기 케이스 미발견 — 스킵")
        return
    cand, res, s0, g0 = hit
    print(f"[abandon] {cand['id']} w_eff={cand['w_eff']:.2f} "
          f"성공={res.get('success')} 이유={res.get('reason')} "
          f"판정={len(res.get('verdicts', []))}곳")
    episode_frames(c, floors, cand, res,
                   os.path.join(out_dir, common.out_name(
                       building_dir, "case_abandon.png")), building_dir,
                   allow_fail=True, goal=(f0, *g0))


def case_undertable(c, floors, nav, robots, out_dir, building_dir):
    """케이스: 언더테이블 통과 (낮은 로봇만 가능한 높이 게이트)."""
    f0 = sorted(floors)[0]
    low = min(robots, key=lambda r: r["h"])
    g5 = None
    for g in common.floor_gates(floors[f0]):
        if (low["h"] + 0.15 <= g["height"] <= 1.0
                and g["width"] >= low["w_eff"] + 0.15):
            g5 = g
            break
    if g5 is None:
        print("[undertable] 조건 맞는 게이트 없음")
        return
    dx, dz = approach_dir(floors[f0], g5)
    okl = floors[f0].passable(low["w_eff"], low["h"])
    s0 = _near_passable(floors[f0], okl, g5["x"] + dx * 2, g5["z"] + dz * 2)
    g0 = _near_passable(floors[f0], okl, g5["x"] - dx * 2, g5["z"] - dz * 2)
    if not s0 or not g0:
        print("[undertable] 양측 접근 셀 없음")
        return
    res = nav.plan(low, (f0, *s0), (f0, *g0))
    print(f"[undertable] {low['id']} h={low['h']:.2f} "
          f"게이트 h={g5['height']:.2f} 성공={res.get('success')}")
    episode_frames(c, floors, low, res,
                   os.path.join(out_dir, common.out_name(
                       building_dir, "case_undertable.png")), building_dir)


def elevator_only_case(c, building_dir, robots, out_dir):
    """케이스 6: 엘베만 합성 씬 — 계단 문 entry 제거(벽이 자동으로 막힘) 후
    legged가 규칙 폴백으로 엘리베이터를 타는지 계획+렌더로 검증."""
    import glob as _glob

    os.makedirs(out_dir, exist_ok=True)
    floors2 = {}
    for fp in sorted(_glob.glob(os.path.join(building_dir, "floor_*.json"))):
        no = int(os.path.basename(fp).split("_")[1].split(".")[0])
        d = json.load(open(fp))
        d["doors"] = [x for x in d["doors"] if "stair" not in
                      (x.get("room0", "") + x.get("room1", "")).lower()]
        floors2[no] = common.FloorGeometry(d, no)
    nav2 = common.BuildingNav(floors2)
    legged = next(r for r in robots if r["stairs_ok"] and abs(r["h"] - 1.5) < 0.5)
    f0, f1 = sorted(floors2)[0], sorted(floors2)[1]
    ok0 = floors2[f0].passable(legged["w_eff"], legged["h"])
    ok1 = floors2[f1].passable(legged["w_eff"], legged["h"])
    s0 = _near_passable(floors2[f0], ok0, 14.0, 8.0, 3.0)
    g1 = _near_passable(floors2[f1], ok1, 13.0, 12.0, 3.0)
    res = nav2.plan(legged, (f0, *s0), (f1, *g1))
    modes = [t["mode"] for t in res.get("transitions", [])]
    print(f"[elevator_only] {legged['id']} (legged) 성공={res.get('success')} "
          f"전환={modes}  — 계단 문 제거 합성 씬")
    episode_frames(c, floors2, legged, res,
                   os.path.join(out_dir, common.out_name(
                       building_dir, "case_elevator_only.png")),
                   building_dir)


UNITS = ("gates", "episode", "elevator", "abandon", "undertable",
         "elevator_only")


def run_unit(unit, args):
    robots = common.robot_pool()
    if unit == "elevator_only":
        c = common.launch_controller(width=RES, height=RES, render=True)
        try:
            elevator_only_case(c, args.building, robots,
                               os.path.join(args.out, "cases"))
        finally:
            c.stop()
        return
    floors = common.load_building(args.building)
    nav = common.BuildingNav(floors)
    c = common.launch_controller(width=RES, height=RES, render=True)
    try:
        if unit == "gates":
            by_w = sorted(robots, key=lambda r: r["w_eff"])
            reps = [by_w[0], by_w[len(by_w) // 2], by_w[-1]]
            g1 = floors[min(floors)]
            common.load_floor_scene(c, g1.scene)
            c.step(action="AddThirdPartyCamera", position=dict(x=0, y=1, z=0),
                   rotation=dict(x=0, y=0, z=0), fieldOfView=90)
            snapshot_cards(c, g1, pick_gates(g1, robots, args.n_gates), reps,
                           os.path.join(args.out, "gates"), args.building)
        elif unit == "episode":
            legged = next(r for r in robots
                          if r["stairs_ok"] and abs(r["h"] - 1.5) < 0.5)
            f0, f1 = sorted(floors)[0], sorted(floors)[1]
            ok0 = floors[f0].passable(legged["w_eff"], legged["h"])
            ok1 = floors[f1].passable(legged["w_eff"], legged["h"])
            iz0, ix0 = [v[len(v) // 3] for v in np.where(ok0)]
            iz1, ix1 = [v[len(v) // 2] for v in np.where(ok1)]
            res = nav.plan(legged, (f0, int(iz0), int(ix0)),
                           (f1, int(iz1), int(ix1)))
            os.makedirs(os.path.join(args.out, "episode"), exist_ok=True)
            episode_frames(c, floors, legged, res,
                           os.path.join(args.out, "episode",
                                        common.out_name(
                                            args.building,
                                            "episode_strip.png")),
                           args.building)
        else:
            fn = {"elevator": case_elevator, "abandon": case_abandon,
                  "undertable": case_undertable}[unit]
            os.makedirs(os.path.join(args.out, "cases"), exist_ok=True)
            fn(c, floors, nav, robots, os.path.join(args.out, "cases"),
               args.building)
    finally:
        c.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gates", type=int, default=3)
    ap.add_argument("--only", choices=UNITS)
    ap.add_argument("--workers", type=int, default=4,
                    help="병렬 서브프로세스 수 (GPU 0~3 분산)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.only:
        run_unit(args.only, args)
        return

    # 병렬: 자기 자신을 --only 단위로 분기, GPU 0~3 순환 배정
    import subprocess
    import time as _time

    pending = list(UNITS)
    running = []  # (Popen, unit)
    gpu_i = 0
    while pending or running:
        while pending and len(running) < max(1, args.workers):
            unit = pending.pop(0)
            env = dict(os.environ, MANSION_GPU=str(gpu_i % 4))
            gpu_i += 1
            p = subprocess.Popen(
                ["python", os.path.abspath(__file__),
                 "--building", args.building, "--out", args.out,
                 "--n-gates", str(args.n_gates), "--only", unit],
                env=env)
            running.append((p, unit))
            print(f"[병렬] {unit} 시작 (GPU {env['MANSION_GPU']})")
        for p, unit in running[:]:
            if p.poll() is not None:
                print(f"[병렬] {unit} 종료 (rc={p.returncode})")
                running.remove((p, unit))
        _time.sleep(2)


if __name__ == "__main__":
    main()

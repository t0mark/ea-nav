"""관측 기반 expert 주행 파일럿 (Phase 1) + belief 높이 규약 전·후 비교.

expert 궤적은 전지적 최단이 아니라 관측 기반이다. 로봇은 건물 배치는 알지만
"내 몸이 이 게이트를 통과할 수 있는가"는 카메라에 게이트가 잡힌 시점에
판정하고, 못 지나면 그 자리에서 재계획한다(부분 궤적 + 판정 지점 = 불가능
판정 GT). 낙관 가정은 폭과 높이 모두에 적용된다 — 자기 높이로 belief를 세우면
천장이 낮아 막힌 곳만은 보지 않고 미리 아는 셈이 되어 규약이 깨진다.

이 파일럿은 같은 에피소드를 두 규약으로 풀어 비교한다.
  before: belief_h = 로봇 자기 높이 (높이 제약을 미리 아는 구판)
  after : belief_h = BELIEF_H     (폭과 동일하게 관측으로 판정)
경로가 달라진 비율이 곧 주행 에피소드 재렌더 규모다.

실행:
  docker exec airlab_hw_mansion bash -c 'cd /workspace/research/scripts/datasets/MANSION \
    && python 01_expert.py --building "/data/MansionWorld/mansionworld/public_hotel_dormitory_4f_300_fp001#0" \
    --out /workspace/research/check/MANSION/01_expert'
"""

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import time

import cv2
import numpy as np

import common

COLORS = [(60, 220, 60), (60, 160, 255), (230, 120, 60), (200, 80, 220)]
BEFORE_COLOR = (150, 150, 150)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _xzh(geom, cells, stride=4, ahead_m=1.0, tail=None):
    """셀 경로 → [(x, z, heading°)] — heading = 약 1 m 전방 지점 방향.

    인접 표본점(10 cm) 접선은 지터·꼭짓점 스파이크가 생긴다. 도착 heading의
    의미("도착 후 이어서 갈 방향")에 맞게 look-ahead로 정의한다.
    tail = 실패 에피소드의 정지점 너머 belief 잔여(게이트 틈→goal 방향) —
    look-ahead 대상에만 포함해, 말미 heading이 실행 경로 끝에서 잘려 goal
    지향성을 잃지 않게 한다(포즈 자체는 실행 경로까지만).
    """
    pts = [geom.to_world(iz, ix) for iz, ix in cells[::stride]]
    n = len(pts)
    if tail:
        pts += [geom.to_world(iz, ix) for iz, ix in tail[::stride]]
    step = max(1, int(round(ahead_m / max(stride * common.GRID, 1e-9))))
    out = []
    for i in range(n):
        x, z = pts[i]
        j = min(i + step, len(pts) - 1)
        dx, dz = pts[j][0] - x, pts[j][1] - z
        h = (math.degrees(math.atan2(dx, dz)) if (dx or dz)
             else (out[-1][2] if out else 0.0))
        out.append([round(x, 2), round(z, 2), round(h, 1)])
    return out


def pick_representatives(robots):
    """대표 로봇: wheeled 최소/최대 폭, humanoid 최대 h, multileg 중간."""
    wh = sorted([r for r in robots if r["cls"] == "wheeled"],
                key=lambda r: r["w_eff"])
    hu = sorted([r for r in robots if r["cls"] == "humanoid"],
                key=lambda r: r["h"])
    ml = sorted([r for r in robots if r["cls"] == "multileg"],
                key=lambda r: r["w_eff"])
    seen, out = set(), []
    for r in [wh[0], wh[-1], hu[-1], ml[len(ml) // 2]]:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def far_free_cell(geom, mask_ok, rng, avoid=None, min_d=4.0):
    """주행 가능 셀 중 랜덤 (avoid에서 min_d 이상)."""
    iz, ix = np.where(mask_ok)
    for _ in range(200):
        k = rng.integers(len(iz))
        cell = (int(iz[k]), int(ix[k]))
        if avoid is None:
            return cell
        ax, az = geom.to_world(*avoid)
        bx, bz = geom.to_world(*cell)
        if (ax - bx) ** 2 + (az - bz) ** 2 >= min_d ** 2:
            return cell
    return None


def route_cells(res):
    """결과의 전체 셀 경로를 (층, 셀) 집합으로 — 겹침율 비교용."""
    return {(fl, c) for fl, cells in res.get("segments", []) for c in cells}


def changed(a, b):
    """두 결과가 실질적으로 다른가 = 성공 여부·층 전환 수단·경로 겹침율."""
    if a.get("success") != b.get("success"):
        return True, "success"
    ma = ",".join(t["mode"] for t in a.get("transitions", []))
    mb = ",".join(t["mode"] for t in b.get("transitions", []))
    if ma != mb:
        return True, "mode"
    ca, cb = common._coarse_cells(a), common._coarse_cells(b)
    if not ca and not cb:
        return False, "same"
    iou = len(ca & cb) / max(len(ca | cb), 1)
    return (iou < common.DIVERGENCE_IOU, "path" if iou < common.DIVERGENCE_IOU
            else "same")


def draw_episode(floors, robots, results, before, title, out_png, building_dir,
                 start=None, goal=None):
    """층별 패널 몽타주. 회색 = before(구 규약), 색 = after(신 규약)."""
    panels = []
    for no, geom in sorted(floors.items()):
        img, w2p = common.sim_view(geom, building_dir)
        for tag, pos in (("START", start), ("GOAL", goal)):
            if pos is not None and pos[0] == no:
                u, v = w2p(*geom.to_world(*pos[1:]))
                cv2.drawMarker(img, (u, v), (255, 255, 255),
                               cv2.MARKER_TILTED_CROSS, 22, 3)
                cv2.putText(img, tag, (u + 12, v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for k, (r, res, pre) in enumerate(zip(robots, results, before)):
            for src, color, width in ((pre, BEFORE_COLOR, 5),
                                      (res, COLORS[k % len(COLORS)], 3)):
                for fl, cells in src.get("segments", []):
                    if fl != no:
                        continue
                    sm = common.natural_path(geom, r, cells)
                    pts = common.cells_to_px(geom, sm, w2p)
                    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False,
                                  color, width, cv2.LINE_AA)
            common.draw_transitions(img, geom, w2p,
                                    res.get("transitions", []),
                                    COLORS[k % len(COLORS)])
            common.draw_climb(img, geom, w2p, res.get("transitions", []),
                              COLORS[k % len(COLORS)])
            for vfl, vd in res.get("verdicts", []):
                if vfl != no:
                    continue
                u, v = w2p(*geom.to_world(*vd))
                cv2.drawMarker(img, (u, v), (0, 255, 255),
                               cv2.MARKER_DIAMOND, 16, 3)
        cv2.putText(img, f"F{no}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        panels.append(img)
    hmax = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, hmax - p.shape[0], 0, 8,
                                 cv2.BORDER_CONSTANT) for p in panels]
    img = np.concatenate(panels, axis=1)
    bar = np.zeros((26 * (len(robots) + 2), img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    cv2.putText(bar, "gray = before (belief_h = robot h)", (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BEFORE_COLOR, 1)
    for k, (r, res, pre) in enumerate(zip(robots, results, before)):
        ch, why = changed(pre, res)
        s = (f"{r['id']} {r['cls']} w_eff={r['w_eff']:.2f} h={r['h']:.2f} -> "
             + (f"cost={res['cost']:.1f} "
                + ",".join(t["mode"] for t in res["transitions"])
                if res.get("success") else f"FAIL({res.get('reason')})")
             + (f"  [CHANGED:{why}]" if ch else "  [same]"))
        cv2.putText(bar, s, (8, 70 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COLORS[k % len(COLORS)], 1)
    cv2.imwrite(out_png, np.concatenate([bar, img], axis=0))


def episode_job(job):
    """에피소드 하나를 두 규약으로 풀고 카드까지 그린다."""
    building, name, start, goal, reps, out_dir = job
    floors = common.load_building(building)
    nav_new = common.BuildingNav(floors)
    nav_old = common.BuildingNav(floors, belief_h=None)
    after = [nav_new.plan(r, start, goal) for r in reps]
    before = []
    for r in reps:
        nav_old.belief_h = r["h"]  # 구 규약 = 자기 높이로 belief
        before.append(nav_old.plan(r, start, goal))

    sx, sz = floors[start[0]].to_world(*start[1:])
    gx, gz = floors[goal[0]].to_world(*goal[1:])
    title = (f"{name}  start=F{start[0]}({sx:.1f},{sz:.1f}) "
             f"goal=F{goal[0]}({gx:.1f},{gz:.1f})")
    draw_episode(floors, reps, after, before, title,
                 os.path.join(out_dir, common.out_name(
                     building, f"{name}.png")), building,
                 start=start, goal=goal)

    rec = {"episode": name,
           "start": {"floor": start[0], "x": round(sx, 3), "z": round(sz, 3)},
           "goal": {"floor": goal[0], "x": round(gx, 3), "z": round(gz, 3)},
           "results": []}
    for r, res, pre in zip(reps, after, before):
        ch, why = changed(pre, res)
        rec["results"].append({
            "robot": r["id"], "cls": r["cls"], "h": round(r["h"], 2),
            "success": res.get("success", False),
            "success_before": pre.get("success", False),
            "cost": round(res.get("cost", -1), 2) if res.get("success") else None,
            "cost_before": round(pre.get("cost", -1), 2)
            if pre.get("success") else None,
            "transitions": res.get("transitions", []),
            "reason": res.get("reason"),
            "n_verdicts": len(res.get("verdicts", [])),
            "n_verdicts_before": len(pre.get("verdicts", [])),
            "changed": ch, "change_kind": why,
            # 포즈 = (x, z, heading°). look-ahead 거리는 로봇별
            # max(1m, (cam_h−0.05)×1.15) — 02의 waypoint 시인 규칙과 동일
            "path_xzh": [[fl, _xzh(
                floors[fl], cells,
                ahead_m=max(1.0, (r["cam_h"] - 0.05) * 1.15),
                tail=(res.get("belief_tail", (None,))[1]
                      if li == len(res["segments"]) - 1
                      and res.get("belief_tail", (None,))[0] == fl
                      else None))]
                for li, (fl, cells) in enumerate(res.get("segments", []))],
        })
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-same", type=int, default=12)
    ap.add_argument("--n-cross", type=int, default=6)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    robots = common.robot_pool()
    reps = pick_representatives(robots)
    log("대표 로봇: " + ", ".join(
        f"{r['id']}({r['cls']} w={r['w_eff']:.2f} h={r['h']:.2f})"
        for r in reps))

    floors = common.load_building(args.building)
    inter = {no: np.logical_and.reduce(
        [g.passable(r["w_eff"], r["h"]) for r in reps])
        for no, g in floors.items()}
    for no, m in sorted(inter.items()):
        log(f"F{no}: 공통 주행 가능 셀 {int(m.sum())}")

    episodes, f0 = [], min(floors)
    for i in range(args.n_same):
        s = far_free_cell(floors[f0], inter[f0], rng)
        g = far_free_cell(floors[f0], inter[f0], rng, avoid=s, min_d=6.0)
        episodes.append((f"same_floor_{i}", (f0, *s), (f0, *g)))
    tops = sorted(floors)[1:]
    for i in range(args.n_cross):
        ft = tops[i % len(tops)]
        s = far_free_cell(floors[f0], inter[f0], rng)
        g = far_free_cell(floors[ft], inter[ft], rng)
        episodes.append((f"cross_{f0}to{ft}_{i}", (f0, *s), (ft, *g)))

    jobs = [(args.building, n, s, g, reps, args.out) for n, s, g in episodes]
    nw = args.workers or min(len(jobs), max(1, mp.cpu_count() // 3))
    log(f"에피소드 {len(jobs)}개 × 로봇 {len(reps)}대 × 2규약 — 워커 {nw}")

    t0, records = time.time(), []
    with mp.Pool(nw) as pool:
        for i, rec in enumerate(pool.imap_unordered(episode_job, jobs), 1):
            records.append(rec)
            el = time.time() - t0
            nch = sum(1 for r in rec["results"] if r["changed"])
            log(f"  [{i}/{len(jobs)}] {rec['episode']} — 변화 {nch}/"
                f"{len(rec['results'])} | {el:.0f}s 경과 / ETA "
                f"{el / i * (len(jobs) - i):.0f}s")
    records.sort(key=lambda r: r["episode"])

    flat = [x for r in records for x in r["results"]]
    n = len(flat)
    nch = sum(1 for x in flat if x["changed"])
    print(f"\n=== belief 높이 규약 전·후 (경로 {n}개) ===")
    print(f"변화한 경로: {nch}/{n} ({nch / n:.1%}) = 재렌더 대상 비율")
    for kind in ("success", "mode", "path"):
        k = sum(1 for x in flat if x["changed"] and x["change_kind"] == kind)
        print(f"  {kind:8s}: {k}")
    print("로봇별 변화율:")
    for r in reps:
        mine = [x for x in flat if x["robot"] == r["id"]]
        c = sum(1 for x in mine if x["changed"])
        print(f"  {r['id']:<28} h={r['h']:.2f}: {c}/{len(mine)} "
              f"({c / max(len(mine), 1):.0%})")
    vb = sum(x["n_verdicts_before"] for x in flat)
    va = sum(x["n_verdicts"] for x in flat)
    print(f"관측 분기(불가능 판정) 총합: {vb} → {va}")
    sb = sum(1 for x in flat if x["success_before"])
    sa = sum(1 for x in flat if x["success"])
    print(f"성공 경로: {sb} → {sa}")

    with open(os.path.join(args.out, common.out_name(
            args.building, "episodes.json")), "w") as fh:
        json.dump({"belief_h": common.BELIEF_H,
                   "n_paths": n, "n_changed": nch,
                   "verdicts_before": vb, "verdicts_after": va,
                   "success_before": sb, "success_after": sa,
                   "episodes": records}, fh, ensure_ascii=False, indent=1)
    log(f"저장: {args.out}/*_episodes.json + 에피소드 PNG "
        f"(총 {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

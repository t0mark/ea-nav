"""게이트 추출 + 통과성 라벨 행렬 (Phase 1 파일럿).

게이트 = 통과 여부가 embodiment에 따라 갈릴 수 있는 "통로" 지점.
추출 로직은 common(door_gates/choke_gates — expert 주행의 관측 판정과 공유).
  - door  : 씬 JSON doors (holePolygon 폭·높이, doorSegment 위치)
  - choke : 가구 협착부 — 스켈레톤 국소 폭 최소(< GATE_WIDTH_MAX) 중,
            막았을 때 주변 자유 공간이 둘로 갈라지는 통로만 (막다른 틈 제외).
            탐색 높이 GATE_H_LEVELS 4단의 합집합 — 가구 아래로 지나가는
            통로의 높이 제약을 키 큰 로봇의 표본으로 확보한다.

라벨 = common.gate_label — 로봇 자기 높이 단면의 국소 통과 폭 기준
  (기록된 폭이 아니라 width_map(h)에서 읽어 expert 주행 판정과 일치시킴).

검수 지표: 단일 높이 탐색과 나란히 찍은 "키 구간별 차단율"과 차단 원인
  분해(높이 때문인가 폭 때문인가). 높이 축이 실제로 살아났는지 보는 자리다.

파일럿 실행 (data/에 저장하지 않고 check/에만):
  docker exec airlab_hw_mansion bash -c 'cd /workspace/research/scripts/datasets/MANSION \
    && python 00_gates.py --building "/data/MansionWorld/mansionworld/public_hotel_dormitory_4f_300_fp001#0" \
    --out /workspace/research/check/MANSION/00_gates'
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import cv2
import numpy as np

import common

# 진단용 가상 로봇: 폭을 고정하고 키만 스윕해 높이 효과만 분리해 본다
DIAG_HEIGHTS = (0.4, 0.7, 1.0, 1.3, 1.6, 1.9)
DIAG_WIDTHS = (0.3, 0.5)
BANDS = ((0.3, 0.7), (0.7, 1.1), (1.1, 1.5), (1.5, 2.0))
# 카드 색 = 최초로 잡힌 탐색 높이. 바닥에서만 보이던 게이트와 높이를 올려야
# 드러나는 게이트를 그림에서 바로 구분한다.
H_COLOR = {0.15: (0, 165, 255), 0.6: (0, 230, 230),
           1.1: (255, 200, 0), 1.6: (255, 90, 200)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def draw_card(geom, building, gates, out_dir, no):
    img, w2p = common.sim_view(geom, building)
    for g in gates:
        u, v = w2p(g["x"], g["z"])
        color = ((0, 0, 255) if g["type"] == "door"
                 else H_COLOR.get(g["h_ref"], (0, 165, 255)))
        cv2.circle(img, (u, v), 9, color, 2)
        tag = (f"{g['width']:.2f}" if g["type"] == "door"
               else f"{g['width']:.2f}@{g['h_ref']}")
        for th, cc in ((2, (255, 255, 255)), (1, color)):
            cv2.putText(img, tag, (u + 11, v + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, cc, th)
    legend = [("door", (0, 0, 255))] + [(f"choke h={k}", v)
                                        for k, v in H_COLOR.items()]
    for i, (name, cc) in enumerate(legend):
        cv2.circle(img, (24, 28 + i * 26), 9, cc, 2)
        cv2.putText(img, name, (40, 33 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cc, 2)
    cv2.imwrite(os.path.join(out_dir, common.out_name(
        building, f"F{no}.png")), img)


def floor_job(args):
    """층 하나: 게이트 추출(단일·4단) + 라벨 행렬 + 진단 집계 + 카드."""
    building, no, robots, out_dir = args
    geom = common.load_floor(building, no)
    doors = common.door_gates(geom)
    single, _ = common.choke_gates(geom, doors, h_levels=(0.15,))
    multi, n_dead = common.choke_gates(geom, doors)
    g_single, g_multi = doors + single, doors + multi

    lab = np.array([[common.gate_label(geom, g, r) for r in robots]
                    for g in g_multi], dtype=np.float32)
    lab_s = np.array([[common.gate_label(geom, g, r) for r in robots]
                      for g in g_single], dtype=np.float32)

    # 진단: 폭 고정·키 스윕 차단율과 차단 원인(높이/폭) 분해
    diag = {}
    for tag, gs in (("single", g_single), ("multi", g_multi)):
        for w in DIAG_WIDTHS:
            diag[f"{tag}_w{w}"] = [
                sum(1 for g in gs
                    if common.gate_label(geom, g, {"w_eff": w, "h": h}) < 0.1)
                for h in DIAG_HEIGHTS]
    cause = []
    for h in DIAG_HEIGHTS:
        hb = common.robot_bucket({"w_eff": 0.5, "h": h})[1]
        wm = geom.width_map(hb)
        by_h = by_w = 0
        for g in g_multi:
            iz, ix = common.gate_cell(geom, g)
            iz, ix = min(max(iz, 0), geom.nz - 1), min(max(ix, 0), geom.nx - 1)
            v = float(wm[iz, ix])
            if v <= 0.0:
                by_h += 1
            elif common.soft_label(v - 0.5) < 0.1:
                by_w += 1
        cause.append((by_h, by_w))

    clear = []
    for g in g_multi:
        iz, ix = common.gate_cell(geom, g)
        iz, ix = min(max(iz, 0), geom.nz - 1), min(max(ix, 0), geom.nx - 1)
        clear.append(float(geom.clearance[iz, ix]))

    draw_card(geom, building, g_multi, out_dir, no)
    for g in g_multi:
        g["floor"] = no
    by_level = {h: sum(1 for g in multi if g["h_ref"] == h)
                for h in common.GATE_H_LEVELS}
    return {"floor": no, "gates": g_multi, "labels": lab, "labels_single": lab_s,
            "n_door": len(doors), "n_single": len(single), "n_multi": len(multi),
            "n_dead": n_dead, "by_level": by_level, "diag": diag,
            "cause": cause, "clear": clear, "y_err": geom.y_align_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    bname = os.path.basename(args.building.rstrip("/")).split("#")[0]

    robots = common.robot_pool()
    log(f"URDF 풀 {len(robots)}개 "
        f"(w_eff {min(r['w_eff'] for r in robots):.2f}~"
        f"{max(r['w_eff'] for r in robots):.2f} m, "
        f"h {min(r['h'] for r in robots):.2f}~"
        f"{max(r['h'] for r in robots):.2f} m) "
        f"— 협착부 탐색 높이 {common.GATE_H_LEVELS} m")

    nos = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                 for p in __import__("glob").glob(
                     os.path.join(args.building, "floor_*.json")))
    jobs = [(args.building, no, robots, args.out) for no in nos]
    nw = args.workers or min(len(jobs), max(1, mp.cpu_count() // 2))
    log(f"층 {len(nos)}개 × (단일+4단) 추출·라벨 — 워커 {nw}")

    t0, res = time.time(), []
    with mp.Pool(nw) as pool:
        for i, r in enumerate(pool.imap_unordered(floor_job, jobs), 1):
            res.append(r)
            el = time.time() - t0
            log(f"  [{i}/{len(jobs)}] F{r['floor']} 완료 — door {r['n_door']} / "
                f"choke {r['n_single']}→{r['n_multi']} "
                f"(막다른 틈 제외 {r['n_dead']}) 최초 발견 높이별 {r['by_level']} "
                f"| {el:.0f}s 경과 / ETA {el / i * (len(jobs) - i):.0f}s")
    res.sort(key=lambda r: r["floor"])

    gates = [g for r in res for g in r["gates"]]
    L = np.concatenate([r["labels"] for r in res])
    Ls = np.concatenate([r["labels_single"] for r in res])
    n_single = sum(r["n_door"] + r["n_single"] for r in res)
    y = np.abs(np.concatenate([np.array(r["y_err"]) for r in res]))
    log(f"y 정렬 감사: 메시 최저점 |y|<5cm 비율 {np.mean(y < 0.05):.0%} "
        f"(중앙값 {np.median(y):.3f} m)")

    split = ((L > 0.9).sum(1) > 0) & ((L < 0.1).sum(1) > 0)
    log(f"게이트 {n_single}개(단일) → {len(gates)}개(4단), "
        f"갈림 게이트 {int(split.sum())}개 ({split.mean():.0%})")
    wbins = np.histogram([g["width"] for g in gates],
                         bins=[0, .4, .6, .8, 1.0, 1.2, 1.5, 2.0, 9])[0]
    log(f"폭 분포 [0,.4,.6,.8,1,1.2,1.5,2,+]: {wbins.tolist()}")

    print("\n실제 풀 기준 키 구간별 차단율 (라벨<0.1) — 단일 → 4단")
    rows = []
    for lo, hi in BANDS:
        idx = [j for j, r in enumerate(robots) if lo <= r["h"] < hi]
        if not idx:
            continue
        s, m = (Ls[:, idx] < 0.1).mean(), (L[:, idx] < 0.1).mean()
        rows.append((f"{lo}~{hi}", len(idx), float(s), float(m)))
        print(f"  h {lo}~{hi} m ({len(idx):3d}대): {s:6.1%} → {m:6.1%}")

    print("\n폭 고정·키 스윕 차단율 (다른 변수 없이 높이 효과만)")
    diag_out = {}
    for tag in ("single", "multi"):
        for w in DIAG_WIDTHS:
            tot = np.sum([r["diag"][f"{tag}_w{w}"] for r in res], axis=0)
            n = n_single if tag == "single" else len(gates)
            diag_out[f"{tag}_w{w}"] = (tot / n).round(4).tolist()
            print(f"  [{tag:6s} 폭 {w}m] "
                  + " ".join(f"h{h}={v / n:.0%}"
                             for h, v in zip(DIAG_HEIGHTS, tot)))

    print("\n차단 원인 분해 (폭 0.5 m 고정, 4단 게이트)")
    cause = np.sum([r["cause"] for r in res], axis=0)
    for h, (bh, bw) in zip(DIAG_HEIGHTS, cause):
        print(f"  h={h}m: 높이차단 {bh:3d} / 폭차단 {bw:3d} / 통과 "
              f"{len(gates) - bh - bw:3d}")

    clear = np.array([c for r in res for c in r["clear"]])
    print("\n게이트 셀 클리어런스 분포 (높이 제약이 걸릴 여지)")
    for t in (0.5, 0.9, 1.3, 1.7, 2.1, 3.0):
        print(f"  < {t}m: {int((clear < t).sum()):3d} ({(clear < t).mean():.0%})")

    with open(os.path.join(args.out, common.out_name(
            args.building, "gates.json")), "w") as fh:
        json.dump({"building": bname, "gates": gates,
                   "robots": [r["id"] for r in robots],
                   "h_levels": list(common.GATE_H_LEVELS),
                   "n_gates_single": n_single,
                   "block_rate_by_height": [
                       {"band": b, "n_robots": n, "single": round(s, 4),
                        "multi": round(m, 4)} for b, n, s, m in rows],
                   "diag_height_sweep": {"heights": list(DIAG_HEIGHTS),
                                         **diag_out},
                   "cause_h_vs_w": np.array(cause).tolist(),
                   "clearance_hist": np.histogram(
                       clear, bins=[0, .5, .9, 1.3, 1.7, 2.1, 3.0, 9]
                   )[0].tolist(),
                   "labels": np.round(L, 3).tolist()}, fh,
                  ensure_ascii=False, indent=1)
    log(f"저장: {args.out}/*_gates.json + 층별 PNG "
        f"(총 {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

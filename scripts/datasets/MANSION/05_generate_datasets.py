"""통합 데이터셋 생성기 (파일럿/본생성 공용) — 00~04 파일럿 로직의 데이터 모드.

산출(카드가 아니라 학습용 원자료 — 이미지·프롬프트·GT 쌍):
  gates/        통과성 튜플: gates.json(게이트×로봇 soft 라벨) +
                snaps.npz(게이트×9포즈×카메라높이 RGB-D)          → loss (a)
  topomap/      로봇별 기억 그래프 json + 노드 스냅샷 RGB-D npz
                (프리픽스 컷은 t 규칙으로 파생)                    → 결합형·평가
  episodes/     goal 주도 에피소드(divergence 배합 55/45):
                ep_*.json(로봇별 결과·verdict·goal 메타) +
                ep_*_<robot>.npz(프레임 RGB-D·포즈·waypoint GT 빈) → loss (b)
  instructions.json  3형태 지시 + 역파싱 결과                      → 지시 GT

모드: --mode pilot = 소규모(출력 check/MANSION/05_generate — data/ 금지 규칙),
      --mode full  = 본생성(출력 /data/EVLN_dataset). full 규모 수치는
      SCALE["full"] — **본생성 전 결정 대기 항목 반영 후 확정**(현재 잠정).
병렬: --only <스테이지>로 자기 분기(02와 동일), GPU 0~3 순환.
      스테이지 의존성: instructions는 episodes·topomap 완료 후.
모니터링: 스테이지별 progress.json(단계·완료/전체·초당 처리·타임스탬프) 갱신.

제안기 GT는 픽셀 히트맵(common.heat_gt) — 도달 가능 자유 공간의 래스터.
투영 규약은 models/EA_Nav.py pixel_to_pose와 동일(256²·HFOV 90°·pitch 0·
planar z-depth·픽셀 중심 = 인덱스+0.5). 규약 변경 시 양쪽 동시 수정.

실행(파일럿):
  docker exec -e OPENAI_API_KEY=... airlab_hw_mansion bash -c \
    'cd /workspace/research/scripts/datasets/MANSION && \
     python 05_generate_datasets.py --mode pilot \
     --building "/data/MansionWorld/mansionworld/public_hotel_dormitory_4f_300_fp001#0" \
     --out /workspace/research/check/MANSION/05_generate'
"""

import argparse
import glob
import importlib
import json
import math
import os
import subprocess
import time

import cv2
import numpy as np

import common

m02 = importlib.import_module("02_rendering")
m03 = importlib.import_module("03_topomap")
m04 = importlib.import_module("04_instructions")

RES = common.RES
FRAME_STEP = 1.5

SCALE = {
    "pilot": dict(n_gates=6, n_episodes=6, nondiv_pair_ratio=0.45,
                  n_impossible=1, topomap_robots=2, n_goal=4, n_r2r=2,
                  n_combo=3, n_notfound=2, height_buckets=4, bucket_w=0.05,
                  hard_ratio=0.2),
    # full 확정치(2026-07-26): 무갈림 배합 = **로봇쌍 단위 45%**(에피소드
    # 단위는 대표 4대 극단 치수 탓에 성립 불가 — 파일럿 실증. 쌍 단위가
    # 원 의도(z 과잉 반응 방지)에 부합). 높이 버킷 10cm(35→~18버킷, 렌더
    # 절반 — 10cm 시점차는 미미). 빌딩 단위 실행(호텔→오피스→주택 순차).
    "full": dict(n_gates=None, n_episodes=120, nondiv_pair_ratio=0.45,
                 n_impossible=12, topomap_robots=4, n_goal=70, n_r2r=18,
                 n_combo=30, n_notfound=10, height_buckets=None,
                 bucket_w=0.10, hard_ratio=0.2),
    # 매트릭스 부속 실행(엘베만 합성·주택·오피스)용 절반 규모 — 총 예산을
    # 실행당 200으로 곱하지 않고 배분(ETA 과대 방지, 사용자 지적)
    "full_half": dict(n_gates=None, n_episodes=60, nondiv_pair_ratio=0.45,
                      n_impossible=6, topomap_robots=2, n_goal=35,
                      n_r2r=9, n_combo=15, n_notfound=5,
                      height_buckets=None, bucket_w=0.10, hard_ratio=0.2),
}
STAGES = ("gates", "topomap", "episodes_plan", "episodes_plan_merge",
          "episodes_render", "gt_rederive", "instructions", "combined_eps",
          "evalsets")


def load_bld(args):
    """빌딩 로드 — 합성 변형은 common.load_building의 MANSION_VARIANT로 위임."""
    if getattr(args, "variant", "none") != "none":
        os.environ["MANSION_VARIANT"] = args.variant
    return common.load_building(args.building)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def progress(args, stage, done, total, extra=None):
    p = {"stage": stage, "done": done, "total": total,
         "t": time.strftime("%H:%M:%S"), **(extra or {})}
    with open(os.path.join(args.out, common.out_name(
            args.building, f"progress_{stage}.json")), "w") as fh:
        json.dump(p, fh)


def reps4(robots):
    """대표 4대 (01과 동일 규칙: 최소/최대 wheeled, 최대 humanoid, 중간 multileg)."""
    wh = sorted([r for r in robots if r["cls"] == "wheeled"],
                key=lambda r: r["w_eff"])
    hu = sorted([r for r in robots if r["cls"] == "humanoid"],
                key=lambda r: r["h"])
    ml = sorted([r for r in robots if r["cls"] == "multileg"],
                key=lambda r: r["w_eff"])
    seen, out = set(), []
    for r in (wh[0], wh[-1], hu[-1], ml[len(ml) // 2]):
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


class Renderer:
    """THOR 렌더 세션 — 일시 타임아웃(CreateHouse 경합 등, 실측 2회) 시
    컨트롤러를 재생성하고 현재 씬을 복구해 1회 재시도. 장시간 샤드가
    일시 오류 한 방에 죽는 것을 방지(본생성 필수 내성)."""

    def __init__(self):
        self.c = common.launch_controller(width=RES, height=RES, render=True)
        self.scene = None

    def _recreate(self, with_scene):
        try:
            self.c.stop()
        except Exception:
            pass
        time.sleep(5)
        self.c = common.launch_controller(width=RES, height=RES, render=True)
        if with_scene and self.scene is not None:
            common.load_floor_scene(self.c, self.scene)
            self._add_cam()

    def _add_cam(self):
        self.c.step(action="AddThirdPartyCamera",
                    position=dict(x=0, y=1, z=0),
                    rotation=dict(x=0, y=0, z=0), fieldOfView=90)

    def load(self, scene):
        for attempt in (0, 1):
            try:
                common.load_floor_scene(self.c, scene)
                self._add_cam()
                self.scene = scene
                return
            except Exception as e:
                if attempt:
                    raise
                log(f"THOR 씬 로드 실패({type(e).__name__}) — 재생성 재시도")
                self._recreate(with_scene=False)

    def cam(self, x, y, z, yaw):
        for attempt in (0, 1):
            try:
                return common.update_cam(self.c, x, y, z, yaw)
            except Exception as e:
                if attempt:
                    raise
                log(f"THOR 렌더 실패({type(e).__name__}) — 재생성 재시도")
                self._recreate(with_scene=True)

    def step(self, **kw):
        return self.c.step(**kw)

    def stop(self):
        try:
            self.c.stop()
        except Exception:
            pass


# ---------------------------------------------------------------- gates

def stage_gates(args, sc):
    """통과성 튜플: 라벨 행렬 + 게이트 스냅샷 RGB-D (loss (a))."""
    out = os.path.join(args.out, "gates")
    os.makedirs(out, exist_ok=True)
    floors = load_bld(args)
    robots = reps4(common.robot_pool())
    # 카메라 높이 버킷 — 비슷한 높이는 렌더 공유(확정 설계). 폭은 스케일
    # 파라미터(full 10cm — 렌더 수 절반, 시점차 미미)
    bw = sc.get("bucket_w", 0.05)
    pool_all = common.robot_pool()
    buckets = sorted({round(r["cam_h"] / bw) * bw for r in pool_all})
    if sc["height_buckets"]:
        buckets = buckets[:sc["height_buckets"]]

    gates_all, labels = [], []
    for no, geom in sorted(floors.items()):
        gs = m02.pick_gates(geom, robots, sc["n_gates"] or 10 ** 9)
        for g in gs:
            g["floor"] = no
        gates_all += gs
    # 라벨 행렬 = 게이트 × 전체 풀. 폭은 로봇 높이 단면에서 읽는다(gate_label)
    for g in gates_all:
        geom = floors[g["floor"]]
        labels.append([round(common.gate_label(geom, g, r), 3)
                       for r in pool_all])
    if args.shard == 0:  # gates.json은 대표 샤드만 기록 (내용 동일)
        with open(os.path.join(out, common.out_name(
                args.building, "gates.json")), "w") as fh:
            json.dump({"robots": [{"id": r["id"], "w_eff": r["w_eff"],
                                   "h": r["h"], "cam_h": r["cam_h"]}
                                  for r in pool_all],
                       "gates": gates_all, "labels": labels,
                       "snap_dists": m02.SNAP_DISTS,
                       "snap_angs": m02.SNAP_ANGS,
                       "cam_buckets": buckets,
                       "nshards": args.nshards}, fh, indent=1)

    r = Renderer()
    rgbs, deps, keys = [], [], []
    total = len(gates_all) * len(m02.SNAP_DISTS) * len(m02.SNAP_ANGS) \
        * len(buckets)
    t0, done, cur_fl = time.time(), 0, None
    my_gates = [(gi, g) for gi, g in enumerate(gates_all)
                if gi % args.nshards == args.shard]
    total = len(my_gates) * len(m02.SNAP_DISTS) * len(m02.SNAP_ANGS) \
        * len(buckets)
    for gi, g in my_gates:
        geom = floors[g["floor"]]
        if g["floor"] != cur_fl:
            r.load(geom.scene)
            cur_fl = g["floor"]
        adx, adz = m02.approach_dir(geom, g)
        base = math.degrees(math.atan2(-adx, -adz))  # 접근점→게이트 방향
        for di, dist in enumerate(m02.SNAP_DISTS):
            for ai, da in enumerate(m02.SNAP_ANGS):
                x = g["x"] + adx * dist
                z = g["z"] + adz * dist
                yaw = base + da
                for bi, ch in enumerate(buckets):
                    cam_y = min(ch, geom.ceil_h - 0.15)
                    rgb, dep = r.cam(x, cam_y, z, yaw)
                    rgbs.append(rgb[:, :, ::-1])  # BGR→RGB 저장
                    deps.append(dep.astype(np.float16))
                    keys.append((gi, di, ai, bi))
                    done += 1
        progress(args, f"gates_s{args.shard}", done, total,
                 {"sec_per_item": round((time.time() - t0) / max(done, 1), 3)})
        log(f"gates[s{args.shard}] {done}/{total}")
    r.stop()
    np.savez_compressed(os.path.join(out, common.out_name(
        args.building, f"snaps_shard{args.shard}.npz")),
                        rgb=np.stack(rgbs), depth=np.stack(deps),
                        key=np.array(keys, dtype=np.int32))
    log(f"gates[s{args.shard}] 완료: 게이트 {len(my_gates)}, 스냅샷 {done}")


# ---------------------------------------------------------------- topomap

def topomap_robots(n):
    """기억 그래프 로봇 n대: 03 대표 2대(최소 wheeled·중간 multileg) +
    최대 humanoid·최대 wheeled (full=4, half=2 앞 2대)."""
    pool = common.robot_pool()
    base = m03.pick_robots(pool)
    hu = sorted([r for r in pool if r["cls"] == "humanoid"],
                key=lambda r: r["h"])
    wh = sorted([r for r in pool if r["cls"] == "wheeled"],
                key=lambda r: r["w_eff"])
    seen, out = set(), []
    for r in base + [hu[-1], wh[-1]]:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out[:n]


def stage_topomap(args, sc):
    """기억 그래프 + 노드 스냅샷 RGB-D (결합형·평가 세트·기억 랭킹 소재)."""
    out = os.path.join(args.out, "topomap")
    os.makedirs(out, exist_ok=True)
    floors = load_bld(args)
    nav = common.BuildingNav(floors)
    robots = topomap_robots(sc["topomap_robots"])
    if args.robot:  # 로봇 단위 병렬 분기
        robots = [r for r in robots if r["id"] == args.robot]
    for robot in robots:
        legs, visited, small, failed = m03.explore_route(nav, floors, robot)
        gb = m03.GraphBuilder(floors, robot)  # anchor_scan=False (회전 불채택)
        for res in legs:
            gb.add_leg(res)
        nodes, edges = gb.nodes, gb.edges
        n_snap = sum(len(nd["headings"]) for nd in nodes)
        r = Renderer()
        rgbs, deps, keys = [], [], []
        cur_fl, t0, done = None, time.time(), 0
        for nd in nodes:
            geom = floors[nd["floor"]]
            if nd["floor"] != cur_fl:
                r.load(geom.scene)
                cur_fl = nd["floor"]
            cam_y = min(robot["cam_h"], geom.ceil_h - 0.15)
            for hi, hd in enumerate(nd["headings"]):
                rgb, dep = r.cam(nd["x"], cam_y, nd["z"], hd)
                rgbs.append(rgb[:, :, ::-1])
                deps.append(dep.astype(np.float16))
                keys.append((nd["id"], hi))
                done += 1
            progress(args, "topomap", done, n_snap,
                     {"robot": robot["id"]})
        r.stop()
        np.savez_compressed(os.path.join(out, common.out_name(
            args.building, f"{robot['id']}_snaps.npz")),
                            rgb=np.stack(rgbs), depth=np.stack(deps),
                            key=np.array(keys, dtype=np.int32))
        cuts = {str(f): int(math.ceil(len(nodes) * f))
                for f in m03.PREFIX_CUTS}
        # 커버리지 = 기억이 담은 가구 비율. 목표형 goal은 관측된 대상에서만
        # 뽑으므로 확보 가능한 goal 수의 상한이기도 하다(회전 불채택 영향 지표)
        cov = m03.object_coverage(floors, robot, nodes)
        with open(os.path.join(out, common.out_name(
                args.building, f"{robot['id']}.json")), "w") as fh:
            json.dump({"robot": robot["id"], "cls": robot["cls"],
                       "visited_rooms": visited, "failed_rooms": failed,
                       "prefix_cuts": cuts, "anchor_scan": False,
                       "coverage": cov, "nodes": nodes, "edges": edges},
                      fh, ensure_ascii=False, indent=1)
        log(f"topomap {robot['id']}: 노드 {len(nodes)} 스냅샷 {done} "
            f"| 가구 관측 {cov['objects_seen']}/{cov['objects_total']} "
            f"방 {cov['rooms_with_any']}/{cov['rooms_total']} "
            f"({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- episodes

def _drive_frames(r, floors, robot, res):
    """expert 결과 → 프레임 데이터 (02 episode_frames의 무장식 데이터 모드).

    02와 GT 규칙 동일(호길이 1.5m 샘플·belief tail 연장·엘베 문 개폐·
    대기/내부 hold·verdict) — 02 갱신 시 동기 필수. 제안기 GT는 히트맵
    (common.heat_gt), wp_world는 시선·heading look-ahead 지점.
    반환: dict of arrays.
    """
    wp_far = FRAME_STEP          # 시선·heading look-ahead 거리 (m)
    verd = {(fl, tuple(v)) for fl, v in res.get("verdicts", [])}
    climbs = {min(t["floor"], t["to"]): t["climb"]
              for t in res.get("transitions", []) if t["mode"] == "stairs"}
    elev_waits = {(t["floor"], tuple(t["from_cell"])): t.get("from_door")
                  for t in res.get("transitions", [])
                  if t["mode"] == "elevator" and t.get("from_cell")}
    elev_holds = {(t["floor"], tuple(t["wait_cell_in"])):
                  (t.get("from_door"), t.get("wait_steps", 3))
                  for t in res.get("transitions", [])
                  if t["mode"] == "elevator" and t.get("wait_cell_in")}
    entry_ends = {(t["floor"], tuple(t["wait_cell_in"]))
                  for t in res.get("transitions", [])
                  if t["mode"] == "elevator" and t.get("wait_cell_in")}
    exit_starts = {(t["to"], tuple(t["wait_cell_out"]))
                   for t in res.get("transitions", [])
                   if t["mode"] == "elevator" and t.get("wait_cell_out")}

    def door_ids():
        ev = r.step(action="Pass")
        return [o["objectId"] for o in ev.metadata["objects"]
                if o["objectId"].startswith("elevator_doors")]

    def set_doors(open_):
        for oid in door_ids():
            r.step(action="DisableObject" if open_ else "EnableObject",
                   objectId=oid)

    F = dict(rgb=[], depth=[], pose=[], heat=[], wp_world=[], special=[])

    def emit(rgb, dep, fl, x, z, yaw, cam_y, wp=None, wph=None, special=0):
        F["rgb"].append(rgb[:, :, ::-1])
        F["depth"].append(dep.astype(np.float16))
        F["pose"].append((fl, x, z, yaw, cam_y))
        # 제안기 GT = 도달 가능 자유 공간의 픽셀 래스터 (구 단일 타깃 폐기)
        geom = floors[fl]
        reach = common.reach_mask(geom, robot, x, z)
        F["heat"].append(common.heat_gt(geom, reach, x, z, yaw, cam_y, dep))
        F["wp_world"].append((wp[0], wp[1], wph) if wp is not None
                             else (0.0, 0.0, 0.0))
        F["special"].append(special)

    cur_fl = None
    for fl, cells in res["segments"]:
        geom = floors[fl]
        if fl != cur_fl:
            r.load(geom.scene)
            cur_fl = fl
        sm = common.natural_path(geom, robot, cells)
        boarding = (fl, tuple(sm[-1])) in entry_ends
        exiting = (fl, tuple(sm[0])) in exit_starts
        if boarding or exiting:
            set_doors(open_=True)
        pts = [geom.to_world(iz, ix) for iz, ix in sm]
        la_pts = list(pts)
        bt = res.get("belief_tail")
        if bt and bt[0] == fl and bt[1] and cells is res["segments"][-1][1]:
            acc, prev = 0.0, pts[-1]
            for czx in bt[1][::8]:
                w = geom.to_world(*czx)
                acc += math.hypot(w[0] - prev[0], w[1] - prev[1])
                prev = w
                la_pts.append(w)
                if acc > 4.0:
                    break
        climb = climbs.get(fl)
        cl_pts = ([(geom.to_world(cz, cx), y) for cz, cx, y in climb]
                  if climb else [])
        seglens = [math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(pts[:-1], pts[1:])]
        total_len = sum(seglens)
        la_seglens = [math.hypot(b[0] - a[0], b[1] - a[1])
                      for a, b in zip(la_pts[:-1], la_pts[1:])]
        la_total = sum(la_seglens)

        def arc_at(tq):
            kq, acc_ = 0, 0.0
            while kq < len(la_seglens) - 1 and acc_ + la_seglens[kq] < tq:
                acc_ += la_seglens[kq]
                kq += 1
            f = (tq - acc_) / max(la_seglens[kq], 1e-9)
            return (la_pts[kq][0] + (la_pts[kq + 1][0] - la_pts[kq][0]) * f,
                    la_pts[kq][1] + (la_pts[kq + 1][1] - la_pts[kq][1]) * f)

        t = 0.0
        while t <= total_len and seglens:
            x0, z0 = arc_at(t)
            tw = min(t + wp_far, la_total)
            wp = arc_at(tw)
            yaw = (m02.yaw_deg(wp[0] - x0, wp[1] - z0)
                   if math.hypot(wp[0] - x0, wp[1] - z0) > 0.15 else 0.0)
            h_off = 0.0
            iz, ix = geom.to_idx(x0, z0)
            if cl_pts and geom.stair_mask[min(iz, geom.nz - 1),
                                          min(ix, geom.nx - 1)]:
                h_off = min(cl_pts, key=lambda p: (p[0][0] - x0) ** 2
                            + (p[0][1] - z0) ** 2)[1]
            cam_y = min(h_off + robot["cam_h"], geom.ceil_h - 0.15)
            rgb, dep = r.cam(x0, cam_y, z0, yaw)
            wp2 = arc_at(min(tw + 0.8, la_total))
            hdx, hdz = wp2[0] - wp[0], wp2[1] - wp[1]
            if math.hypot(hdx, hdz) < 0.1:
                hdx = la_pts[-1][0] - la_pts[-2][0]
                hdz = la_pts[-1][1] - la_pts[-2][1]
            wph = m02.yaw_deg(hdx, hdz)
            sp = 1 if boarding else (2 if exiting else 0)
            emit(rgb, dep, fl, x0, z0, yaw, cam_y, wp, wph, sp)
            t += FRAME_STEP
        if boarding or exiting:
            set_doors(open_=False)
        end_cell = sm[-1]
        if any((fl, tuple(v)) in verd for v in sm[-3:]):
            ex, ez = geom.to_world(*end_cell)
            vp = arc_at(min(total_len + wp_far, la_total))
            vyaw = (m02.yaw_deg(vp[0] - ex, vp[1] - ez)
                    if math.hypot(vp[0] - ex, vp[1] - ez) > 0.3 else 0.0)
            cam_y = min(robot["cam_h"], geom.ceil_h - 0.15)
            rgb, dep = r.cam(ex, cam_y, ez, vyaw)
            emit(rgb, dep, fl, ex, ez, vyaw, cam_y, special=5)
        wd = elev_waits.get((fl, tuple(end_cell)))
        if wd:
            ax, az = geom.to_world(*end_cell)
            dxw, dzw = ax - wd[0], az - wd[1]
            L = max(math.hypot(dxw, dzw), 1e-6)
            ex, ez = wd[0] + dxw / L * 1.3, wd[1] + dzw / L * 1.3
            wyaw = m02.yaw_deg(wd[0] - ex, wd[1] - ez)
            cam_y = min(robot["cam_h"], geom.ceil_h - 0.15)
            for _ in range(3):
                rgb, dep = r.cam(ex, cam_y, ez, wyaw)
                emit(rgb, dep, fl, ex, ez, wyaw, cam_y, special=3)
        hd_ = elev_holds.get((fl, tuple(end_cell)))
        if hd_ and hd_[0]:
            ax, az = geom.to_world(*end_cell)
            wyaw = m02.yaw_deg(hd_[0][0] - ax, hd_[0][1] - az)
            cam_y = min(robot["cam_h"], geom.ceil_h - 0.15)
            for _ in range(int(hd_[1])):
                rgb, dep = r.cam(ax, cam_y, az, wyaw)
                emit(rgb, dep, fl, ax, az, wyaw, cam_y, special=4)
    return {k: np.array(v) for k, v in F.items()}


def _pair_counts(robots, results):
    """로봇쌍 갈림/무갈림 수 — 무갈림 배합은 쌍 단위(확정 규칙)."""
    d = nd = 0
    for i, a in enumerate(robots):
        for b in robots[i + 1:]:
            ra, rb = results[a["id"]], results[b["id"]]
            if not (ra.get("success") or ra.get("verdicts")) and \
                    not (rb.get("success") or rb.get("verdicts")):
                continue  # 양쪽 다 사전 불가 — 학습 표본 아님
            if common.divergent(ra, rb):
                d += 1
            else:
                nd += 1
    return d, nd


def stage_episodes_plan(args, sc):
    """에피소드 샘플링·계획 (렌더 없음) → episodes.json + plan.pkl(샤드 입력).

    무갈림 배합 = 로봇쌍 단위 45%(확정): 에피소드 단위는 대표 4대 극단
    치수라 성립 불가(파일럿 실증). 초반 1/3은 자유 수용(다양성), 이후
    무갈림 쌍 비율이 목표 미달이면 무갈림 쌍 우세 에피소드만 수용.
    """
    import pickle
    out = os.path.join(args.out, "episodes")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    floors = load_bld(args)
    nav = common.BuildingNav(floors)
    robots = reps4(common.robot_pool())
    registry = m04.goal_registry(floors)

    ns, shard = max(1, args.nshards), args.shard
    target = (sc["n_episodes"] + ns - 1) // ns
    n_imp_total = sc.get("n_impossible", 0)
    n_imp_quota = n_imp_total // ns + (1 if shard < n_imp_total % ns else 0)
    if ns > 1:  # 샤드별 독립 rng (병렬 계획 — plan이 벽시계 병목, 실측)
        rng = np.random.default_rng(args.seed * 1000 + shard * 7 + 1)
    ratio = sc["nondiv_pair_ratio"]
    eps, tried, p_div, p_nd, n_rej = [], 0, 0, 0, 0
    warmup = max(2, target // 10)
    f0 = min(floors)
    while len(eps) < target and tried < target * 50:
        tried += 1
        share = p_nd / max(p_div + p_nd, 1)
        g = m04.sample_goals(registry, rng, 1)[0]
        # 무갈림 쌍은 사실상 동일층 경로에서만 나옴(층 전환 수단 차이가
        # 곧 갈림) — 비율이 뒤처지면 시작층 골로 표집을 편향해 눈먼
        # 거절 공회전을 차단 (호텔 plan 35분 무진행 실측 → 수정)
        if len(eps) >= warmup and share < ratio and rng.random() < 0.85:
            for _ in range(40):
                if g["floor"] == f0:
                    break
                g = m04.sample_goals(registry, rng, 1)[0]
        s = m04.start_cell(floors, robots[0], rng, f0)
        results = {}
        for r in robots:
            ap = m04.approach_cell(floors[g["floor"]], r, g["x"], g["z"])
            results[r["id"]] = (nav.plan(r, s, (g["floor"], *ap))
                                if ap else {"success": False,
                                            "reason": "no_approach"})
        ok = [r for r in robots if results[r["id"]].get("success")
              or results[r["id"]].get("verdicts")]
        if not ok:
            continue
        ed, end_ = _pair_counts(robots, results)
        # 배합 제어를 **처음부터** 유지 — "후반 보정" 방식은 초반에 갈림만
        # 쌓이면 수 시간 계획 후 게이트에서 늦게 실패(시작 단계 버그).
        # 수용 후 예상 비율이 목표의 85% 밑으로 떨어지는 갈림 우세
        # 에피소드는 워ム업(10%) 이후 거절.
        proj = (p_nd + end_) / max(p_div + ed + p_nd + end_, 1)
        # 하한 0.85배는 0.38~0.40 평형 → 게이트(0.40) 아슬 미달 실측.
        # 0.95배(≈0.43)로 상향 — 게이트 위에서 평형하도록.
        if len(eps) >= warmup and end_ <= ed and proj < ratio * 0.95:
            n_rej += 1
            if n_rej % 10 == 0:
                log(f"배합 제어: 갈림 우세 거절 누적 {n_rej} (시도 "
                    f"{tried}, 무갈림 {share:.2f}/목표 {ratio})")
            continue
        p_div += ed
        p_nd += end_
        eid = (f"ep{len(eps):04d}" if ns == 1
               else f"s{shard}e{len(eps):04d}")
        eps.append({"id": eid, "goal": g, "start": s,
                    "divergent": ed > 0, "results": results,
                    "render_robots": [r["id"] for r in ok]})
        log(f"episode 샘플 {len(eps)}/{target} (쌍 갈림 {p_div}/무갈림 "
            f"{p_nd}, 무갈림 비율 {p_nd / max(p_div + p_nd, 1):.2f})")
        progress(args, "episodes_plan", len(eps), target,
                 {"pair_nondiv_share":
                  round(p_nd / max(p_div + p_nd, 1), 3)})
    if p_nd / max(p_div + p_nd, 1) < ratio:
        log(f"경고: 무갈림 쌍 비율 {p_nd / max(p_div + p_nd, 1):.2f} < "
            f"{ratio} (시도 한도)")

    # 불가능(embodiment) 쿼터: 관측 후 포기(verdict) 궤적이 상태 GT의
    # special=5·impossible_embodiment 사례를 실증해야 함 (사용자 지적 —
    # 성공 위주 표집이면 파일럿에 불가능 사례가 0건이 됨)
    n_imp = sum(1 for e in eps for rid in e["render_robots"]
                if e["results"][rid].get("verdicts")
                and not e["results"][rid].get("success"))
    tried_imp = 0
    by_width = sorted(robots, key=lambda r: -r["w_eff"])
    while n_imp < n_imp_quota and tried_imp < 60:
        tried_imp += 1
        g = m04.sample_goals(registry, rng, 1)[0]
        hit = None
        # 폭 내림차순 탐색 — 최대 폭은 대개 사전 판정(주행 없음)이라
        # "주행하며 관측 후 판정"하는 로봇을 찾아야 함 (02 abandon과 동일)
        for cand in by_width:
            ap = m04.approach_cell(floors[g["floor"]], cand,
                                   g["x"], g["z"])
            if ap is None:
                continue
            s_ = m04.start_cell(floors, cand, rng, min(floors))
            res = nav.plan(cand, s_, (g["floor"], *ap))
            if not res.get("success") and res.get("verdicts"):
                hit = (cand, s_, res)
                break
        if hit is None:
            continue
        cand, s_, res = hit
        eid = (f"ep{len(eps):04d}" if ns == 1
               else f"s{shard}e{len(eps):04d}")
        eps.append({"id": eid, "goal": g, "start": s_,
                    "divergent": True, "results": {cand["id"]: res},
                    "render_robots": [cand["id"]]})
        n_imp += 1
        log(f"불가능(embodiment) 에피소드 확보 {n_imp}/{n_imp_quota} "
            f"({cand['id']}, goal={g['gid']})")

    def state_of(rr):
        """에피소드×로봇 상태 GT (상태 출력 3종 규약)."""
        if rr.get("success"):
            return ("driving_with_wait"
                    if any(t["mode"] == "elevator"
                           for t in rr.get("transitions", []))
                    else "driving")
        # 둘 다 원인은 로봇 크기. 차이는 주행 궤적이 남느냐뿐이다.
        if rr.get("verdicts"):
            return "impossible_embodiment"   # 주행 후 포기 (궤적 있음)
        return "impossible_noroute"          # 이어진 길 자체가 없음 (궤적 0)

    metas = []
    for e in eps:
        metas.append({
            "id": e["id"], "divergent": e["divergent"],
            "goal": e["goal"],
            "start": {"floor": e["start"][0],
                      "cell": list(e["start"][1:])},
            "results": {rid: {
                "success": rr.get("success", False),
                "state": state_of(rr),  # 명시 state GT (사용자 지적)
                "reason": rr.get("reason"),
                "transitions": [
                    {k: v for k, v in t.items() if k != "climb"}
                    for t in rr.get("transitions", [])],
                "verdicts": rr.get("verdicts", []),
                "verdict_gates": rr.get("verdict_gates", []),
            } for rid, rr in e["results"].items()}})
    ej = ("episodes.json" if ns == 1
          else f"episodes_shard{shard}.json")
    pk = "plan.pkl" if ns == 1 else f"plan_shard{shard}.pkl"
    with open(os.path.join(out, common.out_name(
            args.building, ej)), "w") as fh:
        json.dump({"robots": [r["id"] for r in robots],
                   "pair_stats": {"div": p_div, "nondiv": p_nd},
                   "episodes": metas}, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out, common.out_name(
            args.building, pk)), "wb") as fh:
        pickle.dump({"episodes": eps,
                     "robots": [r["id"] for r in robots]}, fh)
    log(f"episodes_plan 완료: {len(eps)}개, 궤적 "
        f"{sum(len(e['render_robots']) for e in eps)}, 쌍 무갈림 비율 "
        f"{p_nd / max(p_div + p_nd, 1):.2f}")


def stage_episodes_plan_merge(args, sc):
    """plan 샤드 병합: ep id 재부여 + pair_stats 합산 → 최종 episodes.json/plan.pkl."""
    import pickle
    out = os.path.join(args.out, "episodes")
    tag = common.env_prefix(args.building)
    sj = sorted(glob.glob(os.path.join(
        out, f"{tag}_episodes_shard*.json")))
    sp = sorted(glob.glob(os.path.join(out, f"{tag}_plan_shard*.pkl")))
    assert sj and len(sj) == len(sp), f"샤드 불일치 {len(sj)}/{len(sp)}"
    metas, eps, div, nd, robots = [], [], 0, 0, None
    for jf, pf in zip(sj, sp):
        j = json.load(open(jf))
        with open(pf, "rb") as fh:
            p = pickle.load(fh)
        robots = j["robots"]
        div += j["pair_stats"]["div"]
        nd += j["pair_stats"]["nondiv"]
        for m, e in zip(j["episodes"], p["episodes"]):
            nid = f"ep{len(eps):04d}"
            m = dict(m, id=nid)
            e = dict(e, id=nid)
            metas.append(m)
            eps.append(e)
    with open(os.path.join(out, common.out_name(
            args.building, "episodes.json")), "w") as fh:
        json.dump({"robots": robots,
                   "pair_stats": {"div": div, "nondiv": nd},
                   "episodes": metas}, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out, common.out_name(
            args.building, "plan.pkl")), "wb") as fh:
        pickle.dump({"episodes": eps, "robots": robots}, fh)
    log(f"plan 병합: 샤드 {len(sj)} → ep {len(eps)}, 쌍 무갈림 "
        f"{nd / max(div + nd, 1):.2f}")


def stage_episodes_render(args, sc):
    """plan.pkl의 궤적을 샤드 분할 렌더 (GPU 병렬의 단위)."""
    import pickle
    out = os.path.join(args.out, "episodes")
    floors = load_bld(args)
    robots = {r["id"]: r for r in reps4(common.robot_pool())}
    with open(os.path.join(out, common.out_name(
            args.building, "plan.pkl")), "rb") as fh:
        plan = pickle.load(fh)
    trajs = [(e, rid) for e in plan["episodes"]
             for rid in e["render_robots"]]
    mine = [t for i, t in enumerate(trajs)
            if i % args.nshards == args.shard]
    r = Renderer()
    t0 = time.time()
    for k, (e, rid) in enumerate(mine):
        F = _drive_frames(r, floors, robots[rid], e["results"][rid])
        np.savez_compressed(
            os.path.join(out, common.out_name(
                args.building, f"{e['id']}_{rid}.npz")), **F)
        progress(args, f"episodes_render_s{args.shard}", k + 1,
                 len(mine), {"sec_per_traj":
                             round((time.time() - t0) / (k + 1), 1)})
        log(f"episodes_render[s{args.shard}] {k + 1}/{len(mine)} "
            f"({e['id']} {rid}, 프레임 {len(F['rgb'])})")
    r.stop()
    log(f"episodes_render[s{args.shard}] 완료: {len(mine)}궤적")


# ---------------------------------------------------------------- 지시

def _rederive_one(job):
    """에피소드 하나에 제안기·전역 GT를 다시 붙인다 (렌더 없음).

    dry면 npz를 건드리지 않고 통계만 돌려준다 — 양성 비율·분기 분포를 먼저
    확인하고 거리 상한을 정한 뒤에 실제로 덮어쓰기 위한 측정 모드.
    """
    path, building_dir, robot, dry, tm_nodes = job
    z = dict(np.load(path, allow_pickle=True))
    pose, depth = z["pose"], z["depth"].astype(np.float32)
    floors, heats, kinds, ratios, node_why = {}, [], [], [], []
    ghead, dyaws = [], []
    for t in range(len(pose)):
        fl = int(pose[t][0])
        if fl not in floors:
            floors[fl] = common.load_floor(building_dir, fl)
        geom = floors[fl]
        x, z_, yaw, cam_y = (float(pose[t][1]), float(pose[t][2]),
                             float(pose[t][3]), float(pose[t][4]))
        reach = common.reach_mask(geom, robot, x, z_)
        h = common.heat_gt(geom, reach, x, z_, yaw, cam_y, depth[t])
        heats.append(h)
        kinds.append(common.global_gt(pose, t, h))
        valid = h != common.HEAT_IGNORE
        ratios.append(float((h[valid] == 1).mean()) if valid.any() else 0.0)
        gh = common.global_head_gt(pose, t)
        ghead.append(-999.0 if gh is None else gh)
        # 홉별 heading 변화 — FOV 절반(±45°)을 넘으면 곡선 접근으로 못 돈다.
        # 초과분은 기억 노드를 골라 도는 경로가 있어야 진행이 막히지 않으므로,
        # 새 heading 방향(±45°, 8 m)에 노드가 있었는지 함께 기록한다.
        if t < len(pose) - 1 and int(pose[t + 1][0]) == fl:
            dh = abs((float(pose[t + 1][3]) - yaw + 180) % 360 - 180)
            nh = float(pose[t + 1][3])
            has_node = False
            for nd in tm_nodes:
                if int(nd["floor"]) != fl:
                    continue
                dx_, dz_ = nd["x"] - x, nd["z"] - z_
                r_ = math.hypot(dx_, dz_)
                if not 0.5 < r_ <= 8.0:
                    continue
                b_ = math.degrees(math.atan2(dx_, dz_))
                if abs((b_ - nh + 180) % 360 - 180) <= 45.0:
                    has_node = True
                    break
            dyaws.append((dh, has_node))
        # node 분기 내역 = 제안기 GT와 전역 GT의 정합 점검.
        #   offscreen — 다음 홉이 시야 밖(정상적인 방향 전환)
        #   zone      — 계단·엘베 존 진출입(존은 free_mask에서 빠지므로 정답이
        #               원래 노드 선택)
        #   other     — 그 밖. 두 GT가 어긋났다는 결함 신호이므로 0이어야 한다
        if kinds[-1][0] == common.GSEL["node"] and t < len(pose) - 1 \
                and int(pose[t + 1][0]) == fl:
            nx_, nz_ = float(pose[t + 1][1]), float(pose[t + 1][2])
            uv = common.heat_pixel(x, z_, yaw, cam_y, nx_, nz_)
            zone = geom.stair_mask | geom.elev_mask
            cells = [geom.to_idx(nx_, nz_), geom.to_idx(x, z_)]
            in_zone = any(zone[min(max(a, 0), geom.nz - 1),
                               min(max(b, 0), geom.nx - 1)] for a, b in cells)
            node_why.append("offscreen" if uv is None
                            else "zone" if in_zone else "other")
    if not dry:
        z["heat"] = np.array(heats, dtype=np.uint8)
        z["gsel"] = np.array([k[0] for k in kinds], dtype=np.int16)
        z["gsel_uv"] = np.array([k[1] for k in kinds], dtype=np.int16)
        z["gsel_head"] = np.array(ghead, dtype=np.float16)
        for k in ("wp_bin", "wp_px"):
            z.pop(k, None)
        np.savez_compressed(path, **z)
    return (os.path.basename(path), [k[0] for k in kinds], ratios, node_why,
            dyaws)


def _episode_instruction(out_dir, building, ep_name):
    """에피소드에 붙은 지시문 (없으면 빈 문자열)."""
    p = os.path.join(out_dir, common.out_name(building, "instructions.json"))
    if not os.path.exists(p):
        return ""
    try:
        recs = json.load(open(p)).get("records", [])
    except (OSError, ValueError):
        return ""
    eid = ep_name.split("_ep")[-1].split("_")[0] if "_ep" in ep_name else ""
    for r in recs:
        if eid and eid in str(r.get("episode", "")):
            return f"[{r.get('form', '')}] {r.get('instruction', '')}"
    return f"[{recs[0].get('form', '')}] {recs[0].get('instruction', '')}" \
        if recs else ""


def _gt_card(path, building_dir, robot, out_png, cols=8, instr=""):
    """재유도 GT 검수 카드 — 프레임별 히트맵 오버레이 + 전역 분기 표기.

    초록 = 도달 가능(양성), 회색 = 경계 무시, 노란 십자 = forward 정답 픽셀,
    화살표 = 그 지점의 도착 heading. 렌더 없이 rgb·depth·pose만으로 그린다.
    """
    z = np.load(path, allow_pickle=True)
    pose, depth, rgb = z["pose"], z["depth"].astype(np.float32), z["rgb"]
    wpw = z["wp_world"] if "wp_world" in z.files else None
    floors, tiles = {}, []
    heats = []
    for t in range(len(pose)):
        fl = int(pose[t][0])
        if fl not in floors:
            floors[fl] = common.load_floor(building_dir, fl)
        geom = floors[fl]
        x, z_, yaw, cam_y = (float(pose[t][1]), float(pose[t][2]),
                             float(pose[t][3]), float(pose[t][4]))
        reach = common.reach_mask(geom, robot, x, z_)
        heats.append(common.heat_gt(geom, reach, x, z_, yaw, cam_y, depth[t]))
    inv = {v: k for k, v in common.GSEL.items()}
    for t in range(len(pose)):
        im = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR).copy()
        R = im.shape[0]
        big = cv2.resize(heats[t], (R, R), interpolation=cv2.INTER_NEAREST)
        im[big == 1] = (0.45 * im[big == 1]
                        + 0.55 * np.array([60, 220, 60])).astype(np.uint8)
        im[big == common.HEAT_IGNORE] = (
            0.6 * im[big == common.HEAT_IGNORE]
            + 0.4 * np.array([200, 200, 200])).astype(np.uint8)
        k = R // common.HEAT_HW
        x, z_, yaw, cam_y = (float(pose[t][1]), float(pose[t][2]),
                             float(pose[t][3]), float(pose[t][4]))
        kind, uv = common.global_gt(pose, t, heats[t])
        gh = common.global_head_gt(pose, t)
        if kind == common.GSEL["forward"]:
            cx, cy = int((uv[0] + 0.5) * k), int((uv[1] + 0.5) * k)
            cv2.drawMarker(im, (cx, cy), (255, 255, 255),
                           cv2.MARKER_CROSS, 18, 2)
            # 도착 heading = 홉당 1개(선택기 예측 대상), GT는 pose[t+1] 실측
            if gh is not None:
                dx = int(22 * math.sin(math.radians(gh - yaw)))
                dy = -int(22 * math.cos(math.radians(gh - yaw)))
                cv2.arrowedLine(im, (cx, cy), (cx + dx, cy + dy),
                                (60, 170, 255), 2, cv2.LINE_AA, tipLength=0.35)
        valid = heats[t] != common.HEAT_IGNORE
        pos = (heats[t][valid] == 1).mean() if valid.any() else 0.0
        cv2.putText(im, f"t{t} {inv[kind]} pos={pos:.0%}", (6, R - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        tiles.append(im)
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(row, axis=1))
    img = np.concatenate(rows, axis=0)
    # 머리말 = 로봇·지시문. 지시가 무엇이었는지 없으면 GT만 봐서는 검수가 안 됨
    head = np.zeros((56, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(head, os.path.basename(out_png), (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(head, (instr or "(지시문 없음)")[:150], (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 220), 1)
    for i, (txt, col) in enumerate((
            ("green=reachable", (60, 220, 60)),
            ("orange=arrival heading", (60, 170, 255)),
            ("white cross=global GT", (255, 255, 255)))):
        cv2.putText(head, txt, (img.shape[1] - 900 + i * 230, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.imwrite(out_png, np.concatenate([head, img], axis=0))


def stage_gt_rederive(args, sc):
    """기존 에피소드에 제안기 히트맵·전역 선택 GT를 오프라인으로 재유도.

    RGB-D 렌더는 그대로 두고 GT만 갈아끼운다 — GT 규칙이 바뀌어도 재렌더가
    필요 없도록 렌더 단계와 분리한 스테이지(재생성 범위 규약).
    """
    import multiprocessing as mp
    dry = getattr(args, "dry_run", False)
    pool = common.robot_pool()
    eps = sorted(glob.glob(os.path.join(
        args.out, "episodes", common.env_prefix(args.building) + "*.npz")))
    jobs = []
    for p_ in eps:
        rid = "_".join(os.path.basename(p_).rsplit(".", 1)[0].split("_")[-3:])
        rb = next((r for r in pool if r["id"] == rid), None)
        if rb is not None:
            tmp = os.path.join(args.out, "topomap", common.out_name(
                args.building, rb["id"] + ".json"))
            nodes = (json.load(open(tmp))["nodes"]
                     if os.path.exists(tmp) else [])
            jobs.append((p_, args.building, rb, dry, nodes))
    if getattr(args, "limit", 0):
        jobs = jobs[:args.limit]
    # 워커마다 층 기하를 들고 있으므로 코어 수만큼 띄우면 메모리가 샌다
    n_proc = max(1, min(len(jobs), 32, mp.cpu_count() - 1))
    log(f"gt_rederive{'[측정]' if dry else ''}: 에피소드 {len(jobs)}개, "
        f"워커 {n_proc}, 히트맵 {common.HEAT_HW}² / 최대 {common.HEAT_MAX_M} m")
    t0, kinds, ratios, node_why, dyaw = time.time(), [], [], [], []
    with mp.Pool(n_proc) as mpool:
        for i, (name, kk, rr, nw, dy) in enumerate(
                mpool.imap_unordered(_rederive_one, jobs), 1):
            kinds += kk
            ratios += rr
            node_why += nw
            dyaw += dy
            if i % 20 == 0 or i == len(jobs):
                el = time.time() - t0
                log(f"  [{i}/{len(jobs)}] 프레임 {len(kinds)} | {el:.0f}s 경과 "
                    f"/ ETA {el / i * (len(jobs) - i):.0f}s")
                progress(args, "gt_rederive", i, len(jobs))
    n = max(len(kinds), 1)
    log("전역 선택 분기: " + " / ".join(
        f"{k} {kinds.count(v)} ({kinds.count(v) / n:.0%})"
        for k, v in common.GSEL.items()))
    a = np.array(ratios) if ratios else np.zeros(1)
    log(f"제안기 양성 비율: 중앙값 {np.median(a):.1%} / 평균 {a.mean():.1%} "
        f"/ 사분위 {np.percentile(a, 25):.1%}~{np.percentile(a, 75):.1%} "
        f"/ 5~95% {np.percentile(a, 5):.1%}~{np.percentile(a, 95):.1%}")
    log("  구간별 프레임 비율: " + " ".join(
        f"{lo:.0%}~{hi:.0%}={np.mean((a >= lo) & (a < hi)):.0%}"
        for lo, hi in ((0, .05), (.05, .2), (.2, .35), (.35, .5), (.5, 1.01))))
    if dry and args.card_dir and jobs:
        os.makedirs(args.card_dir, exist_ok=True)
        for p_, _, rb, _, _tm in jobs[:args.n_cards]:
            name = os.path.basename(p_).rsplit(".", 1)[0]
            _gt_card(p_, args.building, rb,
                     os.path.join(args.card_dir, name + "_gt.png"),
                     instr=_episode_instruction(args.out, args.building, name))
            log(f"  카드: {name}_gt.png")
    if node_why:
        nn = len(node_why)
        log("  node 분기 내역(같은 층): " + " / ".join(
            f"{k} {node_why.count(k)} ({node_why.count(k) / nn:.0%})"
            for k in ("offscreen", "zone", "other")))
    dy = np.array([d for d, _ in dyaw]) if dyaw else np.zeros(1)
    over = [h for d, h in dyaw if d > 45.0]
    log(f"홉별 heading 변화: 중앙값 {np.median(dy):.1f}° / 90퍼센타일 "
        f"{np.percentile(dy, 90):.1f}° / ±45°(FOV 절반) 초과 "
        f"{np.mean(dy > 45):.1%}")
    if over:
        log(f"  초과 홉 {len(over)}건 중 그 방향에 기억 노드 있음: "
            f"{sum(over) / len(over):.0%} (노드 선택으로 돌 수 있는 비율)")


def stage_instructions(args, sc):
    """생성된 에피소드·기억 그래프에 3형태 지시 부착 (04 로직 재사용)."""
    from openai import OpenAI
    client = OpenAI()
    rng = np.random.default_rng(args.seed)
    floors = load_bld(args)
    robots = reps4(common.robot_pool())
    rmap = {r["id"]: r for r in robots}
    registry = m04.goal_registry(floors)
    with open(os.path.join(args.out, "episodes", common.out_name(
            args.building, "episodes.json"))) as fh:
        eps = json.load(fh)["episodes"]

    records, n_pass, done = [], 0, 0
    total = min(sc["n_goal"], len(eps)) + sc["n_r2r"] * 2 + sc["n_combo"]

    def parse_goal(instr, g):
        idx = rng.choice(len(registry), size=9, replace=False)
        cands = [registry[i] for i in idx if registry[i]["gid"] != g["gid"]][:9]
        cands.append(dict(g))
        rng.shuffle(cands)
        gt_i = next(i for i, cc in enumerate(cands) if cc["gid"] == g["gid"])
        p = m04.llm_json(client, m04.PARSE_MODEL, m04.PARSE_GOAL_SYS,
                         json.dumps({"instruction": instr, "candidates": [
                             {"number": i, "object": cc["name"],
                              "room": cc["rtype"], "floor": cc["floor"]}
                             for i, cc in enumerate(cands)]}))
        return p, p.get("choice") == gt_i

    # 목표형 — 에피소드 goal에 부착. HARD_RATIO만큼 어려운 표현으로 생성하고
    # 거짓 서수는 폐기·재생성한다(04와 같은 규약).
    goal_eps = eps[:sc["n_goal"]]
    n_hard = int(round(len(goal_eps) * sc["hard_ratio"]))
    hard_of = {i: m04.HARD_KINDS[i % len(m04.HARD_KINDS)]
               for i in range(len(goal_eps) - n_hard, len(goal_eps))}
    for i_, e in enumerate(goal_eps):
        g = e["goal"]
        ctx = [x["name"] for x in registry
               if x["rid"] == g["rid"] and x["gid"] != g["gid"]][:5]
        kind = hard_of.get(i_)
        sys_p = m04.GEN_GOAL_SYS + (" " + m04.GEN_HARD_SYS[kind] if kind
                                    else "")
        for attempt in range(m04.GEN_RETRY + 1):
            o = m04.llm_json(client, m04.GEN_MODEL, sys_p, json.dumps(
                {"goal": {"object": g["name"], "room_type": g["rtype"],
                          "floor": g["floor"]}, "other_objects_in_room": ctx}))
            instr = o.get("instruction", "")
            conf = m04.ordinal_conflict(instr, g, registry)
            p, ok = parse_goal(instr, g)
            if conf is None and ok:
                break
        records.append({"form": "goal", "episode": e["id"], "goal": g,
                        "hard": kind, "conflict": conf, "retries": attempt,
                        "instruction": instr, "parse": p,
                        "pass": ok and conf is None})
        n_pass += ok and conf is None
        done += 1
        progress(args, "instructions", done, total)

    # R2R형 — 갈림 에피소드에서 성공 로봇쌍
    div_eps = [e for e in eps if e["divergent"]][:sc["n_r2r"]]
    nav = common.BuildingNav(floors)
    for e in div_eps:
        for rid, rr in e["results"].items():
            if not rr.get("success") or rid not in rmap:
                continue
            robot = rmap[rid]
            ap = m04.approach_cell(floors[e["goal"]["floor"]], robot,
                                   e["goal"]["x"], e["goal"]["z"])
            res = nav.plan(robot, (e["start"]["floor"],
                                   *e["start"]["cell"]),
                           (e["goal"]["floor"], *ap))
            if not res.get("success"):
                continue
            seq = m04.route_landmarks(floors, registry, res, e["goal"])
            seq_full = seq + [{"goal": e["goal"]["name"],
                              "room": e["goal"]["rtype"]}]
            o = m04.llm_json(client, m04.GEN_MODEL, m04.GEN_R2R_SYS,
                             json.dumps({"sequence": seq_full}))
            instr = o.get("instruction", "")
            vocab = sorted({x["landmark"] for x in seq if "landmark" in x}
                           | {"stairs", "elevator", e["goal"]["name"]})
            p = m04.llm_json(client, m04.PARSE_MODEL, m04.PARSE_R2R_SYS,
                             json.dumps({"instruction": instr,
                                         "vocabulary": vocab}))
            gt_seq = [x.get("landmark") or x.get("event") for x in seq]
            ratio = m04.lcs_ratio(gt_seq,
                                  [str(x) for x in p.get("sequence", [])])
            records.append({"form": "r2r", "episode": e["id"],
                            "robot": rid, "goal": e["goal"],
                            "sequence": seq_full, "instruction": instr,
                            "sub_instructions":
                                o.get("sub_instructions", []),
                            "lcs": round(ratio, 2), "pass": ratio >= 0.6})
            n_pass += records[-1]["pass"]
            done += 1
            progress(args, "instructions", done, total)

    # 결합형 — topomap 프리픽스 관측 goal (04 규칙)
    tm_path = os.path.join(args.out, "topomap")
    wh_id = m03.pick_robots(common.robot_pool())[0]["id"]
    wh_full = m03.pick_robots(common.robot_pool())[0]
    tm = json.load(open(os.path.join(tm_path, common.out_name(
        args.building, wh_id + ".json"))))
    k = tm["prefix_cuts"]["0.5"]
    known_rooms = set()
    for nd in tm["nodes"][:k]:
        geom = floors[nd["floor"]]
        iz, ix = geom.to_idx(nd["x"], nd["z"])
        ri = geom.room_grid[min(iz, geom.nz - 1), min(ix, geom.nx - 1)]
        if ri >= 0:
            known_rooms.add((nd["floor"], geom.rooms[ri]["id"]))
    import re as _re
    from collections import Counter as _C
    kc = _C((no, _re.sub(r"\s*\d+$", "", m04.pretty_room(rid, no)[0]))
            for no, rid in known_rooms)
    known = sorted(f"{rt} (floor {no})" + (f" x{c_}" if c_ > 1 else "")
                   for (no, rt), c_ in kc.items())
    known_modes = sorted({e_["mode"] for e_ in tm["edges"]
                          if e_["mode"] != "walk" and e_["a"] < k
                          and e_["b"] < k})
    torch, dev = common.torch_dev()
    obs_from = {}    # 프리픽스 내 최초 관측 노드 (결합형 goal 풀)
    obs_full = {}    # 전체 기억 그래프 가시 노드 목록 (goal grounding GT)
    wh = rmap.get(wh_id) or wh_full
    for fl in sorted(floors):
        geom = floors[fl]
        objs = [g for g in registry if g["floor"] == fl and g["unique"]]
        nds = [nd for nd in tm["nodes"] if nd["floor"] == fl]
        if not objs or not nds:
            continue
        # LOS 목표 = 오브젝트 앞 최근접 자유 셀 — 중심 셀은 침대·콘솔처럼
        # 큰 가구에서 자기 몸에 가려짐(제외 반경 3셀 ≪ 풋프린트, 실측:
        # visible_from_nodes가 빈 리스트)
        tiny = {"w_eff": common.BELIEF_W, "h": 0.4}
        cells_obj = []
        for g in objs:
            ap = m04.approach_cell(geom, tiny, g["x"], g["z"])
            cells_obj.append(ap if ap else geom.to_idx(g["x"], g["z"]))
        block_t = torch.from_numpy(
            common.sight_block(geom, wh["cam_h"])).to(dev)
        for nd in nds:
            for hd in nd["headings"]:
                vis = common.gates_in_view(block_t,
                                           geom.to_idx(nd["x"], nd["z"]),
                                           math.radians(hd), cells_obj)
                for g_, v in zip(objs, vis):
                    if not v:
                        continue
                    obs_full.setdefault(g_["gid"], [])
                    if nd["id"] not in obs_full[g_["gid"]]:
                        obs_full[g_["gid"]].append(nd["id"])
                    if nd["id"] < k and g_["gid"] not in obs_from:
                        obs_from[g_["gid"]] = nd["id"]
    pool = [g for g in registry if g["gid"] in obs_from]
    # 결합형 start = 탐사 중단 지점(프리픽스 마지막 노드) — 로봇이 논리적으로
    # 서 있을 수 있는 위치(사용자 지적: 임의 시작은 기억과 모순). goal은
    # 그 위치에서 실제 도달 가능한 것만(Not found 평가 케이스는 별도).
    end_nd = tm["nodes"][k - 1]
    s_cell = floors[end_nd["floor"]].to_idx(end_nd["x"], end_nd["z"])
    start_c = (end_nd["floor"], *s_cell)
    pool2 = []
    for g in pool:
        ap = m04.approach_cell(floors[g["floor"]], wh, g["x"], g["z"])
        if ap is None:
            continue
        res = nav.plan(wh, start_c, (g["floor"], *ap))
        if res.get("success"):
            pool2.append(g)
    log(f"결합형 goal 풀: 관측 {len(pool)} → 도달 가능 {len(pool2)}")
    pool = pool2
    idxs = rng.choice(len(pool), size=min(sc["n_combo"], len(pool)),
                      replace=False)
    for g in (pool[j] for j in idxs):
        o = m04.llm_json(client, m04.GEN_MODEL, m04.GEN_COMBO_SYS,
                         json.dumps({"known_rooms": known,
                                     "known_floor_transitions": known_modes,
                                     "goal": {"object": g["name"],
                                              "room_type": g["rtype"],
                                              "floor": g["floor"]}}))
        p, ok = parse_goal(o.get("instruction", ""), g)
        records.append({"form": "combined", "goal": g,
                        "memory_robot": wh_id, "memory_prefix": "0.5",
                        "start_node": end_nd["id"],
                        "start": {"floor": end_nd["floor"],
                                  "x": end_nd["x"], "z": end_nd["z"]},
                        "start_reachable": True,
                        "observed_from_node": obs_from[g["gid"]],
                        "instruction": o.get("instruction", ""),
                        "parse": p, "pass": ok})
        n_pass += ok
        done += 1
        progress(args, "instructions", done, total)

    # goal grounding GT: goal형·결합형 레코드에 "기억 그래프 어느 노드
    # 스냅샷에 보이는가" 부착 (CLIP grounding·기억 랭킹 평가 GT)
    for r in records:
        if r["form"] in ("goal", "combined"):
            r["visible_from_nodes"] = obs_full.get(r["goal"]["gid"], [])
            r["grounding_memory_robot"] = wh_id
    with open(os.path.join(args.out, common.out_name(
            args.building, "instructions.json")), "w") as fh:
        json.dump({"gen_model": m04.GEN_MODEL,
                   "parse_model": m04.PARSE_MODEL,
                   "n": len(records), "n_pass": n_pass,
                   "records": records}, fh, ensure_ascii=False, indent=1)
    log(f"instructions 완료: {n_pass}/{len(records)} 역파싱 통과")


# ------------------------------------------- 결합형 궤적 + 평가 세트

def stage_combined_eps(args, sc):
    """결합형 expert 궤적 렌더: start=프리픽스 끝 노드 → 관측된 goal.

    instructions.json의 combined 레코드(도달성 검증 완료)를 그대로 궤적화 —
    기억 구간 주행 + 미탐색 꼬리 접근이 loss (b)의 결합형 표본.
    """
    out = os.path.join(args.out, "episodes_combined")
    os.makedirs(out, exist_ok=True)
    floors = load_bld(args)
    nav = common.BuildingNav(floors)
    with open(os.path.join(args.out, common.out_name(
            args.building, "instructions.json"))) as fh:
        recs = [r for r in json.load(fh)["records"]
                if r["form"] == "combined"]
    pool = {rb["id"]: rb for rb in common.robot_pool()}
    rend = Renderer()
    metas, t0 = [], time.time()
    for i, r in enumerate(recs):
        robot = pool[r["memory_robot"]]
        st = r["start"]
        s_cell = floors[st["floor"]].to_idx(st["x"], st["z"])
        g = r["goal"]
        ap = m04.approach_cell(floors[g["floor"]], robot, g["x"], g["z"])
        res = nav.plan(robot, (st["floor"], *s_cell), (g["floor"], *ap))
        if not res.get("success"):
            log(f"combined_eps {i}: 계획 실패({res.get('reason')}) — 스킵")
            continue
        F = _drive_frames(rend, floors, robot, res)
        np.savez_compressed(os.path.join(out, common.out_name(
            args.building, f"cep{i:04d}.npz")), **F)
        metas.append({"id": f"cep{i:04d}", "instruction_idx": i,
                      "goal": g, "robot": robot["id"],
                      "start_node": r["start_node"], "start": st,
                      "memory_prefix": r["memory_prefix"],
                      "transitions": [
                          {k: v for k, v in t.items() if k != "climb"}
                          for t in res.get("transitions", [])]})
        progress(args, "combined_eps", i + 1, len(recs),
                 {"sec_per_traj": round((time.time() - t0) / (i + 1), 1)})
        log(f"combined_eps {i + 1}/{len(recs)} (프레임 {len(F['rgb'])})")
    rend.stop()
    with open(os.path.join(out, common.out_name(
            args.building, "combined_eps.json")), "w") as fh:
        json.dump({"episodes": metas}, fh, ensure_ascii=False, indent=1)
    log(f"combined_eps 완료: {len(metas)}/{len(recs)}")


def stage_evalsets(args, sc):
    """평가 전용 세트 3종 (조합·메타 수준, 렌더 없음 — Not found만 LLM 지시).

    · zswap: 갈림 에피소드의 로봇쌍 — 같은 (시작,목표,기억)에서 z만 교체 시
      기대 갈림 GT (z 스왑 통제 세트)
    · instr_conflict: 로봇 A의 r2r 지시를 로봇 B 에피소드에 결합 — GT 행동 =
      B 자신의 state (자가 우회 or 불가능 판정)
    · notfound: 프리픽스 기억에 관측된 적 없는 goal + 지시 → GT =
      impossible_unexplored (Not found 보고가 정답)
    """
    rng = np.random.default_rng(args.seed + 5)
    floors = load_bld(args)
    registry = m04.goal_registry(floors)
    with open(os.path.join(args.out, "episodes", common.out_name(
            args.building, "episodes.json"))) as fh:
        eps = json.load(fh)["episodes"]
    with open(os.path.join(args.out, common.out_name(
            args.building, "instructions.json"))) as fh:
        instr = json.load(fh)["records"]

    zswap = []
    for e in eps:
        rids = list(e["results"])
        for i, a in enumerate(rids):
            for b in rids[i + 1:]:
                ra, rb = e["results"][a], e["results"][b]
                if ra["state"] == rb["state"] == "impossible_noroute":
                    continue
                ma = [t["mode"] for t in ra["transitions"]]
                mb = [t["mode"] for t in rb["transitions"]]
                if ra["state"] != rb["state"] or ma != mb:
                    zswap.append({"episode": e["id"], "robot_a": a,
                                  "robot_b": b,
                                  "expect_a": {"state": ra["state"],
                                               "modes": ma},
                                  "expect_b": {"state": rb["state"],
                                               "modes": mb}})

    conflict = []
    for ri, r in enumerate(instr):
        if r["form"] != "r2r":
            continue
        e = next((e for e in eps if e["id"] == r.get("episode")), None)
        if e is None:
            continue
        for rid, rr in e["results"].items():
            if rid == r["robot"]:
                continue
            src_modes = [t["mode"] for t in
                         e["results"][r["robot"]]["transitions"]]
            tgt_modes = [t["mode"] for t in rr["transitions"]]
            if rr["state"].startswith("impossible"):
                gt = "report_impossible"
            elif src_modes != tgt_modes:
                gt = "self_detour"   # 지시 수단과 달라도 자기 경로로 도달
            else:
                gt = "follow"
            conflict.append({"instruction_idx": ri, "episode": e["id"],
                             "instruction_robot": r["robot"],
                             "target_robot": rid, "gt_behavior": gt,
                             "target_state": rr["state"]})

    # notfound: 프리픽스 기억에 없는 goal (기억 로봇 기준)
    tmdir = os.path.join(args.out, "topomap")
    wh_id = m03.pick_robots(common.robot_pool())[0]["id"]
    tm = json.load(open(os.path.join(tmdir, common.out_name(
        args.building, wh_id + ".json"))))
    k = tm["prefix_cuts"]["0.5"]
    seen_gids = set()
    for r in instr:
        if r["form"] in ("goal", "combined"):
            for nid in r.get("visible_from_nodes", []):
                if nid < k:
                    seen_gids.add(r["goal"]["gid"])
    end_nd = tm["nodes"][k - 1]
    cand = [g for g in registry if g["unique"]
            and g["gid"] not in seen_gids]
    idx = rng.choice(len(cand), size=min(sc.get("n_notfound", 0),
                                         len(cand)), replace=False)
    notfound = []
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception:
        client = None
    for j in idx:
        g = cand[int(j)]
        instr_txt = ""
        if client is not None:
            o = m04.llm_json(client, m04.GEN_MODEL, m04.GEN_GOAL_SYS,
                             json.dumps({"goal": {"object": g["name"],
                                                  "room_type": g["rtype"],
                                                  "floor": g["floor"]},
                                         "other_objects_in_room": []}))
            instr_txt = o.get("instruction", "")
        notfound.append({"goal": g, "instruction": instr_txt,
                         "memory_robot": wh_id, "memory_prefix": "0.5",
                         "start_node": end_nd["id"],
                         "gt_state": "impossible_unexplored"})

    with open(os.path.join(args.out, common.out_name(
            args.building, "evalsets.json")), "w") as fh:
        json.dump({"zswap": zswap, "instr_conflict": conflict,
                   "notfound": notfound}, fh, ensure_ascii=False, indent=1)
    log(f"evalsets 완료: zswap {len(zswap)} / conflict {len(conflict)} / "
        f"notfound {len(notfound)}")


# ---------------------------------------------------------------- 러너

def _spawn(args, stage, gpu, extra=None, logname=None):
    """스테이지 서브프로세스 (unbuffered, 로그 파일 즉시 기록 — 실시간 모니터링)."""
    env = dict(os.environ, MANSION_GPU=str(gpu % 4))
    cmd = ["python", "-u", os.path.abspath(__file__), "--mode", args.mode,
           "--building", args.building, "--out", args.out,
           "--seed", str(args.seed), "--variant", args.variant,
           "--only", stage] + (extra or [])
    logdir = os.path.join(args.out, "logs")
    os.makedirs(logdir, exist_ok=True)
    name = logname or stage
    lf = open(os.path.join(logdir, common.out_name(
        args.building, name + ".log")), "a")
    p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    log(f"[병렬] {name} 시작 (GPU {env['MANSION_GPU']})")
    return p, name


def _wait(procs):
    fail = False
    for p, name in procs:
        rc = p.wait()
        log(f"[병렬] {name} 종료 (rc={rc})")
        fail |= rc != 0
    return fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full", "full_half"),
                    default="pilot")
    ap.add_argument("--building", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=STAGES)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--robot", default=None,
                    help="topomap 로봇 단위 병렬 분기용")
    ap.add_argument("--variant", choices=("none", "elevator_only"),
                    default="none", help="합성 씬 변형 (층 전환 매트릭스)")
    ap.add_argument("--skip", nargs="+", default=[],
                    help="건너뛸 스테이지 (예: 변형 실행에서 gates)")
    ap.add_argument("--dry-run", action="store_true",
                    help="gt_rederive 측정 전용 — npz를 쓰지 않고 통계만 낸다")
    ap.add_argument("--limit", type=int, default=0,
                    help="gt_rederive 처리 에피소드 수 상한 (0=전체)")
    ap.add_argument("--card-dir", default=None,
                    help="측정 모드에서 GT 검수 카드를 저장할 폴더")
    ap.add_argument("--n-cards", type=int, default=3)
    ap.add_argument("--gate-shards", type=int, default=0,
                    help="게이트 렌더 샤드 수 (0=워커 수, 최대 4). 기존 샤드 "
                         "수와 맞추면 잔재 없이 전량 덮어쓴다")
    args = ap.parse_args()
    if args.variant != "none":  # 접두사·로딩 공용 스위치 (파싱 직후 고정)
        os.environ["MANSION_VARIANT"] = args.variant
    os.makedirs(args.out, exist_ok=True)
    sc = SCALE[args.mode]

    fns = {"gates": stage_gates, "topomap": stage_topomap,
           "episodes_plan": stage_episodes_plan,
           "episodes_plan_merge": stage_episodes_plan_merge,
           "episodes_render": stage_episodes_render,
           "gt_rederive": stage_gt_rederive,
           "instructions": stage_instructions,
           "combined_eps": stage_combined_eps,
           "evalsets": stage_evalsets}
    if args.only:
        fns[args.only](args, sc)
        return

    # 이전 실행 progress 잔재 정리 — 모니터링 혼선 방지
    for f in os.listdir(args.out) if os.path.isdir(args.out) else []:
        if "progress_" in f:
            os.remove(os.path.join(args.out, f))
    with open(os.path.join(args.out, common.out_name(
            args.building, "meta.json")), "w") as fh:
        json.dump({"mode": args.mode, "scale": sc, "seed": args.seed,
                   "building": os.path.basename(args.building.rstrip("/")),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh,
                  indent=1)
    # robots.json: 로봇 id → URDF 경로·capability — z 인코더(URDFEncoder) 학습이
    # 데이터셋만으로 robot.urdf에 도달 가능해야 함.
    pool = common.robot_pool()
    with open(os.path.join(args.out, common.out_name(
            args.building, "robots.json")), "w") as fh:
        json.dump({"pool_root": common.URDF_POOL_DIR,
                   "robots": [{"id": r["id"], "cls": r["cls"],
                               "urdf": f"{r['cls']}/{r['id']}/robot.urdf",
                               "w_eff": r["w_eff"], "h": r["h"],
                               "cam_h": r["cam_h"],
                               "stairs_ok": r["stairs_ok"]}
                              for r in pool]}, fh, ensure_ascii=False,
                  indent=1)

    # 웨이브 오케스트레이션 — **웨이브당 동시 렌더러 ≤4** (동시 CreateHouse
    # 과적이 100초 타임아웃 유발, 실측 2회. 무렌더 스테이지는 병치 가능)
    # --skip으로 재사용 자산의 스테이지를 건너뛴다: GT 규칙만 바뀐 재생성에서
    # 주행 에피소드 렌더는 그대로 쓰고 GT만 오프라인 재유도한다.
    W = max(1, args.workers)
    gpu = 0
    t_all = time.time()
    # W1: gates 샤드(렌더 ≤4) ∥ episodes_plan / gt_rederive(무렌더)
    procs = []
    n_gs = args.gate_shards or min(W, 4)
    if "gates" not in args.skip:
        for sh in range(n_gs):
            procs.append(_spawn(args, "gates", gpu,
                                ["--shard", str(sh),
                                 "--nshards", str(n_gs)],
                                logname=f"gates_s{sh}"))
            gpu += 1
    if "episodes_plan" not in args.skip:
        procs.append(_spawn(args, "episodes_plan", gpu))
        gpu += 1
    if "gt_rederive" not in args.skip:
        procs.append(_spawn(args, "gt_rederive", gpu))
        gpu += 1
    if _wait(procs):
        log("W1 실패 — 중단")
        return
    log(f"[웨이브] W1 완료 ({time.time() - t_all:.0f}s 누적)")
    # W2: topomap 로봇별(렌더 ≤4)
    procs = []
    topo_robots = ([] if "topomap" in args.skip
                   else topomap_robots(sc["topomap_robots"]))
    for tr in topo_robots:
        procs.append(_spawn(args, "topomap", gpu, ["--robot", tr["id"]],
                            logname=f"topomap_{tr['id']}"))
        gpu += 1
    if _wait(procs):
        log("W2 실패 — 중단")
        return
    log(f"[웨이브] W2 완료 ({time.time() - t_all:.0f}s 누적)")
    # W3: episodes_render 샤드(렌더 4) ∥ instructions(무렌더, topomap 의존)
    procs = []
    if "episodes_render" not in args.skip:
        for sh in range(W):
            procs.append(_spawn(args, "episodes_render", gpu,
                                ["--shard", str(sh), "--nshards", str(W)],
                                logname=f"episodes_render_s{sh}"))
            gpu += 1
    if "instructions" not in args.skip:
        stage_instructions(args, sc)  # 현 프로세스(API 키 상속), 렌더와 병행
    if _wait(procs):
        log("W3 실패 — 중단")
        return
    log(f"[웨이브] W3 완료 ({time.time() - t_all:.0f}s 누적)")
    # W4: combined_eps(렌더 1) ∥ evalsets(무렌더)
    procs = ([] if "combined_eps" in args.skip
             else [_spawn(args, "combined_eps", 0)])
    if "evalsets" not in args.skip:
        stage_evalsets(args, sc)
    if _wait(procs):
        log("combined_eps 실패")
        return
    log(f"전체 완료 (총 {time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()

"""R2R-CE → EA-Nav 학습 데이터 변환 (Go2 단일 embodiment).

MANSION 생성 파이프라인과 독립이며, 산출 규약만 동일하다. 즉 loaders와
train_waypoint·train_global이 태그만 바꿔 그대로 읽는다.

기존 /data/R2R_go2 덤프는 구 wp_bin(각도·거리 빈) 규약이라 히트맵 제안기
학습에 쓸 수 없다. 여기서는 제안기 GT를 MANSION과 같은 정의 — **로봇이 실제로
갈 수 있는 자유 공간의 픽셀 래스터** — 로 다시 유도한다. 격자는 AI2-THOR
점유 격자 대신 habitat navmesh(Go2 반경·높이로 재계산)의 탑다운 뷰를 쓰며,
현재 위치와 이어진 연결 성분만 남기는 규칙은 동일하다.

산출 (루트 기본 /data/R2R_EVLN, 태그 r2rce_go2):
  episodes/{tag}_episodes.json      에피소드 목록 + 로봇별 상태
  {tag}_instructions.json           R2R 원문 지시 (form=r2r)
  {tag}_robots.json                 Go2 1대
  episodes/{tag}_{epid}_{rid}.npz   rgb·depth·pose·heat·wp_world·gsel*

사용 (hw_vln_ce 컨테이너):
  python /research/scripts/datasets/R2RCE/build_go2.py \
      --split train --shard 0 --nshards 4 [--limit 20]
"""
import argparse
import glob
import gzip
import json
import math
import os
import sys
import time

import cv2
import numpy as np

# ------------------------------------------------------------ 렌더·라벨 규약
# 값은 MANSION 생성부(common.py)와 반드시 같아야 한다 — 같은 모델이 두
# 데이터셋을 함께 학습하므로 라벨 도메인이 어긋나면 조용히 망가진다.
RES = 256               # 전방 RGB-D 렌더 해상도
FX = RES / 2.0          # HFOV 90° 핀홀 초점거리 (px)
HEAT_HW = 32            # 제안기 히트맵 해상도
HEAT_IGNORE = 2         # 양·음 경계 1픽셀 = 라벨 무시
HEAT_MAX_M = 5.0        # 도달 가능 판정 최대 거리 (m)
HEAT_GROUND_TOL = 0.12  # 바닥 판정 허용 오차 (m)
GRID = 0.025            # 도달 격자 해상도 (m)
GSEL_FORWARD, GSEL_STOP = 0, 2
HEAD_NONE = -999.0

FRAME_STEP_M = 1.5      # 프레임(=홉) 간격 — MANSION 홉 이동거리 중앙값과 일치
LOOKAHEAD_M = 1.5       # wp_world 룩어헤드
GO2_DIR = "/data/URDF/real_robots/multileg/unitree_go2"
VLNCE = "/workspace/VLN-CE/data"
TAG = "r2rce_go2"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def yaw_deg(dx, dz):
    """월드 방향 → yaw (0°=+z, 90°=+x). MANSION 규약과 동일."""
    return math.degrees(math.atan2(dx, dz))


# ------------------------------------------------------------------ 로봇 사양

def robot_spec(urdf_dir=GO2_DIR):
    """Go2 사양 — cam_h는 sensor_rgb 링크의 FK 실높이(스탠딩)."""
    import yourdfpy

    with open(os.path.join(urdf_dir, "meta.json")) as fh:
        meta = json.load(fh)
    urdf = os.path.join(urdf_dir, "robot.urdf")
    u = yourdfpy.URDF.load(urdf, load_meshes=False,
                           build_collision_scene_graph=False)
    cam_h = None
    for link in u.link_map:
        if link.startswith("sensor_rgb"):
            cam_h = float(meta["base_z"]
                          + u.get_transform(link, u.base_link)[2, 3])
            break
    assert cam_h is not None, "sensor_rgb 링크 없음"
    w, l, h = (float(meta["measured"][k]) for k in ("w", "l", "h"))
    return {"id": meta["id"], "urdf_dir": urdf_dir, "urdf": urdf,
            "cam_h": round(cam_h, 4), "w_eff": round(min(w, l), 4),
            "h": round(h, 4), "radius": round(min(w, l) / 2.0, 4)}


# -------------------------------------------------------------- 시뮬 구성

def make_sim(scene, cam_h, gpu=0):
    """habitat 시뮬 — 전방 RGB-D 1뷰(256², HFOV 90), 센서 높이 = Go2 cam_h."""
    import habitat_sim

    bc = habitat_sim.SimulatorConfiguration()
    bc.scene_id = scene
    bc.enable_physics = False
    bc.gpu_device_id = gpu
    specs = []
    for name, stype in (("rgb", habitat_sim.SensorType.COLOR),
                        ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.SensorSpec()
        s.uuid = name
        s.sensor_type = stype
        s.resolution = [RES, RES]
        s.position = [0.0, cam_h, 0.0]
        s.parameters["hfov"] = "90"
        specs.append(s)
    ac = habitat_sim.agent.AgentConfiguration()
    ac.sensor_specifications = specs
    return habitat_sim.Simulator(habitat_sim.Configuration(bc, [ac]))


def rebuild_navmesh(sim, robot):
    """Go2 반경·높이로 navmesh 재계산 — 기본 .navmesh는 반경 0.1 사전계산본."""
    import habitat_sim

    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius = robot["radius"]
    ns.agent_height = robot["h"]
    return bool(sim.recompute_navmesh(sim.pathfinder, ns))


# ------------------------------------------------------------------ 도달 격자

class ReachGrid:
    """navmesh 탑다운 뷰 → 현재 위치와 이어진 도달 가능 셀.

    MANSION의 reach_mask와 같은 의미다(로봇 침식 격자의 연결 성분). navmesh가
    이미 로봇 반경으로 침식된 결과이므로 별도 침식은 하지 않는다.
    """

    def __init__(self, pathfinder, height):
        view = pathfinder.get_topdown_view(GRID, height)
        self.free = np.asarray(view, dtype=np.uint8)
        lo = pathfinder.get_bounds()[0]
        self.x0, self.z0 = float(lo[0]), float(lo[2])
        self.nz, self.nx = self.free.shape
        n, lab = cv2.connectedComponents(self.free, connectivity=8)
        self.lab = lab
        self.n_comp = n

    def idx(self, x, z):
        ix = int(np.clip((x - self.x0) / GRID, 0, self.nx - 1))
        iz = int(np.clip((z - self.z0) / GRID, 0, self.nz - 1))
        return iz, ix

    def component(self, x, z, snap_m=1.0):
        """(x, z)가 속한 연결 성분 라벨 — 격자 밖이면 근처를 훑어 스냅."""
        iz, ix = self.idx(x, z)
        if self.lab[iz, ix]:
            return int(self.lab[iz, ix])
        r = int(snap_m / GRID)
        z0, z1 = max(0, iz - r), min(self.nz, iz + r + 1)
        x0, x1 = max(0, ix - r), min(self.nx, ix + r + 1)
        sub = self.lab[z0:z1, x0:x1]
        nz = sub[sub > 0]
        return int(np.bincount(nz).argmax()) if len(nz) else 0

    def mask_of(self, comp):
        return (self.lab == comp) if comp else np.zeros_like(self.lab, bool)


# -------------------------------------------------------------------- 히트맵

def heat_gt(reach, mask, x, z, yaw, cam_y, depth):
    """제안기 GT — 도달 가능 자유 공간의 픽셀 래스터 (MANSION과 같은 정의).

    구현은 정투영이다. MANSION은 픽셀마다 depth로 3D 점을 복원해 바닥 여부를
    보지만, 그 판정은 높이 허용오차(±0.12 m)가 픽셀 공간에서 차지하는 두께가
    카메라가 낮을수록 얇아진다. Go2는 cam_h가 0.363 m라 바닥이 지평선 근처로
    압축돼 1024칸 중 88칸만 걸렸다(실측). 그래서 여기서는 도달 격자의 셀을
    **이미지로 투영**해 래스터화한다 — 같은 집합을 표집 손실 없이 얻는다.
    가려짐은 렌더 depth와 셀까지의 전방 거리를 비교해 걸러낸다.
    """
    ry = math.radians(yaw)
    r = int(HEAT_MAX_M / GRID)
    iz0, ix0 = reach.idx(x, z)
    z0, z1 = max(0, iz0 - r), min(reach.nz, iz0 + r + 1)
    x0, x1 = max(0, ix0 - r), min(reach.nx, ix0 + r + 1)
    sub = mask[z0:z1, x0:x1]
    if not sub.any():
        return np.zeros((HEAT_HW, HEAT_HW), np.uint8)
    czi, cxi = np.nonzero(sub)
    wz = reach.z0 + (czi + z0 + 0.5) * GRID
    wx = reach.x0 + (cxi + x0 + 0.5) * GRID
    dx, dz = wx - x, wz - z
    fwd = dx * math.sin(ry) + dz * math.cos(ry)
    rgt = dx * math.cos(ry) - dz * math.sin(ry)
    rng = np.hypot(dx, dz)
    ok = (fwd > 0.2) & (rng <= HEAT_MAX_M)
    if not ok.any():
        return np.zeros((HEAT_HW, HEAT_HW), np.uint8)
    fwd, rgt, wxo, wzo = fwd[ok], rgt[ok], wx[ok], wz[ok]
    u = rgt * FX / fwd + RES / 2
    v = cam_y * FX / fwd + RES / 2          # 바닥은 카메라보다 cam_y 아래
    ins = (u >= 0) & (u < RES) & (v >= 0) & (v < RES)
    if not ins.any():
        return np.zeros((HEAT_HW, HEAT_HW), np.uint8)
    ui, vi = u[ins].astype(np.int32), v[ins].astype(np.int32)
    # 가려짐 판정 — 렌더 depth가 셀보다 앞이면 벽·가구에 가린 것
    dep = np.asarray(depth, dtype=np.float32)[vi, ui]
    vis = dep >= fwd[ins] - 0.25
    k = RES // HEAT_HW
    pos = np.zeros((HEAT_HW, HEAT_HW), bool)
    pos[(vi[vis] // k), (ui[vis] // k)] = True
    out = pos.astype(np.uint8)
    p8 = out.copy()
    kern = np.ones((3, 3), np.uint8)
    out[cv2.dilate(p8, kern) != cv2.erode(p8, kern)] = HEAT_IGNORE
    return out


def project(pt, cam, yaw):
    """월드 점 → 렌더 픽셀 (u, v). 시야 뒤/밖이면 None."""
    ry = math.radians(yaw)
    dx, dy, dz = pt[0] - cam[0], pt[1] - cam[1], pt[2] - cam[2]
    fwd = dx * math.sin(ry) + dz * math.cos(ry)
    if fwd <= 0.05:
        return None
    right = dx * math.cos(ry) - dz * math.sin(ry)
    u = right * FX / fwd + RES / 2
    v = -dy * FX / fwd + RES / 2
    if not (0 <= u < RES and 0 <= v < RES):
        return None
    return int(u), int(v)


def heat_pixel(x, z, yaw, cam_y, wx, wz):
    uv = project((wx, 0.0, wz), (x, cam_y, z), yaw)
    if uv is None:
        return None
    k = RES // HEAT_HW
    return uv[0] // k, uv[1] // k


# ---------------------------------------------------------------- 경로 표집

def geodesic_polyline(pathfinder, pts):
    """참조 경로 각 구간을 navmesh 최단경로로 이어 붙인 폴리라인."""
    import habitat_sim

    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(a, np.float32)
        sp.requested_end = np.asarray(b, np.float32)
        if pathfinder.find_path(sp) and len(sp.points) >= 2:
            seg = [np.asarray(p, np.float64) for p in sp.points]
        else:
            seg = [np.asarray(a, np.float64), np.asarray(b, np.float64)]
        out += seg[:-1] if out else seg[:-1]
    out.append(np.asarray(pts[-1], np.float64))
    ded = [out[0]]
    for p in out[1:]:
        if np.linalg.norm(p - ded[-1]) > 1e-3:
            ded.append(p)
    return ded


def resample(poly, step):
    """폴리라인을 등간격으로 재표집 (끝점 포함)."""
    if len(poly) < 2:
        return list(poly)
    seg = [np.linalg.norm(b - a) for a, b in zip(poly[:-1], poly[1:])]
    total = float(sum(seg))
    if total < 1e-6:
        return [poly[0]]
    out, target, acc, i = [poly[0]], step, 0.0, 0
    while target < total and i < len(seg):
        while i < len(seg) and acc + seg[i] < target:
            acc += seg[i]
            i += 1
        if i >= len(seg):
            break
        t = (target - acc) / max(seg[i], 1e-9)
        out.append(poly[i] + t * (poly[i + 1] - poly[i]))
        target += step
    out.append(poly[-1])
    return out


# ------------------------------------------------------------------ 에피소드

def build_episode(sim, reach_cache, robot, ep, ep_id):
    """한 에피소드 → 프레임 배열 dict. 표본이 안 되면 None."""
    import habitat_sim
    import quaternion  # noqa: F401  (habitat이 요구)

    pf = sim.pathfinder
    ref = [np.asarray(p, np.float64) for p in ep["reference_path"]]
    if len(ref) < 2:
        return None
    poly = geodesic_polyline(pf, ref)
    samples = resample(poly, FRAME_STEP_M)
    if len(samples) < 2:
        return None

    y = float(np.median([p[1] for p in samples]))
    key = round(y, 1)
    if key not in reach_cache:
        reach_cache[key] = ReachGrid(pf, y)
    reach = reach_cache[key]

    agent = sim.get_agent(0)
    F = {k: [] for k in ("rgb", "depth", "pose", "heat", "wp_world",
                         "special", "gsel", "gsel_uv", "gsel_head")}
    n = len(samples)
    for t in range(n):
        p = samples[t]
        nxt = samples[min(t + 1, n - 1)]
        d = nxt - p
        yaw = yaw_deg(d[0], d[2]) if np.linalg.norm(d[[0, 2]]) > 1e-3 else (
            F["pose"][-1][3] if F["pose"] else 0.0)
        # habitat 기본 전방은 -Z — yaw(0°=+z)로 맞추려면 Y축 (yaw+180°) 회전
        st = habitat_sim.AgentState()
        st.position = np.asarray(p, np.float32)
        half = math.radians(yaw + 180.0) / 2.0
        st.rotation = np.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)
        agent.set_state(st)
        obs = sim.get_sensor_observations()
        rgb = np.asarray(obs["rgb"])[..., :3].astype(np.uint8)
        dep = np.asarray(obs["depth"], dtype=np.float32)

        comp = reach.component(float(p[0]), float(p[2]))
        mask = reach.mask_of(comp)
        F["rgb"].append(rgb)
        F["depth"].append(dep.astype(np.float16))
        F["pose"].append([0.0, float(p[0]), float(p[2]), float(yaw),
                          robot["cam_h"]])
        F["heat"].append(heat_gt(reach, mask, float(p[0]), float(p[2]),
                                 yaw, robot["cam_h"], dep))
        F["special"].append(0)
        # 룩어헤드 지점 = 다음 표본 (마지막은 자기 자신)
        F["wp_world"].append([float(nxt[0]), float(nxt[2]), float(yaw)])
        last = (t == n - 1)
        F["gsel"].append(GSEL_STOP if last else GSEL_FORWARD)
        uv = heat_pixel(float(p[0]), float(p[2]), yaw, robot["cam_h"],
                        float(nxt[0]), float(nxt[2]))
        F["gsel_uv"].append(list(uv) if uv else [-1, -1])
        if last or t + 2 >= n:
            F["gsel_head"].append(HEAD_NONE)
        else:
            d2 = samples[t + 2] - nxt
            F["gsel_head"].append(yaw_deg(d2[0], d2[2]))

    return {"rgb": np.asarray(F["rgb"], np.uint8),
            "depth": np.asarray(F["depth"], np.float16),
            "pose": np.asarray(F["pose"], np.float64),
            "wp_world": np.asarray(F["wp_world"], np.float64),
            "special": np.asarray(F["special"], np.int64),
            "heat": np.asarray(F["heat"], np.uint8),
            "gsel": np.asarray(F["gsel"], np.int16),
            "gsel_uv": np.asarray(F["gsel_uv"], np.int16),
            "gsel_head": np.asarray(F["gsel_head"], np.float16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="/data/R2R_EVLN")
    args = ap.parse_args()

    robot = robot_spec()
    log(f"로봇 {robot['id']}: 반경 {robot['radius']}m 높이 {robot['h']}m "
        f"cam_h {robot['cam_h']}m")
    epdir = os.path.join(args.out, "episodes")
    os.makedirs(epdir, exist_ok=True)

    src = os.path.join(VLNCE, "datasets/R2R_VLNCE_v1-3_preprocessed",
                       args.split, f"{args.split}.json.gz")
    eps = json.load(gzip.open(src))["episodes"]
    eps = sorted(eps, key=lambda e: int(e["episode_id"]))
    if args.limit:
        eps = eps[:args.limit]
    # R2R은 한 경로에 지시 3개를 준다 — 경로 단위로 묶어 한 번만 렌더하고
    # 지시는 전부 붙인다(로더가 지시 목록을 그대로 받는다). 3배 절약.
    trajs = {}
    for e in eps:
        trajs.setdefault((e["scene_id"], int(e["trajectory_id"])), []).append(e)
    keys = sorted(trajs)
    mine_keys = keys[args.shard::args.nshards]
    by_scene = {}
    for k in mine_keys:
        by_scene.setdefault(k[0], []).append(trajs[k])
    n_traj = len(mine_keys)
    log(f"{args.split} 담당 경로 {n_traj}/{len(keys)} (에피소드 {len(eps)}) · "
        f"씬 {len(by_scene)}개")

    recs, meta_eps, t0, done, kept = [], [], time.time(), 0, 0
    for si, (scene, group) in enumerate(sorted(by_scene.items())):
        path = os.path.join(VLNCE, "scene_datasets", scene)
        if not os.path.exists(path):
            log(f"씬 없음 건너뜀: {scene}")
            continue
        sim = make_sim(path, robot["cam_h"], args.gpu)
        if not rebuild_navmesh(sim, robot):
            log(f"navmesh 재계산 실패: {scene}")
            sim.close()
            continue
        reach_cache = {}
        for variants in group:
            e = variants[0]
            ep_id = f"tr{int(e['trajectory_id']):06d}"
            try:
                out = build_episode(sim, reach_cache, robot, e, ep_id)
            except Exception as exc:        # 씬 하나 실패로 전체가 죽지 않게
                log(f"  실패 {ep_id}: {type(exc).__name__} {exc}")
                out = None
            done += 1
            if out is not None and len(out["pose"]) >= 3:
                np.savez_compressed(
                    os.path.join(epdir, f"{TAG}_{ep_id}_{robot['id']}.npz"),
                    **out)
                meta_eps.append({"id": ep_id, "scene": scene,
                                 "results": {robot["id"]:
                                             {"state": "driving"}}})
                for v in variants:
                    recs.append({"episode": ep_id, "robot": robot["id"],
                                 "form": "r2r", "pass": True,
                                 "instruction": v["instruction"]
                                 ["instruction_text"].strip()})
                kept += 1
            if done % 50 == 0:
                el = time.time() - t0
                per = el / done
                eta = per * (n_traj - done) / 60
                log(f"  {done}/{n_traj} (유효 {kept}) 씬 {si+1}/"
                    f"{len(by_scene)} · {per:.2f}s/경로 · ETA {eta:.0f}분")
        sim.close()

    suffix = "" if args.nshards == 1 else f".shard{args.shard}"
    with open(os.path.join(epdir,
                           f"{TAG}_episodes.json{suffix}"), "w") as fh:
        json.dump({"episodes": meta_eps}, fh)
    with open(os.path.join(args.out,
                           f"{TAG}_instructions.json{suffix}"), "w") as fh:
        json.dump({"records": recs}, fh)
    if args.shard == 0:
        with open(os.path.join(args.out, f"{TAG}_robots.json"), "w") as fh:
            json.dump({"pool_root": "/data/URDF/real_robots",
                       "robots": [{"id": robot["id"],
                                   "urdf": os.path.relpath(
                                       robot["urdf"], "/data/URDF/real_robots"),
                                   "w_eff": robot["w_eff"], "h": robot["h"],
                                   "cam_h": robot["cam_h"],
                                   "stairs_ok": True}]}, fh)
    log(f"완료 — 유효 {kept}/{done} → {epdir}")


if __name__ == "__main__":
    main()

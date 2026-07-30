"""R2R-CE 폐루프 벤치마크 — EA-Nav 자체 스택 (Go2 단일 embodiment).

기존 r2r_ce.py는 원본 ETPNav 스택(파노라마 12뷰 + 원본 waypoint predictor +
CMT + release_r2r 가중치)을 돌리고 우리 g만 얹는다. 이 파일은 반대로 **우리가
학습한 모델을 그대로 주행**시킨다: 전방 단일 RGB-D → WaypointNet 히트맵 →
이동 후보 → GraphPlanning 선택 → 이동. 두 러너는 같은 에피소드·같은 지표를
쓰므로 수치가 직접 비교된다.

이동은 navmesh 최단경로를 따라간다. 저수준 제어기는 본 논문의 범위가 아니고
(계층 구조의 아랫단), 계획 단계 성능만 보려면 실행이 결정적이어야 한다.

지표 = R2R-CE 표준: SR(측지 ≤3m), SPL, NE, TL, nDTW.

사용 (hw_vln_ce 컨테이너):
  python /research/scripts/benchmark/r2r_ce_eanav.py \
      --split val_unseen --ckpt-wp /data/R2R_ckpt/waypoint_g.pt \
      --ckpt-nav /data/R2R_ckpt/g_nog/global.pt \
      --shard 0 --nshards 4 --gpu 0 [--limit 20]
"""
import argparse
import glob
import gzip
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/research/scripts/datasets/R2RCE")
sys.path.insert(0, "/research/scripts/models")
sys.path.insert(0, "/research/models")

import build_go2 as B          # 씬 구성·navmesh·로봇 사양을 생성부와 공유

VLNCE = "/workspace/VLN-CE/data"
ARRIVE_M = 3.0        # R2R-CE 성공 반경 (측지)
STEP_M = 0.25         # 이동 스텝 환산 — TL 집계용
MAX_HOPS = 30
MAX_STEPS = 500
MIN_HOP_M = 0.25      # 이보다 못 움직인 홉은 진행으로 치지 않는다
MIN_HOP_DEG = 15.0    # 변위가 없어도 이만큼 돌면 진행 — 방향 전환은 정상
                      # 선택지다(시야가 바뀌어야 다음 후보가 생긴다). 제자리
                      # 재선택을 실패로 처리하면 후보가 빈 홉에서 종료 행동만
                      # 남아 첫 홉에 STOP으로 떨어진다(실측: 6/6 에피소드)
HOP_FAIL_LIMIT = 3
FLOOR_SLACK = 0.25    # navmesh 바닥과 렌더 바닥의 어긋남 여유 (실측 최대 0.17)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ 주행 환경

class HabitatEnv:
    """habitat 위의 최소 주행 환경 — 포즈·관측·측지 이동만 제공."""

    def __init__(self, sim, robot):
        self.sim = sim
        self.pf = sim.pathfinder
        self.robot = robot
        self.steps = 0
        self.path = []

    def reset(self, pos, yaw):
        self.steps = 0
        self.path = [np.asarray(pos, np.float64)]
        self._set(pos, yaw)

    def _set(self, pos, yaw):
        import habitat_sim
        st = habitat_sim.AgentState()
        st.position = np.asarray(pos, np.float32)
        half = math.radians(yaw + 180.0) / 2.0
        st.rotation = np.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)
        self.sim.get_agent(0).set_state(st)
        self.pos = np.asarray(pos, np.float64)
        self.yaw = float(yaw)

    @property
    def x(self):
        return float(self.pos[0])

    @property
    def z(self):
        return float(self.pos[2])

    def observe(self):
        o = self.sim.get_sensor_observations()
        return (np.asarray(o["rgb"])[..., :3].astype(np.uint8),
                np.asarray(o["depth"], dtype=np.float32))

    def go(self, target_xz, heading):
        """navmesh 최단경로로 이동. 반환 = (이동거리 m, 회전각 deg)."""
        import habitat_sim
        turn = abs((heading - self.yaw + 180.0) % 360.0 - 180.0)
        snap = self.pf.snap_point(
            np.array([target_xz[0], self.pos[1], target_xz[1]], np.float32))
        if not np.all(np.isfinite(snap)):
            self._set(self.pos, heading)
            return 0.0, turn
        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(self.pos, np.float32)
        sp.requested_end = np.asarray(snap, np.float32)
        if not self.pf.find_path(sp) or len(sp.points) < 2:
            self._set(self.pos, heading)
            return 0.0, turn
        pts = [np.asarray(p, np.float64) for p in sp.points]
        dist = float(sum(np.linalg.norm(b - a)
                         for a, b in zip(pts[:-1], pts[1:])))
        self.steps += max(1, int(dist / STEP_M))
        self.path += pts[1:]
        self._set(pts[-1], heading)
        return dist, turn


# ------------------------------------------------------------------ 지표

def geo(pf, a, b):
    import habitat_sim
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, np.float32)
    sp.requested_end = np.asarray(b, np.float32)
    pf.find_path(sp)
    return float(sp.geodesic_distance)


def ndtw(path, ref, thr=3.0):
    """nDTW — 표준 정의(측지 대신 유클리드 근사, 같은 층이라 차이 미미)."""
    n, m = len(path), len(ref)
    if n == 0 or m == 0:
        return 0.0
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = float(np.linalg.norm(np.asarray(path[i - 1])
                                     - np.asarray(ref[j - 1])))
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(math.exp(-D[n, m] / (m * thr)))


# ------------------------------------------------------------------ 주행 루프

def run_episode(policy, env, gm_cls, z, txt, tmask, ep, device, cam_h):
    """한 에피소드 폐루프. 반환 = (상태, 홉수, 경로)."""
    import torch

    gm = gm_cls()
    policy.reset_episode()
    start = np.asarray(ep["reference_path"][0], np.float64)
    ref = [np.asarray(p, np.float64) for p in ep["reference_path"]]
    nxt = ref[1] if len(ref) > 1 else start
    yaw = B.yaw_deg(nxt[0] - start[0], nxt[2] - start[2])
    env.reset(start, yaw)

    hops, fails, vp = 0, 0, None
    while env.steps < MAX_STEPS and hops < MAX_HOPS:
        rgb, dep = env.observe()
        with torch.no_grad():
            r = torch.from_numpy(np.ascontiguousarray(rgb))[None]
            d = torch.from_numpy(dep)[None].to(device)
            obs = policy.observe(r, d)
            res = policy.wp(obs["depth_hist"], z, depth_m=obs["depth_m"])
            cand = policy.wp.propose(res)
            lifted = policy.lift(cand, obs,
                                 (0.0, env.x, env.z, env.yaw, cam_h),
                                 floor_y=0.0, floor_slack=FLOOR_SLACK)
            w = lifted["world"][0].numpy()[lifted["valid"][0].numpy()]
            e = lifted["embed"][0][
                torch.from_numpy(lifted["valid"][0].numpy())].to(device)
            ne = obs["node_embed"][0].to(device)
            vp = gm.update(np.array([env.x, 0.0, env.z]), ne,
                           list(w), list(e), floor=0)
            inp = gm.gmap_inputs(vp, np.array([env.x, 0.0, env.z]),
                                 math.radians(env.yaw), device=device,
                                 cur_floor=0)
            nav = policy("navigation", z=z, txt_embeds=txt, txt_masks=tmask,
                         mask_visited=False, g_bias=True, **inp)
            lo = nav["global_logits"][0]
            head = nav["heading"][0]

        banned, moved = set(), False
        while True:
            order = [i for i in range(len(lo)) if i not in banned]
            if not order:
                break
            pick = max(order, key=lambda i: float(lo[i]))
            sel = inp["gmap_vpids"][pick]
            if sel is None:                       # 종료 행동
                return "stop", hops, env.path
            hd = math.degrees(math.atan2(float(head[pick, 0]),
                                         float(head[pick, 1])))
            p = gm.node_pos.get(sel)
            if p is None:
                banned.add(pick)
                continue
            dist, turn = env.go((float(p[0]), float(p[2])), hd)
            if dist >= MIN_HOP_M or turn >= MIN_HOP_DEG:
                moved = True
                break
            banned.add(pick)
        hops += 1
        if not moved:
            fails += 1
            if fails >= HOP_FAIL_LIMIT:
                return "stuck", hops, env.path
        else:
            fails = 0
    return "budget", hops, env.path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--ckpt01", default="/data/EVLN_ckpt/traversability.pt")
    ap.add_argument("--ckpt-wp", default="/data/R2R_ckpt/waypoint_g.pt")
    ap.add_argument("--ckpt-nav", default="/data/R2R_ckpt/g_nog/global.pt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/data/R2R_ckpt/benchmark")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import common
    import weights as lp
    from EA_Nav import EANav, tokenizer
    from modules.GraphPlanning import GraphMap

    device = "cuda"
    robot = B.robot_spec()
    policy = EANav(clip_device=device).to(device)
    policy.load_pretrained(finetuned=True, verbose=False)
    common.load_into(args.ckpt01, urdf_enc=policy.urdf_enc, g=policy.g)
    for path, mod, key in ((args.ckpt_wp, policy.wp, "wp"),
                           (args.ckpt_nav, policy.nav, "nav")):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        lp.load_compat(mod, ck[key] if key in ck else ck)
    policy.eval()
    log(f"모델 로드 — 제안기 {args.ckpt_wp} · 선택기 {args.ckpt_nav}")

    import loaders
    gc = loaders.RobotGraphCache.__new__(loaders.RobotGraphCache)
    from modules.URDFEncoder import collate_graphs, urdf_to_graph
    g1 = urdf_to_graph(robot["urdf"])
    gb = {k: (v.to(device) if torch.is_tensor(v) else v)
          for k, v in collate_graphs([g1]).items()}
    with torch.no_grad():
        z, _ = policy.urdf_enc(gb)

    src = os.path.join(VLNCE, "datasets/R2R_VLNCE_v1-3_preprocessed",
                       args.split, f"{args.split}.json.gz")
    eps = sorted(json.load(gzip.open(src))["episodes"],
                 key=lambda e: int(e["episode_id"]))
    if args.limit:
        eps = eps[:args.limit]
    mine = eps[args.shard::args.nshards]
    by_scene = {}
    for e in mine:
        by_scene.setdefault(e["scene_id"], []).append(e)
    log(f"{args.split} 담당 {len(mine)}/{len(eps)}에피소드 · 씬 {len(by_scene)}")

    tok = tokenizer()
    recs, t0, done = [], time.time(), 0
    for scene, group in sorted(by_scene.items()):
        path = os.path.join(VLNCE, "scene_datasets", scene)
        if not os.path.exists(path):
            continue
        sim = B.make_sim(path, robot["cam_h"], args.gpu)
        B.rebuild_navmesh(sim, robot)
        env = HabitatEnv(sim, robot)
        for e in group:
            instr = e["instruction"]["instruction_text"].strip()
            enc = tok(instr, return_tensors="pt", truncation=True,
                      max_length=80)
            with torch.no_grad():
                txt = policy("language", txt_ids=enc["input_ids"].to(device),
                             txt_masks=enc["attention_mask"].to(device))
            tmask = enc["attention_mask"].to(device)
            try:
                state, hops, traj = run_episode(
                    policy, env, GraphMap, z, txt, tmask, e, device,
                    robot["cam_h"])
            except Exception as exc:
                log(f"  실패 ep{e['episode_id']}: {type(exc).__name__} {exc}")
                done += 1
                continue
            goal = np.asarray(e["goals"][0]["position"], np.float64)
            ne_ = geo(env.pf, env.pos, goal)
            gd = float(e["info"]["geodesic_distance"])
            tl = float(sum(np.linalg.norm(b - a)
                           for a, b in zip(traj[:-1], traj[1:])))
            ok = np.isfinite(ne_) and ne_ <= ARRIVE_M
            recs.append({
                "ep": int(e["episode_id"]), "state": state, "hops": hops,
                "ne": ne_ if np.isfinite(ne_) else None, "tl": tl,
                "success": bool(ok),
                "spl": (gd / max(tl, gd) if ok else 0.0),
                "ndtw": ndtw(traj, [np.asarray(p, np.float64)
                                    for p in e["reference_path"]])})
            done += 1
            if done % 20 == 0:
                el = time.time() - t0
                sr = np.mean([r["success"] for r in recs])
                log(f"  {done}/{len(mine)} SR {sr:.3f} "
                    f"({el/done:.1f}s/ep, ETA {el/done*(len(mine)-done)/60:.0f}분)")
        sim.close()

    os.makedirs(args.out, exist_ok=True)
    fin = [r for r in recs if r["ne"] is not None]
    summ = {"n": len(recs), "n_finite": len(fin),
            "SR": float(np.mean([r["success"] for r in recs])) if recs else 0.0,
            "SPL": float(np.mean([r["spl"] for r in recs])) if recs else 0.0,
            "NE": float(np.mean([r["ne"] for r in fin])) if fin else None,
            "TL": float(np.mean([r["tl"] for r in recs])) if recs else 0.0,
            "nDTW": float(np.mean([r["ndtw"] for r in recs])) if recs else 0.0,
            "ckpt_wp": args.ckpt_wp, "ckpt_nav": args.ckpt_nav,
            "split": args.split}
    with open(os.path.join(args.out,
                           f"eanav_{args.split}_{args.shard}.json"), "w") as fh:
        json.dump({"summary": summ, "records": recs}, fh)
    log(f"완료 — {summ}")


if __name__ == "__main__":
    main()

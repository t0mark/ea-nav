"""통합 평가 CLI — 학습 부품·시스템 폐루프 평가.

사용: python eval.py <sub> [옵션...]
  traversability  통과성 g 홀드아웃 평가 (BCE·acc·AUC, 환경·클래스 분해)
  waypoint        제안기 히트맵 홀드아웃 평가 (BCE·IoU·peak 적중·양성비)
  rollout         폐루프 decision 평가 — zswap 갈림·지시충돌·notfound
                  (+--drive 실주행 표본). 논문 메인 실험 경로
  unseen          미본 치수 로봇 일반화 (프로브·이중미본 통과성)
  objreg          notfound 판정용 관측 객체 레지스트리 빌드(1회성 도구)
구 test/01~04 4파일 통합 — 역할은 서브커맨드로 구분.

상태 집합 = {주행, 대기중, 도착, 불가능(미탐색), 불가능(embodiment)}.
계획 단계 예측은 pred_state(주행/대기중/불가능), 폐루프 결과는
drive_state(도착/주행/stalled)가 낸다. 도착 = 전역 STOP + 반경
ma.ARRIVE_M(벤치 SR과 동일값) + 목표형은 레지스트리 검증.
실주행은 벤치와 같은 ma.HopDriver·MansionSimEnv를 쓴다 — 평가와 벤치가
서로 다른 정책을 재지 않게 하기 위한 것.
"""
import sys as _sys

_SUBS = ("traversability", "waypoint", "rollout", "unseen", "objreg")

# ==================================================================
# 서브커맨드: traversability (구 01_eval_passability.py)
# ==================================================================
import argparse
import glob
import os
import sys
import time

import numpy as np

import common


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main_traversability():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--ckpt", default="/data/EVLN_ckpt/traversability.pt")
    ap.add_argument("--cache-dir", default="/data/EVLN_ckpt/featcache")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split", choices=("val", "train", "all"),
                    default="val")
    args = ap.parse_args()
    common.pick_gpu(args.gpu)

    import torch
    import loaders
    from modules.TraversabilityHead import TraversabilityHead
    from modules.URDFEncoder import URDFEncoder
    device = "cuda"
    tags = args.envs or loaders.tags()

    g = TraversabilityHead(768, 128).to(device)
    enc = URDFEncoder(z_dim=128).to(device)
    ck = common.load_into(args.ckpt, g=g, urdf_enc=enc)
    g.eval(); enc.eval()
    log(f"ckpt 로드 {args.ckpt} (학습 시 지표: {ck.get('meta')})")

    ds = loaders.TraversabilityDataset(env_tags=tags)
    rob_ids = [r["id"] for r in ds.envs[0]["robots"]]
    gc = loaders.RobotGraphCache(tags[0])
    gbatch = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in gc.collate(rob_ids).items()}
    with torch.no_grad():
        z, _ = enc(gbatch)

    feat_cache = {}
    for tag in tags:
        for f in sorted(glob.glob(os.path.join(
                loaders.DS, "gates", f"{tag}_snaps_shard*.npz"))):
            feat_cache[f] = np.load(os.path.join(
                args.cache_dir, os.path.basename(f) + ".feat.npy"))

    per = {}
    rid_cls = {r["id"]: r["id"].split("_")[0]
               for r in ds.envs[0]["robots"]}
    with torch.no_grad():
        for tag in tags:
            env = next(e for e in ds.envs if e["tag"] == tag)
            rows, las, ridx = [], [], []
            for (ei, gi, ri, bi) in ds.samples:
                if ds.envs[ei]["tag"] != tag:
                    continue
                v = common.is_val_gate(tag, gi, args.val_frac)
                if (args.split == "val" and not v) or \
                        (args.split == "train" and v):
                    continue
                for di in range(env["n_dist"]):
                    for ai in range(env["n_ang"]):
                        f, row = env["key2loc"][(gi, di, ai, bi)]
                        rows.append((f, row))
                        las.append(env["labels"][gi, ri])
                        ridx.append(ri)
            if not rows:
                continue
            feats = torch.from_numpy(np.stack(
                [feat_cache[f][r] for f, r in rows])).float().to(device)
            la = torch.tensor(las, device=device)
            zi = z[torch.tensor(ridx, device=device)]
            lo = torch.cat([g(feats[s:s + 8192], zi[s:s + 8192])
                            for s in range(0, len(feats), 8192)])
            hard = (la > 0.5).float()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                lo, la).item()
            acc = ((lo > 0) == hard.bool()).float().mean().item()
            per[tag] = dict(n=len(la), bce=bce, acc=acc,
                            auc=common.auc(lo, hard),
                            cls={})
            for c in sorted(set(rid_cls.values())):
                m = torch.tensor([rid_cls[env["robots"][i]["id"]] == c
                                  for i in ridx], device=device)
                if m.any():
                    per[tag]["cls"][c] = float(
                        ((lo[m] > 0) == hard[m].bool()).float().mean())

    print(f"\n== 통과성 평가 (split={args.split}) ==")
    for tag, r in per.items():
        print(f"{tag}\n  n {r['n']}, bce {r['bce']:.4f}, acc {r['acc']:.4f},"
              f" auc {r['auc']:.4f}")
        print("  클래스별 acc:", {k: round(v, 4)
                                for k, v in r["cls"].items()})



# ==================================================================
# 서브커맨드: waypoint (구 02_eval_waypoint.py)
# ==================================================================
import argparse
import os
import sys
import time

import common


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main_waypoint():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--ckpt", default="/data/EVLN_ckpt/waypoint.pt")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split", choices=("val", "train", "all"),
                    default="val")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    common.pick_gpu(args.gpu)

    import torch
    from torch.utils.data import DataLoader, Subset
    import loaders
    from EA_Nav import EANav
    device = "cuda"
    tags = args.envs or loaders.tags()

    meta = torch.load(args.ckpt, map_location="cpu",
                      weights_only=False).get("meta", {})
    inject = meta.get("inject", "g")   # 변형별 wp 구조가 다름 — meta 우선
    policy = EANav(clip_device=device, waypoint_inject=inject).to(device)
    common.load_into(args.ckpt, wp=policy.wp, g=policy.g,
                     urdf_enc=policy.urdf_enc)
    policy.eval()
    log(f"ckpt 로드 {args.ckpt} (inject={inject}, 학습 시 지표: {meta})")

    ds = loaders.EpisodeFrameDataset(env_tags=tags,
                                     hist_k=meta.get("hist_k",
                                                     loaders.HIST_K))
    if ds.skipped_legacy:
        log(f"구 스키마 궤적 {ds.skipped_legacy}개 건너뜀 (gt_rederive 미완)")
    keep = []
    for i, (f, t, _, rid, tag, st) in enumerate(ds.frames):
        v = common.is_val_traj(tag, os.path.basename(f), args.val_frac)
        if (args.split == "val" and v) or \
                (args.split == "train" and not v) or args.split == "all":
            keep.append(i)
    log(f"평가 프레임 {len(keep)} (split={args.split})")
    gc = loaders.RobotGraphCache(tags[0])
    dl = DataLoader(Subset(ds, keep), batch_size=args.bs, shuffle=False,
                    num_workers=args.workers,
                    collate_fn=lambda b: loaders.collate_frames(b, gc))

    # (tag|state) → Meter 묶음. peak_hit = 최고 점수 셀이 양성인가
    # (제안기가 실제로 쓰이는 방식 — NMS 상위 후보의 적중), iou는 0.5 임계
    groups = {}

    def met(key):
        if key not in groups:
            groups[key] = {k: common.Meter() for k in
                           ("bce", "iou", "prec", "recall", "auc",
                            "peak_hit", "pos_rate")}
        return groups[key]

    with torch.no_grad():
        for b in dl:
            if "heat" not in b:
                raise KeyError(
                    "collate에 heat 없음 — 제안기 GT는 히트맵(32²)이다. "
                    "loaders 히트맵 대응 후 재실행할 것")
            hist = b["depth_hist"].to(device)
            df = policy.depth_enc(hist)
            gb = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in b["robot_graph"].items()}
            z, _ = policy.urdf_enc(gb)
            # 학습과 같은 입력 — 히스토리 K프레임 + 현재 프레임 depth 스템.
            # 둘 중 하나라도 빠지면 학습 분포와 달라져 지표가 낮게 나온다
            lo = policy.wp(df, z, depth_m=hist[:, 0])["heat"][:, 0]
            gt = b["heat"].to(device)
            for i in range(len(gt)):
                # 손실·지표 정의는 common 한 곳에만 둔다 — 학습 로그와
                # 평가 보고가 같은 수를 말하게 하려면 정의가 하나여야 한다
                m = common.heat_metrics(lo[i:i + 1], gt[i:i + 1])
                if not m:
                    continue
                m["bce"] = float(common.heat_loss(lo[i:i + 1], gt[i:i + 1]))
                for key in ("전체", f"env:{b['tag'][i]}",
                            f"state:{b['state'][i]}"):
                    g_ = met(key)
                    for k_ in g_:
                        if k_ in m:
                            g_[k_].add(m[k_])

    print(f"\n== 제안기(히트맵) 평가 (split={args.split}) ==")
    for key in sorted(groups):
        g_ = groups[key]
        print(f"{key:44} n {g_['bce'].n:6} bce {g_['bce'].avg:.4f} "
              f"IoU {g_['iou'].avg:.4f} prec {g_['prec'].avg:.4f} "
              f"recall {g_['recall'].avg:.4f} auc {g_['auc'].avg:.4f} "
              f"peak적중 {g_['peak_hit'].avg:.4f} "
              f"| GT양성 {g_['pos_rate'].avg:.3f}")



# ==================================================================
# 서브커맨드: rollout (구 03_eval_rollout.py)
# ==================================================================
import argparse
import json
import math
import os
import sys
import time

import numpy as np

import common
import mansion_adapter as ma


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


COLLAPSE = {"impossible_noroute": "impossible",
            "impossible_embodiment": "impossible"}

# 레지스트리 검증 임계 — 목표형 도착 판정의 보조 조건(GDINO 검출 점수).
# notfound ROC로 재보정할 잠정값이라, 판정을 뒤집을 때마다 로그를 남긴다.
REG_TAU = 0.3
# GDINO 프롬프트당 명칭 수 (objreg)
CH = 12


def pred_state(route):
    """계획 단계 상태 예측 — GT 라벨(driving/driving_with_wait/impossible).

    도착·대기중은 폐루프 결과라 여기 들어오지 않는다(drive_state 참조).
    """
    if not route["reachable"]:
        return "impossible"
    return ("driving_with_wait" if "elevator" in route["modes"]
            else "driving")


def registry_hit(tag, mem, goal):
    """목표 객체가 기억 레지스트리에 있는지 — 목표형 도착 검증.

    탐사 로봇 전체 레지스트리 중 최고 점수를 쓴다(표준 기억이 합집합이라
    어느 로봇이 봤든 기억에 있는 것으로 친다).
    """
    best = 0.0
    for rob in mem.graphs:
        try:
            best = max(best, ma.objreg_query(tag, rob, goal, 10 ** 9))
        except (FileNotFoundError, OSError, KeyError):
            continue
    return best


def drive_state(res, verified=None):
    """폐루프 결과 → 상태 집합 {도착, 주행, stalled}.

    도착 = 전역 STOP + 반경(ma.ARRIVE_M = 벤치 SR과 동일값) + 목표형은
    레지스트리 검증까지. 검증에서 뒤집히면 도착이 아니라 arrival_error다.
    """
    if res["state"] != "arrived":
        return res["state"], res["reason"]
    if verified is False:
        return "driving", "registry_reject"
    return "arrived", "-"


def main_rollout():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--ckpt01", default="/data/EVLN_ckpt/traversability.pt")
    ap.add_argument("--ckpt02", default="/data/EVLN_ckpt/waypoint_g.pt")
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--goal-radius", type=float, default=4.0)
    ap.add_argument("--drive", type=int, default=0,
                    help="실주행 표본 쌍 수 (0=decision만)")
    ap.add_argument("--ckpt03", default="/data/EVLN_ckpt/global.pt",
                    help="전역 선택기 — --drive에 필요(학습 3단계 산출)")
    args = ap.parse_args()
    common.pick_gpu(args.gpu)

    import torch
    import loaders
    from EA_Nav import EANav
    import weights as lp
    device = "cuda"
    tags = args.envs or loaders.tags()

    policy = EANav(clip_device=device).to(device)
    policy.load_pretrained(finetuned=True, verbose=False)
    common.load_into(args.ckpt01, urdf_enc=policy.urdf_enc, g=policy.g)
    if args.drive:
        # 홉 루프에 필요한 학습분 — 제안기(wp)·전역 선택기(nav).
        # 학습 4단계 전에는 없을 수 있고, 그때는 수치가 무효다.
        for path, mod, key in ((args.ckpt02, policy.wp, "wp"),
                               (args.ckpt03, policy.nav, "nav")):
            if not os.path.exists(path):
                log(f"경고: {key} ckpt 없음({path}) — 무작위 초기값이라 "
                    f"--drive 수치 무효(스모크 전용)")
                continue
            ck = torch.load(path, map_location="cpu", weights_only=False)
            lp.load_compat(mod, ck[key] if key in ck else ck)
    policy.eval()

    gc = loaders.RobotGraphCache(tags[0])
    zs_cache = {}

    def z_of(rid):
        if rid not in zs_cache:
            gb = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in gc.collate([rid]).items()}
            with torch.no_grad():
                z, aux = policy.urdf_enc(gb)
            zs_cache[rid] = (z, float(torch.sigmoid(
                aux["stairs_ok_logit"])[0]))
        return zs_cache[rid]

    tot = {"state": common.Meter(), "modes": common.Meter(),
           "pair": common.Meter(), "div": common.Meter(),
           "conflict": common.Meter()}
    nf_all = {"pos": [], "neg": []}

    for tag in tags:
        log(f"== {tag}")
        mem = ma.UnionMemory(tag)
        feats = ma.MemoryFeatures(policy, mem, device)
        bank = ma.GateBank(tag)
        eg_map = ma.edge_gate_map(mem, bank)
        meta_r = loaders.robots_meta(tag)
        floors = ma.load_floors(tag)
        ej = json.load(open(os.path.join(
            ma.DS, "episodes", f"{tag}_episodes.json")))
        eps = {e["id"]: e for e in ej["episodes"]}
        ev = json.load(open(os.path.join(ma.DS, f"{tag}_evalsets.json")))
        log(f"기억: 노드 {len(mem.nodes)}(원본 "
            f"{sum(len(g['nodes']) for g in mem.graphs.values())}), "
            f"엣지 {len(mem.edges)}, zswap {len(ev['zswap'])}")

        def route_of(rid, e):
            z, sok = z_of(rid)
            if rid not in edge_block_cache:
                if rid not in bank_sc_cache:
                    bank_sc_cache[rid] = bank.scores(
                        policy, z, meta_r[rid]["cam_h"], device)
                edge_block_cache[rid] = ma.differential_block(
                    bank_sc_cache, eg_map, rid, args.tau)
            st = e["start"]
            sx, sz = floors[st["floor"]].to_world(*st["cell"])
            su = mem.nearest(st["floor"], sx, sz)
            gl = e["goal"]
            gu = mem.nodes_near(gl["floor"], gl["x"], gl["z"],
                                args.goal_radius)
            if not gu:
                n = mem.nearest(gl["floor"], gl["x"], gl["z"], 8.0)
                gu = [n] if n is not None else []
            # 판정 전용 경로 — 실행(홉 루프)은 이 결과를 쓰지 않는다.
            # 차단은 z-차등 엣지 하나로 통일(judge_state 규약)
            return ma.judge_state(mem, su, gu, sok > 0.5,
                                  tau=args.tau,
                                  edge_block=edge_block_cache[rid]
                                  )["route"] or {"reachable": False,
                                                 "modes": [], "path": [],
                                                 "dist": None}

        edge_block_cache = {}
        bank_sc_cache = {}
        # 차등 차단 기준을 위해 등장 로봇 전체의 뱅크 점수를 선계산
        rids_all = {it[f"robot_{sd}"] for it in ev["zswap"]
                    for sd in ("a", "b")}
        rids_all |= {it["target_robot"] for it in ev["instr_conflict"]}
        for r_ in sorted(rids_all):
            z_, _ = z_of(r_)
            bank_sc_cache[r_] = bank.scores(
                policy, z_, meta_r[r_]["cam_h"], device)
            edge_block_cache.pop(r_, None)
        env_m = {k: common.Meter() for k in
                 ("state", "modes", "pair", "div")}
        for it in ev["zswap"]:
            e = eps[it["episode"]]
            preds, gts = [], []
            for side in ("a", "b"):
                rid = it[f"robot_{side}"]
                exp = it[f"expect_{side}"]
                r = route_of(rid, e)
                ps = pred_state(r)
                gs = COLLAPSE.get(exp["state"], exp["state"])
                s_ok = ps == gs
                m_ok = (set(r["modes"]) == set(exp["modes"])
                        if gs != "impossible" else True)
                env_m["state"].add(s_ok); tot["state"].add(s_ok)
                env_m["modes"].add(m_ok); tot["modes"].add(m_ok)
                preds.append((ps, tuple(sorted(r["modes"])), r))
                gts.append((gs, tuple(sorted(exp["modes"]))))
            pair_ok = all(p[:2] == g or (g[0] == "impossible"
                          and p[0] == "impossible")
                          for p, g in zip(preds, gts))
            gt_div = gts[0] != gts[1]
            pd_div = preds[0][:2] != preds[1][:2]
            env_m["pair"].add(pair_ok); tot["pair"].add(pair_ok)
            env_m["div"].add(pd_div == gt_div); tot["div"].add(pd_div == gt_div)
        log(f"zswap: 상태 {env_m['state'].avg:.3f} 수단 "
            f"{env_m['modes'].avg:.3f} 쌍 {env_m['pair'].avg:.3f} "
            f"갈림재현 {env_m['div'].avg:.3f} (n={env_m['div'].n})")

        # ---- 지시-충돌 ----
        cm = common.Meter()
        for it in ev["instr_conflict"]:
            e = eps[it["episode"]]
            ir = e["results"][it["instruction_robot"]]
            instr_modes = set(t["mode"] for t in ir.get("transitions", []))
            r = route_of(it["target_robot"], e)
            if not r["reachable"]:
                pb = "report_impossible"
            elif set(r["modes"]) == instr_modes:
                pb = "follow"
            else:
                pb = "self_detour"
            ok = pb == it["gt_behavior"]
            cm.add(ok); tot["conflict"].add(ok)
        log(f"conflict: gt_behavior 일치 {cm.avg:.3f} (n={cm.n})")

        # ---- notfound grounding (CLIP, prefix 기억) ----
        ins = json.load(open(os.path.join(
            ma.DS, f"{tag}_instructions.json")))
        # notfound = 관측 객체 레지스트리 조회 (문맥 필터)
        for it in ev["notfound"]:
            nf_all["neg"].append(ma.objreg_query(
                tag, it["memory_robot"], it["goal"],
                int(it["start_node"])))
        for r in ins["records"]:
            if r["form"] == "combined" and r.get("pass"):
                nf_all["pos"].append(ma.objreg_query(
                    tag, r["memory_robot"], r["goal"],
                    int(r["start_node"])))

        # ---- 실주행 표본 ----
        if args.drive:
            drive_sample(args, policy, mem, feats, floors, eps, ev,
                         tag, z_of, device)

    import torch as _t
    pos = _t.tensor(nf_all["pos"]); neg = _t.tensor(nf_all["neg"])
    sc = _t.cat([pos, neg])
    lb = _t.cat([_t.ones(len(pos)), _t.zeros(len(neg))])
    thr = (pos.mean() + neg.mean()) / 2
    acc = float(((sc >= thr) == lb.bool()).float().mean())
    print("\n== 종합 ==")
    print(f"zswap 상태 {tot['state'].avg:.3f} | 수단 {tot['modes'].avg:.3f}"
          f" | 쌍 정합 {tot['pair'].avg:.3f} | 갈림 재현 "
          f"{tot['div'].avg:.3f} (쌍 {tot['div'].n})")
    print(f"conflict gt_behavior 일치 {tot['conflict'].avg:.3f} "
          f"(n={tot['conflict'].n})")
    print(f"notfound: 관측goal simmax {pos.mean():.3f}±{pos.std():.3f} vs "
          f"미관측 {neg.mean():.3f}±{neg.std():.3f}, AUC "
          f"{common.auc(sc, lb):.3f}, acc@중점 {acc:.3f} "
          f"(pos {len(pos)}/neg {len(neg)})")


def drive_sample(args, policy, mem, feats, floors, eps, ev, tag,
                 z_of, device):
    """zswap 표본 폐루프 주행 — 벤치와 같은 드라이버·같은 도착 판정.

    구판은 ThorDriver를 직접 몰면서 충돌·엘베 FSM 없이 노드를 이어 붙였고
    성공 반경(4.0m)도 벤치 SR(3.0m)과 달라 서로 다른 정책을 재고 있었다.
    여기서는 MansionSimEnv + ma.HopDriver를 그대로 쓴다.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "benchmark"))
    import loaders
    from EA_Nav import tokenizer
    from mansion_sim_env import MansionSimEnv
    meta_r = loaders.robots_meta(tag)
    tok = tokenizer()
    # 목표형 지시 — 전역 선택기는 지시 조건부(없으면 무지시 토큰)
    texts = {}
    ip = os.path.join(ma.DS, f"{tag}_instructions.json")
    if os.path.exists(ip):
        for r_ in json.load(open(ip)).get("records", []):
            if r_.get("form") == "goal" and r_.get("pass") \
                    and r_.get("episode"):
                texts.setdefault(r_["episode"], r_["instruction"])
    env = MansionSimEnv(tag, gpu=args.gpu, tol_cells=1, render=True)
    done = 0
    ok_n = 0
    try:
        for it in ev["zswap"]:
            if done >= args.drive:
                break
            e = eps[it["episode"]]
            st = e["start"]
            sx, sz = floors[st["floor"]].to_world(*st["cell"])
            gl = e["goal"]
            goal = (int(gl["floor"]), float(gl["x"]), float(gl["z"]))
            txt = None
            if texts.get(e["id"]):
                txt = encode_instruction(policy, tok, texts[e["id"]],
                                         device)
            for side in ("a", "b"):
                rid = it[f"robot_{side}"]
                if it[f"expect_{side}"]["state"].startswith("impossible"):
                    continue
                z, sok = z_of(rid)
                r = meta_r[rid]
                env.reset(dict(w_eff=r["w_eff"], h=r["h"],
                               cam_h=r["cam_h"],
                               stairs_ok=r["stairs_ok"]),
                          (st["floor"], sx, sz, 0.0), goal, wait_steps=3)
                drv = ma.HopDriver(policy, device, mem, feats=feats,
                                   tau=args.tau, stairs_ok=sok > 0.5,
                                   cam_h=r["cam_h"], log=log)
                drv.start(z, txt)
                res = drv.run(env, goal)
                verified = None
                if res["state"] == "arrived":
                    sc = registry_hit(tag, mem, gl)
                    verified = sc >= REG_TAU
                    if not verified:
                        log(f"  레지스트리 기각 {gl.get('name')} "
                            f"score {sc:.3f} < {REG_TAU}")
                state, reason = drive_state(res, verified)
                ok = state == "arrived"
                ok_n += ok
                done += 1
                log(f"drive {tag} {it['episode']} {rid}: {state} "
                    f"({reason}) hops={res['hops']} steps={env.steps}")
                if done >= args.drive:
                    break
    finally:
        env.close()
    log(f"drive 표본: 도착 {ok_n}/{done}")


def encode_instruction(policy, tok, text, device):
    """지시 → (txt_embeds, masks) — 전역 선택기 언어 입력(벤치와 동일 규약)."""
    import torch
    enc = tok(text, padding="max_length", truncation=True, max_length=80,
              return_tensors="pt")
    ids, msk = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    with torch.no_grad():
        return policy("language", txt_ids=ids, txt_masks=msk), msk




# ==================================================================
# 서브커맨드: unseen (구 04_eval_unseen.py)
# ==================================================================
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

import common
import mansion_adapter as ma


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def soft(m, ramp=0.15):
    return np.clip(0.5 + m / (2.0 * ramp), 0.0, 1.0)


def main_unseen():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pool", default="/data/URDF/val_unseen_dims")
    ap.add_argument("--ckpt01", default="/data/EVLN_ckpt/traversability.pt")
    ap.add_argument("--cache-dir", default="/data/EVLN_ckpt/featcache")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()
    common.pick_gpu(args.gpu)

    import torch
    import loaders
    from modules.TraversabilityHead import TraversabilityHead
    from modules.URDFEncoder import URDFEncoder, collate_graphs, urdf_to_graph
    device = "cuda"

    g = TraversabilityHead(768, 128).to(device)
    enc = URDFEncoder(z_dim=128).to(device)
    common.load_into(args.ckpt01, g=g, urdf_enc=enc)
    g.eval(); enc.eval()

    # 미본 풀 GT — 학습 풀과 동일 코드 경로 (root 인자 오버라이드)
    mc = ma._mc()
    pool = mc.robot_pool(args.pool)
    log(f"미본 풀 로봇 {len(pool)} ({args.pool})")

    gb = collate_graphs([urdf_to_graph(os.path.join(
        args.pool, r["cls"], r["id"], "robot.urdf")) for r in pool])
    gb = {k: (v.to(device) if torch.is_tensor(v) else v)
          for k, v in gb.items()}
    with torch.no_grad():
        z, aux = enc(gb)

    # ---- A. 프로브 일반화 ----
    ch = torch.tensor([r["cam_h"] for r in pool], device=device)
    w = torch.tensor([r["w_eff"] for r in pool], device=device)
    h = torch.tensor([r["h"] for r in pool], device=device)
    sok = torch.tensor([float(r["stairs_ok"]) for r in pool],
                       device=device)
    print("\n== A. z 프로브 (미본 60 로봇)")
    print(f"  cam_h MAE {float((aux['cam_h'] - ch).abs().mean()):.3f}m "
          f"(범위 {float(ch.min()):.2f}~{float(ch.max()):.2f})")
    print(f"  w MAE {float((aux['wlh'][:, 0] - w).abs().mean()):.3f}m, "
          f"h MAE {float((aux['wlh'][:, 2] - h).abs().mean()):.3f}m")
    sacc = float((((aux["stairs_ok_logit"] > 0).float()) == sok)
                 .float().mean())
    print(f"  stairs_ok acc {sacc:.3f}")

    # ---- B. 통과성 일반화 ----
    print("\n== B. 통과성 g(미본 z) vs 규약 라벨")
    for tag in loaders.tags():
        bank = ma.GateBank(tag, args.cache_dir)
        gates = bank.gates
        widths = np.array([gt["width"] for gt in gates])
        heights = np.array([gt["height"] for gt in gates])
        rows = {"all": ([], []), "val": ([], [])}
        with torch.no_grad():
            for ri, r in enumerate(pool):
                sc = bank.scores(
                    type("P", (), {"g": g})(), z[ri:ri + 1],
                    r["cam_h"], device)
                lab = np.minimum(soft(widths - r["w_eff"]),
                                 soft(heights - r["h"]))
                for gi, s in sc.items():
                    rows["all"][0].append(s)
                    rows["all"][1].append(lab[gi])
                    if common.is_val_gate(tag, gi, args.val_frac):
                        rows["val"][0].append(s)
                        rows["val"][1].append(lab[gi])
        for split, (ss, ll) in rows.items():
            if not ss:
                continue
            s_ = torch.tensor(ss)
            l_ = torch.tensor(ll)
            hard = (l_ > 0.5).float()
            acc = float(((s_ > 0) == hard.bool()).float().mean())
            auc = common.auc(s_, hard)
            name = "이중 미본(로봇+게이트)" if split == "val" else "전체 게이트"
            print(f"  {tag}\n    [{name}] n {len(ss)}, acc {acc:.3f}, "
                  f"auc {auc:.3f}")



# ==================================================================
# 서브커맨드: objreg (구 objreg.py)
# ==================================================================

def main_objreg():
    """관측 객체 레지스트리 빌드 — notfound 판정 입력 (1회성 도구).

    토폴로지 맵 노드 스냅샷에 GDINO를 돌려 검출 박스를 모으고, 크롭을
    CLIP으로 임베딩해 노드별 레지스트리를 만든다.
    """
    import glob
    import json
    import time

    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/data/EVLN_ckpt/evalcache")
    ap.add_argument("--box-thresh", type=float, default=0.15)
    args = ap.parse_args()
    common.pick_gpu(args.gpu)

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection
    from transformers import AutoProcessor
    import loaders
    from modules.VisionEncoder import CLIPEncoder

    tag, out_dir = args.tag, args.out
    os.makedirs(out_dir, exist_ok=True)
    mid = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        mid).to("cuda").eval()

    # 벤치마크 어휘 (전 환경 합집합, 정렬 고정)
    vocab = set()
    for T in loaders.tags():
        ins = json.load(open(os.path.join(loaders.DS,
                                          f"{T}_instructions.json")))
        ev = json.load(open(os.path.join(loaders.DS,
                                         f"{T}_evalsets.json")))
        vocab |= {r["goal"]["name"] for r in ins["records"] if "goal" in r
                  and isinstance(r["goal"], dict)}
        vocab |= {it["goal"]["name"] for it in ev["notfound"]}
    vocab = sorted(vocab)
    log(f"어휘 {len(vocab)}개")
    clip = CLIPEncoder("cuda")

    def detect(imgs, names):
        text = " . ".join(f"a {n}" for n in names) + " ."
        inp = proc(images=imgs, text=[text] * len(imgs),
                   return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model(**inp)
        return proc.post_process_grounded_object_detection(
            o, inp.input_ids, box_threshold=args.box_thresh,
            text_threshold=args.box_thresh,
            target_sizes=[im.size[::-1] for im in imgs])

    snaps = sorted(glob.glob(os.path.join(
        loaders.DS, "topomap", f"{tag}_*_snaps.npz")))
    for tf in snaps:
        rob = os.path.basename(tf)[len(tag) + 1:-len("_snaps.npz")]
        if "custom" in rob:
            continue
        of = os.path.join(out_dir, f"{tag}_{rob}_objreg.npz")
        if os.path.exists(of):
            log(f"skip {rob}")
            continue
        z = np.load(tf)
        rgb, key = z["rgb"], z["key"]
        rows = {"node": [], "name": [], "score": [], "box": [], "crop": []}
        t0 = time.time()
        for s in range(0, len(rgb), 16):
            imgs = [Image.fromarray(r) for r in rgb[s:s + 16]]
            for c0 in range(0, len(vocab), CH):
                names = vocab[c0:c0 + CH]
                res = detect(imgs, names)
                for bi, r in enumerate(res):
                    vi = s + bi
                    for lab, sc, box in zip(r["labels"], r["scores"],
                                            r["boxes"]):
                        hits = [n for n in names if n in lab]
                        if not hits:
                            continue
                        x0, y0, x1, y1 = [int(v) for v in box]
                        x0, y0 = max(0, x0), max(0, y0)
                        x1, y1 = min(256, x1), min(256, y1)
                        if x1 - x0 < 8 or y1 - y0 < 8:
                            continue
                        crop = Image.fromarray(
                            rgb[vi][y0:y1, x0:x1]).resize((112, 112))
                        for n in hits:
                            rows["node"].append(int(key[vi, 0]))
                            rows["name"].append(vocab.index(n))
                            rows["score"].append(float(sc))
                            rows["box"].append([x0, y0, x1, y1])
                            rows["crop"].append(np.asarray(crop))
        crops = (np.stack(rows["crop"]) if rows["crop"]
                 else np.zeros((0, 112, 112, 3), np.uint8))
        embs = []
        with torch.no_grad():
            for s in range(0, len(crops), 64):
                embs.append(clip(torch.from_numpy(crops[s:s + 64]))
                            .half().cpu().numpy())
        emb = (np.concatenate(embs) if embs
               else np.zeros((0, 512), np.float16))
        np.savez_compressed(
            of, node=np.array(rows["node"], np.int32),
            name=np.array(rows["name"], np.int32),
            score=np.array(rows["score"], np.float32),
            box=np.array(rows["box"], np.int32).reshape(-1, 4),
            emb=emb, vocab=np.array(vocab))
        log(f"{rob}: 크롭 {len(crops)} ({time.time()-t0:.0f}s)")
    log(f"DONE {tag}")


def _dispatch():
    if len(_sys.argv) < 2 or _sys.argv[1] not in _SUBS:
        print(__doc__)
        _sys.exit(1)
    sub = _sys.argv.pop(1)
    globals()[f"main_{sub}"]()


if __name__ == "__main__":
    _dispatch()

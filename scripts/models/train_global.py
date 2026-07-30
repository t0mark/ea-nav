"""학습 3단계 — 전역 선택기(홉 단위 후보 선택 + STOP + 도착 heading).

에피소드를 홉 순서대로 재생하면서 GraphMap을 실제로 굴리고, 매 홉의 후보
집합(방문 노드 + 고스트 + STOP)에서 정답 하나를 맞히도록 학습한다. 지시가
경로에 관여하고 z 토큰에 기울기가 흐르는 유일한 통로다.

2단계 구조 (제안기가 동결이라 고스트를 매 에폭 다시 뽑을 이유가 없다):
  1) 캐시(--build-cache): 동결 제안기로 프레임마다 고스트 K개의 월드 위치·
     feature와 노드 임베딩을 1회 계산해 에피소드별 npz로 저장
  2) 학습: 캐시만 읽어 GraphMap 재생 → 선택기 forward. RGB/depth 인코더와
     CLIP은 학습 루프에서 아예 돌지 않는다

정답 3분기(데이터 생성부 common.global_gt와 대응):
  forward — pose[t+1] 위치에 가장 가까운 고스트
  node    — pose[t+1] 위치에 가장 가까운 방문 노드(백트래킹·방향 전환·층 전환)
  stop    — 후보 0번 슬롯
후보 매칭이 tol 안에서 실패한 홉은 표본에서 제외하고 수를 보고한다.

손실 = 후보 CE + heading(1 − cos Δθ, 정답 후보 위치에서만). z 토큰은 첫
20% 스텝 동안 선형 개방(set_token_warmup). 지시 없는 프레임은 무지시 토큰.

사용:
  python train_global.py --gpu 0 --build-cache      # 고스트 캐시 1회
  python train_global.py --gpu 0                    # 학습
출력: {out}/global.pt (nav + 지표)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import common


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# 층 간 y 간격(m) — GraphMap이 층을 위치로도 분리하도록 두는 값
FLOOR_Y_M = 3.0
# 엘베 스킬 구간(special) — 전역 선택 미호출
SKILL_SPECIAL = (1, 2, 3, 4)
# 정답 후보 매칭 허용 반경(m)
MATCH_TOL = 1.0


def episode_list(ds):
    """EpisodeFrameDataset.frames → 궤적 단위 [(npz, rid, tag, instr)].

    전역 라벨(gsel)이 없는 궤적은 제외한다 — 제안기 학습에는 쓰이지만
    선택기 학습에는 정답이 없다.
    """
    seen, out = set(), []
    for f, t, instrs, rid, tag, st in ds.frames:
        if f in seen or f in ds.no_global:
            continue
        seen.add(f)
        instr = next((s for s in instrs if s), "")
        out.append((f, rid, tag, instr))
    return out


def build_cache(policy, eps, z_by_rid, cache_dir, device,
                floor_slack=0.0):
    """에피소드별 고스트·노드 임베딩 캐시 (동결 제안기 1회 실행).

    프레임 순서대로 observe를 호출해야 히스토리 버퍼가 학습 때와 같은
    상태가 된다. 저장: world (T,K,3) f32 / feat (T,K,768) f16 /
    valid (T,K) bool / node_embed (T,768) f16.
    """
    import torch
    os.makedirs(cache_dir, exist_ok=True)
    policy.eval()
    todo = [e for e in eps if not os.path.exists(
        os.path.join(cache_dir, os.path.basename(e[0]) + ".ghost.npz"))]
    log(f"캐시 대상 {len(todo)}/{len(eps)}궤적")
    t0 = time.time()
    for i, (f, rid, tag, _) in enumerate(todo):
        with np.load(f) as z:
            rgb, dep, pose = z["rgb"], z["depth"], z["pose"]
        zz = z_by_rid[rid]
        policy.reset_episode()
        W, F, V, E = [], [], [], []
        with torch.no_grad():
            for t in range(len(pose)):
                r = torch.from_numpy(np.ascontiguousarray(rgb[t]))[None]
                d = torch.from_numpy(
                    dep[t].astype(np.float32))[None].to(device)
                obs = policy.observe(r, d)
                res = policy.wp(obs["depth_hist"], zz,
                                depth_m=obs["depth_m"])
                cand = policy.wp.propose(res)
                fl = int(pose[t][0])
                lifted = policy.lift(
                    cand, obs, tuple(float(x) for x in pose[t]),
                    floor_y=fl * FLOOR_Y_M, floor_slack=floor_slack)
                W.append(lifted["world"][0].numpy())
                F.append(lifted["embed"][0].half().cpu().numpy())
                V.append(lifted["valid"][0].numpy())
                E.append(obs["node_embed"][0].half().cpu().numpy())
        np.savez_compressed(
            os.path.join(cache_dir, os.path.basename(f) + ".ghost.npz"),
            world=np.stack(W).astype(np.float32), feat=np.stack(F),
            valid=np.stack(V), node_embed=np.stack(E))
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            log(f"캐시 {i+1}/{len(todo)} ({el/(i+1):.1f}s/궤적, "
                f"ETA {el/(i+1)*(len(todo)-i-1)/60:.1f}분)")


def target_index(inp, gm, kind, cur, nxt, tol=MATCH_TOL):
    """정답 후보 인덱스 — 못 찾으면 None(표본 제외).

    정답은 **위치로** 찾는다: pose[t+1]에 가장 가까운 후보(고스트든 방문
    노드든). 저장된 kind는 "그 홉의 다음 위치가 현재 프레임 히트맵에
    보였는가"를 기록한 것이지 후보의 종류를 못박는 값이 아니다 — 이전 뷰에서
    만들어져 살아 있는 고스트가 정답인 경우가 흔하다.

    **층이 바뀌는 홉**은 정답 좌표가 다음 층이라 현재 후보 집합에서 매칭될
    수 없다 — 이때는 현재 위치에서 가장 가까운 전환 노드(계단·엘베)가
    정답이다(그 노드를 고르면 스킬이 층 전환까지 수행하는 실행 규약).
    """
    if kind == 2:                                   # stop
        return 0
    ids = inp["gmap_vpids"]
    cross = kind == 1 and int(nxt[0]) != int(cur[0])
    if cross:
        best, bi = float("inf"), None
        for i, vp in enumerate(ids):
            if vp is None or vp.startswith("g"):
                continue
            if gm.node_type.get(vp, 0) not in (1, 2):
                continue
            p = gm.node_pos[vp]
            d = float(np.hypot(p[0] - float(cur[1]), p[2] - float(cur[2])))
            if d < best:
                best, bi = d, i
        return bi
    tgt = np.array([float(nxt[1]), float(nxt[2])])
    best, bi = tol, None
    for i, vp in enumerate(ids):
        if vp is None:
            continue
        pos = (gm.ghost_mean_pos[vp] if vp.startswith("g")
               else gm.node_pos[vp])
        d = float(np.hypot(pos[0] - tgt[0], pos[2] - tgt[1]))
        if d < best:
            best, bi = d, i
    return bi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=16,
                    help="홉 단위 기울기 누적 (후보 수가 홉마다 달라 배치 대신)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--hist-k", type=int, default=4)
    ap.add_argument("--warmup-frac", type=float, default=0.2,
                    help="z 토큰 선형 개방 구간(전체 스텝 대비)")
    ap.add_argument("--head-w", type=float, default=1.0, help="heading 손실 가중")
    ap.add_argument("--g-bias", type=int, default=1)
    ap.add_argument("--init-g", default="/data/EVLN_ckpt/traversability.pt")
    ap.add_argument("--init-wp", default="/data/EVLN_ckpt/waypoint_g.pt")
    ap.add_argument("--cache-dir", default="/data/EVLN_ckpt/ghostcache")
    ap.add_argument("--floor-slack", type=float, default=0.0,
                    help="렌더 바닥과 발 높이의 어긋남 여유 — habitat(R2R)은 "
                         "navmesh가 바닥 위에 떠 0.25 필요, MANSION은 0")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--out", default="/data/EVLN_ckpt")
    ap.add_argument("--limit-eps", type=int, default=0)
    ap.add_argument("--sched-sampling", type=float, default=0.0,
                    help="4단계 회복 학습 — 정답 대신 모델 선택을 소비할 "
                         "최종 확률(0→이 값까지 선형 증가, 0이면 미사용)")
    ap.add_argument("--preload-memory", type=int, default=0,
                    help="표준 기억 그래프를 선등록하고 재생 — 배포(폐루프)와 "
                         "같은 후보 집합 크기로 학습")
    ap.add_argument("--init-nav", default="",
                    help="이어서 학습할 선택기 ckpt (3단계 산출)")
    args = ap.parse_args()
    common.pick_gpu(args.gpu)

    import torch
    import loaders
    from EA_Nav import EANav, tokenizer
    from modules.GraphPlanning import GraphMap
    device = "cuda"
    tags = args.envs or loaders.tags()

    policy = EANav(clip_device=device, hist_k=args.hist_k).to(device)
    policy.load_pretrained(finetuned=True, verbose=False)
    if args.init_g:
        common.load_into(args.init_g, urdf_enc=policy.urdf_enc, g=policy.g)
    if args.init_wp and os.path.exists(args.init_wp):
        ck = torch.load(args.init_wp, map_location="cpu", weights_only=False)
        import weights as lp
        lp.load_compat(policy.wp, ck["wp"])
        log(f"2단계 제안기 로드: {args.init_wp}")
    else:
        log("경고: 제안기 ckpt 없음 — 미학습 제안기로 고스트를 뽑게 된다")
    policy.freeze_stage1()

    ds = loaders.EpisodeFrameDataset(env_tags=tags, hist_k=args.hist_k)
    if ds.skipped_legacy:
        log(f"구 스키마 궤적 {ds.skipped_legacy}개 건너뜀")
    if ds.no_global:
        log(f"전역 라벨(gsel) 없는 궤적 {len(ds.no_global)}개 제외 "
            f"— episodes_combined는 gt_rederive 대상 밖")
    eps = episode_list(ds)
    if args.limit_eps:
        eps = eps[:args.limit_eps]
    if not eps:
        raise SystemExit("궤적 0 — GT 재유도(05 gt_rederive) 필요")
    log(f"궤적 {len(eps)}개")

    # z는 로봇별 상수 — 동결 인코더로 1회 계산
    gc = loaders.RobotGraphCache(tags[0])
    rids = sorted({r for _, r, _, _ in eps})
    with torch.no_grad():
        gb = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in gc.collate(rids).items()}
        zs, _ = policy.urdf_enc(gb)
    z_by_rid = {r: zs[i:i + 1] for i, r in enumerate(rids)}

    if args.build_cache:
        build_cache(policy, eps, z_by_rid, args.cache_dir, device,
                    args.floor_slack)
        log(f"고스트 캐시 완료 → {args.cache_dir}")
        return

    missing = [f for f, _, _, _ in eps if not os.path.exists(
        os.path.join(args.cache_dir, os.path.basename(f) + ".ghost.npz"))]
    if missing:
        raise SystemExit(f"고스트 캐시 없음 {len(missing)}개 — "
                         f"--build-cache 먼저 실행")

    # 표준 기억 선등록 재료 (태그별 1회 로드)
    memcache = {}
    if args.preload_memory:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "benchmark"))
        import mansion_adapter as ma
        for tg in sorted({e[2] for e in eps}):
            mm = ma.UnionMemory(tg)
            ff = ma.MemoryFeatures(policy, mm, device)
            nodes = [((float(n["x"]), float(n["floor"]) * FLOOR_Y_M,
                       float(n["z"])),
                      torch.from_numpy(ff.node_embed_mean(mm, uid)).to(device),
                      int(n["floor"]),
                      2 if n.get("kind") == "elevator" else 0)
                     for uid, n in enumerate(mm.nodes)]
            edges = [(u, v, float(L)) for (u, v, _m, L) in mm.edges]
            memcache[tg] = (nodes, edges)
            log(f"기억 선등록 준비 {tg[:28]}: 노드 {len(nodes)} 엣지 {len(edges)}")

    tok = tokenizer()
    tr = [e for e in eps
          if not common.is_val_traj(e[2], os.path.basename(e[0]),
                                    args.val_frac)]
    tr_set = {e[0] for e in tr}
    va = [e for e in eps if e[0] not in tr_set]
    log(f"궤적 train {len(tr)} / val {len(va)}")

    # 학습 = 선택기(nav)만. 제안기·인코더·g는 동결.
    if args.init_nav and os.path.exists(args.init_nav):
        common.load_into(args.init_nav, nav=policy.nav)
        log(f"3단계 선택기 이어받기: {args.init_nav}")
    common.freeze(policy)
    common.freeze(policy.nav, False)
    policy.freeze_stage1()          # g는 nav 안에서도 공유되므로 다시 동결
    params = [p for p in policy.nav.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    log(f"학습 파라미터 {sum(p.numel() for p in params)/1e6:.2f}M (선택기)")

    def episode_pass(ep, train, step_ref, ss_p=0.0):
        f, rid, tag, instr = ep
        with np.load(f) as z:
            pose, gsel = z["pose"], z["gsel"]
            guv, ghead, special = z["gsel_uv"], z["gsel_head"], z["special"]
        cache = dict(np.load(os.path.join(
            args.cache_dir, os.path.basename(f) + ".ghost.npz")))
        zz = z_by_rid[rid]
        if instr:
            enc = tok(instr, return_tensors="pt", truncation=True,
                      max_length=80)
            txt = policy("language", txt_ids=enc["input_ids"].to(device),
                         txt_masks=enc["attention_mask"].to(device))
            tmask = enc["attention_mask"].to(device)
        else:
            txt, tmask = policy.nav.null_instruction(1, device)

        gm = GraphMap()
        if tag in memcache:
            gm.preload(*memcache[tag])
        stats = {"n": 0, "hit": 0, "skip": 0, "skip_fwd": 0, "skip_node": 0,
                 "loss": 0.0, "head": 0.0}
        prev_fl, prev_vp = None, None
        for t in range(len(pose)):
            fl = int(pose[t][0])
            w = cache["world"][t][cache["valid"][t]]
            e = torch.from_numpy(cache["feat"][t][cache["valid"][t]]
                                 .astype(np.float32)).to(device)
            ne = torch.from_numpy(
                cache["node_embed"][t].astype(np.float32)).to(device)
            ntype = 2 if int(special[t]) in SKILL_SPECIAL else 0
            vp = gm.update(
                np.array([pose[t][1], fl * FLOOR_Y_M, pose[t][2]]),
                ne, list(w), list(e), node_type=ntype, floor=fl)
            # 계단 등반은 special 마커가 없어 층 변화로만 알 수 있다 —
            # 직전 노드를 계단 전환 노드로 소급 표시해야 층 전환 홉의
            # 정답(전환 노드)이 후보 집합에 존재한다
            if (prev_fl is not None and fl != prev_fl
                    and int(special[t]) not in SKILL_SPECIAL):
                gm.node_type[prev_vp] = 1
            prev_fl, prev_vp = fl, vp
            if int(special[t]) in SKILL_SPECIAL:
                continue                     # 스킬 구간 = 전역 미호출
            inp = gm.gmap_inputs(
                vp, np.array([pose[t][1], fl * FLOOR_Y_M, pose[t][2]]),
                np.radians(float(pose[t][3])), device=device, cur_floor=fl)
            kind = int(gsel[t])
            nxt = pose[min(t + 1, len(pose) - 1)]
            ti = target_index(inp, gm, kind, pose[t], nxt)
            if ti is None:
                # 매칭 실패 = 정답 위치 근처에 후보가 없는 홉. forward 실패는
                # 제안기 품질, node 실패는 기억 그래프 커버리지 문제다
                stats["skip"] += 1
                stats["skip_fwd" if kind == 0 else "skip_node"] += 1
                continue
            nav = policy("navigation", z=zz, txt_embeds=txt, txt_masks=tmask,
                         mask_visited=False, g_bias=bool(args.g_bias), **inp)
            lo = nav["global_logits"]
            tgt = torch.tensor([ti], device=device)
            loss = torch.nn.functional.cross_entropy(lo, tgt)
            hd = float(ghead[t])
            if hd > -900:
                loss = loss + args.head_w * common.heading_loss(
                    nav["heading"][0, ti:ti + 1], [hd])
                stats["head"] += 1
            if train:
                (loss / args.accum).backward()
                step_ref[0] += 1
                if step_ref[0] % args.accum == 0:
                    opt.step(); opt.zero_grad()
                if args.warmup_frac > 0:
                    frac = min(1.0, step_ref[0]
                               / max(1.0, step_ref[1] * args.warmup_frac))
                    policy.set_token_warmup(frac)
            stats["n"] += 1
            pick = int(lo[0].argmax())
            stats["hit"] += int(pick == ti)
            stats["loss"] += float(loss)
            # 다음 홉으로 진행 = 선택된 고스트 소비. 회복 학습에서는 정답
            # 대신 **모델이 고른 후보**를 소비해, 그래프가 모델의 실수까지
            # 반영한 상태로 자라게 한다(폐루프와 같은 조건). 감독은 그대로
            # 정답으로 주므로 어긋난 상태에서 복귀하는 법을 배운다.
            take = ti
            if train and ss_p > 0.0 and pick != ti:
                import random as _rnd
                if _rnd.random() < ss_p:
                    take = pick
                    stats["dev"] = stats.get("dev", 0) + 1
            sel = inp["gmap_vpids"][take]
            if sel and sel.startswith("g"):
                gm.delete_ghost(sel)
        return stats

    # 워밍업 분모용 총 스텝 추정 (궤적당 평균 홉 수를 20으로 가정) —
    # 정확한 값이 아니어도 되는 이유는 frac을 1.0에서 자르기 때문
    step_ref = [0, max(1, args.epochs * len(tr) * 20)]
    best = None
    for ep_i in range(args.epochs):
        policy.nav.train()
        agg = {"n": 0, "hit": 0, "skip": 0, "skip_fwd": 0,
               "skip_node": 0, "loss": 0.0}
        opt.zero_grad()
        # 회복 확률은 0에서 목표치까지 선형 증가 — 초반부터 흔들면 기본
        # 경로조차 못 배운다
        ss_p = (args.sched_sampling * (ep_i + 1) / max(args.epochs, 1)
                if args.sched_sampling > 0 else 0.0)
        t_ep = time.time()
        for i_e, e in enumerate(tr):
            s = episode_pass(e, True, step_ref, ss_p)
            for k in agg:
                agg[k] += s[k]
            # 중간 점검 — 침묵 구간을 만들지 않는다
            if (i_e + 1) % 100 == 0 or i_e + 1 == len(tr):
                el = time.time() - t_ep
                per = el / (i_e + 1)
                rest = per * (len(tr) - i_e - 1)
                left_ep = args.epochs - ep_i - 1
                eta = (rest + (per * len(tr) + el * 0.15) * left_ep) / 60
                log(f"  ep{ep_i+1} {i_e+1}/{len(tr)}궤적 "
                    f"top1 {agg['hit']/max(agg['n'],1):.3f} "
                    f"({per:.2f}s/궤적, 이번 에폭 잔여 {rest/60:.1f}분, "
                    f"전체 ETA {eta:.0f}분)")
        policy.nav.eval()
        vagg = {"n": 0, "hit": 0, "skip": 0, "skip_fwd": 0,
                "skip_node": 0, "loss": 0.0}
        with torch.no_grad():
            for e in va:
                s = episode_pass(e, False, step_ref)
                for k in vagg:
                    vagg[k] += s[k]
        tr_acc = agg["hit"] / max(agg["n"], 1)
        va_acc = vagg["hit"] / max(vagg["n"], 1)
        log(f"ep {ep_i+1}/{args.epochs} train top1 {tr_acc:.4f} "
            f"loss {agg['loss']/max(agg['n'],1):.4f} | val top1 {va_acc:.4f} "
            f"| 매칭실패 train {agg['skip']}(fwd {agg['skip_fwd']}/"
            f"node {agg['skip_node']}) val {vagg['skip']} | 회복확률 {ss_p:.2f}")
        if best is None or va_acc > best:
            best = va_acc
            common.save_ckpt(os.path.join(args.out, "global.pt"),
                             nav=policy.nav,
                             meta={"val_top1": va_acc, "epoch": ep_i + 1,
                                   "tags": tags, "hist_k": args.hist_k})
    log(f"완료 — best val top1 {best:.4f} → {args.out}/global.pt")


if __name__ == "__main__":
    main()

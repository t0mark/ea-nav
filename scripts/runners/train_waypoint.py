"""학습 2단계 — 로컬 제안기(픽셀 히트맵) imitation.

입력 = depth 히스토리 K프레임(동결 DepthEncoder)과 z(URDF GNN).
출력 = 32² 히트맵 로짓, 손실 = 도달가능 자유공간 래스터 BCE 단일
(IGNORE 경계 픽셀 제외). 구 각도·거리·heading 빈 CE는 폐기됐고 도착
heading은 선택기가 예측하므로 여기서 학습하지 않는다.

동결 정책: 1단계에서 학습한 URDF GNN·g는 동결한다 — z 분포가 흔들리면
g가 깨지고, g는 픽셀·노드·스냅샷·고스트 네 소비처가 공유한다. 따라서
학습 대상은 wp 본체(g 인스턴스 제외)뿐이다.

검증 = 궤적 단위 홀드아웃(common.is_val_traj) — 같은 궤적 프레임 누수 없음.
지표 = precision·recall·IoU·AUC(임계 0.5) + GT 양성 비율.

사용: python train_waypoint.py --gpu 0 [--init /data/EVLN_ckpt/traversability.pt]
출력: {out}/waypoint_{inject}.pt (wp + 지표)
"""
import argparse
import os
import time

import numpy as np

import common


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--hist-k", type=int, default=4)
    ap.add_argument("--pos-weight", type=float, default=0.0,
                    help="BCE 양성 가중(0 = 미사용)")
    ap.add_argument("--init", default="/data/EVLN_ckpt/traversability.pt",
                    help="1단계 ckpt ('' = 생략)")
    ap.add_argument("--out", default="/data/EVLN_ckpt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-steps", type=int, default=0)
    ap.add_argument("--inject", choices=("g", "film", "both", "none"),
                    default="g", help="z 주입 ablation (WaypointNet)")
    args = ap.parse_args()
    common.pick_gpu(args.gpu)

    import torch
    from torch.utils.data import DataLoader, Subset
    import loaders
    from EA_Nav import EANav
    device = "cuda"
    tags = args.envs or loaders.tags()

    policy = EANav(clip_device=device, waypoint_inject=args.inject,
                   hist_k=args.hist_k).to(device)
    policy.load_pretrained(finetuned=True, verbose=False)
    if args.init:
        common.load_into(args.init, urdf_enc=policy.urdf_enc, g=policy.g)
        log(f"1단계 ckpt 로드: {args.init}")

    # 학습 = wp 본체만. g·urdf_enc는 동결(공유 손상 방지), 트렁크도 동결.
    common.freeze(policy)
    common.freeze(policy.wp, False)
    policy.freeze_stage1()
    policy.nav.eval(); policy.depth_enc.eval()
    train_params = [p for p in policy.wp.parameters() if p.requires_grad]
    log(f"학습 파라미터 {common.count_trainable(policy)/1e6:.2f}M "
        f"(g·urdf_enc 동결)")
    opt = torch.optim.AdamW(train_params, lr=args.lr)
    pw = args.pos_weight or None

    ds = loaders.EpisodeFrameDataset(env_tags=tags, hist_k=args.hist_k)
    if ds.skipped_legacy:
        log(f"구 스키마(heat 없음) 궤적 {ds.skipped_legacy}개 건너뜀 "
            f"— 05 gt_rederive 미완 구간")
    if not ds.frames:
        raise SystemExit("학습 프레임 0 — GT 재유도(05 gt_rederive) 필요")
    tr, va = [], []
    for i, (f, t, _, rid, tag, st) in enumerate(ds.frames):
        (va if common.is_val_traj(tag, os.path.basename(f),
                                  args.val_frac) else tr).append(i)
    log(f"프레임 train {len(tr)} / val {len(va)} (궤적 홀드아웃)")
    gc = loaders.RobotGraphCache(tags[0])
    coll = lambda b: loaders.collate_frames(b, gc)
    dl_tr = DataLoader(Subset(ds, tr), batch_size=args.bs, shuffle=True,
                       num_workers=args.workers, collate_fn=coll)
    dl_va = DataLoader(Subset(ds, va), batch_size=args.bs, shuffle=False,
                       num_workers=args.workers, collate_fn=coll)

    def fwd(batch):
        hist = batch["depth_hist"].to(device)
        with torch.no_grad():
            # (B,K,256,256) → (B,K,128,4,4), z는 동결 인코더
            df = policy.depth_enc(hist)
            gb = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in batch["robot_graph"].items()}
            z, _ = policy.urdf_enc(gb)
        out = policy.wp(df, z, depth_m=hist[:, 0])
        return out["heat"], batch["heat"].to(device)

    def run_eval():
        policy.wp.eval()
        acc = {}
        n = 0
        with torch.no_grad():
            for b in dl_va:
                lo, tgt = fwd(b)
                m = common.heat_metrics(lo, tgt)
                for k, v in m.items():
                    acc[k] = acc.get(k, 0.0) + v * len(tgt)
                n += len(tgt)
        policy.wp.train()
        return {k: v / max(n, 1) for k, v in acc.items()}

    best = None
    for ep in range(args.epochs):
        m = common.Meter()
        for si, b in enumerate(dl_tr):
            if args.limit_steps and si >= args.limit_steps:
                break
            lo, tgt = fwd(b)
            loss = common.heat_loss(lo, tgt, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
            m.add(loss.item(), len(tgt))
        v = run_eval()
        log(f"ep {ep+1}/{args.epochs} train bce {m.avg:.4f} | val "
            f"iou {v.get('iou', 0):.4f} prec {v.get('prec', 0):.4f} "
            f"recall {v.get('recall', 0):.4f} auc {v.get('auc', 0):.4f} "
            f"(GT 양성 {v.get('pos_rate', 0):.3f})")
        if best is None or v.get("iou", 0) > best:
            best = v.get("iou", 0)
            # g·urdf_enc는 동결이지만 함께 저장한다 — 평가·다음 단계가
            # ckpt 하나로 같은 모델을 정확히 복원하게 하기 위해
            common.save_ckpt(
                os.path.join(args.out, f"waypoint_{args.inject}.pt"),
                wp=policy.wp, g=policy.g, urdf_enc=policy.urdf_enc,
                meta={"val": v, "epoch": ep + 1, "tags": tags,
                      "inject": args.inject, "hist_k": args.hist_k})
    log(f"완료 — best val IoU {best:.4f} → "
        f"{args.out}/waypoint_{args.inject}.pt")


if __name__ == "__main__":
    main()

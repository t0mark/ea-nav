"""loss (a) 통과성 사전학습 — URDF GNN(z) + 통과성 헤드 g.

2단계 구조(시각 트렁크 동결을 이용한 feature 캐시):
  1) 캐시: 게이트 스냅샷 RGB-D → policy.observe() node_embed(768)를 1회
     계산해 npy로 저장(동결 경로라 재계산 낭비 — 52k장 수 분).
  2) 학습: 캐시 feature 전체를 GPU에 상주시키고 (feature, z(로봇), soft
     라벨) 미니배치로 g+GNN만 학습. z는 매 스텝 480 로봇 그래프 배치로
     재계산(기울기 통과).

검증 = 게이트 단위 홀드아웃(common.is_val_gate, 해시 결정적) — 같은
게이트의 9뷰·480로봇이 통째로 빠져 뷰 누수 없음.

사용: python 01_pretrain_passability.py --gpu 0 [--epochs 20 --bs 4096]
출력: {out}/traversability.pt (urdf_enc·g + 지표)
"""
import argparse
import glob
import json
import os
import time

import numpy as np

# 경로 주입 (research/models·data)
import common


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_cache(policy, tags, cache_dir, device, bs=64):
    """샤드 npz별 node_embed(float16) 캐시 — 파일명 대응 .feat.npy."""
    import torch
    import loaders
    os.makedirs(cache_dir, exist_ok=True)
    # 동결 트렁크 dropout 차단 (동등성 함정 방지)
    policy.eval()
    for tag in tags:
        for f in sorted(glob.glob(os.path.join(
                loaders.DS, "gates", f"{tag}_snaps_shard*.npz"))):
            out = os.path.join(cache_dir,
                               os.path.basename(f) + ".feat.npy")
            with np.load(f) as z:
                rgb, dep = z["rgb"], z["depth"]
            # 스냅샷이 재생성되면 캐시는 무효 — 존재만 보고 건너뛰면 옛
            # feature로 학습된다. 행 수와 mtime을 모두 확인한다
            if os.path.exists(out):
                old_n = np.load(out, mmap_mode="r").shape[0]
                if (old_n == len(rgb)
                        and os.path.getmtime(out) >= os.path.getmtime(f)):
                    continue
                log(f"캐시 갱신 필요: {os.path.basename(out)} "
                    f"(행 {old_n}→{len(rgb)})")
            feats = []
            with torch.no_grad():
                for s in range(0, len(rgb), bs):
                    r = torch.from_numpy(rgb[s:s + bs]).to(device)
                    d = torch.from_numpy(
                        dep[s:s + bs].astype(np.float32)).to(device)
                    # 게이트 스냅샷은 서로 독립이라 히스토리를 쌓지 않는다
                    feats.append(policy.observe(r, d, push_history=False)
                                 ["node_embed"].half().cpu().numpy())
            np.save(out, np.concatenate(feats))
            log(f"캐시 {os.path.basename(out)} ({len(rgb)}장)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="/data/EVLN_ckpt")
    ap.add_argument("--cache-dir", default="/data/EVLN_ckpt/featcache")
    ap.add_argument("--cache-only", action="store_true",
                    help="feature 캐시만 만들고 종료 — 환경별 병렬 캐싱용")
    ap.add_argument("--limit-steps", type=int, default=0,
                    help="스모크: 에포크당 스텝 상한")
    ap.add_argument("--augment", type=int, default=1,
                    help="학습 시 그래프 구조 잡음 증강 (0=끔)")
    args = ap.parse_args()
    common.pick_gpu(args.gpu)

    import torch
    import loaders
    from EA_Nav import EANav
    device = "cuda"
    tags = args.envs or loaders.tags()

    policy = EANav(clip_device=device).to(device)
    policy.load_pretrained(finetuned=True, verbose=False)
    build_cache(policy, tags, args.cache_dir, device)
    if args.cache_only:
        log("캐시 전용 모드 — 학습 생략")
        return

    # ---- 표본 인덱스 조립: (feat행, 라벨, 로봇idx, val여부) ----
    ds = loaders.TraversabilityDataset(env_tags=tags)
    feat_files, feats = {}, []
    off = 0
    for tag in tags:
        for f in sorted(glob.glob(os.path.join(
                loaders.DS, "gates", f"{tag}_snaps_shard*.npz"))):
            a = np.load(os.path.join(args.cache_dir,
                                     os.path.basename(f) + ".feat.npy"))
            feat_files[f] = off
            feats.append(a)
            off += len(a)
    # (N,768) f16
    feats = torch.from_numpy(np.concatenate(feats)).to(device)
    log(f"feature 캐시 {feats.shape} GPU 상주")

    rob_ids = [r["id"] for r in ds.envs[0]["robots"]]
    rid2idx = {r: i for i, r in enumerate(rob_ids)}
    rows, labels, robs, vals = [], [], [], []
    for (ei, gi, ri, bi) in ds.samples:
        env = ds.envs[ei]
        v = common.is_val_gate(env["tag"], gi, args.val_frac)
        for di in range(env["n_dist"]):
            for ai in range(env["n_ang"]):
                f, row = env["key2loc"][(gi, di, ai, bi)]
                rows.append(feat_files[f] + row)
                labels.append(env["labels"][gi, ri])
                robs.append(rid2idx[env["robots"][ri]["id"]])
                vals.append(v)
    rows = torch.tensor(rows, device=device)
    labels = torch.tensor(labels, device=device)
    robs = torch.tensor(robs, device=device)
    vals = torch.tensor(vals, device=device)
    tr_idx = torch.where(~vals)[0]
    va_idx = torch.where(vals)[0]
    log(f"표본 train {len(tr_idx)} / val {len(va_idx)} (게이트 홀드아웃)")

    # ---- 로봇 480 그래프 배치 (z 재계산용) ----
    # 평가용은 무증강 1회 조립. 증강 켜면 학습 스텝마다 구조 잡음 재조립.
    gc = loaders.RobotGraphCache(tags[0])
    gbatch = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in gc.collate(rob_ids).items()}
    aug_rng = np.random.default_rng(0) if args.augment else None

    def train_gbatch():
        if aug_rng is None:
            return gbatch
        return {k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in gc.collate(rob_ids, augment_rng=aug_rng).items()}

    # 보조 프로브 GT (plan 확정: capability 라벨 = z 보조 loss GT 이중 사용)
    meta_r = loaders.robots_meta(tags[0])
    stairs_t = torch.tensor([float(meta_r[r]["stairs_ok"])
                             for r in rob_ids], device=device)
    w_t = torch.tensor([meta_r[r]["w_eff"] for r in rob_ids],
                       device=device)
    h_t = torch.tensor([meta_r[r]["h"] for r in rob_ids], device=device)
    ch_t = torch.tensor([meta_r[r]["cam_h"] for r in rob_ids],
                        device=device)

    g, enc = policy.g, policy.urdf_enc
    common.freeze(policy)
    common.freeze(g, False)
    common.freeze(enc, False)
    log(f"학습 파라미터 {common.count_trainable(policy)/1e6:.2f}M (g+GNN)")
    opt = torch.optim.AdamW([*g.parameters(), *enc.parameters()],
                            lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()

    def run_eval(idx):
        g.eval(); enc.eval()
        outs, ls = [], []
        with torch.no_grad():
            z, _ = enc(gbatch)
            for s in range(0, len(idx), args.bs):
                b = idx[s:s + args.bs]
                lo = g(feats[rows[b]].float(), z[robs[b]])
                outs.append(lo)
                ls.append(labels[b])
        g.train(); enc.train()
        lo, la = torch.cat(outs), torch.cat(ls)
        hard = (la > 0.5).float()
        acc = ((lo > 0) == hard.bool()).float().mean()
        return (bce(lo, la).item(), acc.item(),
                common.auc(lo, hard))

    best = None
    for ep in range(args.epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        m = common.Meter()
        for si, s in enumerate(range(0, len(perm), args.bs)):
            if args.limit_steps and si >= args.limit_steps:
                break
            b = perm[s:s + args.bs]
            z, aux = enc(train_gbatch())
            logit = g(feats[rows[b]].float(), z[robs[b]])
            aux_l = (torch.nn.functional.binary_cross_entropy_with_logits(
                aux["stairs_ok_logit"], stairs_t)
                + torch.nn.functional.mse_loss(aux["wlh"][:, 0], w_t)
                + torch.nn.functional.mse_loss(aux["wlh"][:, 2], h_t)
                + torch.nn.functional.mse_loss(aux["cam_h"], ch_t))
            loss = bce(logit, labels[b]) + 0.1 * aux_l
            opt.zero_grad(); loss.backward(); opt.step()
            m.add(loss.item(), len(b))
        vb, va_, vauc = run_eval(va_idx)
        with torch.no_grad():
            _, aux = enc(gbatch)
            sacc = float((((aux["stairs_ok_logit"] > 0).float()
                           == stairs_t)).float().mean())
            ch_mae = float((aux["cam_h"] - ch_t).abs().mean())
        log(f"ep {ep+1}/{args.epochs} train bce {m.avg:.4f} | "
            f"val bce {vb:.4f} acc {va_:.4f} auc {vauc:.4f} | "
            f"stairs_ok {sacc:.3f} cam_h MAE {ch_mae:.3f}m")
        # 베스트 기준 = val AUC (BCE는 과신 벌점 탓에 acc/AUC 정점과 어긋남 실측)
        if best is None or vauc > best:
            best = vauc
            common.save_ckpt(os.path.join(args.out, "traversability.pt"),
                             urdf_enc=enc, g=g,
                             meta={"val_bce": vb, "val_acc": va_,
                                   "val_auc": vauc, "epoch": ep + 1,
                                   "tags": tags, "robots": rob_ids})
    log(f"완료 — best val auc {best:.4f} → {args.out}/traversability.pt")


if __name__ == "__main__":
    main()

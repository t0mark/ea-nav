"""R2R-CE 폐루프 벤치마크 — ETPNav 원본 평가 파이프라인 + Go2 embodiment 조건.

원본 repo(research_assets/ETPNav — 컨테이너 /workspace/ETPNav)는 무수정. 이 파일이 원본 SS-ETP 트레이너의
eval 경로(파노라마 12뷰 + 원본 waypoint predictor + CMT + release_r2r ckpt)를
그대로 구동하되, 다음 3개만 서브클래스/몽키패치로 주입한다:
  ① ThreadedVectorEnv 강제(NUM_ENVIRONMENTS=1) — 몽키패치가 워커 프로세스로
     전파되지 않는 문제를 회피(in-process 실행). 단일 env라 성능 손실 없음.
  ② VLNCEDaggerEnv.reset 래핑 — 씬 로드마다 Go2 반경(w_eff/2=0.155m)·높이로
     navmesh 재계산(기본 .navmesh는 반경 0.1 사전계산본이라 config만으론 무효).
  ③ (--variant ours) policy.net.forward 래핑 — mode='navigation'의
     global_logits에서 고스트(미방문 후보) 노드를 g(gmap_img_fts, z) < τ면
     −inf 차단. 주입 지점 = CMT 후보 스코어링 직후의 후보 feature(768,
     forward_panorama 계열 임베딩 — 우리 g의 학습 입력과 같은 규격).
     embodiment 토큰류(nav_token)는 일절 사용하지 않는다(폐기 결정 준수).

variant:
  base : 원본 ETPNav 그대로 + Go2 embodiment 조건(반경·카메라 높이 0.363m
         (sensor_rgb FK 실높이)·depth 256² HFOV90 10m — 센서 12뷰 전부 반영)
  ours : base + z 주입(ckpt01의 urdf_enc→z, g 게이팅 τ 기본 2.5)

에피소드 = 표준 R2R-CE val_unseen 1,839개(R2R_VLNCE_v1-3, VLN-CE-v1 로더).
샤딩 = gt json 키 순서 [shard::nshards] — 기존 베이스라인 4-GPU 평가의
rank 분배([local_rank::GPU_NUMBERS])와 동일 규칙이라 서브셋이 그대로 대응.

지표 = 원본 rollout('eval') 산출(stat_eps)을 그대로 채택 — SR(geodesic ≤3m),
SPL, NE(geodesic), TL, nDTW(vln_metrics와 동일 정의). geodesic inf(단절
navmesh) 가드: inf인 NE는 집계에서 제외하고 별도 카운트, SPL nan은 0 처리.

사용(hw_vln_ce 컨테이너, GPU 0~3 병렬 — 전 샤드 동일 설정 필수):
  python /research/scripts/benchmark/r2r_ce.py \
      --variant ours --gpu S --shard S --nshards 4 [--limit 8]
결과 = /data/EVLN_ckpt/benchmark/r2rce_{variant}_{shard}.json
(meta에 ckpt·τ·limit·로봇·설정 해시 기록 — 샤드 설정 혼합 방지).
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.abspath(os.path.join(_HERE, "..", "..", "models"))
ETPNAV_DIR = "/workspace/ETPNav"   # research_assets (hw_vln_ce 마운트)

GO2_DIR = "/data/URDF/real_robots/multileg/unitree_go2"
ETPNAV_CKPT = "/data/ETPNav_weights/logs/checkpoints/release_r2r/" \
              "ckpt.iter12000.pth"
DATA_PATH = ("data/datasets/R2R_VLNCE_v1-3_BERTidx/"
             "{split}/{split}_bertidx.json.gz")
GT_PATH = ("data/datasets/R2R_VLNCE_v1-3_preprocessed/"
           "{split}/{split}_gt.json.gz")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def robot_spec(urdf_dir):
    """로봇 사양 — cam_h = sensor_rgb FK 실높이(스탠딩, robot_pool 규약)."""
    import yourdfpy

    with open(os.path.join(urdf_dir, "meta.json")) as fh:
        meta = json.load(fh)
    urdf = os.path.join(urdf_dir, "robot.urdf")
    u = yourdfpy.URDF.load(urdf, load_meshes=False,
                           build_collision_scene_graph=False)
    cam_h = None
    for link in u.link_map:
        if link.startswith("sensor_rgb"):
            cam_h = float(meta["base_z"] + u.get_transform(
                link, u.base_link)[2, 3])
            break
    assert cam_h is not None, "sensor_rgb 링크 없음"
    w, l, h = (float(meta["measured"][k]) for k in ("w", "l", "h"))
    return {"id": meta["id"], "urdf": urdf, "cam_h": round(cam_h, 4),
            "radius": round(min(w, l) / 2.0, 4), "height": h}


def build_z_and_g(ckpt01, urdf_path, device):
    """ckpt01(traversability.pt)의 urdf_enc·g 로드 → (g 헤드, 로봇 z)."""
    import torch
    sys.path.insert(0, _MODELS)
    from modules.TraversabilityHead import TraversabilityHead
    from modules.URDFEncoder import URDFEncoder, collate_graphs, urdf_to_graph
    ck = torch.load(ckpt01, map_location="cpu", weights_only=False)
    g = TraversabilityHead(768, 128)
    g.load_state_dict(ck["g"])
    enc = URDFEncoder(z_dim=128)
    enc.load_state_dict(ck["urdf_enc"])
    g.to(device).eval()
    enc.to(device).eval()
    with torch.no_grad():
        batch = collate_graphs([urdf_to_graph(urdf_path)])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        z, _aux = enc(batch)
    return g, z.float()


def attach_g_gate(net, g_head, z, tau, stats):
    """③ CMT 후보 스코어링에 g 게이트 주입 — 고스트 노드 로짓 τ 미만 차단."""
    import torch
    orig = net.forward

    def fwd(*a, mode=None, **kw):
        out = orig(*a, mode=mode, **kw) if a else orig(mode=mode, **kw)
        if mode == "navigation":
            ghost = kw["gmap_masks"] & kw["gmap_visited_masks"].logical_not()
            # 인덱스 0 = stop 토큰
            ghost[:, 0] = False
            with torch.no_grad():
                glog = g_head(kw["gmap_img_fts"].float(),
                              z.expand(kw["gmap_img_fts"].shape[0], -1))
            block = ghost & (glog < tau)
            out["global_logits"] = out["global_logits"].masked_fill(
                block, -float("inf"))
            stats["n_cand"] += int(ghost.sum())
            stats["n_block"] += int(block.sum())
            # g 로짓 분포 기록(τ 캘리브 근거). 표본을 모아 분위수까지 남긴다
            # — g는 MANSION 관측으로 학습해 ETPNav 파노라마 feature에서는
            # 척도가 옮겨가므로, τ를 이식하지 말고 여기 분포에서 정해야 한다.
            if ghost.any():
                gv = glog[ghost]
                stats["g_sum"] += float(gv.sum())
                stats["g_min"] = min(stats["g_min"], float(gv.min()))
                stats["g_max"] = max(stats["g_max"], float(gv.max()))
                s = stats.setdefault("g_samples", [])
                if len(s) < 200000:
                    s.extend(gv.detach().cpu().numpy().tolist())
        return out

    net.forward = fwd


# ==================================================================
# 롤: report — R2R-CE embodiment 파생 데이터셋 리포트 — 서브셋 분리 채점 (재작성).
# ==================================================================
import argparse
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

DATA = "/data/R2R"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_meta(path):
    with open(os.path.join("/data", path)) as f:
        m = json.load(f)
    return m.get("plan", {})


def main_report():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="/data/EVLN_ckpt/benchmark")
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in
            open(os.path.join(DATA, "splits", f"{args.split}.jsonl"))]
    mp3d = [r for r in rows if str(r["scene_id"]).startswith("mp3d")]
    mans = [r for r in rows if not str(r["scene_id"]).startswith("mp3d")]
    log(f"{args.split}: 전체 {len(rows)} = MP3D {len(mp3d)} + MANSION {len(mans)}")

    # ---- MP3D 서브셋: 차단 발생률 + clean 무결성 -------------------------
    def bucket(r):
        if r["blocked"]:
            return "blocked"
        return "partial_only" if r["partial"] else "clean"

    def category(r):
        return "holonomic" if r["morphology"][4] >= 0.5 else "diffdrive"

    mp_stat = defaultdict(Counter)
    for r in mp3d:
        mp_stat[category(r)][bucket(r)] += 1
        mp_stat["전체"][bucket(r)] += 1

    clean_rows = [r for r in mp3d if bucket(r) == "clean"]
    with ThreadPoolExecutor(args.workers) as ex:
        plans = list(ex.map(read_meta, (r["meta_path"] for r in clean_rows)))
    snap = sorted(p.get("goal_offset_m") for p in plans
                  if p.get("goal_offset_m") is not None)
    integrity = {
        "n_clean_meta": len(snap),
        "snap_mean_m": sum(snap) / max(len(snap), 1),
        "snap_max_m": snap[-1] if snap else None,
        "snap_over_0.5m": sum(1 for s in snap if s > 0.5),
    }

    mp_report = {}
    for cat, c in mp_stat.items():
        n = sum(c.values())
        mp_report[cat] = {
            "n": n,
            "clean": c["clean"], "blocked": c["blocked"],
            "partial_only": c["partial_only"],
            "차단률": c["blocked"] / n,
            "차단+절단률": (c["blocked"] + c["partial_only"]) / n,
        }

    # ---- MANSION 서브셋: plan.success / refusal 상태 판정 ---------------
    with ThreadPoolExecutor(args.workers) as ex:
        mplans = list(ex.map(read_meta, (r["meta_path"] for r in mans)))
    # refusal 없음 → 주행 에피소드
    drive = Counter()
    # refusal_kind → 불가능 판정 GT 풀
    refusal = Counter()
    by_robot = defaultdict(Counter)
    for r, p in zip(mans, mplans):
        kind = p.get("refusal_kind") or ""
        if kind:
            refusal[kind] += 1
            by_robot[r["robot"]]["refusal"] += 1
            if p.get("success"):        # refusal인데 success=true → 모순
                refusal["_inconsistent_success"] += 1
        else:
            ok = bool(p.get("success"))
            drive["success" if ok else "fail"] += 1
            by_robot[r["robot"]]["success" if ok else "fail"] += 1

    n_drive = drive["success"] + drive["fail"]
    man_report = {
        "n": len(mans),
        "주행_에피소드": {"n": n_drive, "success": drive["success"],
                       "성공률": drive["success"] / max(n_drive, 1)},
        "불가능판정_GT풀": dict(refusal),
        "by_robot": {k: dict(v) for k, v in sorted(by_robot.items())},
    }

    report = {"split": args.split,
              "mp3d_차단발생률": mp_report,
              "mp3d_clean_무결성": integrity,
              "mansion_상태판정": man_report,
              "주의": "두 서브셋은 합산 금지 — MP3D는 expert 재생(정책 "
                      "평가 아님), MANSION refusal은 SR 분모 제외"}

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"r2r_report_{args.split}.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log(f"저장: {out}")
    log(f"MP3D 차단률 {mp_report['전체']['차단률']:.3f} / "
        f"차단+절단 {mp_report['전체']['차단+절단률']:.3f} / "
        f"clean 스냅 max {integrity['snap_max_m']}")
    log(f"MANSION 주행 성공률 {man_report['주행_에피소드']['성공률']:.3f} "
        f"({n_drive}건) / refusal {sum(refusal.values())}건 {dict(refusal)}")



# ==================================================================
# 롤: prep_frames — R2R-CE train → 우리 학습 포맷 변환 (Go2 시점, 혼합 재학습용).
# ==================================================================
import argparse
import gzip
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/research/models")
sys.path.insert(0, "/workspace/ETPNav")

GO2_CAM_H = 0.363
STEP_M = 1.0
# N_D = 14 (거리빈 상한 3.5m 규약과 일치). 기존 /data/R2R_go2 덤프는
# 12빈 산물(d≤11 — 14빈에서도 유효 인덱스, 3.0m 초과 룩어헤드만 보수적).
# wp_world 미저장이라 오프라인 재계산 불가 — 정밀화하려면 재덤프.
N_A, N_D, N_H = 30, 14, 12
HFOV = 90.0
OUT_ROOT = "/data/R2R_go2"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bins_of(dx, dz, hd_deg):
    """로봇 프레임 (우+, 전+) → (각도, 거리, heading) 빈. 시야 밖 = None."""
    ang = math.degrees(math.atan2(dx, dz))
    if abs(ang) >= HFOV / 2:
        return None
    dist = math.hypot(dx, dz)
    a = int((ang + HFOV / 2) / (HFOV / N_A))
    d = int(round(dist / 0.25)) - 1
    if not (0 <= d < N_D):
        d = min(max(d, 0), N_D - 1)
    h = int(round((hd_deg % 360) / (360 / N_H))) % N_H
    return min(a, N_A - 1), d, h


def main_prep_frames():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

    import torch
    import habitat
    from EA_Nav import EANav
    device = "cuda"
    policy = EANav(clip_device=device).to(device)
    policy.load_pretrained(finetuned=True, verbose=False)
    policy.eval()

    os.chdir("/workspace/ETPNav")
    from vlnce_baselines.config.default import get_config
    cfg = get_config("run_r2r/iter_train.yaml", opts=[
        "TASK_CONFIG.DATASET.DATA_PATH",
        "data/datasets/R2R_VLNCE_v1-3_BERTidx/{split}/{split}_bertidx"
        ".json.gz",
        "TASK_CONFIG.DATASET.SPLIT", args.split])
    tc = cfg.TASK_CONFIG.clone()
    tc.defrost()
    tc.DATASET.SPLIT = args.split
    tc.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
    tc.SIMULATOR.RGB_SENSOR.POSITION = [0.0, GO2_CAM_H, 0.0]
    tc.SIMULATOR.DEPTH_SENSOR.POSITION = [0.0, GO2_CAM_H, 0.0]
    tc.freeze()
    ds = habitat.make_dataset(tc.DATASET.TYPE, config=tc.DATASET)
    # 씬 단위 그룹 후 샤딩(씬 로드 오버헤드 최소화)
    by_scene = {}
    for e in ds.episodes:
        by_scene.setdefault(e.scene_id, []).append(e)
    scenes = sorted(by_scene)
    my_eps = []
    for i, sc in enumerate(scenes):
        if i % args.nshards == args.shard:
            my_eps += by_scene[sc]
    if args.limit:
        my_eps = my_eps[:args.limit]
    ds.episodes = my_eps
    env = habitat.Env(config=tc, dataset=ds)
    with gzip.open(f"data/datasets/R2R_VLNCE_v1-3_preprocessed/"
                   f"{args.split}/{args.split}_gt.json.gz") as fh:
        gt = json.load(fh)

    os.makedirs(f"{OUT_ROOT}/{args.split}", exist_ok=True)
    deps, embs, wpb, poses, epi = [], [], [], [], []
    jpgs = []
    import cv2
    meta = []
    t0 = time.time()
    for k in range(len(my_eps)):
        obs = env.reset()
        ep = env.current_episode
        locs = gt.get(str(ep.episode_id), {}).get("locations")
        if not locs or len(locs) < 2:
            continue
        pts = [np.array(q, dtype=np.float64) for q in locs]
        # 1.0m 간격 재표집
        line = [pts[0]]
        acc = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            seg = float(np.linalg.norm((b - a)[[0, 2]]))
            if seg < 1e-6:
                continue
            n = max(1, int(seg / STEP_M))
            for j in range(1, n + 1):
                line.append(a + (b - a) * (j / n))
        goal = pts[-1]
        for t, p in enumerate(line[:-1]):
            # 룩어헤드 지점(≤3m 앞) — MANSION waypoint GT 규약의 근사
            la = None
            rem = 0.0
            for q in line[t + 1:]:
                rem = float(np.linalg.norm((q - p)[[0, 2]]))
                la = q
                if rem >= 3.0:
                    break
            if la is None:
                break
            nxt = line[t + 1]
            yaw = math.atan2(-(nxt[0] - p[0]), -(nxt[2] - p[2]))
            rot = None
            import quaternion as nq
            rot = nq.quaternion(math.cos(yaw / 2), 0.0,
                                math.sin(yaw / 2), 0.0)
            o = env.sim.get_observations_at(
                np.asarray(p, np.float32), rot, False)
            # 월드→로봇 프레임 (전방 -z, 우 +x)
            rel = la - p
            fx, fz = -math.sin(yaw), -math.cos(yaw)
            rx, rz = math.cos(yaw), -math.sin(yaw)
            dz_ = rel[0] * fx + rel[2] * fz
            dx_ = rel[0] * rx + rel[2] * rz
            j = min(t + 2, len(line) - 1)
            hd_vec = line[j] - la
            hd = math.degrees(math.atan2(
                hd_vec[0] * rx + hd_vec[2] * rz,
                hd_vec[0] * fx + hd_vec[2] * fz))
            b_ = bins_of(dx_, dz_, hd)
            if b_ is None:
                continue
            dep = torch.from_numpy(
                o["depth"][..., 0].astype(np.float32) * 10.0
            )[None].to(device)
            rgbt = torch.from_numpy(o["rgb"].copy())[None].to(device)
            import torch.nn.functional as F
            rgbt = F.interpolate(rgbt.permute(0, 3, 1, 2).float(),
                                 size=(256, 256)
                                 ).permute(0, 2, 3, 1).byte()
            with torch.no_grad():
                # 오프라인 덤프 — 에피소드 히스토리 버퍼를 갱신하지 않는다
                # (폐루프 주행이 아니라 프레임 단위 featurize)
                emb = policy.observe(rgbt, dep,
                                     push_history=False)["node_embed"][0]
            okj, jb = cv2.imencode(".jpg", o["rgb"][..., ::-1],
                                   [cv2.IMWRITE_JPEG_QUALITY, 92])
            jpgs.append(jb.tobytes() if okj else b"")
            deps.append(o["depth"][..., 0].astype(np.float16) * 10.0)
            embs.append(emb.half().cpu().numpy())
            wpb.append(b_)
            poses.append((float(p[0]), float(p[2]), math.degrees(yaw)))
            epi.append(k)
        meta.append({"episode_id": ep.episode_id,
                     "instruction": obs["instruction"]["text"],
                     "n_frames": int(sum(1 for e_ in epi if e_ == k)),
                     "goal": [float(goal[0]), float(goal[2])]})
        if (k + 1) % 50 == 0:
            log(f"{k+1}/{len(my_eps)} eps, 프레임 {len(deps)} "
                f"({time.time()-t0:.0f}s)")
    out = f"{OUT_ROOT}/{args.split}/shard{args.shard}.npz"
    np.savez_compressed(
        out, depth=np.stack(deps) if deps else np.zeros((0, 256, 256),
                                                        np.float16),
        embed=np.stack(embs) if embs else np.zeros((0, 768), np.float16),
        wp_bin=np.array(wpb, np.int64).reshape(-1, 3),
        pose=np.array(poses, np.float32).reshape(-1, 3),
        ep_idx=np.array(epi, np.int32),
        jpg=np.frombuffer(b"".join(jpgs), dtype=np.uint8),
        jpg_off=np.cumsum([0] + [len(b) for b in jpgs]).astype(np.int64))
    with open(f"{OUT_ROOT}/{args.split}/shard{args.shard}_meta.jsonl",
              "w") as fh:
        for m in meta:
            fh.write(json.dumps(m) + "\n")
    log(f"저장 {out}: 에피소드 {len(meta)}, 프레임 {len(deps)}")



def main():
    for tool in ("report", "prep_frames"):
        if "--role" in sys.argv and \
                sys.argv[sys.argv.index("--role") + 1] == tool:
            i = sys.argv.index("--role")
            del sys.argv[i:i + 2]
            return globals()[f"main_{tool}"]()
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["base", "ours"], required=True)
    ap.add_argument("--no-embodiment", action="store_true",
                    help="Go2 조건(카메라 높이·에이전트 반경·navmesh 재계산)을 "
                         "빼고 원본 ETPNav 설정 그대로 — 공식 가중치의 재현 "
                         "기준선 행. 성능 하락이 방법 탓인지 embodiment 조건 "
                         "탓인지 가르는 대조군이다.")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--ckpt01", default="/data/EVLN_ckpt/traversability.pt")
    ap.add_argument("--etpnav-ckpt", default=ETPNAV_CKPT)
    ap.add_argument("--out", default="/data/EVLN_ckpt/benchmark")
    ap.add_argument("--urdf-dir", default=GO2_DIR,
                    help="로봇 URDF 폴더(robot.urdf+meta.json) — 기본 Go2, "
                         "학습 분포 내 생성 URDF 대조 실험용")
    ap.add_argument("--suffix", default="",
                    help="결과 파일명 접미(로봇 구분용, 예: _quad122)")
    ap.add_argument("--viz", type=int, default=0,
                    help="앞 N에피소드 탑뷰 궤적 카드(GT vs 실주행) — "
                         "MANSION 벤치 카드와 동일 규약, check/benchmark/")
    ap.add_argument("--viz-dir", default="/research/check/benchmark")
    args = ap.parse_args()
    assert 0 <= args.gpu <= 3, "GPU는 0~3만"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import random

    import torch
    robot = robot_spec(args.urdf_dir)
    log(f"variant={args.variant} 로봇 {robot['id']}: cam_h {robot['cam_h']}m"
        f", radius {robot['radius']}m, height {robot['height']}m")

    os.chdir(ETPNAV_DIR)
    sys.path.insert(0, ETPNAV_DIR)
    import habitat
    import habitat_extensions  # noqa: F401 — VLN-CE-v1·Sim-v1·measures 등록
    import vlnce_baselines  # noqa: F401 — SS-ETP 트레이너·정책 등록
    from vlnce_baselines.config.default import get_config

    workdir = f"/tmp/r2rce_eval_{args.variant}_s{args.shard}"
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    # 원본 재현 기준선은 ETPNav 기본 센서·에이전트 규격을 그대로 쓴다
    cam_pos = ([0.0, 1.5, 0.0] if args.no_embodiment
               else [0.0, robot["cam_h"], 0.0])
    agent_r = 0.1 if args.no_embodiment else robot["radius"]
    agent_h = 1.5 if args.no_embodiment else robot["height"]
    config = get_config("run_r2r/iter_train.yaml", opts=[
        "GPU_NUMBERS", 1, "NUM_ENVIRONMENTS", 1,
        "SIMULATOR_GPU_IDS", [0], "TORCH_GPU_IDS", [0], "TORCH_GPU_ID", 0,
        "TASK_CONFIG.DATASET.DATA_PATH", DATA_PATH,
        "TASK_CONFIG.TASK.NDTW.GT_PATH", GT_PATH,
        "TASK_CONFIG.TASK.SDTW.GT_PATH", GT_PATH,
        "TASK_CONFIG.SIMULATOR.RGB_SENSOR.POSITION", cam_pos,
        "TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION", cam_pos,
        "TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH", 10.0,
        "TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH", True,
        "TASK_CONFIG.SIMULATOR.AGENT_0.RADIUS", agent_r,
        "TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT", agent_h,
        "EVAL.SPLIT", args.split, "EVAL.CKPT_PATH_DIR", args.etpnav_ckpt,
        "EVAL.EPISODE_COUNT", -1, "EVAL.SAVE_RESULTS", False,
        "RESULTS_DIR", workdir, "TENSORBOARD_DIR", workdir,
        "CHECKPOINT_FOLDER", workdir, "EVAL_CKPT_PATH_DIR", workdir,
        "VIDEO_DIR", workdir,
    ])
    config.defrost()
    config.local_rank = 0
    config.freeze()
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    # ① 몽키패치 전파를 위해 in-process(스레드) VectorEnv 강제
    import vlnce_baselines.common.env_utils as env_utils
    env_utils.habitat = types.SimpleNamespace(
        VectorEnv=habitat.ThreadedVectorEnv,
        ThreadedVectorEnv=habitat.ThreadedVectorEnv)

    # ② 씬 로드마다 Go2 반경·높이로 navmesh 재계산 (+ --viz 기록 훅)
    import habitat_sim
    from vlnce_baselines.common.environments import VLNCEDaggerEnv
    orig_reset = VLNCEDaggerEnv.reset
    orig_step = VLNCEDaggerEnv.step
    # [{ep_id, grid, x0, z0, res, positions, ref, ...}]
    viz_eps = []
    viz_cur = {}

    def _nav_grid(sim, y_ref, res=0.05):
        """pathfinder 바운드 기반 자체 탑뷰 점유 그리드 — 구버전 maps API
        (COORDINATE_MIN 고정 상수·x/z 순서 함정) 의존을 피해 변환을 직접
        소유한다. 픽셀 = ((x-x0)/res, (z-z0)/res)."""
        lo, hi = sim.pathfinder.get_bounds()
        xs = np.arange(lo[0], hi[0], res)
        zs = np.arange(lo[2], hi[2], res)
        g = np.zeros((len(zs), len(xs)), np.uint8)
        for iz, zv in enumerate(zs):
            for ix, xv in enumerate(xs):
                if sim.pathfinder.is_navigable([xv, y_ref, zv]):
                    g[iz, ix] = 1
        return g, float(lo[0]), float(lo[2]), res

    def reset_with_navmesh(self):
        obs = orig_reset(self)
        scene = self._env.current_episode.scene_id
        if args.no_embodiment:
            pass
        elif getattr(self, "_evln_navmesh_scene", None) != scene:
            ns = habitat_sim.NavMeshSettings()
            ns.set_defaults()
            ns.agent_radius = robot["radius"]
            ns.agent_height = robot["height"]
            ok = self._env.sim.recompute_navmesh(self._env.sim.pathfinder,
                                                 ns)
            self._evln_navmesh_scene = scene
            log(f"navmesh 재계산 {os.path.basename(scene)}: "
                f"{'성공' if ok else '실패(기본 유지)'}")
        if args.viz and len(viz_eps) < args.viz:
            ep = self._env.current_episode
            st = self._env.sim.get_agent_state().position
            grid, x0, z0, res = _nav_grid(self._env.sim, float(st[1]))
            viz_cur.clear()
            viz_cur.update(dict(
                ep_id=str(ep.episode_id), grid=grid, x0=x0, z0=z0, res=res,
                positions=[(float(st[0]), float(st[2]))],
                ref=[(float(p[0]), float(p[2])) for p in ep.reference_path],
                goal=(float(ep.goals[0].position[0]),
                      float(ep.goals[0].position[2]))))
            viz_eps.append(viz_cur.copy())
            viz_cur["rec"] = viz_eps[-1]
        else:
            viz_cur.clear()
        return obs

    def step_with_record(self, *a, **kw):
        out = orig_step(self, *a, **kw)
        rec = viz_cur.get("rec")
        if rec is not None:
            p = self._env.sim.get_agent_state().position
            rec["positions"].append((float(p[0]), float(p[2])))
        return out

    VLNCEDaggerEnv.reset = reset_with_navmesh
    VLNCEDaggerEnv.step = step_with_record

    # ③ ours: g 게이트 준비 (base는 원본 그대로)
    gate_stats = {"n_cand": 0, "n_block": 0,
                  "g_sum": 0.0, "g_min": float("inf"),
                  "g_max": float("-inf")}
    g_head = z_go2 = None
    if args.variant == "ours":
        g_head, z_go2 = build_z_and_g(args.ckpt01, robot["urdf"], "cuda:0")
        log(f"z({robot['id']}) 준비 완료, g 게이트 τ={args.tau}")

    from vlnce_baselines.ss_trainer_ETP import RLTrainer

    class R2RCEBenchTrainer(RLTrainer):
        def collect_val_traj(self):
            # GPU_NUMBERS=1 → 전체
            traj = super().collect_val_traj()
            # 기존 rank 분배와 동일
            traj = traj[args.shard::args.nshards]
            if args.limit:
                traj = traj[:args.limit]
            log(f"샤드 {args.shard}/{args.nshards} 에피소드 {len(traj)}")
            return traj

        def _initialize_policy(self, *a, **kw):
            super()._initialize_policy(*a, **kw)
            if args.variant == "ours":
                attach_g_gate(self.policy.net, g_head, z_go2, args.tau,
                              gate_stats)
                log("g 게이트 주입 완료 (policy.net.forward 래핑)")

        def rollout(self, mode, *a, **kw):
            t = time.time()
            r = super().rollout(mode, *a, **kw)
            done = len(self.stat_eps) if hasattr(self, "stat_eps") else 0
            log(f"에피소드 {done}/{len(self.traj)} 완료 "
                f"({time.time() - t:.0f}s/rollout)")
            return r

    t0 = time.time()
    trainer = R2RCEBenchTrainer(config)
    trainer.eval()
    elapsed = time.time() - t0

    # ---- 결과 집계 (geodesic inf 가드: NE 집계 제외 + 별도 카운트) ----
    rows = []
    for ep_id, m in sorted(trainer.stat_eps.items(), key=lambda x: int(x[0])):
        ne = float(m["distance_to_goal"])
        fin = math.isfinite(ne)
        spl = float(m["spl"])
        rows.append({
            "episode_id": int(ep_id), "SR": float(m["success"]),
            "SPL": spl if math.isfinite(spl) else 0.0,
            "NE": ne if fin else None, "geo_inf": not fin,
            "TL": float(m["path_length"]), "nDTW": float(m["ndtw"]),
            "oracle_SR": float(m["oracle_success"]),
            "steps": float(m["steps_taken"]),
            "collisions": float(m["collisions"]),
            "ghost_cnt": float(m["ghost_cnt"])})
    # ---- --viz: 탑뷰 궤적 카드 (MANSION 벤치 카드와 동일 규약) ----
    if args.viz and viz_eps:
        import cv2
        os.makedirs(args.viz_dir, exist_ok=True)
        met = {str(k): v for k, v in trainer.stat_eps.items()}
        for rec in viz_eps:
            g = rec["grid"]
            img = np.full((*g.shape, 3), 40, np.uint8)
            img[g == 1] = (205, 205, 205)

            def px(pt):
                return (int((pt[0] - rec["x0"]) / rec["res"]),
                        int((pt[1] - rec["z0"]) / rec["res"]))

            for a_, b_ in zip(rec["ref"][:-1], rec["ref"][1:]):
                cv2.line(img, px(a_), px(b_), (0, 200, 0), 2)
            for a_, b_ in zip(rec["positions"][:-1], rec["positions"][1:]):
                cv2.line(img, px(a_), px(b_), (0, 0, 255), 2)
            if rec["positions"]:
                cv2.circle(img, px(rec["positions"][0]), 7, (255, 0, 0), -1)
            cv2.circle(img, px(rec["goal"]), 9, (0, 255, 255), 2)
            img = cv2.flip(img, 0)
            m = met.get(rec["ep_id"], {})
            bar = np.zeros((56, max(img.shape[1], 640), 3), np.uint8)
            if img.shape[1] < bar.shape[1]:
                img = cv2.copyMakeBorder(
                    img, 0, 0, 0, bar.shape[1] - img.shape[1],
                    cv2.BORDER_CONSTANT)
            cv2.putText(bar, f"ep{rec['ep_id']} {args.variant} "
                        f"SR={m.get('success', -1):.0f} "
                        f"NE={m.get('distance_to_goal', -1):.2f} "
                        f"nDTW={m.get('ndtw', -1):.2f}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)
            x = 8
            for col, name in (((0, 200, 0), "GT ref"), ((0, 0, 255),
                              "agent"), ((255, 0, 0), "start"),
                              ((0, 255, 255), "goal")):
                cv2.line(bar, (x, 44), (x + 22, 44), col, 3)
                cv2.putText(bar, name, (x + 27, 49),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)
                x += 27 + 9 * len(name) + 18
            cv2.imwrite(os.path.join(
                args.viz_dir,
                f"r2rce_{args.variant}{args.suffix}_ep{rec['ep_id']}.png"),
                cv2.vconcat([bar, img]))
        log(f"viz 카드 {len(viz_eps)}장 → {args.viz_dir}")

    fin_rows = [r for r in rows if not r["geo_inf"]]
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("SR", "SPL", "TL", "nDTW", "oracle_SR")}
    agg["NE"] = (float(np.mean([r["NE"] for r in fin_rows]))
                 if fin_rows else None)
    agg["n"] = len(rows)
    agg["n_geo_inf"] = len(rows) - len(fin_rows)
    agg["sec_per_ep"] = round(elapsed / max(1, len(rows)), 1)

    settings = {"variant": args.variant, "split": args.split,
                "nshards": args.nshards, "limit": args.limit,
                "tau": args.tau if args.variant == "ours" else None,
                "ckpt01": args.ckpt01 if args.variant == "ours" else None,
                "etpnav_ckpt": args.etpnav_ckpt, "robot": robot,
                "data_path": DATA_PATH, "pipeline": "ETPNav-original-eval",
                "nav_token": "unused", "max_traj_len": config.IL.max_traj_len}
    cfg_hash = hashlib.md5(
        json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    if gate_stats["n_cand"]:
        gate_stats["g_mean"] = round(
            gate_stats["g_sum"] / gate_stats["n_cand"], 3)
    smp = gate_stats.pop("g_samples", None)
    if smp:
        q = np.percentile(smp, [1, 5, 10, 25, 50, 75, 90])
        gate_stats["g_pct"] = {k: round(float(v), 3) for k, v in
                               zip(("p1", "p5", "p10", "p25", "p50", "p75",
                                    "p90"), q)}
        gate_stats["n_sampled"] = len(smp)
    for k in ("g_min", "g_max"):
        if not math.isfinite(gate_stats[k]):
            gate_stats[k] = None
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out,
                       f"r2rce_{args.variant}{args.suffix}_{args.shard}.json")
    with open(out, "w") as fh:
        json.dump({"meta": {**settings, "shard": args.shard,
                            "config_hash": cfg_hash,
                            "gate_stats": gate_stats,
                            "date": time.strftime("%Y-%m-%d %H:%M")},
                   "aggregate": agg, "episodes": rows}, fh, indent=1)
    shutil.rmtree(workdir, ignore_errors=True)
    log(f"저장 {out}")
    log(f"집계 {json.dumps(agg, ensure_ascii=False)} | 게이트 {gate_stats}")


if __name__ == "__main__":
    main()

"""EVLN 데이터셋 PyTorch 로더 — research/data/ 심볼릭 링크 기준 경로.

경로 규약: 이 파일과 같은 폴더의 EVLN_dataset/·URDF/ 심볼릭 링크가
호스트(~/jairlab/data)와 컨테이너(/data) 어디서든 동일하게 풀리므로
절대 경로를 쓰지 말 것.

구성 (학습 loss 2개 + 평가 세트에 대응):
  TraversabilityDataset   loss (a) 통과성 튜플 — (게이트 스냅샷 RGB-D, 로봇)
                       → soft 라벨 [0,1]. 스냅샷은 로봇 cam_h와 최근접
                       높이 버킷의 9뷰(거리 3×각도 3).
  EpisodeFrameDataset  loss (b)·(c) 프레임 단위 — (RGB-D, 히스토리 스택,
                       포즈, 지시) → heat(32² 도달가능 래스터)와 전역
                       선택 라벨(gsel/gsel_uv/gsel_head). 구 wp_bin·wp_px
                       빈 라벨은 폐기됐다.
  RobotGraphCache      로봇 id → URDF 그래프(models/modules/URDFEncoder) 캐시.
                       collate에서 z 인코더 입력 배치로 변환.
  load_evalsets/…      평가 JSON(zswap·instr_conflict·notfound)·topomap
                       패스스루.

토크나이즈는 트레이너 책임(policy.py tokenizer) — 로더는 원문 텍스트 반환.
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_HERE, "..", "..", "data")   # 데이터 앵커(심볼릭 링크)
# 데이터셋 루트는 환경변수로 갈아끼운다 — R2R-CE(Go2) 산출물을 MANSION과
# 같은 규약으로 두되 다른 루트에 두어, 태그 혼입 없이 따로 학습한다.
DS = os.environ.get("EVLN_DS", os.path.join(ROOT, "EVLN_dataset"))
URDF_ROOT = os.path.join(ROOT, "URDF")

TRAJ_STATES = ("driving", "driving_with_wait", "impossible_embodiment")

# 전역 선택 라벨 종류 (데이터 생성부 common.GSEL과 같은 값 유지)
GSEL_FORWARD, GSEL_NODE, GSEL_STOP = 0, 1, 2
# gsel_head 무효 센티널 (마지막 프레임 등 다음 포즈가 없는 홉)
HEAD_NONE = -999.0
# 이미지 히스토리 기본 길이 — models/modules/WaypointNet.N_HIST_MAX와 일치
HIST_K = 4


def tags():
    """환경 태그 목록 (episodes.json 존재 기준, 접두사 규약)."""
    fs = glob.glob(os.path.join(DS, "episodes", "*_episodes.json"))
    return sorted(os.path.basename(f)[: -len("_episodes.json")] for f in fs
                  if "shard" not in os.path.basename(f))


def robots_meta(tag):
    """{tag}_robots.json — id→(urdf 상대경로, 치수, cam_h, stairs_ok)."""
    j = json.load(open(os.path.join(DS, f"{tag}_robots.json")))
    split = os.path.basename(j["pool_root"].rstrip("/"))  # 절대경로 불신
    return {r["id"]: dict(r, urdf_abs=os.path.join(URDF_ROOT, split,
                                                   r["urdf"]))
            for r in j["robots"]}


class RobotGraphCache:
    """로봇 id → urdf_to_graph 결과 캐시 (풀 480개, 1회 파싱 후 재사용)."""

    def __init__(self, tag):
        self.meta = robots_meta(tag)
        self._cache = {}

    def __call__(self, rid):
        if rid not in self._cache:
            import sys
            sys.path.insert(0, os.path.join(_HERE, "..", "..", "models"))
            from modules.URDFEncoder import urdf_to_graph
            self._cache[rid] = urdf_to_graph(self.meta[rid]["urdf_abs"])
        return self._cache[rid]

    def collate(self, rids, augment_rng=None):
        import sys
        sys.path.insert(0, os.path.join(_HERE, "..", "..", "models"))
        from modules.URDFEncoder import augment_graph, collate_graphs
        gs = [self(r) for r in rids]
        # 학습 전용 — 구조 잡음 증강(라벨 보존)
        if augment_rng is not None:
            gs = [augment_graph(g, augment_rng) for g in gs]
        return collate_graphs(gs)


# ---------------------------------------------------------------- loss (a)

class TraversabilityDataset(Dataset):
    """통과성 튜플: (스냅샷 RGB-D, 로봇 id) → soft 라벨.

    표본 단위 = (게이트, 로봇, 뷰). 뷰 9개(거리 3×각도 3)는 로봇 cam_h
    최근접 버킷에서 취함. 샤드 npz는 통짜 배열이라 LRU 캐시(워커당
    max_shards개)로 읽음 — 무작위 셔플이 캐시를 튕기므로 학습 시
    ShardBatchSampler 사용 권장.
    """

    def __init__(self, env_tags=None, views_per_item=9, max_shards=2):
        # (env_i, gate_i, robot_i, bucket_i)
        self.samples = []
        self.envs = []
        self.views = views_per_item
        self._max_shards = max_shards
        for tag in (env_tags or tags()):
            gj = json.load(open(os.path.join(DS, "gates",
                                             f"{tag}_gates.json")))
            labels = np.asarray(gj["labels"], dtype=np.float32)
            buckets = np.asarray(gj["cam_buckets"], dtype=np.float64)
            # 키(gate,dist,ang,bucket) → (샤드 파일, 행) 인덱스
            key2loc = {}
            for f in sorted(glob.glob(os.path.join(
                    DS, "gates", f"{tag}_snaps_shard*.npz"))):
                with np.load(f) as z:
                    for row, k in enumerate(z["key"]):
                        key2loc[tuple(int(v) for v in k)] = (f, row)
            ei = len(self.envs)
            self.envs.append(dict(tag=tag, gates=gj["gates"],
                                  robots=gj["robots"], labels=labels,
                                  n_dist=len(gj["snap_dists"]),
                                  n_ang=len(gj["snap_angs"]),
                                  key2loc=key2loc))
            for gi in range(len(gj["gates"])):
                for ri, r in enumerate(gj["robots"]):
                    bi = int(np.abs(buckets - r["cam_h"]).argmin())
                    self.samples.append((ei, gi, ri, bi))

    def __len__(self):
        return len(self.samples) * self.views

    # 수동 LRU — 워커당 max_shards개 상주
    def _shard(self, path):
        if not hasattr(self, "_lru"):
            self._lru = {}
        if path not in self._lru:
            if len(self._lru) >= self._max_shards:
                self._lru.pop(next(iter(self._lru)))
            with np.load(path) as z:
                self._lru[path] = {"rgb": z["rgb"], "depth": z["depth"]}
        return self._lru[path]

    def __getitem__(self, idx):
        si, vi = divmod(idx, self.views)
        ei, gi, ri, bi = self.samples[si]
        env = self.envs[ei]
        di, ai = divmod(vi, env["n_ang"])
        f, row = env["key2loc"][(gi, di, ai, bi)]
        sh = self._shard(f)
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(sh["rgb"][row])),
            "depth": torch.from_numpy(
                sh["depth"][row].astype(np.float32)),
            "label": float(env["labels"][gi, ri]),
            "robot_id": env["robots"][ri]["id"],
            "tag": env["tag"], "gate_idx": gi,
        }

    def shard_of(self, idx):
        si, vi = divmod(idx, self.views)
        ei, gi, ri, bi = self.samples[si]
        env = self.envs[ei]
        di, ai = divmod(vi, env["n_ang"])
        return env["key2loc"][(gi, di, ai, bi)][0]


class ShardBatchSampler(Sampler):
    """샤드 셔플 → 샤드 내 셔플 배치 — LRU 캐시 적중 유지용."""

    def __init__(self, ds, batch_size, seed=0):
        self.ds, self.bs, self.seed, self.epoch = ds, batch_size, seed, 0
        by_shard = {}
        for i in range(len(ds)):
            by_shard.setdefault(ds.shard_of(i), []).append(i)
        self.groups = list(by_shard.values())

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        for gi in torch.randperm(len(self.groups), generator=g):
            idxs = self.groups[gi]
            perm = torch.randperm(len(idxs), generator=g)
            for s in range(0, len(idxs), self.bs):
                yield [idxs[p] for p in perm[s:s + self.bs]]

    def __len__(self):
        return sum((len(x) + self.bs - 1) // self.bs for x in self.groups)


# ---------------------------------------------------------------- loss (b)

class EpisodeFrameDataset(Dataset):
    """에피소드 프레임: (RGB-D, 히스토리, 포즈, 지시, 로봇) → heat·전역 라벨.

    지시 결합 — goal형: episode 일치(로봇 무관), r2r형: episode+robot
    일치, combined형: cep 궤적(episodes_combined) 자체에 결합. 지시가 없는
    프레임은 빈 문자열이고, 트레이너가 무지시 토큰으로 대체한다(표본 폐기
    금지).

    프레임 필터는 없다 — 계단 등반처럼 heat가 전 픽셀 음성인 프레임도
    "여기서는 고스트를 내지 말라"는 유효 라벨이다. 전역 학습에서 제외할
    구간(엘베 스킬 등)은 special로 트레이너가 거른다.

    hist_k > 1이면 depth 히스토리를 함께 낸다: 인덱스 0 = 현재, 이후 과거
    순이며 에피소드 초반은 가장 오래된 프레임을 복제해 항상 K장을 채운다
    (EANav.observe의 패딩 규약과 동일).
    """

    def __init__(self, env_tags=None, include_combined=True,
                 states=TRAJ_STATES, hist_k=HIST_K):
        # (npz 경로, 프레임 t, 지시 후보 튜플, rid, tag)
        self.frames = []
        self.hist_k = hist_k
        self.skipped_legacy = 0
        # 전역 선택 라벨이 없는 궤적 (제안기 학습에는 사용 가능)
        self.no_global = set()
        self._npz_cache = (None, None)
        for tag in (env_tags or tags()):
            ej = json.load(open(os.path.join(
                DS, "episodes", f"{tag}_episodes.json")))
            ins = json.load(open(os.path.join(
                DS, f"{tag}_instructions.json")))
            by_ep, by_ep_rob = {}, {}
            for r in ins["records"]:
                if not r.get("pass"):
                    continue
                if r["form"] == "goal":
                    by_ep.setdefault(r["episode"], []).append(
                        r["instruction"])
                elif r["form"] == "r2r":
                    by_ep_rob.setdefault(
                        (r["episode"], r["robot"]), []).append(
                        r["instruction"])
            for e in ej["episodes"]:
                for rid, rr in e["results"].items():
                    if rr["state"] not in states:
                        continue
                    f = os.path.join(DS, "episodes",
                                     f"{tag}_{e['id']}_{rid}.npz")
                    if not os.path.exists(f):
                        continue
                    instrs = tuple(by_ep.get(e["id"], [])
                                   + by_ep_rob.get((e["id"], rid), []))
                    self._add_frames(f, instrs, rid, tag, rr["state"])
            if include_combined:
                recs = [r for r in ins["records"]
                        if r["form"] == "combined" and r.get("pass")]
                cj_path = os.path.join(
                    DS, "episodes_combined", f"{tag}_combined_eps.json")
                if os.path.exists(cj_path):
                    cj = json.load(open(cj_path))
                    for ce in cj["episodes"]:
                        f = os.path.join(DS, "episodes_combined",
                                         f"{tag}_{ce['id']}.npz")
                        if not os.path.exists(f):
                            continue
                        it = (recs[ce["instruction_idx"]]["instruction"],
                              ) if ce["instruction_idx"] < len(recs) else ()
                        self._add_frames(f, it, ce["robot"], tag,
                                         "combined")

    def _add_frames(self, f, instrs, rid, tag, state):
        with np.load(f) as z:
            # heat(제안기 GT)는 렌더 단계, gsel*(전역 GT)는 재유도 단계에서
            # 붙는다 — 둘의 유무가 다를 수 있으므로 따로 본다. heat조차 없는
            # 구 스키마만 건너뛰고, 전역 라벨이 없는 궤적은 제안기 학습에는
            # 그대로 쓴다(전역 학습 쪽에서 has_global로 거른다).
            if "heat" not in z.files:
                self.skipped_legacy += 1
                return
            if "gsel" not in z.files:
                self.no_global.add(f)
            n = len(z["pose"])
        for t in range(n):
            self.frames.append((f, int(t), instrs, rid, tag, state))

    def __len__(self):
        return len(self.frames)

    def _npz(self, f):
        # 순차 접근 시 궤적 1개 캐시
        if self._npz_cache[0] != f:
            with np.load(f) as z:
                self._npz_cache = (f, {k: z[k] for k in z.files})
        return self._npz_cache[1]

    def __getitem__(self, i):
        f, t, instrs, rid, tag, state = self.frames[i]
        z = self._npz(f)
        instr = instrs[t % len(instrs)] if instrs else ""
        # 히스토리: 현재 → 과거 순, 모자라면 가장 오래된 프레임 복제
        idx = [max(0, t - k) for k in range(self.hist_k)]
        dep = z["depth"][idx].astype(np.float32)
        nxt = z["pose"][min(t + 1, len(z["pose"]) - 1)]
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(z["rgb"][t])),
            "depth": torch.from_numpy(dep[0]),
            "depth_hist": torch.from_numpy(dep),
            "pose": torch.from_numpy(z["pose"][t].astype(np.float32)),
            "next_pose": torch.from_numpy(nxt.astype(np.float32)),
            "heat": torch.from_numpy(z["heat"][t].astype(np.int64)),
            # 전역 라벨 없는 궤적은 센티널(-1)로 채운다 — 트레이너가 거른다
            "gsel": int(z["gsel"][t]) if "gsel" in z else -1,
            "gsel_uv": torch.from_numpy(
                z["gsel_uv"][t].astype(np.int64) if "gsel_uv" in z
                else np.array([-1, -1], dtype=np.int64)),
            "gsel_head": (float(z["gsel_head"][t]) if "gsel_head" in z
                          else HEAD_NONE),
            "special": int(z["special"][t]),
            "instruction": instr, "robot_id": rid,
            "tag": tag, "state": state, "traj": os.path.basename(f),
            "frame": t,
        }


def collate_frames(batch, graph_cache=None, augment_rng=None):
    """프레임 배치 collate — 텍스트는 리스트로, z 그래프는 선택 배치."""
    out = {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "depth": torch.stack([b["depth"] for b in batch]),
        "depth_hist": torch.stack([b["depth_hist"] for b in batch]),
        "pose": torch.stack([b["pose"] for b in batch]),
        "next_pose": torch.stack([b["next_pose"] for b in batch]),
        "heat": torch.stack([b["heat"] for b in batch]),
        "gsel": torch.tensor([b["gsel"] for b in batch]),
        "gsel_uv": torch.stack([b["gsel_uv"] for b in batch]),
        "gsel_head": torch.tensor([b["gsel_head"] for b in batch]),
        "special": torch.tensor([b["special"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
        "robot_id": [b["robot_id"] for b in batch],
        "tag": [b["tag"] for b in batch],
        "state": [b["state"] for b in batch],
        "traj": [b["traj"] for b in batch],
        "frame": [b["frame"] for b in batch],
    }
    if graph_cache is not None:
        out["robot_graph"] = graph_cache.collate(out["robot_id"],
                                                 augment_rng=augment_rng)
    return out


def collate_traversability(batch, graph_cache=None):
    out = {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "depth": torch.stack([b["depth"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
        "robot_id": [b["robot_id"] for b in batch],
        "tag": [b["tag"] for b in batch],
    }
    if graph_cache is not None:
        out["robot_graph"] = graph_cache.collate(out["robot_id"])
    return out


# ------------------------------------------------------- R2R(Go2) 프레임

class R2RFrameDataset(Dataset):
    """R2R-CE→Go2 변환 덤프(prep_r2r_frames) 프레임 로더 — 혼합 재학습용.

    구 빈(wp_bin) 규약 덤프라 히트맵 제안기 학습에는 쓰지 않는다. 쓰려면
    도달가능 래스터를 R2R 쪽에서 다시 유도해야 한다.

    항목 = (depth, wp_bin[, embed]) — z는 Go2 고정이라 로더는 로봇 id만
    상수로 제공. 샤드 npz가 커서(≈2GB) LRU 1개 상주 + 샤드 순회 권장.
    """

    ROBOT_ID = "unitree_go2"

    def __init__(self, split="train", root="/data/R2R_go2",
                 with_embed=False):
        self.files = sorted(glob.glob(
            os.path.join(root, split, "shard*.npz")))
        self.with_embed = with_embed
        # (파일 idx, 행)
        self.index = []
        self.counts = []
        for fi, f in enumerate(self.files):
            with np.load(f) as z:
                n = len(z["wp_bin"])
            self.counts.append(n)
            self.index += [(fi, r) for r in range(n)]
        self._lru = (None, None)

    def __len__(self):
        return len(self.index)

    def _shard(self, fi):
        if self._lru[0] != fi:
            with np.load(self.files[fi]) as z:
                keep = {"depth": z["depth"], "wp_bin": z["wp_bin"]}
                if self.with_embed:
                    keep["embed"] = z["embed"]
            self._lru = (fi, keep)
        return self._lru[1]

    def __getitem__(self, i):
        fi, r = self.index[i]
        sh = self._shard(fi)
        out = {"depth": torch.from_numpy(
                   sh["depth"][r].astype(np.float32)),
               "wp_bin": torch.from_numpy(sh["wp_bin"][r]),
               "robot_id": self.ROBOT_ID}
        if self.with_embed:
            out["embed"] = torch.from_numpy(
                sh["embed"][r].astype(np.float32))
        return out

    def shard_of(self, i):
        return self.index[i][0]


# ---------------------------------------------------------------- 평가 세트

def load_evalsets(tag):
    return json.load(open(os.path.join(DS, f"{tag}_evalsets.json")))


def load_topomap(tag, robot_id):
    j = json.load(open(os.path.join(DS, "topomap",
                                    f"{tag}_{robot_id}.json")))
    npz = os.path.join(DS, "topomap", f"{tag}_{robot_id}_snaps.npz")
    return j, npz


def urdf_splits():
    """URDF 풀 스플릿 매니페스트 (train/val_unseen_dims/test_unseen_form)."""
    return json.load(open(os.path.join(URDF_ROOT, "splits.json")))

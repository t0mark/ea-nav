"""AI2-THOR MANSION 폐루프 어댑터 — 기억·판정·홉 드라이버·THOR 렌더러.

벤치(scripts/benchmark/mansion.py)와 평가(eval.py)가 공유하는 부품.
"expert가 아니라 모델이 고르는" 경로만 여기서 만든다.

구성:
  UnionMemory     환경의 로봇별 topomap을 합집합으로 병합한 표준 기억
                  그래프 — z스왑은 "같은 기억, z만 교체"가 전제인데 개별
                  로봇 기억은 자기 규칙의 전환 수단만 담고 있어(legged=
                  stairs만, wheeled=elevator만) 병합해야 두 선택지가
                  기억 안에 공존한다.
  MemoryFeatures  기억 스냅샷의 node_embed(768, g·전역 후보 입력)와 CLIP
                  이미지 임베딩(512, goal grounding) 사전 계산·캐시.
  GateBank        전 게이트의 통과성 점수(재렌더 없이 g로 채점).
  judge_state     불가능 판정 전용 진입점. 아래 "판정과 실행의 분리" 참조.
  HopDriver       홉 단위 전역 선택 주행 — 벤치·평가 공용 유일 주행 경로.
  ThorDriver      씬 로드·URDF 실높이 렌더 (렌더러 수명 정책 담당).

판정과 실행의 분리:
  plan_route(다익스트라)는 **판정 전용**이다. 실행 경로에서 호출하지
  않는다 — 행선 선택은 학습된 전역 선택기(HopDriver)가 한다. 규칙
  플래너를 실행에 쓰면 "goal 선택→탐색" 구조가 온라인 탐사·층 이동
  경유에서 케이스별 규칙 패치를 증식시킨다.
  판정 호출은 judge_state 하나로 모은다 — 게이트 차단 인자를 호출부마다
  달리 넘겨 같은 정책을 서로 다른 조건으로 재던 문제를 없앤다.

  단, 홉의 행선이 확정된 뒤 그 노드까지 이동하는 백트래킹 경로는
  기억 그래프의 최단경로를 쓴다. 이는 **실행 보조**이며 계획·판정이
  아니다(이미 주행한 엣지만 따르고, 행선 자체는 선택기가 정한다).
"""
import glob
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "..", "models"),
           os.path.join(_HERE, "..", "..", "data")):
    sys.path.insert(0, _p)

_MC = None


def _mc():
    """datasets/MANSION/common — train/common.py와 이름 충돌이라 경로 로딩."""
    global _MC
    if _MC is None:
        import importlib.util
        p = os.path.join(_HERE, "..", "datasets", "MANSION",
                         "common.py")
        spec = importlib.util.spec_from_file_location("mansion_common", p)
        _MC = importlib.util.module_from_spec(spec)
        sys.modules["mansion_common"] = _MC
        spec.loader.exec_module(_MC)
    return _MC

DS = os.path.join(_HERE, "..", "..", "data", "EVLN_dataset")
MW = "/data/MansionWorld/mansionworld"
CACHE = "/data/EVLN_ckpt/evalcache"

# 주행 규약 (벤치·평가 공용)
# STEP_M: 저수준 스텝당 최대 전진, NODE_PASS_M: 경유 노드 통과 반경,
# GHOST_PASS_M: 고스트 도착 반경(고스트 병합 반경과 같은 값 — 노드 반경을
# 쓰면 후보가 이미 그 안에 있어 0스텝 홉이 된다),
# MIN_HOP_M / MIN_HOP_DEG: 홉이 진행으로 인정되는 최소 순변위 또는 방향 변화
# (방향 전환 홉은 변위 없이 heading만 바뀌는 것이 정상),
# ARRIVE_M: 도착 판정 반경 = vln_metrics.SUCCESS_M(벤치 SR과 동일값 의무),
# HOP_STEPS: 홉당 저수준 스텝 상한, HOP_FAIL_LIMIT: 연속 홉 실패 한계,
# FLOOR_Y: GraphMap이 층 분리에 쓰는 y 오프셋(층당 높이)
STEP_M = 0.5
NODE_PASS_M = 1.2
GHOST_PASS_M = 0.5
MIN_HOP_M = 0.25
MIN_HOP_DEG = 20.0
ARRIVE_M = 3.0
HOP_STEPS = 40
HOP_FAIL_LIMIT = 3
MAX_HOPS = 80
FLOOR_Y = 3.0


# 태그 → (빌딩 경로, MANSION_VARIANT)
def building_of(tag):
    if tag.endswith("_custom_elevonly"):
        base = tag[: -len("_custom_elevonly")]
        return os.path.join(MW, base + "#0"), "elevator_only"
    return os.path.join(MW, tag + "#0"), "none"


def load_floors(tag):
    """common.load_building (variant 반영). 기하·좌표 변환용."""
    mc = _mc()
    bld, var = building_of(tag)
    old = os.environ.get("MANSION_VARIANT")
    if var != "none":
        os.environ["MANSION_VARIANT"] = var
    elif "MANSION_VARIANT" in os.environ:
        del os.environ["MANSION_VARIANT"]
    try:
        return mc.load_building(bld)
    finally:
        if old is not None:
            os.environ["MANSION_VARIANT"] = old
        elif "MANSION_VARIANT" in os.environ:
            del os.environ["MANSION_VARIANT"]


class UnionMemory:
    """로봇별 topomap 합집합 — 1.0m·동일층 노드 병합(생성기와 같은 반경)."""

    def __init__(self, tag, merge_m=1.0):
        self.tag = tag
        self.graphs = {}
        for f in sorted(glob.glob(os.path.join(DS, "topomap",
                                               f"{tag}_*.json"))):
            rest = os.path.basename(f)[len(tag) + 1:-5]
            if "custom" in rest:      # 접두사 겹침(호텔 vs elevonly) 배제
                continue
            self.graphs[rest] = json.load(open(f))
        # 노드 병합
        # {floor,x,z,kind,refs:[(robot,nid)]}
        self.nodes = []
        # (robot,nid) → union id
        self.map = {}
        for rob, g in self.graphs.items():
            for n in g["nodes"]:
                uid = None
                for i, u in enumerate(self.nodes):
                    if (u["floor"] == n["floor"]
                            and (u["x"] - n["x"]) ** 2
                            + (u["z"] - n["z"]) ** 2 <= merge_m ** 2):
                        uid = i
                        break
                if uid is None:
                    uid = len(self.nodes)
                    self.nodes.append({"floor": n["floor"], "x": n["x"],
                                       "z": n["z"], "kind": n["kind"],
                                       "refs": []})
                if n["kind"] == "gate":   # 게이트 성격 우선(통과성 게이팅)
                    self.nodes[uid]["kind"] = "gate"
                self.nodes[uid]["refs"].append((rob, int(n["id"])))
                self.map[(rob, int(n["id"]))] = uid
        # 엣지 합집합 (u,v,mode) 단위 최단 길이
        ed = {}
        for rob, g in self.graphs.items():
            for e in g["edges"]:
                u = self.map[(rob, int(e["a"]))]
                v = self.map[(rob, int(e["b"]))]
                if u == v:
                    continue
                k = (min(u, v), max(u, v), e["mode"])
                ed[k] = min(ed.get(k, 1e9), float(e["length_m"]))
        self.edges = [(u, v, m, L) for (u, v, m), L in ed.items()]

    def nearest(self, floor, x, z, max_m=1e9):
        best, bd = None, max_m ** 2
        for i, n in enumerate(self.nodes):
            if n["floor"] != floor:
                continue
            d = (n["x"] - x) ** 2 + (n["z"] - z) ** 2
            if d < bd:
                best, bd = i, d
        return best

    def nodes_near(self, floor, x, z, radius_m):
        return [i for i, n in enumerate(self.nodes)
                if n["floor"] == floor
                and (n["x"] - x) ** 2 + (n["z"] - z) ** 2
                <= radius_m ** 2]

    def transition_nodes(self):
        """전환 노드 → 수단 — 계단·엘베 엣지의 양 끝점.

        스킬 구간(등반·탑승) 진입 판정에 쓴다. 전역 선택기가 다른 층
        노드를 고르면 이 노드를 경유해야 층이 바뀐다.
        """
        out = {}
        for u, v, m, _L in self.edges:
            if m in ("stairs", "elevator"):
                out[u] = m
                out[v] = m
        return out


class MemoryFeatures:
    """스냅샷 → node_embed(g·전역 후보 입력)·CLIP 임베딩. npz 단위 캐시."""

    def __init__(self, policy, mem, device="cuda", bs=64):
        import torch
        os.makedirs(CACHE, exist_ok=True)
        # (robot,nid) → (K,768) 스냅샷별
        self.embed = {}
        # (robot,nid) → (K,512)
        self.clip = {}
        policy.eval()
        for rob in mem.graphs:
            npz = os.path.join(DS, "topomap",
                               f"{mem.tag}_{rob}_snaps.npz")
            cf = os.path.join(CACHE, f"{mem.tag}_{rob}_memfeat.npz")
            if not os.path.exists(cf):
                with np.load(npz) as z:
                    rgb, dep, key = z["rgb"], z["depth"], z["key"]
                es, cs = [], []
                with torch.no_grad():
                    for s in range(0, len(rgb), bs):
                        r = torch.from_numpy(rgb[s:s + bs]).to(device)
                        d = torch.from_numpy(
                            dep[s:s + bs].astype(np.float32)).to(device)
                        # 오프라인 배치 featurize — 에피소드 히스토리
                        # 버퍼를 오염시키지 않는다
                        o = policy.observe(r, d, push_history=False)
                        es.append(o["node_embed"].half().cpu().numpy())
                        cs.append(policy.clip(r).half().cpu().numpy())
                np.savez_compressed(cf, embed=np.concatenate(es),
                                    clip=np.concatenate(cs), key=key)
            with np.load(cf) as z:
                key, emb, cl = z["key"], z["embed"], z["clip"]
            for nid in np.unique(key[:, 0]):
                m = key[:, 0] == nid
                self.embed[(rob, int(nid))] = emb[m]
                self.clip[(rob, int(nid))] = cl[m]

    def node_embeds(self, mem, uid):
        return np.concatenate([self.embed[r] for r in
                               mem.nodes[uid]["refs"] if r in self.embed]
                              or [np.zeros((0, 768), np.float16)])

    def node_clips(self, mem, uid):
        return np.concatenate([self.clip[r] for r in
                               mem.nodes[uid]["refs"] if r in self.clip]
                              or [np.zeros((0, 512), np.float16)])

    def node_embed_mean(self, mem, uid):
        """전역 후보 토큰용 노드 대표 feature (스냅샷 평균)."""
        e = self.node_embeds(mem, uid)
        if not len(e):
            return np.zeros(768, np.float32)
        return e.astype(np.float32).mean(axis=0)


class GateBank:
    """본생성 gates 스냅샷을 표준 기억에 편입 — 게이트 통과성 판정용.

    커버리지 병목 해소: topomap은 탐사자가 노드를 둔
    게이트만 알지만, gates.json + 학습 feature 캐시(01이 만든
    featcache/*.feat.npy)는 전 게이트(환경당 25~105)를 로봇 버킷 규약
    그대로 담고 있다. 재렌더·재학습 없이 g로 채점만 한다.
    """

    def __init__(self, tag, feat_cache_dir="/data/EVLN_ckpt/featcache"):
        gj = json.load(open(os.path.join(DS, "gates",
                                         f"{tag}_gates.json")))
        self.gates = gj["gates"]
        self.buckets = np.asarray(gj["cam_buckets"], dtype=np.float64)
        n_d, n_a = len(gj["snap_dists"]), len(gj["snap_angs"])
        # key(gate,di,ai,bucket) → featcache 행
        self.feat = {}
        for f in sorted(glob.glob(os.path.join(
                DS, "gates", f"{tag}_snaps_shard*.npz"))):
            cf = os.path.join(feat_cache_dir,
                              os.path.basename(f) + ".feat.npy")
            emb = np.load(cf)
            with np.load(f) as z_:
                key = z_["key"]
            # 게이트를 재추출하면 스냅샷 행이 늘어나는데 featcache는 그대로라
            # 행 대응이 깨진다 — 조용히 엉뚱한 feature를 쓰지 않도록 막는다
            if len(key) != len(emb):
                raise RuntimeError(
                    f"featcache 불일치: {os.path.basename(f)} key {len(key)}"
                    f"행 vs feat {len(emb)}행 — 게이트 재추출 후 통과성"
                    f" feature 캐시를 다시 만들어야 한다"
                    f"(train_traversability 캐싱 단계)")
            for row, k in enumerate(key):
                self.feat[tuple(int(v) for v in k)] = emb[row]
        self.n_d, self.n_a = n_d, n_a

    def scores(self, policy, z, cam_h, device="cuda"):
        """로봇(z, cam_h) → 게이트별 g 최고 뷰 로짓 (학습과 동일 버킷 규약)."""
        import torch
        bi = int(np.abs(self.buckets - cam_h).argmin())
        out = {}
        with torch.no_grad():
            for gi in range(len(self.gates)):
                fs = [self.feat[(gi, di, ai, bi)]
                      for di in range(self.n_d) for ai in range(self.n_a)
                      if (gi, di, ai, bi) in self.feat]
                if not fs:
                    continue
                f = torch.from_numpy(np.stack(fs).astype(np.float32)
                                     ).to(device)
                out[gi] = float(policy.g(f, z.expand(len(f), -1)).max())
        return out


def _seg_dist2(px, pz, ax, az, bx, bz):
    """점-선분 거리 제곱."""
    vx, vz = bx - ax, bz - az
    L2 = vx * vx + vz * vz
    if L2 < 1e-9:
        return (px - ax) ** 2 + (pz - az) ** 2
    t = max(0.0, min(1.0, ((px - ax) * vx + (pz - az) * vz) / L2))
    cx, cz = ax + t * vx, az + t * vz
    return (px - cx) ** 2 + (pz - cz) ** 2


def edge_gate_map(mem, bank, pass_m=0.8):
    """union 엣지 → 그 엣지가 관통하는 게이트 목록 (walk 엣지·동일층)."""
    out = {}
    for ei, (u, v, m, L) in enumerate(mem.edges):
        if m != "walk":
            continue
        nu, nv = mem.nodes[u], mem.nodes[v]
        if nu["floor"] != nv["floor"]:
            continue
        hits = [gi for gi, g in enumerate(bank.gates)
                if int(g["floor"]) == int(nu["floor"])
                and _seg_dist2(g["x"], g["z"], nu["x"], nu["z"],
                               nv["x"], nv["z"]) <= pass_m ** 2]
        if hits:
            out[ei] = hits
    return out


def differential_block(bank_scores, eg_map, rid, tau):
    """z-차등 엣지 차단 — 이 로봇만 τ 미달이고 누군가는 통과하는 게이트.

    전원 차단(높이 협착)은 라벨엔 있지만 expert 주행 GT(2D 폭)엔 없어
    모순이므로 환경 결함으로 보고 세지 않는다. 판정 전용.

    "누군가 통과"의 기준은 bank_scores에 들어 있는 **다른 로봇들**이다.
    따라서 호출부는 비교 대상 로봇의 점수를 **미리 모두 채운 뒤** 호출해야
    한다 — 한 대씩 채우며 호출하면 기준이 비어 차단이 통째로 비고(실측:
    폭 1m 로봇도 194개 중 0개 차단), 게다가 호출 순서에 따라 결과가
    달라진다. 기준이 비면 조용히 통과시키지 않고 즉시 실패시킨다.
    """
    others = [o for k, o in bank_scores.items() if k != rid]
    if not others:
        raise ValueError(
            f"차등 차단 기준 로봇 없음(rid={rid}) — bank_scores에 비교 대상 "
            f"로봇 점수를 먼저 채울 것")
    bad = {gi for gi, s in bank_scores[rid].items()
           if s < tau and any(o.get(gi, -99) >= tau for o in others)}
    return {ei for ei, gis in eg_map.items() if any(g in bad for g in gis)}


def objreg_query(tag, mrob, goal, upto, near_m=10.0,
                 cache_dir=CACHE):
    """관측 객체 레지스트리 조회 — notfound·도착 검증 점수 (GOAT식).

    검출 명칭 일치 + **인스턴스 문맥 필터**(지시문의 층 + 근접) — 이름만
    매칭하면 동명 인스턴스에 걸려 우연 수준(0.52), 문맥 필터로 AUC 0.865
    (실측). near_m은 방 수준 문맥의 대리(관측 사거리 10m).
    """
    rf = os.path.join(cache_dir, f"{tag}_{mrob}_objreg.npz")
    with np.load(rf) as z_:
        reg = {k: z_[k] for k in z_.files}
    tm = json.load(open(os.path.join(DS, "topomap",
                                     f"{tag}_{mrob}.json")))
    nf = {int(n["id"]): (int(n["floor"]), n["x"], n["z"])
          for n in tm["nodes"]}
    vocab = list(reg["vocab"])
    name, gfl = goal["name"], int(goal["floor"])
    gx, gz = float(goal["x"]), float(goal["z"])
    best = 0.0
    if name in vocab:
        m = (reg["node"] <= upto) & (reg["name"] == vocab.index(name))
        for i in np.where(m)[0]:
            fl, nx, nz = nf[int(reg["node"][i])]
            if fl == gfl and (nx - gx) ** 2 + (nz - gz) ** 2 \
                    <= near_m ** 2:
                best = max(best, float(reg["score"][i]))
    return best


def gate_scores(policy, mem, feats, z, device="cuda"):
    """union 게이트 노드별 g 최고 뷰 로짓 (z 조건)."""
    import torch
    out = {}
    with torch.no_grad():
        for i, n in enumerate(mem.nodes):
            if n["kind"] != "gate":
                continue
            e = feats.node_embeds(mem, i)
            if not len(e):
                continue
            f = torch.from_numpy(e.astype(np.float32)).to(device)
            s = policy.g(f, z.expand(len(f), -1))
            out[i] = float(s.max())
    return out


# ------------------------------------------------------------ 판정 전용

def plan_route(mem, start_uid, goal_uids, stairs_ok, gate_sc,
               tau=0.0, elev_penalty=20.0, edge_block=None):
    """다익스트라 — **판정 전용**(실행 경로 호출 금지, judge_state 참조).

    edge_block=차단 엣지 인덱스 집합(게이트 관통 차단).
    """
    import heapq
    adj = {}
    for ei, (u, v, m, L) in enumerate(mem.edges):
        if m == "stairs" and not stairs_ok:
            continue
        if edge_block and ei in edge_block:
            continue
        w = L + (elev_penalty if m == "elevator" else 0.0)
        adj.setdefault(u, []).append((v, w, m))
        adj.setdefault(v, []).append((u, w, m))
    blocked = {i for i, s in gate_sc.items() if s < tau}
    blocked.discard(start_uid)
    goal_set = set(goal_uids) - blocked
    if start_uid is None or not goal_set:
        return {"reachable": False, "modes": [], "path": [], "dist": None}
    dist = {start_uid: 0.0}
    prev = {}
    pq = [(0.0, start_uid)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e18):
            continue
        if u in goal_set:
            path, modes = [u], []
            while u in prev:
                u, m = prev[u]
                path.append(u)
                modes.append(m)
            path.reverse()
            uniq = [m for m in dict.fromkeys(reversed(modes))
                    if m != "walk"]
            return {"reachable": True, "modes": uniq, "path": path,
                    "dist": d}
        for v, w, m in adj.get(u, []):
            if v in blocked:
                continue
            nd = d + w
            if nd < dist.get(v, 1e18):
                dist[v] = nd
                prev[v] = (u, m)
                heapq.heappush(pq, (nd, v))
    return {"reachable": False, "modes": [], "path": [], "dist": None}


def judge_state(mem, start_uid, goal_uids, stairs_ok, gate_sc=None,
                tau=0.0, edge_block=None):
    """불가능 판정 — 상태 축의 유일한 진입점 (실행 플래닝 아님).

    반환 state: "ok"(계획 가능) / "notfound"(기억에 목적지 없음) /
    "embodiment"(기억엔 있으나 이 몸으로 도달 불가).

    통과성 차단 규약 = **z-차등 엣지 차단 하나**(differential_block).
    노드 단위 gate_sc는 기본으로 쓰지 않는다 — 차등 가드가 없어서 "전원
    차단" 게이트(높이 협착)까지 세는데, 그건 라벨엔 있지만 expert 주행
    GT(2D 폭)엔 없어 모순이라 false_impossible을 만든다. GateBank는 환경의
    전 게이트를 덮으므로 커버리지 손실도 없다. gate_sc는 진단용으로만 남긴다.
    """
    if not goal_uids:
        return {"state": "notfound", "modes": [], "route": None}
    route = plan_route(mem, start_uid, goal_uids, stairs_ok,
                       gate_sc or {}, tau=tau, edge_block=edge_block)
    if not route["reachable"]:
        return {"state": "embodiment", "modes": [], "route": route}
    return {"state": "ok", "modes": route["modes"], "route": route}


# ------------------------------------------------------------ 주행 부품

def avoid(env, wx, wz):
    """LiDAR 가정 로컬 회피(1스텝 MPC류) — 제어 층 전용.

    희망 월드 변위가 footprint상 막히면 ±15~90° 중 희망 방향에 가장
    가까운 통과 가능 변위로 대체, 전부 막히면 정지. 행선 선택(모델)에는
    불개입. 거리 소스 = 시뮬 점유 그리드(2D LiDAR 등가).
    """
    d = math.hypot(wx, wz)
    if d < 1e-6:
        return wx, wz
    allow = any(k[0] == env.fl for k in env._door_open)
    if env._free(env.fl, env.x + wx, env.z + wz, allow_elev=allow):
        return wx, wz
    b = math.atan2(wx, wz)
    for deg in (15, 30, 45, 60, 75, 90):
        for sgn in (1, -1):
            nb = b + math.radians(sgn * deg)
            cx, cz = d * math.sin(nb), d * math.cos(nb)
            if env._free(env.fl, env.x + cx, env.z + cz, allow_elev=allow):
                return cx, cz
    return 0.0, 0.0


def move_action(env, wx, wz, arrive_yaw=None, step_m=STEP_M):
    """월드 목표점 → env 액션 (dx, dz, dyaw). 막히면 None.

    회전 선적용 규약 보정: env가 dyaw를 먼저 적용하므로 월드 변위를
    회전 **후** 프레임으로 환산해 넘긴다. arrive_yaw가 주어지면 그 값이
    이 스텝의 도착 heading(전역 선택기 예측)이고, 없으면 진행 방향을 본다.
    순수 회전(dx=dz=0, dyaw≠0)은 만들지 않는다 — 막히면 None을 돌려
    홉 실패로 처리한다.
    """
    dxw, dzw = wx - env.x, wz - env.z
    d = math.hypot(dxw, dzw)
    if d < 1e-9:
        return None
    mag = min(step_m, d)
    ux, uz = dxw / d * mag, dzw / d * mag
    ux, uz = avoid(env, ux, uz)
    if ux == 0.0 and uz == 0.0:
        return None
    tgt = (math.degrees(math.atan2(ux, uz)) if arrive_yaw is None
           else arrive_yaw)
    dyaw = (tgt - env.yaw + 180) % 360 - 180
    ry = math.radians(env.yaw + dyaw)
    return (ux * math.cos(ry) - uz * math.sin(ry),
            ux * math.sin(ry) + uz * math.cos(ry), dyaw)


class HopDriver:
    """홉 단위 전역 선택 주행 — 벤치·평가 공용 주행 경로.

    한 홉 = 전역 선택 1회 + 확정된 행선까지의 저수준 이동(이동 중에는
    선택 미호출). 절차는 EA_Nav 홉 루프 규약 그대로:
      observe → propose → lift → GraphMap.update → gmap_inputs →
      navigation → argmax → (고스트면 delete_ghost 후) 주행

    행선 3종:
      · STOP(vpid None) = 도착 선언 → 상태 판정으로 넘김
      · 고스트 = 미방문 후보. 리프트된 월드 좌표로 pure-pursuit 주행
      · 방문 노드 = 백트래킹. 기억 그래프 최단경로를 따라가고, 층이
        바뀌는 구간은 스킬(엘베 탑승·계단 등반)로 처리한다

    스킬 구간은 전역 선택을 호출하지 않는다 — 엘베는 FSM 진행(대기중),
    계단은 등반 완료까지. 계단 프레임은 바닥 y=0 기준 히트맵이 전 픽셀
    음성이라 후보가 비는 것이 정상이고, 이를 막다른 곳으로 오판하지
    않도록 상태를 스킬 진행으로 고정한다.

    홉 실패(막힘·무진행)는 그 후보를 이번 홉에서 제외하고 즉시 재선택,
    HOP_FAIL_LIMIT 연속 실패면 stalled. 후보가 0개여도 대체 후보를 만들지
    않는다(노드+STOP만 남아 백트래킹이 자연 유도됨).
    """

    def __init__(self, policy, device, mem, feats=None, tau=None,
                 stairs_ok=True, cam_h=0.4, budget=2000,
                 max_hops=MAX_HOPS, hop_steps=HOP_STEPS, log=None):
        import torch
        from modules.GraphPlanning import GraphMap
        self.torch = torch
        self.policy, self.device = policy, device
        self.mem, self.feats = mem, feats
        self.tau, self.stairs_ok, self.cam_h = tau, stairs_ok, cam_h
        self.budget, self.max_hops, self.hop_steps = (budget, max_hops,
                                                      hop_steps)
        self.log = log or (lambda *_a: None)
        self._GraphMap = GraphMap
        self.trans = mem.transition_nodes()
        # 백트래킹 인접(실행 보조) — 이미 주행한 엣지만, 계단은 능력 필요
        self.adj = {}
        for u, v, m, L in mem.edges:
            if m == "stairs" and not stairs_ok:
                continue
            self.adj.setdefault(u, []).append((v, L, m))
            self.adj.setdefault(v, []).append((u, L, m))

    # ---- 에피소드 준비 ----

    def start(self, z=None, txt=None, seed=False):
        """에피소드 초기화 — 히스토리·그래프를 비운다.

        **기본은 빈 그래프**다. 학습(train_global)이 빈 GraphMap에서
        에피소드를 재생하며 노드를 쌓으므로, 추론에서 표준 기억 전체를
        방문 노드로 미리 넣으면 후보 집합 분포가 학습과 어긋난다(실측:
        후보 130여 개 vs 학습 시 홉 수만큼 → 선택기가 첫 홉에서 STOP,
        전량 벤치 TL 1.2m·SR 0.086). seed=True는 그 구조를 실험할 때만.

        규칙 전역 ablation(follow)은 후보 집합을 쓰지 않으므로 무관하다.
        """
        self.z = z
        self.txt = txt
        self.policy.reset_episode()
        self.gm = self._GraphMap()
        self.uid2vp = {}
        self.prev_vp = None
        self.hops = 0
        self.sel_log = []
        if seed and self.feats is not None:
            self._seed_memory()

    def _seed_memory(self):
        """표준 기억을 온라인 그래프에 방문 노드로 적재.

        벤치의 기억은 사전 구축된 표준 기억 그래프다 — 후보 집합에
        들어가야 백트래킹·층 전환·목표형 접근이 성립한다. 공개 API만
        사용(update로 등록 + connect로 실엣지 복원)해 내부 상태를
        직접 건드리지 않는다.
        """
        for uid, n in enumerate(self.mem.nodes):
            emb = self.feats.node_embed_mean(self.mem, uid)
            t = self.torch.from_numpy(emb).to(self.device)
            vp = self.gm.update(self._pos(n["floor"], n["x"], n["z"]), t,
                                [], [], floor=int(n["floor"]),
                                prev_vp=None)
            self.uid2vp[uid] = vp
        for u, v, _m, L in self.mem.edges:
            if u in self.uid2vp and v in self.uid2vp:
                self.gm.connect(self.uid2vp[u], self.uid2vp[v], float(L))

    @staticmethod
    def _pos(floor, x, z):
        return (float(x), float(floor) * FLOOR_Y, float(z))

    # ---- 홉 루프 ----

    def run(self, env, goal, arrival_check=None):
        """폐루프 주행. 반환 = {state, reason, hops, stop_pose}.

        state ∈ {arrived, driving, stalled, impossible_*}. arrival_check는
        목표형·결합형의 레지스트리 검증 콜백(없으면 STOP + 반경만).
        """
        fails = 0
        while env.steps < self.budget and self.hops < self.max_hops:
            obs = self._observe(env)
            if obs is None:      # 렌더 없음(진단 모드)
                return {"state": "driving", "reason": "no_render",
                        "hops": self.hops}
            inp, nav = self._select(env, obs)
            banned = set()
            moved = False
            while True:
                pick = self._argmax(nav["global_logits"][0], banned)
                if pick is None:
                    break
                sel = inp["gmap_vpids"][pick]
                hd = self._heading_deg(nav["heading"][0, pick])
                if sel is None:
                    return self._stop(env, goal, arrival_check)
                self.sel_log.append(sel)
                # 홉 로그 = 진단 + 감독자 심박(로그 mtime 정체로 stall 판정)
                kind = ("고스트" if sel in self.gm.ghost_pos else "노드")
                self.log(f"  홉 {self.hops}: {kind} {sel} "
                         f"heading {hd:.0f}° steps={env.steps} fl={env.fl}")
                before = (env.fl, env.x, env.z, env.yaw)
                ok = self._go(env, sel, hd)
                # 진행 없는 홉은 성공으로 치지 않는다 — 도착 허용 반경 안의
                # 후보를 고르면 0스텝으로 "도착"해 홉 예산만 태운다.
                # 단 heading 변화는 진행으로 인정한다: 방향 전환 홉은 변위가
                # 아니라 시야 변화가 성과이고, 그게 다음 홉의 후보를 만든다.
                if ok and before[0] == env.fl and math.hypot(
                        env.x - before[1], env.z - before[2]) < MIN_HOP_M \
                        and abs((env.yaw - before[3] + 180) % 360 - 180) \
                        < MIN_HOP_DEG:
                    ok = False
                if ok:
                    moved = True
                    break
                banned.add(pick)
            self.hops += 1
            if not moved:
                fails += 1
                self.log(f"  홉 실패 {fails}/{HOP_FAIL_LIMIT} "
                         f"(steps={env.steps} fl={env.fl})")
                if fails >= HOP_FAIL_LIMIT:
                    return {"state": "stalled", "reason": "stalled",
                            "hops": self.hops}
            else:
                fails = 0
        return {"state": "driving",
                "reason": "budget" if env.steps >= self.budget
                else "max_hops", "hops": self.hops}

    def follow(self, env, uids, goal):
        """규칙 전역 ablation 집행 — 주어진 행선열을 그대로 따라간다.

        선택기를 호출하지 않는 대신 저수준 이동·스킬은 학습 팔과 같은
        것을 쓴다 — 그래야 두 팔의 차이가 "전역 선택" 하나로 좁혀진다.
        """
        i = 0
        while i < len(uids):
            i = self._skip_stale(uids, i, env)
            if i >= len(uids):
                break
            n = self.mem.nodes[uids[i]]
            if int(n["floor"]) != env.fl:
                if not self._skill(env, int(n["floor"])):
                    return {"state": "stalled",
                            "reason": "transition_failed",
                            "hops": self.hops}
                i += 1
                continue
            if not self._walk(env, n["x"], n["z"], None, NODE_PASS_M):
                return {"state": "stalled", "reason": "stalled",
                        "hops": self.hops}
            self.hops += 1
            i += 1
        # 마지막 접근 — goal 노드 탐색 반경(4m)이 도착 반경(3m)보다 커서
        # 경로를 다 걸어도 성공 반경 밖일 수 있다. 학습 팔은 선택기가 STOP
        # 시점을 정하므로 이 보정이 없지만, 규칙 팔에는 있어야 구 규칙
        # 플래너와 같은 정책이 된다(ablation 비교 대상 유지).
        if int(env.fl) == int(goal[0]):
            self._walk(env, goal[1], goal[2], None, ARRIVE_M * 0.8)
        return self._stop(env, goal, None)

    def _skip_stale(self, uids, i, env):
        """낡은 중간층 노드 건너뛰기.

        엘베는 목표층으로 직행하므로(FSM이 에피소드 목표층을 탄다) 경로에
        남은 중간층 노드는 이미 지나간 것이 된다. 현재 층 노드가 뒤에
        있으면 다른 층 노드는 소비하고 넘어간다 — 안 그러면 이미 도착한
        층에서 층 전환을 다시 시도해 실패한다.
        """
        while i < len(uids) and \
                int(self.mem.nodes[uids[i]]["floor"]) != env.fl and \
                any(int(self.mem.nodes[u]["floor"]) == env.fl
                    for u in uids[i:]):
            i += 1
        return i

    def _observe(self, env):
        o = env._obs()
        if o is None:
            return None
        t = self.torch
        rgb = t.from_numpy(np.ascontiguousarray(o["rgb"]))[None] \
            .to(self.device)
        dep = t.from_numpy(o["depth"].astype(np.float32))[None] \
            .to(self.device)
        with t.no_grad():
            return self.policy("observe", rgb=rgb, depth=dep,
                               cam_h=self.cam_h,
                               loc_heading_rad=t.tensor(
                                   [math.radians(env.yaw)],
                                   device=self.device))

    def _select(self, env, obs):
        """제안 → 리프트 → 그래프 갱신 → 전역 선택."""
        t = self.torch
        with t.no_grad():
            out = self.policy("propose", obs=obs, z=self.z)
            prop = self.policy.wp.propose(out)
            pose = (env.fl, env.x, env.z, env.yaw, self.cam_h)
            cand = self.policy.lift(prop, obs, pose,
                                    floor_y=env.fl * FLOOR_Y)
            v = cand["valid"][0]
            cpos = cand["world"][0][v].numpy()
            cemb = cand["embed"][0][v.to(cand["embed"].device)]
            vp = self.gm.update(self._pos(env.fl, env.x, env.z),
                                obs["node_embed"][0], cpos, cemb,
                                floor=int(env.fl), prev_vp=self.prev_vp)
            self.prev_vp = vp
            inp = self.gm.gmap_inputs(vp, self._pos(env.fl, env.x, env.z),
                                      math.radians(env.yaw),
                                      device=self.device,
                                      cur_floor=int(env.fl))
            txt = self.txt or self.policy.nav.null_instruction(1,
                                                               self.device)
            # g는 가산 바이어스로만 쓴다(g_tau=None) — 학습(train_global)이
            # 하드 −inf 차단 없이 돌았으므로 추론에서 τ로 후보를 죽이면
            # 학습이 본 적 없는 후보 집합이 된다(실측: 살아 있는 고스트가
            # −inf로 사라져 STOP이 상대적으로 유리해짐). τ 차단은 상태 판정
            # 경로(게이트 차단)의 규약이지 선택기의 규약이 아니다.
            nav = self.policy("navigation", z=self.z,
                              txt_embeds=txt[0], txt_masks=txt[1],
                              mask_visited=False, g_bias=True,
                              g_tau=None, **inp)
        return inp, nav

    @staticmethod
    def _argmax(logits, banned):
        best, bi = None, None
        for i in range(logits.shape[0]):
            if i in banned:
                continue
            v = float(logits[i])
            if v == float("-inf"):
                continue
            if best is None or v > best:
                best, bi = v, i
        return bi

    @staticmethod
    def _heading_deg(hv):
        """선택기 heading 헤드 (sin, cos) → 월드 yaw(도)."""
        s, c = float(hv[0]), float(hv[1])
        return math.degrees(math.atan2(s, c)) % 360.0

    # ---- 행선 집행 ----

    def _go(self, env, sel, hd):
        """확정 행선까지 이동. 성공 True.

        방문 노드는 두 갈래다 — 표준 기억에서 적재한 노드(uid 대응)는
        기억 그래프 최단경로로 백트래킹하고, 이번 에피소드에 생긴 온라인
        노드는 저장 위치로 직접 간다(주행해서 만든 노드라 경로가 자명).
        현재 노드를 다시 고르는 것은 진행이 없으므로 실패로 돌려 재선택을
        유도한다.
        """
        if sel == self.prev_vp:
            # 현재 노드 재선택 = "여기서 방향만 바꿔라". 전방 히트맵이 고갈된
            # 홉에서 선택기가 쓰는 정상 수단이므로 도착 heading으로 정렬한다.
            # 이미 그 방향이면 바뀌는 게 없으니 실패로 돌려 재선택을 유도.
            before = env.yaw
            self._align(env, hd)
            return abs((env.yaw - before + 180) % 360 - 180) >= MIN_HOP_DEG
        if sel in self.gm.ghost_pos:
            p = self.gm.ghost_mean_pos[sel]
            self.gm.delete_ghost(sel)
            return self._walk(env, float(p[0]), float(p[2]), hd,
                              GHOST_PASS_M)
        uid = next((u for u, v in self.uid2vp.items() if v == sel), None)
        if uid is not None:
            return self._backtrack(env, uid, hd)
        if sel not in self.gm.node_pos:
            return False
        p = self.gm.node_pos[sel]
        fl = self.gm.node_floor.get(sel, env.fl)
        if int(fl) != env.fl and not self._skill(env, int(fl)):
            return False
        return self._walk(env, float(p[0]), float(p[2]), hd, NODE_PASS_M)

    def _align(self, env, hd):
        """도착 heading 정렬 — 순수 회전이 없으므로 그 방향으로 짧게 이동."""
        if hd is None or abs((hd - env.yaw + 180) % 360 - 180) < 10.0:
            return
        ry = math.radians(hd)
        act = move_action(env, env.x + 0.25 * math.sin(ry),
                          env.z + 0.25 * math.cos(ry),
                          arrive_yaw=hd, step_m=0.25)
        if act is not None:
            env.step(act, observe=False)

    def _walk(self, env, wx, wz, hd, tol):
        """저수준 pure-pursuit — 도착 heading은 마지막 스텝에 적용.

        이미 허용 반경 안이면 정렬만 한다. 방향 전환 홉(선택기가 가까운
        방문 노드를 고르는 경우)은 변위가 아니라 heading이 성과이므로,
        여기서 그냥 True를 돌려주면 시야가 안 바뀌어 같은 선택이 무한
        반복된다(실측: 홉 3~7이 같은 노드를 재선택하며 정지).
        """
        for _ in range(self.hop_steps):
            d = math.hypot(wx - env.x, wz - env.z)
            if d <= tol:
                self._align(env, hd)
                return True
            act = move_action(env, wx, wz,
                              arrive_yaw=hd if d <= STEP_M else None)
            if act is None:
                return False
            # 홉 내부 이동은 관측을 쓰지 않는다 — 렌더는 홉 경계에서만
            env.step(act, observe=False)
            if env.steps >= self.budget:
                return False
        return math.hypot(wx - env.x, wz - env.z) <= tol

    def _backtrack(self, env, goal_uid, hd):
        """기억 그래프 최단경로 추종(실행 보조) — 층 구간은 스킬로."""
        path = self._shortest(env, goal_uid)
        if not path:
            return False
        i = 0
        while i < len(path):
            i = self._skip_stale(path, i, env)
            if i >= len(path):
                break
            n = self.mem.nodes[path[i]]
            if int(n["floor"]) != env.fl:
                if not self._skill(env, int(n["floor"])):
                    return False
                i += 1
                continue
            last = i == len(path) - 1
            if not self._walk(env, n["x"], n["z"], hd if last else None,
                              NODE_PASS_M):
                return False
            i += 1
        return True

    def _shortest(self, env, goal_uid):
        """현재 위치에서 목표 노드까지 노드열 (다익스트라, 주행 엣지만)."""
        import heapq
        src = self.mem.nearest(env.fl, env.x, env.z)
        if src is None or goal_uid is None:
            return []
        if src == goal_uid:
            return [goal_uid]
        dist, prev = {src: 0.0}, {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, 1e18):
                continue
            if u == goal_uid:
                out = [u]
                while u in prev:
                    u = prev[u]
                    out.append(u)
                return list(reversed(out))[1:]
            for v, L, _m in self.adj.get(u, []):
                nd = d + L
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return []

    def _skill(self, env, to_floor):
        """층 전환 스킬 — 전역 선택 미호출 구간."""
        if env.fl in env.elev:
            return self._ride(env, to_floor)
        if env.fl in env.stairs and self.stairs_ok:
            return self._climb(env)
        return False

    def _ride(self, env, to_floor):
        """엘베 탑승 — 호출·대기·탑승. 대기 중에는 제자리 정지(회전 금지)."""
        ex, ez, nx, nz, _cx, _cz = env.elev[env.fl]
        # 호출 존은 문 앞 0.8m, 도착 heading은 문을 향한다(FSM 성립 조건)
        face = math.degrees(math.atan2(ex - (ex + nx * 0.8),
                                       ez - (ez + nz * 0.8))) % 360.0
        if not self._walk(env, ex + nx * 0.8, ez + nz * 0.8, face, 0.4):
            return False
        # 문 정면 정렬 — FSM 호출은 문 방향 시선을 요구하는데 제자리 회전이
        # 없으므로, 문 쪽으로 조금 이동하면서 도착 heading으로 맞춘다
        if abs((face - env.yaw + 180) % 360 - 180) > 20.0:
            act = move_action(env, ex + nx * 0.6, ez + nz * 0.6,
                              arrive_yaw=face, step_m=0.2)
            if act is not None:
                env.step(act, observe=False)
        start_fl = env.fl
        for _ in range(self.hop_steps * 3):
            if env.fl != start_fl:
                return True
            open_here = any(k[0] == env.fl for k in env._door_open)
            # 문이 열리면 전진해 탑승, 아니면 문을 본 채 정지 대기
            env.step((0.0, 0.4, 0.0) if open_here else (0.0, 0.0, 0.0),
                     observe=False)
            if env.steps >= self.budget:
                return False
        return env.fl != start_fl

    def _climb(self, env):
        """계단 등반 — 존 진입 시 env가 전환 처리. 능력 미달이면 실패."""
        sx, sz = env.stairs[env.fl]
        start_fl = env.fl
        for _ in range(self.hop_steps * 2):
            if env.fl != start_fl:
                return True
            act = move_action(env, sx, sz)
            if act is None:
                return False
            env.step(act, observe=False)
            if env.events and env.events[-1][0] == "stairs_denied":
                return False
            if env.steps >= self.budget:
                return False
        return env.fl != start_fl

    def _stop(self, env, goal, arrival_check):
        """전역 STOP → 도착 판정 (반경 = 벤치 SR과 동일값)."""
        same = int(env.fl) == int(goal[0])
        d = math.hypot(env.x - goal[1], env.z - goal[2])
        ok = same and d <= ARRIVE_M
        if ok and arrival_check is not None:
            ok = bool(arrival_check(env))
        return {"state": "arrived" if ok else "driving",
                "reason": "-" if ok else "arrival_error",
                "hops": self.hops, "stop_dist": d}


class ThorDriver:
    """실주행 롤아웃 — 층별 컨트롤러 풀 + 실높이 렌더 (드라이브 레벨).

    실패 처리 원칙:
    이 클래스는 복구를 시도하지 않는다. THOR 웨지(Unity 바이너리 내부
    결함)는 server_timeout으로 즉시 예외가 되어 실행자 프로세스가 그대로
    종료되고, 벤치 감독자가 프로세스 재기동 + 장부(항목 jsonl) 재개로
    일원 처리한다(감독자-실행자 구조). 여기 남는 것은 수명 정책 2개뿐:
    - 층별 풀: 층당 컨트롤러 1회 로드 유지, 전환 = 스왑(씬 재구축 제거)
    - 선제 재활용: 카메라 갱신 RECYCLE_OPS 누적 시 교체 — 장수 인스턴스
      웨지(실측 ~4,500회 누적)의 유발 조건 제거
    """

    SERVER_TIMEOUT = 120.0
    RECYCLE_OPS = 1200

    def __init__(self, tag, gpu=0):
        mc = _mc()
        self.mc = mc
        # pick_gpu가 CUDA_VISIBLE_DEVICES를 이미 제한 → 컨트롤러엔 항상 0
        os.environ["MANSION_GPU"] = "0"
        bld, var = building_of(tag)
        if var != "none":
            os.environ["MANSION_VARIANT"] = var
        self.floors = load_floors(tag)
        self.bld = bld
        # floor → controller (풀)
        self.ctrls = {}
        self.ctrl = None
        self.cur_floor = None
        # floor → 카메라 갱신 누적(재활용 판단)
        self._ops = {}

    def _launch_floor(self, floor):
        import json as _j
        c = self.mc.launch_controller(width=self.mc.RES, height=self.mc.RES,
                                      render=True,
                                      server_timeout=self.SERVER_TIMEOUT)
        scene = _j.load(open(os.path.join(self.bld,
                                          f"floor_{floor}.json")))
        self.mc.load_floor_scene(c, scene)
        c.step(action="AddThirdPartyCamera",
               position=dict(x=0, y=1, z=0),
               rotation=dict(x=0, y=0, z=0), fieldOfView=90)
        # 렌더 전용 드라이버 — 스텝마다 전송되는 오브젝트 메타데이터
        # JSON을 차단해 카메라 갱신 단가를 낮춘다 (프레임에는 무영향)
        c.step(action="SetObjectFilter", objectIds=[])
        return c

    def goto_floor(self, floor):
        if floor == self.cur_floor:
            return
        if floor not in self.ctrls:
            self.ctrls[floor] = self._launch_floor(floor)
            self._ops[floor] = 0
        self.ctrl = self.ctrls[floor]
        self.cur_floor = floor

    def _retire(self, floor):
        """수명 만료 컨트롤러 정리(복구 아님 — 자원 해제)."""
        c = self.ctrls.pop(floor, None)
        try:
            if c is not None:
                c.stop()
        except Exception:
            pass
        if self.ctrl is c:
            self.ctrl = None
        if self.cur_floor == floor:
            self.cur_floor = None
        self._ops[floor] = 0

    def render(self, floor, x, z, yaw_deg, cam_h):
        if self._ops.get(floor, 0) >= self.RECYCLE_OPS:
            # 선제 재활용(수명 정책)
            self._retire(floor)
        self.goto_floor(floor)
        bgr, dep = self.mc.update_cam(self.ctrl, x, cam_h, z, yaw_deg)
        self._ops[floor] = self._ops.get(floor, 0) + 1
        return bgr[..., ::-1].copy(), dep.copy()

    def free(self, floor, x, z):
        g = self.floors[floor]
        iz, ix = g.to_idx(x, z)
        return (0 <= iz < g.inside.shape[0]
                and 0 <= ix < g.inside.shape[1]
                and bool(g.inside[iz, ix]))

    def close(self):
        for c in list(self.ctrls.values()) + \
                ([self.ctrl] if self.ctrl is not None
                 and self.ctrl not in self.ctrls.values() else []):
            try:
                c.stop()
            except Exception:
                pass

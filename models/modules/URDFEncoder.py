"""URDF GNN 인코더 — robot.urdf → latent z.

그래프 규약 (scripts/datasets/URDF/common.py의 생성 규약과 목록·차원 일치
유지 의무):
  노드 = 링크. 피처 = 타입 원핫(링크 이름 prefix) + primitive 원핫
        + 기하 치수(3) + 질량(log1p) + 관성 대각 ixx,iyy,izz(log1p)
  엣지 = 조인트(양방향, 동일 피처). 피처 = 타입 원핫 + 축(3)
        + origin xyz(3) + origin rpy(3) + 리밋(lower,upper)
        + effort(log1p) + velocity(log1p)

인코더 = 엣지 조건 메시지 패싱(순수 torch, torch_geometric 불사용) × layers
→ mean+max readout → MLP → z. 보조 프로브(z → w·l·h 회귀 + stairs_ok 로짓
+ cam_h 회귀)는 z가 형태 정보를 담는지 감시하는 선택 헤드.

파싱 = yourdfpy(load_meshes=False). 배치 = collate_graphs 블록 결합.
"""

import math

import torch
import torch.nn as nn

# ---- 그래프 규약 상수 ----

NODE_TYPES = [
    "base", "torso", "head", "neck", "arm", "hand", "leg", "foot", "ball",
    "wheel", "caster", "steer", "mast",
    "sensor_rgb", "sensor_depth", "sensor_lidar", "sensor_imu",
]
JOINT_TYPES = ["revolute", "continuous", "prismatic", "fixed", "floating",
               "planar"]
GEOM_TYPES = ["box", "cylinder", "sphere"]

# 노드 피처 = 타입 + primitive + 치수 3 + 질량 1 + 관성 대각 3
NODE_DIM = len(NODE_TYPES) + len(GEOM_TYPES) + 3 + 1 + 3
# 엣지 피처 = 타입 + 축 3 + origin xyz 3 + origin rpy 3 + 리밋 2
#             + effort 1 + velocity 1
EDGE_DIM = len(JOINT_TYPES) + 3 + 3 + 3 + 2 + 1 + 1

# 정규화(고정 링크 병합)에서 병합하지 않는 노드 타입 —
# fixed 조인트여도 기능 의미가 있는 링크들
_PROTECTED = {"sensor_rgb", "sensor_depth", "sensor_lidar", "sensor_imu",
              "wheel", "caster", "steer", "foot", "ball"}


# ---- URDF 파싱 → 중간 표현 ----

def _rpy(R):
    """회전 행렬 → URDF rpy (고정축 XYZ = ZYX 오일러 추출)."""
    import numpy as _np
    r = math.atan2(R[2, 1], R[2, 2])
    p = math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    y = math.atan2(R[1, 0], R[0, 0])
    return _np.array([r, p, y])


def _node_type_idx(link_name):
    """링크 이름 prefix → NODE_TYPES 인덱스. 규약 밖 이름은 base(0)."""
    for t in sorted(NODE_TYPES, key=len, reverse=True):
        if link_name.startswith(t):
            return NODE_TYPES.index(t)
    return 0


def _parse_urdf(urdf_path):
    """URDF → (links{name: dict}, joints[dict]). origin은 4×4 동차변환."""
    import numpy as np
    import yourdfpy

    u = yourdfpy.URDF.load(urdf_path, load_meshes=False,
                           build_collision_scene_graph=False)
    links = {}
    for n, link in u.link_map.items():
        dims, gtype = (0.0, 0.0, 0.0), None
        if link.visuals:
            geo = link.visuals[0].geometry
            if geo.box is not None:
                gtype, dims = "box", tuple(geo.box.size)
            elif geo.cylinder is not None:
                gtype = "cylinder"
                dims = (geo.cylinder.radius, geo.cylinder.length, 0.0)
            elif geo.sphere is not None:
                gtype, dims = "sphere", (geo.sphere.radius, 0.0, 0.0)
        inr = link.inertial.inertia if link.inertial else None
        links[n] = {
            "type": _node_type_idx(n), "gtype": gtype, "dims": dims,
            "mass": float(link.inertial.mass or 0.0) if link.inertial else 0.0,
            "inertia": [max(float(inr[k][k]), 0.0) for k in range(3)]
            if inr is not None else [0.0, 0.0, 0.0]}
    joints = []
    for j in u.joint_map.values():
        if j.parent not in links or j.child not in links:
            continue
        joints.append({
            "type": j.type, "parent": j.parent, "child": j.child,
            "axis": (list(j.axis) if j.axis is not None else [0.0, 0.0, 0.0]),
            "origin": (np.array(j.origin, dtype=float) if j.origin is not None
                       else np.eye(4)),
            "lower": float(j.limit.lower or 0.0) if j.limit else 0.0,
            "upper": float(j.limit.upper or 0.0) if j.limit else 0.0,
            "effort": float(j.limit.effort or 0.0) if j.limit else 0.0,
            "velocity": float(j.limit.velocity or 0.0) if j.limit else 0.0})
    return links, joints


def _canonicalize(links, joints):
    """고정 링크 병합 정규화 — 운동학 불변.

    fixed 조인트의 자식이 _PROTECTED 타입이 아니면 부모로 병합한다:
    질량·관성 합산, 부모에 기하가 없으면 자식 기하 승계, 자식의 하위
    조인트는 origin 합성(T_pc @ T_cd)으로 부모에 재부모화. 수렴까지 반복.
    """
    while True:
        merge = None
        for j in joints:
            if j["type"] == "fixed" and \
                    NODE_TYPES[links[j["child"]]["type"]] not in _PROTECTED:
                merge = j
                break
        if merge is None:
            return links, joints
        p, c = merge["parent"], merge["child"]
        lp, lc = links[p], links[c]
        lp["mass"] += lc["mass"]
        lp["inertia"] = [a + b for a, b in zip(lp["inertia"], lc["inertia"])]
        if lp["gtype"] is None and lc["gtype"] is not None:
            lp["gtype"], lp["dims"] = lc["gtype"], lc["dims"]
        for j2 in joints:
            if j2["parent"] == c:
                j2["parent"] = p
                j2["origin"] = merge["origin"] @ j2["origin"]
        del links[c]
        joints.remove(merge)


# ---- 중간 표현 → 텐서 그래프 ----

def _tensorize(links, joints):
    names = list(links.keys())
    idx = {n: i for i, n in enumerate(names)}
    x = torch.zeros(len(names), NODE_DIM)
    for n, l in links.items():
        i = idx[n]
        x[i, l["type"]] = 1.0
        if l["gtype"]:
            x[i, len(NODE_TYPES) + GEOM_TYPES.index(l["gtype"])] = 1.0
        base = len(NODE_TYPES) + len(GEOM_TYPES)
        x[i, base:base + 3] = torch.tensor(l["dims"])
        x[i, -4] = math.log1p(l["mass"])
        for k in range(3):
            x[i, -3 + k] = math.log1p(l["inertia"][k])
    src, dst, eattr = [], [], []
    for j in joints:
        f = torch.zeros(EDGE_DIM)
        if j["type"] in JOINT_TYPES:
            f[JOINT_TYPES.index(j["type"])] = 1.0
        o = len(JOINT_TYPES)
        f[o:o + 3] = torch.tensor(j["axis"], dtype=torch.float32)
        f[o + 3:o + 6] = torch.tensor(j["origin"][:3, 3],
                                      dtype=torch.float32)
        f[o + 6:o + 9] = torch.tensor(_rpy(j["origin"][:3, :3]),
                                      dtype=torch.float32)
        f[o + 9], f[o + 10] = j["lower"], j["upper"]
        f[o + 11] = math.log1p(j["effort"])
        f[o + 12] = math.log1p(j["velocity"])
        for a, b in ((j["parent"], j["child"]), (j["child"], j["parent"])):
            src.append(idx[a])
            dst.append(idx[b])
            eattr.append(f)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = (torch.stack(eattr) if eattr
                 else torch.zeros(0, EDGE_DIM))
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}


def urdf_to_graph(urdf_path, canonical=True):
    """robot.urdf → {x:(N,NODE_DIM), edge_index:(2,E), edge_attr:(E,EDGE_DIM)}.

    canonical=True면 고정 링크 병합 정규화를 먼저 적용한다 — 보호 타입만
    fixed인 생성 로봇은 사실상 불변이고, 실기 URDF의 마운트·더미 체인이
    제거된다.
    """
    links, joints = _parse_urdf(urdf_path)
    if canonical:
        links, joints = _canonicalize(links, joints)
    return _tensorize(links, joints)


def augment_graph(g, rng, max_insert=8):
    """구조 잡음 증강 — 더미 fixed 리프 링크 0~max_insert개 삽입.

    삽입 링크는 capability(치수·cam_h·stairs_ok)에 영향이 없으므로
    통과성·프로브 라벨이 전부 그대로 유효한 라벨 보존 변환이다.
    """
    k = int(rng.integers(0, max_insert + 1))
    if k == 0:
        return g
    n = g["x"].shape[0]
    xs, srcs, dsts, eas = [g["x"]], [g["edge_index"][0]], \
        [g["edge_index"][1]], [g["edge_attr"]]
    new_x = torch.zeros(k, NODE_DIM)
    new_src, new_dst, new_ea = [], [], []
    for t in range(k):
        parent = int(rng.integers(0, n))
        ntype = 0 if rng.random() < 0.7 else int(g["x"][parent, :len(
            NODE_TYPES)].argmax())
        new_x[t, ntype] = 1.0
        new_x[t, len(NODE_TYPES) + GEOM_TYPES.index("box")] = 1.0
        base = len(NODE_TYPES) + len(GEOM_TYPES)
        new_x[t, base:base + 3] = torch.tensor(
            rng.uniform(0.01, 0.15, 3), dtype=torch.float32)
        new_x[t, -4] = math.log1p(float(rng.uniform(0.01, 0.5)))
        f = torch.zeros(EDGE_DIM)
        f[JOINT_TYPES.index("fixed")] = 1.0
        o = len(JOINT_TYPES)
        f[o + 3:o + 6] = torch.tensor(rng.uniform(-0.2, 0.2, 3),
                                      dtype=torch.float32)
        if rng.random() < 0.5:
            f[o + 6:o + 9] = torch.tensor(rng.uniform(-math.pi, math.pi, 3),
                                          dtype=torch.float32)
        ni = n + t
        new_src += [parent, ni]
        new_dst += [ni, parent]
        new_ea += [f, f]
    xs.append(new_x)
    srcs.append(torch.tensor(new_src, dtype=torch.long))
    dsts.append(torch.tensor(new_dst, dtype=torch.long))
    eas.append(torch.stack(new_ea))
    return {"x": torch.cat(xs),
            "edge_index": torch.stack([torch.cat(srcs), torch.cat(dsts)]),
            "edge_attr": torch.cat(eas)}


def collate_graphs(graphs):
    """그래프 리스트 → 블록 결합 배치. batch 벡터 = 노드→그래프 id."""
    xs, eis, eas, batch = [], [], [], []
    off = 0
    for gi, g in enumerate(graphs):
        n = g["x"].shape[0]
        xs.append(g["x"])
        eis.append(g["edge_index"] + off)
        eas.append(g["edge_attr"])
        batch.append(torch.full((n,), gi, dtype=torch.long))
        off += n
    return {"x": torch.cat(xs), "edge_index": torch.cat(eis, dim=1),
            "edge_attr": torch.cat(eas), "batch": torch.cat(batch),
            "num_graphs": len(graphs)}


# ---- GNN 인코더 ----

class _EdgeConv(nn.Module):
    """엣지 조건 메시지 패싱 1층: h_i ← MLP(h_i, Σ_j MLP([h_j, e_ij]))."""

    def __init__(self, dim, edge_dim):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(dim + edge_dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim))
        self.upd = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        m = self.msg(torch.cat([h[src], edge_attr], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        out = self.upd(torch.cat([h, agg], dim=-1))
        return self.norm(h + out)


class URDFEncoder(nn.Module):
    """URDF 그래프 배치 → z (B, z_dim). aux=True면 보조 프로브 출력 포함."""

    def __init__(self, z_dim=128, hidden=128, layers=4, aux=True):
        super().__init__()
        self.embed = nn.Linear(NODE_DIM, hidden)
        self.convs = nn.ModuleList(
            [_EdgeConv(hidden, EDGE_DIM) for _ in range(layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, z_dim))
        # 보조 프로브: z → 외곽 (w,l,h) 회귀 + stairs_ok 로짓 + cam_h 회귀
        self.probe = (nn.Sequential(
            nn.Linear(z_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 5))
            if aux else None)

    def forward(self, g):
        h = self.embed(g["x"])
        for conv in self.convs:
            h = conv(h, g["edge_index"], g["edge_attr"])
        # readout: 그래프별 mean + max 풀 결합
        B = g["num_graphs"]
        mean = torch.zeros(B, h.shape[-1], device=h.device)
        cnt = torch.zeros(B, 1, device=h.device)
        mean.index_add_(0, g["batch"], h)
        cnt.index_add_(0, g["batch"],
                       torch.ones(h.shape[0], 1, device=h.device))
        mean = mean / cnt.clamp(min=1)
        mx = torch.full((B, h.shape[-1]), -torch.inf, device=h.device)
        mx.index_reduce_(0, g["batch"], h, "amax", include_self=True)
        z = self.readout(torch.cat([mean, mx], dim=-1))
        if self.probe is None:
            return z, None
        p = self.probe(z)
        return z, {"wlh": p[:, :3], "stairs_ok_logit": p[:, 3],
                   "cam_h": p[:, 4]}

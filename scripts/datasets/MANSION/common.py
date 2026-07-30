"""MANSION 씬 GT 기하 공통 유틸 (게이트 추출·expert 플래너 파이프라인).

씬 JSON + 에셋 콜라이더 메시로 층 기하를 직접 조립한다. 시뮬레이터·렌더링에
의존하지 않으므로 결정적이고, 통과성 라벨과 expert 경로가 같은 기하 원천을 쓴다.

좌표계: AI2-THOR/Unity (y-up). 격자 인덱스는 (iz, ix), 월드는 (x, z).
층 전환은 계단/엘리베이터 앵커 간 그래프 엣지로 추상화한다(시뮬레이터의
TakeStairs/UseElevator 텔레포트와 대응). navmesh는 쓰지 않는다.

통과성 판정 규약 (plan 확정 초기값):
  - 로봇 유효 폭 = min(w, l)  (게이트는 몸을 돌려 통과 가능)
  - 셀 통과 조건: 바닥 지지 + 머리 위 클리어런스 >= 로봇 h
  - 폭 조건: 자유 공간 EDT >= 유효 폭 / 2
  - soft 라벨: margin ±0.15 m 선형 램프
  - 계단: legged(humanoid·multileg)만, 엘리베이터: 전원 + 대기 페널티 20 m
"""

import glob
import hashlib
import json
import math
import os

import cv2
import numpy as np
from scipy import ndimage

_torch = None


def torch_dev():
    """torch + GPU 디바이스 (지연 로드). Agent.md 규칙: 연산은 GPU 우선.

    GPU 0~3만 사용(규칙). MANSION_GPU 환경변수로 지정, 기본 0.
    """
    global _torch
    if _torch is None:
        import torch

        _torch = (torch, torch.device(
            f"cuda:{int(os.environ.get('MANSION_GPU', 0))}"))
    return _torch

ASSETS_DIR = "/root/.objathor-assets/2023_09_23/assets"
# URDF 풀: 기본 = 본 풀(train 스플릿, 2026-07-26 생성). 파일럿 재현 등은
# MANSION_URDF_POOL 환경변수로 교체(/tmp/urdf_pilot — seed 42 재생성 가능)
URDF_POOL_DIR = os.environ.get("MANSION_URDF_POOL", "/data/URDF/train")

GRID = 0.025          # 격자 해상도 (m) — plan 확정 초기값
SURFACE_EPS = 0.05    # 이 높이 이하 포인트 = 바닥 접촉 장애물
LABEL_RAMP = 0.15     # soft 라벨 선형 램프 반폭 (m)
GATE_WIDTH_MAX = 1.5  # 협착부 게이트 판정 상한 (m)
H_MIN = 0.40          # embodiment h 최소 — 협착부 탐색용 기준 높이
ELEV_WAIT_COST = 20.0     # 엘리베이터 대기 페널티 (이동거리 환산 m)
STAIR_TRAVERSE_COST = 8.0  # 계단 한 층 통과 비용 (수직+왕복 경로 환산 m)
# 자연스러운 주행 경로: 장애물 여유가 PATH_PREF_MARGIN 미만인 셀은 비용을
# 최대 (1+PATH_HUG_PENALTY)배 가중 → 경로가 복도 중앙을 선호하고 좁은
# 게이트에서만 벽에 붙는다. 순수 최단거리는 벽·가구를 스치며 지나가 부자연.
# 스무딩은 natural_path(DP 단순화)만 사용 — 시선 직선화류는 이 여백을 되돌림.
PATH_PREF_MARGIN = 0.6
PATH_HUG_PENALTY = 6.0

_mesh_cache = {}


# ---------------------------------------------------------------- 에셋 메시

def asset_mesh(asset_id):
    """에셋 콜라이더 메시 (V, F). objathor는 pkl.gz, MANSION 패치는 obj."""
    if asset_id in _mesh_cache:
        return _mesh_cache[asset_id]
    base = os.path.join(ASSETS_DIR, asset_id)
    out = None
    try:
        pkl = os.path.join(base, f"{asset_id}.pkl.gz")
        if os.path.exists(pkl):
            import compress_pickle

            dd = compress_pickle.load(pkl)
            vs, fs, off = [], [], 0
            colliders = dd.get("colliders") or []
            if not colliders:
                colliders = [dd]
            for c in colliders:
                v = np.array([[p["x"], p["y"], p["z"]] for p in c["vertices"]])
                f = np.array(c["triangles"], dtype=np.int64).reshape(-1, 3)
                vs.append(v)
                fs.append(f + off)
                off += len(v)
            out = (np.concatenate(vs), np.concatenate(fs))
        elif os.path.exists(os.path.join(base, f"{asset_id}.obj")):
            vs, fs = [], []
            with open(os.path.join(base, f"{asset_id}.obj")) as fh:
                for line in fh:
                    t = line.split()
                    if not t:
                        continue
                    if t[0] == "v":
                        vs.append([float(t[1]), float(t[2]), float(t[3])])
                    elif t[0] == "f":
                        idx = [int(w.split("/")[0]) - 1 for w in t[1:]]
                        for k in range(1, len(idx) - 1):  # 팬 삼각화
                            fs.append([idx[0], idx[k], idx[k + 1]])
            if vs and fs:
                out = (np.array(vs), np.array(fs, dtype=np.int64))
    except Exception:  # noqa: BLE001 — 깨진 에셋은 AABB 폴백
        out = None
    _mesh_cache[asset_id] = out
    return out


def sample_mesh(asset_id, density=8000, max_pts=40000, with_normals=False):
    """면적 가중 표면 샘플. asset_id 시드로 결정적.

    with_normals=True면 (포인트, 면 법선)을 반환 — 계단 디딤판(위를 향한 면)
    추출 등에 사용.
    """
    m = asset_mesh(asset_id)
    if m is None:
        return None
    v, f = m
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    total = float(area2.sum()) / 2.0
    if total <= 0:
        return None
    seed = int(hashlib.md5(asset_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    n = int(min(max_pts, max(500, total * density)))
    fi = rng.choice(len(f), size=n, p=area2 / area2.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    pts = (tri[fi, 0] * (1 - s)[:, None]
           + tri[fi, 1] * (s * (1 - r2))[:, None]
           + tri[fi, 2] * (s * r2)[:, None])
    if not with_normals:
        return pts
    nrm = cross[fi] / np.maximum(area2[fi][:, None], 1e-12)
    return pts, nrm


def place_points(pts, obj):
    """에셋 로컬 포인트 → 월드. rotation.y 회전 후 footprint AABB 중심·position.y에 정렬."""
    ry = math.radians(obj.get("rotation", {}).get("y", 0.0))
    R = np.array([[math.cos(ry), 0.0, math.sin(ry)],
                  [0.0, 1.0, 0.0],
                  [-math.sin(ry), 0.0, math.cos(ry)]])
    p = pts @ R.T
    verts = np.array(obj["vertices"], dtype=np.float64) / 100.0  # cm → m
    cx = (verts[:, 0].min() + verts[:, 0].max()) / 2
    cz = (verts[:, 1].min() + verts[:, 1].max()) / 2
    p[:, 0] += cx - (p[:, 0].min() + p[:, 0].max()) / 2
    p[:, 2] += cz - (p[:, 2].min() + p[:, 2].max()) / 2
    p[:, 1] += obj["position"]["y"] - (p[:, 1].min() + p[:, 1].max()) / 2
    return p


# ---------------------------------------------------------------- 층 기하

class FloorGeometry:
    """한 층의 격자 기하.

    필드
      inside      방 폴리곤 내부
      wall        벽 (문·열린 벽 개구부는 도려냄)
      clearance   바닥 위 첫 장애물까지 높이 (바닥 접촉 장애물·벽 = 0)
      room_grid   셀별 방 인덱스 (-1 = 없음), rooms = [{id, roomType}]
      stair_mask / elev_mask  계단·엘리베이터 방(+오브젝트 footprint) 셀
    """

    def __init__(self, scene, floor_no):
        self.scene = scene
        self.floor_no = floor_no
        self.ceil_h = float(scene.get("wall_height", 3.2))

        polys = [[(p["x"], p["z"]) for p in r["floorPolygon"]]
                 for r in scene.get("rooms", []) if r.get("floorPolygon")]
        xs = [x for q in polys for x, _ in q]
        zs = [z for q in polys for _, z in q]
        self.bounds = (min(xs), max(xs), min(zs), max(zs))  # 방 폴리곤 bbox
        self.x0, self.z0 = min(xs) - 0.3, min(zs) - 0.3
        self.nx = int((max(xs) - self.x0 + 0.6) / GRID) + 1
        self.nz = int((max(zs) - self.z0 + 0.6) / GRID) + 1
        nz, nx = self.nz, self.nx

        self.rooms = []
        self.room_grid = np.full((nz, nx), -1, dtype=np.int16)
        inside = np.zeros((nz, nx), dtype=np.uint8)
        for r in scene.get("rooms", []):
            if not r.get("floorPolygon"):
                continue
            pts = np.array([self._to_px(p["x"], p["z"])
                            for p in r["floorPolygon"]], dtype=np.int32)
            m = np.zeros((nz, nx), dtype=np.uint8)
            cv2.fillPoly(m, [pts], 1)
            idx = len(self.rooms)
            self.rooms.append({"id": r["id"], "roomType": r.get("roomType", "")})
            self.room_grid[m > 0] = idx
            inside |= m
        self.inside = inside.astype(bool)

        self._build_walls(scene)
        self._build_objects(scene)

    # ---- 좌표 변환 ----
    def _to_px(self, x, z):
        return int((x - self.x0) / GRID), int((z - self.z0) / GRID)

    def to_idx(self, x, z):
        return int((z - self.z0) / GRID), int((x - self.x0) / GRID)

    def to_world(self, iz, ix):
        return self.x0 + (ix + 0.5) * GRID, self.z0 + (iz + 0.5) * GRID

    # ---- 벽 ----
    def _build_walls(self, scene):
        nz, nx = self.nz, self.nx
        wall = np.zeros((nz, nx), dtype=np.uint8)
        for w in scene.get("walls", []):
            pts = {(round(p["x"], 4), round(p["z"], 4))
                   for p in w.get("polygon", [])}
            if len(pts) != 2:
                continue
            (ax, az), (bx, bz) = sorted(pts)
            cv2.line(wall, self._to_px(ax, az), self._to_px(bx, bz), 1,
                     thickness=max(1, int(0.10 / GRID)))
        # 개구부: 세그먼트 위 샘플점에서 법선 방향 ±0.13 m만 도려낸다.
        # cv2.line(두께)로 하면 끝점 캡이 세그먼트 밖으로 넘쳐 직각으로 맞닿은
        # 이웃 벽까지 뚫리고, 법선 한쪽만 지우면 벽 래스터(±0.05 m)가 남는다.
        openings = [d["doorSegment"] for d in scene.get("doors", [])
                    if d.get("doorSegment")]
        openings += (scene.get("open_walls") or {}).get("segments", [])
        offs = np.arange(-0.13, 0.13 + 1e-9, GRID / 2)
        for seg in openings:
            (ax, az), (bx, bz) = seg[0], seg[1]
            length = math.hypot(bx - ax, bz - az)
            if length < 1e-6:
                continue
            nxv, nzv = -(bz - az) / length, (bx - ax) / length
            ts = np.linspace(0.0, 1.0, max(2, int(length / (GRID / 2)) + 1))
            px = ax + (bx - ax) * ts[:, None] + nxv * offs[None, :]
            pz = az + (bz - az) * ts[:, None] + nzv * offs[None, :]
            ix = ((px - self.x0) / GRID).astype(np.int64).ravel()
            iz = ((pz - self.z0) / GRID).astype(np.int64).ravel()
            ok = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
            wall[iz[ok], ix[ok]] = 0
        self.wall = wall.astype(bool)
        self.door_segments = [tuple(map(tuple, d["doorSegment"]))
                              for d in scene.get("doors", [])
                              if d.get("doorSegment")]
        self.doors_meta = [{"segment": tuple(map(tuple, d["doorSegment"])),
                            "rooms": [d.get("room0", ""), d.get("room1", "")]}
                           for d in scene.get("doors", [])
                           if d.get("doorSegment")]

    # ---- 오브젝트 → 클리어런스 ----
    def _build_objects(self, scene):
        nz, nx = self.nz, self.nx
        self.clearance = np.full((nz, nx), self.ceil_h, dtype=np.float32)
        self.clearance[self.wall | ~self.inside] = 0.0
        # top = 셀별 장애물 상단 최고 높이 — 시선 차단 판정용
        # (시선 높이 h에서 차단 = 벽 or 장애물이 [clearance, top]으로 h를 가로지름)
        self.top = np.zeros((nz, nx), dtype=np.float32)
        self.top[self.wall] = self.ceil_h
        self._stair_pts = []  # 계단 메시 월드 포인트 (등반 경로 추출용)
        self.stair_mask = np.zeros((nz, nx), dtype=bool)
        self.elev_mask = np.zeros((nz, nx), dtype=bool)
        self.n_mesh, self.n_aabb = 0, 0
        self.y_align_err = []  # (position.y - 메시 중심 배치시 바닥 오차) 감사용

        for i, r in enumerate(self.rooms):
            rid = r["id"].lower()
            if "stair" in rid:
                self.stair_mask |= self.room_grid == i
            if "elevator" in rid or "elev" in rid:
                self.elev_mask |= self.room_grid == i

        for o in scene.get("objects", []):
            verts = o.get("vertices")
            if not verts:
                continue
            name = (o.get("object_name", "") + " " + o.get("assetId", "")).lower()
            fp = (np.array(verts, dtype=np.float64) / 100.0)
            fp_px = np.array([self._to_px(x, z) for x, z in fp], dtype=np.int32)
            if "stair" in name:
                m = np.zeros((nz, nx), dtype=np.uint8)
                cv2.fillPoly(m, [fp_px], 1)
                self.stair_mask |= m > 0
                self.clearance[m > 0] = 0.0  # 주행 격자에서는 차단, 등반은 별도 경로
                self.top[m > 0] = self.ceil_h  # 계단 구조물 = 시선 불투과
                sm = (sample_mesh(o["assetId"], with_normals=True)
                      if o.get("assetId") else None)
                if sm is not None:
                    pts, nrm = sm
                    # 디딤판 = 위를 향한 면 (법선 y > cos 20°) — 난간·기둥 제외
                    # (y축 회전은 법선 y 성분을 바꾸지 않으므로 배치 전 판정 가능)
                    tread = nrm[:, 1] > math.cos(math.radians(20))
                    if tread.any():
                        self._stair_pts.append(place_points(pts, o)[tread])
                continue
            if "elevator" in name:
                m = np.zeros((nz, nx), dtype=np.uint8)
                cv2.fillPoly(m, [fp_px], 1)
                self.elev_mask |= m > 0
                continue  # 패널 등 — 통행 차단으로 안 봄

            pts = sample_mesh(o["assetId"]) if o.get("assetId") else None
            if pts is None:
                # 메시 없음 → footprint 전체를 바닥 접촉·불투과 장애물로 (보수적)
                m = np.zeros((nz, nx), dtype=np.uint8)
                cv2.fillPoly(m, [fp_px], 1)
                self.clearance[m > 0] = 0.0
                self.top[m > 0] = self.ceil_h
                self.n_aabb += 1
                continue
            p = place_points(pts, o)
            self.y_align_err.append(float(p[:, 1].min()))
            ix = ((p[:, 0] - self.x0) / GRID).astype(np.int64)
            iz = ((p[:, 2] - self.z0) / GRID).astype(np.int64)
            y = p[:, 1]
            ok = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz) & (y < self.ceil_h)
            low = ok & (y <= SURFACE_EPS)
            self.clearance[iz[low], ix[low]] = 0.0
            hi = ok & (y > SURFACE_EPS)
            np.minimum.at(self.clearance, (iz[hi], ix[hi]),
                          y[hi].astype(np.float32))
            np.maximum.at(self.top, (iz[ok], ix[ok]), y[ok].astype(np.float32))
            self.n_mesh += 1

    # ---- 통과성 ----
    def free_mask(self, h):
        """높이 h 로봇이 서 있을 수 있는 셀.

        계단실·엘리베이터 방 내부는 제외 — 층 전환은 문 앞 앵커 + 텔레포트로
        추상화하므로 샤프트 내부는 주행 공간이 아니다 (내부를 열어두면 시작점
        샘플·경로가 샤프트를 지나는 그림이 나온다).
        """
        return (self.inside & ~self.wall & (self.clearance >= h)
                & ~self.stair_mask & ~self.elev_mask)

    def width_map(self, h):
        """free_mask(h)의 EDT×2 = 국소 통과 폭 (m). 높이별 캐시."""
        key = round(h, 3)
        if not hasattr(self, "_width_cache"):
            self._width_cache = {}
        if key not in self._width_cache:
            free = self.free_mask(h).astype(np.uint8)
            dist = cv2.distanceTransform(
                np.pad(free, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            self._width_cache[key] = dist * GRID * 2.0
        return self._width_cache[key]

    def passable(self, w_eff, h):
        """유효 폭 w_eff·높이 h 로봇의 주행 가능 셀 (침식)."""
        return self.width_map(h) >= w_eff

    def stair_climb_path(self):
        """계단 등반 경로: 높이 0.15 m 구간별 메시 포인트 중심을 이은 폴리라인.

        계단은 텔레포트가 아니라 물리 통과(사용자 확정) — legged의 GT 궤적과
        P2 렌더 경로가 이 폴리라인을 따른다. 반환: [(iz, ix, y), ...] 낮은→높은.
        난간 포인트가 중심을 약간 당기는 근사 오차 있음(카드로 검수).
        """
        if not hasattr(self, "_climb_cache"):
            if not self._stair_pts:
                self._climb_cache = []
            else:
                p = np.concatenate(self._stair_pts)
                out = []
                for y0 in np.arange(0.05, self.ceil_h - 0.05, 0.15):
                    sel = p[(p[:, 1] >= y0) & (p[:, 1] < y0 + 0.15)]
                    if len(sel) < 30:
                        continue
                    # 바닥 슬래브 제외: x·z 모두 계단실 전체를 덮는 구간은
                    # 디딤판이 아니라 에셋에 포함된 바닥판 — 중심이 계단실
                    # 정중앙으로 끌려가 시작점·진입이 1/2 지점이 됨(실제 발생).
                    # 랜딩은 한 방향만 넓어서 이 필터에 안 걸림.
                    if (np.ptp(sel[:, 0]) > 1.5 and np.ptp(sel[:, 2]) > 1.5):
                        continue  # numpy 2.x: ndarray.ptp 메서드 없음
                    iz, ix = self.to_idx(float(sel[:, 0].mean()),
                                         float(sel[:, 2].mean()))
                    out.append((iz, ix, round(y0 + 0.075, 3)))
                self._climb_cache = out
        return self._climb_cache

    def zone_door_segment(self, kind):
        """존(stair/elev) 방으로 통하는 문 세그먼트 ((ax,az),(bx,bz)) | None."""
        key = "stair" if kind == "stair" else "elev"
        for d in self.doors_meta:
            if any(key in r.lower() for r in d["rooms"]):
                return d["segment"]
        return None

    def zone_anchor(self, kind, w_eff, h):
        """계단('stair')/엘리베이터('elev') 탑승 지점 = 그 방의 문 앞 셀.

        존 방으로 통하는 문 세그먼트 중점에서 1.5 m 안의 주행 가능 셀 중 가장
        가까운 것. 문이 없으면(개방벽 연결) 존 경계 최근접 셀로 폴백.
        "존에서 가장 가까운 셀"만 쓰면 문 반대편(벽 너머 옆방)이 잡혀서 경로가
        벽을 뚫고 탑승하는 그림이 된다(실제 발생한 버그).
        """
        ok = self.passable(w_eff, h)
        mask = self.stair_mask if kind == "stair" else self.elev_mask
        if not mask.any() or not ok.any():
            return None
        key = "stair" if kind == "stair" else "elev"
        mids = []
        for d in self.doors_meta:
            if any(key in r.lower() for r in d["rooms"]):
                (ax, az), (bx, bz) = d["segment"]
                mids.append(((ax + bx) / 2, (az + bz) / 2))
        cand = None
        for mx, mz in mids:
            miz, mix = self.to_idx(mx, mz)
            r = int(1.5 / GRID)
            z0, z1 = max(0, miz - r), min(self.nz, miz + r + 1)
            x0, x1 = max(0, mix - r), min(self.nx, mix + r + 1)
            sub = ok[z0:z1, x0:x1]
            if not sub.any():
                continue
            zz, xx = np.where(sub)
            dd = (zz + z0 - miz) ** 2 + (xx + x0 - mix) ** 2
            k = int(np.argmin(dd))
            c = (int(zz[k] + z0), int(xx[k] + x0))
            if cand is None or dd[k] < cand[0]:
                cand = (dd[k], c, (mx, mz))
        if cand is not None:
            return cand[1], cand[2]  # (문 앞 셀, 문 중점 월드 좌표)
        # 폴백: 존 경계 최근접 주행 가능 셀 (3 m 이내), 문 중점 없음
        d = ndimage.distance_transform_edt(~mask)
        d[~ok] = np.inf
        iz, ix = np.unravel_index(np.argmin(d), d.shape)
        if not np.isfinite(d[iz, ix]) or d[iz, ix] * GRID > 3.0:
            return None
        return (int(iz), int(ix)), None


def env_prefix(building_dir):
    """산출물 파일 접두사 = 시뮬 제공 빌딩명 그대로(#뒤 제거) [+_custom_변형].

    사용자 확정(2026-07-26): 모든 check 산출물 파일에 예외 없이 부착.
    커스텀 합성(MANSION_VARIANT=elevator_only)은 `_custom_elevonly` 접미.
    """
    base = os.path.basename(building_dir.rstrip("/")).split("#")[0]
    if os.environ.get("MANSION_VARIANT", "none") == "elevator_only":
        base += "_custom_elevonly"
    return base


def out_name(building_dir, name):
    """파일명에 환경 접두사 부착."""
    return f"{env_prefix(building_dir)}_{name}"


def _floor_scene(building_dir, no):
    """층 JSON 로드 + 변형 적용.

    MANSION_VARIANT=elevator_only면 계단 문 entry를 제거한 합성 씬을 만든다
    (층 전환 매트릭스 "엘베만" — 00~05 전 스크립트 공통 스위치).
    """
    with open(os.path.join(building_dir, f"floor_{no}.json")) as fh:
        d = json.load(fh)
    if os.environ.get("MANSION_VARIANT", "none") == "elevator_only":
        d["doors"] = [x for x in d["doors"] if "stair" not in
                      (x.get("room0", "") + x.get("room1", "")).lower()]
    return d


def load_floor(building_dir, no):
    """빌딩 폴더에서 한 층만 FloorGeometry로 로드 (GT 재유도 워커용)."""
    return FloorGeometry(_floor_scene(building_dir, no), no)


def load_building(building_dir):
    """빌딩 폴더 → {층번호: FloorGeometry}. 캐시 없음(층당 수 초)."""
    floors = {}
    for fp in sorted(glob.glob(os.path.join(building_dir, "floor_*.json"))):
        no = int(os.path.basename(fp).split("_")[1].split(".")[0])
        floors[no] = FloorGeometry(_floor_scene(building_dir, no), no)
    return floors


# ---------------------------------------------------------------- URDF 풀

def _rgb_cam_height(urdf_path, base_z):
    """URDF FK(스탠딩=전 관절 0)로 sensor_rgb 카메라의 지면 기준 높이."""
    try:
        import yourdfpy

        u = yourdfpy.URDF.load(urdf_path, load_meshes=False,
                               build_collision_scene_graph=False)
        for link in u.link_map:
            if link.startswith("sensor_rgb"):
                T = u.get_transform(link, u.base_link)
                return float(base_z + T[2, 3])
    except Exception:  # noqa: BLE001
        pass
    return None


def robot_pool(root=URDF_POOL_DIR):
    """URDF 풀 → [{id, cls, form, w, l, h, w_eff, stairs_ok, cam_h}] (id 정렬).

    cam_h = sensor_rgb 실높이 (FK). 관측 시점 판정(시선 차단·FOV)에 사용.
    """
    cache = os.path.join(root, "pool_cache.json")
    metas = sorted(glob.glob(os.path.join(root, "*", "*", "meta.json")))
    if os.path.exists(cache) and metas and \
            os.path.getmtime(cache) > max(map(os.path.getmtime, metas)):
        with open(cache) as fh:
            return json.load(fh)
    out = []
    for mp in metas:
        with open(mp) as fh:
            m = json.load(fh)
        w, l, h = (float(m["measured"][k]) for k in ("w", "l", "h"))
        cam = _rgb_cam_height(os.path.join(os.path.dirname(mp), "robot.urdf"),
                              float(m.get("base_z", 0.0)))
        out.append({
            "id": m["id"], "cls": m["robot_class"], "form": m["form"],
            "w": w, "l": l, "h": h, "w_eff": min(w, l),
            "stairs_ok": m["robot_class"] in ("humanoid", "multileg"),
            "cam_h": cam if cam is not None else 0.9 * h,
        })
    try:  # 캐시 저장 (읽기 전용 마운트 등 실패는 무시)
        with open(cache, "w") as fh:
            json.dump(out, fh)
    except OSError:
        pass
    return out


def soft_label(margin, ramp=LABEL_RAMP):
    """margin(여유, m) → [0,1] 선형 램프. ±ramp 밖은 포화."""
    return float(np.clip(0.5 + margin / (2.0 * ramp), 0.0, 1.0))


def robot_bucket(robot):
    """(유효폭 5 cm, 높이 10 cm) 버킷 — 침식 그리드·통과 폭 캐시 키."""
    return (round(robot["w_eff"] / 0.05) * 0.05,
            round(robot["h"] / 0.1) * 0.1)


# ---------------------------------------------------------------- 게이트

GATE_H_LEVELS = (0.15, 0.6, 1.1, 1.6)  # 협착부 탐색 높이 4단 (m)
GATE_MERGE_R = 0.4    # 이 반경 안에서 다른 높이가 잡히면 같은 협착부로 묶음 (m)
MIN_SIDE_AREA = 0.25  # 게이트 양쪽에 있어야 하는 최소 자유 면적 (m²)


def door_gates(geom):
    out = []
    for d in geom.scene.get("doors", []):
        hp, seg = d.get("holePolygon"), d.get("doorSegment")
        if not hp or len(hp) < 2 or not seg:
            continue
        (ax, az), (bx, bz) = seg
        x, z = (ax + bx) / 2, (az + bz) / 2
        iz, ix = geom.to_idx(x, z)
        out.append({
            "type": "door", "floor": geom.floor_no,
            "x": x, "z": z, "cell": [int(iz), int(ix)],
            "width": round(abs(hp[1]["x"] - hp[0]["x"]), 3),
            "height": round(abs(hp[1]["y"] - hp[0]["y"]), 3),
            "h_ref": 0.0, "h_hits": [],
            "rooms": [d.get("room0", ""), d.get("room1", "")],
        })
    return out


def _is_passage(free, iz, ix, w_px):
    """후보 지점을 막으면 주변 자유 공간이 둘로 갈라지는지 (막다른 틈 제외)."""
    rb = int(1.5 / GRID)
    z0, z1 = max(0, iz - rb), min(free.shape[0], iz + rb + 1)
    x0, x1 = max(0, ix - rb), min(free.shape[1], ix + rb + 1)
    local = free[z0:z1, x0:x1].astype(np.uint8).copy()
    cz, cx = iz - z0, ix - x0
    r_cut = w_px // 2 + max(2, int(0.075 / GRID))
    cv2.circle(local, (cx, cz), r_cut, 0, -1)
    n, lab = cv2.connectedComponents(local)
    if n <= 2:
        return False
    ring = np.zeros_like(local)
    cv2.circle(ring, (cx, cz), r_cut + 2, 1, 2)
    touch = set(np.unique(lab[ring > 0])) - {0}
    min_cells = int(MIN_SIDE_AREA / GRID ** 2)
    return len([t for t in touch if (lab == t).sum() >= min_cells]) >= 2


def _chokes_at(geom, doors, h_ref):
    """탐색 높이 h_ref 단면에서의 가구 협착부. 문 근처(0.5 m)·막다른 틈 제외.

    medial_axis는 rng 고정 필수 — 픽셀 처리 순서 난수화로 결과가 실행마다
    달라진다(실제 발생).
    """
    from scipy.ndimage import minimum_filter
    from skimage.morphology import medial_axis

    free = geom.free_mask(h_ref)
    skel, dist = medial_axis(free, return_distance=True, rng=0)
    width = dist * GRID * 2.0
    cand = skel & (width < GATE_WIDTH_MAX) & (width > 0.05)
    r = int(0.3 / GRID)
    wsk = np.where(skel, width, np.inf)
    cand &= width <= minimum_filter(wsk, size=2 * r + 1, mode="constant",
                                    cval=np.inf) + 0.02
    n_lab, labels = cv2.connectedComponents(
        cv2.dilate(cand.astype(np.uint8), np.ones((5, 5), np.uint8)))
    out, n_dead = [], 0
    for lb in range(1, n_lab):
        m = (labels == lb) & cand
        if not m.any():
            continue
        iz, ix = np.unravel_index(np.argmin(np.where(m, width, np.inf)),
                                  width.shape)
        x, z = geom.to_world(iz, ix)
        if min((math.hypot(x - g["x"], z - g["z"]) for g in doors),
               default=9e9) < 0.5:
            continue
        w_px = max(1, int(width[iz, ix] / GRID))
        if not _is_passage(free, iz, ix, w_px):
            n_dead += 1
            continue
        rr = max(2, w_px // 2)
        z0, z1 = max(0, iz - rr), min(geom.nz, iz + rr + 1)
        x0, x1 = max(0, ix - rr), min(geom.nx, ix + rr + 1)
        patch = geom.clearance[z0:z1, x0:x1]
        h = float(patch[patch > 0].min()) if (patch > 0).any() else 0.0
        ri = geom.room_grid[iz, ix]
        out.append({
            "type": "choke", "floor": geom.floor_no,
            "x": round(x, 3), "z": round(z, 3),
            "cell": [int(iz), int(ix)],
            "width": round(float(width[iz, ix]), 3), "height": round(h, 3),
            "h_ref": h_ref, "h_hits": [h_ref],
            "rooms": [geom.rooms[ri]["id"] if ri >= 0 else ""],
        })
    return out, n_dead


def choke_gates(geom, doors, h_levels=GATE_H_LEVELS):
    """가구 협착부 게이트 — 탐색 높이 여러 단의 합집합.

    바닥 한 높이만 훑으면 가구 위·아래로 지나가는 통로의 높이 제약이 게이트로
    잡히지 않아, 키가 큰 로봇에게는 "높이 때문에 못 지나감" 표본이 거의 생기지
    않는다. 같은 자리를 여러 높이가 잡으면 가장 낮은 높이의 기록만 남기고
    h_hits에 나머지 높이를 적는다 — 라벨은 gate_label이 로봇 자기 높이의 실제
    통과 폭으로 계산하므로 위치만 있으면 되고, 스냅샷 렌더도 중복되지 않는다.
    """
    out, n_dead = [], 0
    for h in h_levels:
        cs, nd = _chokes_at(geom, doors, h)
        n_dead += nd
        for c in cs:
            near = next((o for o in out
                         if math.hypot(c["x"] - o["x"], c["z"] - o["z"])
                         < GATE_MERGE_R), None)
            if near is None:
                out.append(c)
            else:
                near["h_hits"].append(h)
    return out, n_dead


def gate_cell(geom, g):
    """게이트 레코드의 격자 좌표 (구 레코드는 월드 좌표에서 유도)."""
    if g.get("cell"):
        return int(g["cell"][0]), int(g["cell"][1])
    return geom.to_idx(g["x"], g["z"])


def gate_label(geom, g, robot):
    """게이트 × 로봇 통과성 soft 라벨 = min(폭 라벨, 높이 라벨).

    폭은 기록값이 아니라 **로봇 자기 높이 단면의 국소 통과 폭**으로 계산한다
    (expert 주행의 passable과 같은 원천). 다중 높이 게이트에서 "낮은 로봇은
    가구 밑으로 지나가고 큰 로봇은 막힌다"가 라벨에 그대로 반영된다.
    높이 라벨은 문틀처럼 기하 클리어런스에 없는 개구부 높이를 위해 유지한다.
    """
    iz, ix = gate_cell(geom, g)
    iz = min(max(iz, 0), geom.nz - 1)
    ix = min(max(ix, 0), geom.nx - 1)
    hb = robot_bucket(robot)[1]
    w_here = float(geom.width_map(hb)[iz, ix])
    return min(soft_label(w_here - robot["w_eff"]),
               soft_label(g["height"] - robot["h"]))


def floor_gates(geom):
    """층의 전체 게이트 (문 + 협착부). 층당 1회 캐시."""
    if not hasattr(geom, "_gates_cache"):
        d = door_gates(geom)
        c, _ = choke_gates(geom, d)
        geom._gates_cache = d + c
    return geom._gates_cache


# ------------------------------------------------- 관측 기반 expert 주행

CAM_FOV_DEG = 90.0   # 카메라 수평 FOV — P2 렌더 사양 확정 시 같은 값으로 고정
CAM_RANGE = 10.0     # 가시 거리 (m) — MANSION visibilityDistance 준용
DRIVE_TICK = 0.25    # 주행 틱당 전진 거리 (m)
BELIEF_W = 0.10      # 낙관 가정 폭 — 배치(레이아웃)는 알되 "내 몸이 통과
#                      가능한가"는 카메라로 본 시점에 판정한다
BELIEF_H = H_MIN     # 낙관 가정 높이 — 폭과 같은 규약. 자기 높이로 두면 천장이
#                      낮아 막힌 곳은 belief에서도 닫혀 있어, 높이 제약만은
#                      보지 않고 미리 아는 셈이 된다(관측 기반 판정 규약 위반)


def sight_block(geom, cam_h):
    """카메라 높이 cam_h의 시선을 막는 셀 (벽 or 장애물이 시선 높이를 가로지름).

    낮은 가구 너머는 보이고(top < cam_h), 매달린 구조물 아래로도 보인다
    (clearance > cam_h).
    """
    return geom.wall | ((geom.clearance < cam_h) & (geom.top > cam_h))


def gates_in_view(block_t, cell, heading, gate_cells,
                  fov_deg=CAM_FOV_DEG, rng_m=CAM_RANGE):
    """카메라 (셀, 헤딩)에서 각 게이트 중심이 보이는지 (배치 LOS, GPU).

    조건: 거리 ≤ rng_m, 방위각 ≤ FOV/2, 시선 경로에 차단 셀 없음
    (게이트 자체 주변 3셀은 차단 판정에서 제외 — 게이트 가구가 자기를 가림).
    """
    torch, dev = torch_dev()
    if not gate_cells:
        return []
    nz, nx = block_t.shape
    g = torch.tensor(gate_cells, dtype=torch.float32, device=dev)  # (G, 2)
    cz, cx = float(cell[0]), float(cell[1])
    dz, dx = g[:, 0] - cz, g[:, 1] - cx
    dist = torch.sqrt(dz * dz + dx * dx)  # 셀 단위
    ang = torch.atan2(dz, dx) - heading
    ang = torch.atan2(torch.sin(ang), torch.cos(ang)).abs()
    cand = (dist <= rng_m / GRID) & (ang <= math.radians(fov_deg) / 2)
    n = int(torch.clamp(dist.max(), min=2).item()) + 1
    t = torch.linspace(0.0, 1.0, n, device=dev)  # (N,)
    pz = (cz + dz[:, None] * t[None, :]).round().long().clamp(0, nz - 1)
    px = (cx + dx[:, None] * t[None, :]).round().long().clamp(0, nx - 1)
    hit = block_t[pz, px]  # (G, N)
    # 각 게이트의 유효 샘플 = 자기 거리 − 3셀까지
    lim = ((dist - 3.0).clamp(min=1) / dist.clamp(min=1e-6) * (n - 1)).long()
    idx = torch.arange(n, device=dev)[None, :]
    blocked = (hit & (idx <= lim[:, None])).any(dim=1)
    return (cand & ~blocked).cpu().numpy().tolist()


def drive_expert(geom, robot, nav, fl, start, goal, belief_h=BELIEF_H):
    """관측 기반 expert 주행: 분기는 "게이트가 카메라에 잡힌 시점"에 일어난다.

    belief_h = 낙관 가정 높이. 기본값이 규약이며, robot["h"]를 주면 높이 제약을
    미리 아는 구판 동작이 된다(파일럿 전·후 비교 전용).

    가정: 로봇은 자기 몸(z)과 건물 배치(벽·가구 위치)는 알지만, "내 몸이 이
    게이트를 통과할 수 있는가"는 카메라로 그 게이트를 본 시점에 판정한다.
    belief = 실제 통과 가능 맵에서, 내 몸으로 못 지나는 게이트 위치만 원반으로
    낙관적으로 열어둔 맵. 게이트가 시야에 잡히면 원반을 닫고 재계획.
    판정 단위는 이산 게이트 — 연결 성분 방식은 벽 여백 전체가 한 덩어리로
    이어져 하나만 봐도 층 전체가 막히는 오판을 낳는다(실제 발생).

    반환: (성공, 셀 경로, (판정 위치, 게이트) 목록). 실패 시 부분 경로 보존.
    """
    torch, dev = torch_dev()
    wb, hb = nav._bucket(robot)
    true_pass = geom.passable(wb, hb)
    if not true_pass[start]:
        return False, [start], [], []
    tiny = geom.passable(BELIEF_W, belief_h)
    block_t = torch.from_numpy(sight_block(geom, robot["cam_h"])).to(dev)

    # 내 몸으로 못 지나는 게이트 → 낙관 원반 (양쪽을 잇는 통로 폭만큼)
    belief = true_pass.copy()
    disks = []  # (게이트, 원반 마스크, 게이트 셀)
    for g in floor_gates(geom):
        if gate_label(geom, g, robot) >= 0.5:
            continue  # 통과 가능 게이트 — true_pass가 이미 허용
        giz, gix = gate_cell(geom, g)
        # 원반 반경 = 게이트 통로 폭 기준. 높이로만 막힌 게이트는 바닥 폭이
        # 넓게 잡히므로 상한을 둬 층 절반이 낙관으로 열리는 것을 막는다.
        r = int(((min(g["width"], GATE_WIDTH_MAX) + wb) / 2 + 0.1) / GRID)
        disk = np.zeros_like(true_pass, dtype=np.uint8)
        cv2.circle(disk, (gix, giz), r, 1, -1)
        # 낙관으로 "여는" 부분만 관리 — true_pass 셀까지 닫으면 게이트 옆의
        # 멀쩡한 통로가 같이 끊긴다(실제 발생한 버그)
        disk = (disk > 0) & tiny & ~true_pass
        if disk.any():
            belief |= disk
            disks.append((g, disk, (giz, gix)))
    if not belief[goal]:
        return False, [start], [], []

    closed = np.zeros(len(disks), dtype=bool)

    def replan(src):
        ok = belief.copy()
        for k, (_, disk, _) in enumerate(disks):
            if closed[k]:
                ok &= ~disk
        return nav._mask_route(geom, ok, wb, hb, src, goal)

    path, cur = [start], start
    verdicts = []  # (판정 위치, 게이트) — 분기가 일어난 지점
    touch_r = int(0.5 / GRID)
    pending = None  # 마지막 판정 당시 주행 중이던 belief 경로의 잔여 구간
    for _ in range(len(disks) + 2):  # 게이트 수만큼만 재계획 가능
        route = replan(cur)
        if route is None:
            # 도달 불가 확정 — "가던 belief 경로(goal 지향)"를 통과 가능한
            # 데까지 그대로 이어가 진행 한계점에서 정지·판정. 별도 접근 경로를
            # 사후 계산하면 waypoint/heading GT가 goal 지향이 아니라 "판정 후
            # 덧붙인 방향"이 되는 인과 역전이 생김(사용자 지적).
            tail = []
            if verdicts and pending:
                g = verdicts[-1][1]
                ext = []
                for cell in pending[1:]:
                    if not true_pass[cell]:
                        break
                    ext.append(cell)
                if ext:
                    path.extend(ext)
                    verdicts[-1] = (ext[-1], g)
                # 정지점 너머의 belief 잔여(게이트 틈→goal 방향) — 시선/heading
                # GT의 look-ahead 연장용. 로봇은 이 경로를 주행 중이라 믿었으므로
                # 마지막 프레임들의 GT도 이 방향이어야 함(실행 경로 끝 클램프는
                # goal 지향성을 잃음, 사용자 지적).
                tail = pending[len(ext) + 1:]
            return False, path, verdicts, tail  # 관측 결과 도달 불가 판정
        tick = max(1, int(DRIVE_TICK / GRID))
        i, replanned = 0, False
        while i < len(route) - 1:
            j = min(i + tick, len(route) - 1)
            heading = math.atan2(route[j][0] - route[i][0],
                                 route[j][1] - route[i][1])
            open_ks = [k for k in range(len(disks)) if not closed[k]]
            vis = gates_in_view(block_t, route[i], heading,
                                [disks[k][2] for k in open_ks])
            newly = []
            for k, seen in zip(open_ks, vis):
                gc = disks[k][2]
                near = (abs(route[i][0] - gc[0]) <= touch_r
                        and abs(route[i][1] - gc[1]) <= touch_r)
                if seen or near:  # 게이트가 보이거나 코앞이면 판정
                    closed[k] = True
                    newly.append(k)
            if newly:
                remain = route[j:]
                if any(disks[k][1][c] for k in newly for c in remain):
                    # (판정 위치, 판정된 게이트) — 카드에 "어디가 왜 막혔나" 표시용
                    verdicts.append((route[i], disks[newly[0]][0]))
                    pending = route[i:]  # goal 지향 belief 경로의 잔여 구간
                    cur = route[i]
                    replanned = True
                    break
            path.extend(route[i + 1:j + 1])
            i = j
        if not replanned:
            return True, path, verdicts, []
    return False, path, verdicts, []


# ---------------------------------------------------------------- 플래너

class BuildingNav:
    """멀티플로어 expert 플래너 (GT 기하 침식 그리드 + 층 전환 앵커).

    층 내부: 8-이웃 Dijkstra (scipy.sparse.csgraph).
    층 사이: 계단 앵커(legged, STAIR_TRAVERSE_COST) / 엘베 앵커(전원,
    ELEV_WAIT_COST + STAIR_TRAVERSE_COST 상당의 탑승 이동은 앵커 간 0으로 두고
    대기 페널티만 부과).
    """

    def __init__(self, floors, belief_h=BELIEF_H):
        self.floors = floors
        self.belief_h = belief_h  # 관측 기반 주행의 낙관 가정 높이
        self._cache = {}  # (플로어, w버킷, h버킷) -> (mask, graph, ids)

    def _bucket(self, robot):
        return robot_bucket(robot)

    INF = 1.0e9

    @staticmethod
    def _pf(width, wb):
        """클리어런스 가중 계수 (여유 < PATH_PREF_MARGIN에서 최대 1+PENALTY배)."""
        margin = np.clip((width - wb) / 2.0, 0.0, PATH_PREF_MARGIN)
        return (1.0 + PATH_HUG_PENALTY
                * (1.0 - margin / PATH_PREF_MARGIN) ** 2).astype(np.float32)

    def _grid_tensors(self, fl, robot):
        """(층, 버킷)별 ok·pf GPU 텐서 캐시."""
        torch, dev = torch_dev()
        wb, hb = self._bucket(robot)
        key = (fl, wb, hb)
        if key not in self._cache:
            g = self.floors[fl]
            width = g.width_map(hb)
            ok = width >= wb
            self._cache[key] = (
                ok,
                torch.from_numpy(ok).to(dev),
                torch.from_numpy(self._pf(width, wb)).to(dev))
        return self._cache[key]

    @staticmethod
    def _shift(t, dz, dx, fill):
        out = t.new_full(t.shape, fill)
        nz, nx = t.shape
        z0, z1 = max(0, dz), nz + min(0, dz)
        x0, x1 = max(0, dx), nx + min(0, dx)
        out[z0:z1, x0:x1] = t[z0 - dz:z1 - dz, x0 - dx:x1 - dx]
        return out

    def _field(self, ok_t, pf_t, src):
        """GPU 격자 distance field: 8-이웃 가중 반복 완화 (Dijkstra 등가).

        각 반복이 파면을 1셀 전파 — 격자 그래프라 GPU 텐서 연산으로 병렬화
        가능(희소 Dijkstra는 GPU 구현이 없어 이 방식으로 대체).
        """
        torch, dev = torch_dev()
        D = torch.full(ok_t.shape, self.INF, device=dev)
        D[src] = 0.0
        dirs = [(0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
                (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
                (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
        shifted_pf = [(self._shift(pf_t, dz, dx, 0.0), dz, dx, c)
                      for dz, dx, c in dirs]
        check = None
        for it in range(1, 12001):
            newD = D
            for pf_s, dz, dx, c in shifted_pf:
                cand = (self._shift(D, dz, dx, self.INF)
                        + (c * GRID * 0.5) * (pf_t + pf_s))
                newD = torch.minimum(newD, cand)
            D = torch.where(ok_t, newD, torch.tensor(self.INF, device=dev))
            if it % 128 == 0:
                s = float(D[D < self.INF].sum())
                if check is not None and abs(s - check) < 1e-4:
                    break
                check = s
        return D

    @staticmethod
    def _descend(D, ok, src, dst):
        """distance field 탐욕 하강으로 dst→src 경로 복원 (numpy)."""
        if D[dst] >= BuildingNav.INF:
            return None
        nz, nx = D.shape
        path, cur = [dst], dst
        for _ in range(nz * nx):
            if cur == src:
                return path[::-1]
            z, x = cur
            best, bval = None, D[cur]
            for dz in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dx == 0:
                        continue
                    nzi, nxi = z + dz, x + dx
                    if 0 <= nzi < nz and 0 <= nxi < nx and D[nzi, nxi] < bval:
                        bval, best = D[nzi, nxi], (nzi, nxi)
            if best is None:
                return None
            path.append(best)
            cur = best
        return None

    def _mask_route(self, geom, ok, wb, hb, src, dst):
        """주어진 마스크(belief 등) 위 단발 가중 경로 (GPU field).

        가중은 실제 자유폭(기하) 기준 — belief는 연결만 제한한다.
        """
        if not ok[src] or not ok[dst]:
            return None
        torch, dev = torch_dev()
        pf_t = torch.from_numpy(self._pf(geom.width_map(hb), wb)).to(dev)
        ok_t = torch.from_numpy(ok).to(dev)
        D = self._field(ok_t, pf_t, src).cpu().numpy()
        return self._descend(D, ok, src, dst)

    def _route(self, fl, robot, src, dsts):
        """한 층에서 src → 각 dst 비용·경로. dsts는 [(iz,ix)] (GPU field).

        비용 = 기하학적 경로 길이(m). 클리어런스 가중은 경로 모양에만 쓰고
        비용으로 노출하지 않는다 — 가중값을 수단 선택(계단 vs 엘베)에 쓰면
        문 앞 1m 걷기가 4m+로 계상돼 판단이 뒤집힘(실제 발생).
        """
        ok, ok_t, pf_t = self._grid_tensors(fl, robot)
        if src is None or not ok[src]:
            return [(math.inf, None)] * len(dsts)
        D = self._field(ok_t, pf_t, src).cpu().numpy()
        res = []
        for d in dsts:
            if d is None or not ok[d] or D[d] >= self.INF:
                res.append((math.inf, None))
                continue
            cells = self._descend(D, ok, src, d)
            res.append((path_length(cells), cells) if cells
                       else (math.inf, None))
        return res

    @staticmethod
    def _project_door(seg, xz):
        """문 세그먼트 위 xz 최근접 통과점 (문설주 회피 t∈[0.15, 0.85])."""
        (ax, az), (bx, bz) = seg
        vx, vz = bx - ax, bz - az
        L2 = max(vx * vx + vz * vz, 1e-9)
        t = ((xz[0] - ax) * vx + (xz[1] - az) * vz) / L2
        t = min(0.85, max(0.15, t))
        return (ax + vx * t, az + vz * t)

    def _nearest_passable(self, g, pt, wb, hb, r_m=1.2):
        ok = g.passable(wb, hb)
        piz, pix = g.to_idx(*pt)
        r = int(r_m / GRID)
        z0, z1 = max(0, piz - r), min(g.nz, piz + r + 1)
        x0, x1 = max(0, pix - r), min(g.nx, pix + r + 1)
        sub = ok[z0:z1, x0:x1]
        if not sub.any():
            return None
        zz, xx = np.where(sub)
        k = int(np.argmin((zz + z0 - piz) ** 2 + (xx + x0 - pix) ** 2))
        return (int(zz[k] + z0), int(xx[k] + x0))

    def _anchors(self, robot):
        """버킷별 층 앵커 캐시.

        계단은 탑승('sb', 층)/하차('sa', 층) 앵커를 분리 — 탑승은 그 층 등반
        시작점 쪽 문 끝(3/4), 하차는 아래층 등반 꼭대기 쪽 문 끝(1/4).
        문 중앙 앵커 하나로 쓰면 걷기 구간이 문 중앙으로 모였다가 꺾여
        들어가는 그림이 됨(사용자 지적). 값 = ((iz,ix), 문 통과점 xz).
        엘리베이터는 문 중앙 1개 유지(탑승 지점이 실제로 문 중앙).
        """
        key = ("anchors", self._bucket(robot))
        if key in self._cache:
            return self._cache[key]
        wb, hb = self._bucket(robot)
        out = {}
        for fl, g in self.floors.items():
            a = g.zone_anchor("elev", wb, hb)
            if a:
                out[("elev", fl)] = a
            seg = g.zone_door_segment("stair")
            climb = g.stair_climb_path()
            if seg and climb:  # 이 층에서 위층으로 탑승
                pt = self._project_door(seg, g.to_world(climb[0][0],
                                                        climb[0][1]))
                cell = self._nearest_passable(g, pt, wb, hb)
                if cell:
                    out[("sb", fl)] = (cell, pt)
            below = self.floors.get(fl - 1)
            if seg and below is not None:  # 아래층에서 올라와 이 층에 하차
                bclimb = below.stair_climb_path()
                if bclimb:
                    top = below.to_world(bclimb[-1][0], bclimb[-1][1])
                    pt = self._project_door(seg, top)
                    cell = self._nearest_passable(g, pt, wb, hb)
                    if cell:
                        out[("sa", fl)] = (cell, pt)
        self._cache[key] = out
        return out

    def _anchor_edges(self, robot):
        """버킷별 층내 앵커↔앵커 걷기 엣지 캐시."""
        key = ("aedges", self._bucket(robot))
        if key in self._cache:
            return self._cache[key]
        anchors = self._anchors(robot)
        out = {}
        for fl in self.floors:
            ks = [k for k in anchors if k[1] == fl]
            for i, k1 in enumerate(ks):
                res = self._route(fl, robot, anchors[k1][0],
                                  [anchors[k2][0] for k2 in ks[i + 1:]])
                for k2, (c, cells) in zip(ks[i + 1:], res):
                    if cells:
                        out[(k1, k2)] = (c, fl, cells)
        self._cache[key] = out
        return out

    def plan(self, robot, start, goal, observe=True):
        """start=(층, iz, ix), goal=(층, iz, ix) → dict(성공, 비용, 구간, 전환).

        메타 그래프 다익스트라: 노드 = start/goal/층별 계단·엘베 앵커/엘베 카.
        계단 = 인접 층 앵커 엣지(층당 STAIR_TRAVERSE_COST, legged만),
        엘리베이터 = 카 노드 경유(탑승 1회당 ELEV_WAIT_COST + 3 m 상당).

        observe=True(GT 기본): 걷기 구간을 관측 기반 주행(drive_expert)으로
        생성 — 게이트 우회 분기가 "카메라에 잡힌 시점"에 일어나고, 그 지점이
        verdicts로 반환된다. 계단/엘베 선택은 자기 몸 지식(z)이라 사전 결정.
        observe=False: 전지적 최단 (비교·디버그용).
        """
        import heapq

        fs, fg = start[0], goal[0]
        if fs == fg:
            res = self._walk(robot, fs, start[1:], goal[1:], observe)
            if res is None:
                return {"success": False, "reason": "no_path_same_floor"}
            c, cells, (vcells, vgates), okflag, tail = res
            out = {"success": okflag, "cost": c, "transitions": [],
                   "segments": [(fs, cells)],
                   "verdicts": [(fs, v) for v in vcells],
                   "verdict_gates": [(fs, g) for g in vgates]}
            if not okflag:
                out["reason"] = "observed_unreachable"  # 보고 나서 불가 판정
                # 정지점 너머 belief 잔여 — 시선/heading GT look-ahead 연장용
                out["belief_tail"] = (fs, tail)
            return out

        anchors = self._anchors(robot)
        nodes = {"start": (fs, start[1:]), "goal": (fg, goal[1:])}
        doors = {}  # 노드 키 -> 담당 문 통과점 xz
        for k, (cell, mid) in anchors.items():
            if k[0] in ("sb", "sa") and not robot["stairs_ok"]:
                continue
            nodes[k] = (k[1], cell)
            doors[k] = mid

        adj = {}  # n1 -> [(n2, cost, kind, payload)]

        def add(n1, n2, c, kind, payload=None):
            adj.setdefault(n1, []).append((n2, c, kind, payload))
            adj.setdefault(n2, []).append((n1, c, kind, payload))

        # 층내 걷기: 앵커↔앵커는 캐시, start/goal↔같은 층 노드는 즉석 계산
        for (k1, k2), (c, fl, cells) in self._anchor_edges(robot).items():
            if k1 in nodes and k2 in nodes:
                add(k1, k2, c, "walk", (fl, cells))
        for src_name in ("start", "goal"):
            fl, cell = nodes[src_name]
            others = [k for k, (f2, _) in nodes.items()
                      if f2 == fl and k != src_name and k != "start"]
            res = self._route(fl, robot, cell, [nodes[k][1] for k in others])
            for k, (c, cells) in zip(others, res):
                if cells:
                    add(src_name, k, c, "walk", (fl, cells))
        # 계단: 탑승(sb, 아래층) ↔ 하차(sa, 위층) — 무방향이라 하강도 커버
        if robot["stairs_ok"]:
            for fl in self.floors:
                if ("sb", fl) in nodes and ("sa", fl + 1) in nodes:
                    add(("sb", fl), ("sa", fl + 1),
                        STAIR_TRAVERSE_COST, "stairs", (fl, fl + 1))
        # 엘리베이터: 카 노드 경유 (탑승 1회 = 대기 1회)
        elevs = [k for k in nodes if isinstance(k, tuple) and k[0] == "elev"]
        if len(elevs) >= 2:
            for k in elevs:
                add(k, "car", ELEV_WAIT_COST / 2 + 1.5, "elevator", k[1])

        # 메타 다익스트라.
        # 설계 규칙(사용자 확정): legged는 계단·엘베가 둘 다 있으면 무조건
        # 계단 — 비용 비교가 아니라 규칙. 1차 탐색은 엘베 엣지를 제외하고,
        # 계단만으로 도달 불가할 때만(엘베만 있는 층 전환 등) 엘베를 허용.
        def meta_dijkstra(allow_elev):
            dist, prev = {"start": 0.0}, {}
            pq = [(0.0, 0, "start")]
            tie = 1
            while pq:
                d, _, u = heapq.heappop(pq)
                if u == "goal":
                    break
                if d > dist.get(u, math.inf):
                    continue
                for v, c, kind, payload in adj.get(u, []):
                    if kind == "elevator" and not allow_elev:
                        continue
                    nd = d + c
                    if nd < dist.get(v, math.inf):
                        dist[v] = nd
                        prev[v] = (u, kind, payload)
                        heapq.heappush(pq, (nd, tie, v))
                        tie += 1
            return dist, prev

        dist, prev = ({}, {})
        if robot["stairs_ok"]:
            dist, prev = meta_dijkstra(allow_elev=False)
        if "goal" not in dist:
            dist, prev = meta_dijkstra(allow_elev=True)
        if "goal" not in dist:
            return {"success": False, "reason": "no_route_meta"}

        # 경로 복원
        chain, cur = [], "goal"
        while cur != "start":
            u, kind, payload = prev[cur]
            chain.append((u, cur, kind, payload))
            cur = u
        chain.reverse()
        segs, transitions, verdicts, verdict_gates = [], [], [], []
        walk_cost = 0.0
        i = 0
        while i < len(chain):
            u, v, kind, payload = chain[i]
            if kind == "walk":
                fl, cells = payload
                # 캐시 엣지는 (k1→k2) 방향 저장 — 역방향이면 뒤집기
                if nodes[u][1] != tuple(cells[0]) and nodes[u][1] != cells[0]:
                    cells = cells[::-1]
                if observe:
                    c_leg, cells, (vc, vg), okflag, tail = self._walk(
                        robot, fl, tuple(cells[0]), tuple(cells[-1]),
                        observe=True)
                    walk_cost += c_leg
                    verdicts += [(fl, vd) for vd in vc]
                    verdict_gates += [(fl, g) for g in vg]
                    if not okflag:  # 부분 궤적 보존 (보고 포기한 지점까지)
                        segs.append((fl, cells))
                        return {"success": False,
                                "reason": f"observed_block_f{fl}",
                                "segments": segs, "transitions": transitions,
                                "verdicts": verdicts,
                                "verdict_gates": verdict_gates,
                                "belief_tail": (fl, tail)}
                segs.append((fl, cells))
            elif kind == "stairs":
                # 계단은 텔레포트가 아니라 물리 통과 — 하층 씬의 등반 폴리라인을
                # 세그먼트로 삽입 (앵커→계단 진입 연결선 포함)
                lo_fl = min(nodes[u][0], nodes[v][0])
                hi_fl = max(nodes[u][0], nodes[v][0])
                climb = self.floors[lo_fl].stair_climb_path()  # 낮은→높은
                cells2d = [(c[0], c[1]) for c in climb]
                if cells2d:
                    def line(a, b):
                        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1])))
                        return [(int(round(a[0] + (b[0] - a[0]) * t)),
                                 int(round(a[1] + (b[1] - a[1]) * t)))
                                for t in np.linspace(0, 1, n)]

                    # sb 앵커(3/4 지점) → 등반 밑단, 꼭대기 → sa 앵커(1/4 지점)
                    # — 앵커가 이미 문 통과점 옆이라 직결이 곧 올바른 진입/진출
                    lo_anchor = (nodes[u][1] if nodes[u][0] == lo_fl
                                 else nodes[v][1])
                    hi_anchor = (nodes[v][1] if nodes[v][0] == hi_fl
                                 else nodes[u][1])
                    if nodes[u][0] == lo_fl:  # 올라가는 방향
                        segs.append((lo_fl, line(lo_anchor, cells2d[0])
                                     + cells2d))
                        segs.append((hi_fl, line(cells2d[-1], hi_anchor)))
                    else:  # 내려가는 방향
                        segs.append((hi_fl, line(hi_anchor, cells2d[-1])))
                        segs.append((lo_fl, cells2d[::-1]
                                     + line(cells2d[0], lo_anchor)))
                    climb = (climb if nodes[u][0] == lo_fl else climb[::-1])
                transitions.append({"floor": nodes[u][0], "to": nodes[v][0],
                                    "mode": "stairs",
                                    "from_cell": list(nodes[u][1]),
                                    "to_cell": list(nodes[v][1]),
                                    "from_door": doors.get(u),
                                    "to_door": doors.get(v),
                                    "climb": climb})
            elif kind == "elevator":
                # car 진입·진출 두 엣지가 연속 — 한 번의 탑승으로 합침
                f_in = nodes[u][0] if u != "car" else None
                from_cell = list(nodes[u][1]) if u != "car" else None
                from_door = doors.get(u)
                j = i + 1
                f_out, to_cell, to_door = None, None, None
                if j < len(chain) and chain[j][2] == "elevator":
                    v2 = chain[j][1]
                    f_out, to_cell, to_door = (nodes[v2][0], list(nodes[v2][1]),
                                               doors.get(v2))
                    i = j
                # 엘베 내부는 궤적 레벨로 포함(사용자 지시 — 카드 장식이 아니라
                # GT에 진입·내부 대기·진출 포즈가 있어야 대기중 라벨 구간 성립):
                # 진입 = 문 앞 → 카 중심, 진출 = 도착층 카 중심 → 문 앞
                def _line(a, b):
                    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1])))
                    return [(int(round(a[0] + (b[0] - a[0]) * t)),
                             int(round(a[1] + (b[1] - a[1]) * t)))
                            for t in np.linspace(0, 1, n)]

                cell_in = cell_out = None
                if f_in is not None:
                    ez_, ex_ = np.where(self.floors[f_in].elev_mask)
                    if len(ez_):
                        cell_in = (int(ez_.mean()), int(ex_.mean()))
                        segs.append((f_in, _line(tuple(from_cell), cell_in)))
                if f_out is not None:
                    ez_, ex_ = np.where(self.floors[f_out].elev_mask)
                    if len(ez_) and to_cell is not None:
                        cell_out = (int(ez_.mean()), int(ex_.mean()))
                        segs.append((f_out, _line(cell_out, tuple(to_cell))))
                transitions.append({"floor": f_in, "to": f_out,
                                    "mode": "elevator",
                                    "from_cell": from_cell, "to_cell": to_cell,
                                    "from_door": from_door, "to_door": to_door,
                                    "wait_cell_in": (list(cell_in)
                                                     if cell_in else None),
                                    "wait_cell_out": (list(cell_out)
                                                      if cell_out else None),
                                    "wait_steps": 3})
            i += 1
        cost = dist["goal"]
        if observe:  # 걷기 비용을 실제 주행 경로 길이로 대체
            trans_cost = sum(STAIR_TRAVERSE_COST if t["mode"] == "stairs"
                             else ELEV_WAIT_COST + 3.0 for t in transitions)
            cost = walk_cost + trans_cost
        return {"success": True, "cost": cost,
                "transitions": transitions, "segments": segs,
                "verdicts": verdicts, "verdict_gates": verdict_gates}

    def _walk(self, robot, fl, src, dst, observe):
        """한 층 걷기 구간. observe=True면 관측 기반 주행, False면 전지적 최단.

        반환 (비용, 셀 경로, 불가판정 지점들, 성공 여부) 또는 None(시작 불능).
        실패여도 부분 궤적·판정 지점은 보존한다 (불가능 판정 GT).
        """
        if not observe:
            (c, cells), = self._route(fl, robot, src, [dst])
            if cells is None:
                return None
            return c, cells, ([], []), True, []
        okflag, cells, verdicts, tail = drive_expert(
            self.floors[fl], robot, self, fl, src, dst,
            belief_h=self.belief_h)
        vcells = [v[0] for v in verdicts]
        vgates = [v[1] for v in verdicts]
        return path_length(cells), cells, (vcells, vgates), okflag, tail


def path_length(cells):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(cells[:-1], cells[1:])) * GRID


def natural_path(geom, robot, cells, eps_m=0.05):
    """expert 셀 경로 → 일반 주행 경로.

    가중 중앙선 경로를 그대로 두고 Douglas-Peucker 단순화(허용 편차 eps_m)로
    격자 계단 무늬만 제거한다. 문자열 당기기류(시선 직선화)는 쓰지 말 것 —
    허용 여백의 하한선에 딱 붙어 코너를 감는 최단 폴리라인을 만들어, 비용
    가중으로 확보한 중앙선 여백을 되돌린다(두 번 실제 발생한 버그).
    """
    if len(cells) < 3:
        return cells
    eps = eps_m / GRID
    pts = np.array(cells, dtype=np.float64)
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        L = np.hypot(*ab)
        seg = pts[i + 1:j]
        if L < 1e-9:
            d = np.hypot(*(seg - a).T)
        else:
            d = np.abs(np.cross(ab, seg - a)) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[i + 1 + k] = True
            stack.append((i, i + 1 + k))
            stack.append((i + 1 + k, j))
    return [tuple(map(int, p)) for p in pts[keep]]


# ---------------------------------------------------------------- 시각화

TOPVIEW_DIR = "/tmp/mansion_topviews"  # 카드 배경 캐시 — 검수 산출물 아님(check/ 밖)


RES = 256      # 전방 RGB-D 스냅샷 렌더 해상도 (depth 규격, RGB는 팩킹 시 224)
FX = RES / 2.0  # HFOV 90° 핀홀 초점거리(px)


def yaw_deg(dx, dz):
    """월드 방향 → AI2-THOR yaw (0°=+z, 90°=+x)."""
    return math.degrees(math.atan2(dx, dz))


def update_cam(c, x, y, z, yaw, pitch=0.0):
    """서드파티 캠(1개, id 0)을 이동시키고 (RGB BGR, depth m) 반환."""
    ev = c.step(action="UpdateThirdPartyCamera", thirdPartyCameraId=0,
                position=dict(x=x, y=y, z=z),
                rotation=dict(x=pitch, y=yaw, z=0))
    rgb = cv2.cvtColor(ev.third_party_camera_frames[0], cv2.COLOR_RGB2BGR)
    return rgb, ev.third_party_depth_frames[0]


def project(px_pt, cam, yaw):
    """월드 점 → 이미지 픽셀 (pitch 0 가정). 시야 밖이면 None."""
    rel = np.array(px_pt) - np.array(cam)
    ry = math.radians(yaw)
    fwd = rel[0] * math.sin(ry) + rel[2] * math.cos(ry)
    right = rel[0] * math.cos(ry) - rel[2] * math.sin(ry)
    if fwd < 0.2:
        return None
    u = int(RES / 2 + FX * right / fwd)
    v = int(RES / 2 - FX * rel[1] / fwd)
    if 0 <= u < RES and 0 <= v < RES:
        return u, v
    return None


# ------------------------------------------------- 제안기·전역 선택 GT

HEAT_HW = 32          # 제안기 히트맵 해상도 (모델 heat 출력과 동일)
HEAT_IGNORE = 2       # 라벨 무시 센티널 (양·음 경계 1픽셀)
HEAT_MAX_M = 5.0      # 도달 가능 판정 최대 거리 (m)
HEAT_GROUND_TOL = 0.12  # 바닥 판정 허용 오차 (m)


REACH_WIN_M = 2.0 * HEAT_MAX_M  # 연결 성분 계산 창 (양성 판정 거리보다 넉넉히)
REACH_SNAP_M = 1.0              # 로봇 셀이 침식 그리드 밖일 때 최근접 스냅 한도


def reach_mask(geom, robot, x, z):
    """(x, z)에 선 로봇이 실제로 갈 수 있는 셀 — 침식 그리드의 연결 성분.

    제안기 GT의 정의역. 침식만 쓰면 벽 너머의 통과 가능 셀까지 양성이 되므로
    현재 위치와 이어진 성분만 남긴다. 연결 판정 창은 양성 판정 거리(HEAT_MAX_M)
    의 두 배 — 창이 좁으면 모퉁이를 크게 도는 경로가 창 밖으로 나갔다 들어오며
    다른 성분으로 갈려, 정작 expert가 간 지점이 도달 불가로 라벨된다.
    렌더 포즈는 스무딩·재표집을 거쳐 침식 그리드에서 살짝 벗어날 수 있으므로
    로봇 셀은 REACH_SNAP_M 안의 최근접 통과 가능 셀로 스냅한다.
    """
    wb, hb = robot_bucket(robot)
    ok = geom.passable(wb, hb)
    iz, ix = geom.to_idx(x, z)
    rad = int(REACH_WIN_M / GRID)
    z0, z1 = max(0, iz - rad), min(geom.nz, iz + rad + 1)
    x0, x1 = max(0, ix - rad), min(geom.nx, ix + rad + 1)
    sub = ok[z0:z1, x0:x1]
    out = np.zeros_like(ok)
    if not sub.any():
        return out
    cz, cx = iz - z0, ix - x0
    if not (0 <= cz < sub.shape[0] and 0 <= cx < sub.shape[1]):
        return out
    if not sub[cz, cx]:
        zz, xx = np.where(sub)
        d = (zz - cz) ** 2 + (xx - cx) ** 2
        k = int(np.argmin(d))
        if d[k] * GRID ** 2 > REACH_SNAP_M ** 2:
            return out
        cz, cx = int(zz[k]), int(xx[k])
    lab, _ = ndimage.label(sub)
    out[z0:z1, x0:x1] = lab == lab[cz, cx]
    return out


def heat_gt(geom, reach, x, z, yaw, cam_y, depth):
    """제안기 GT 히트맵 (HEAT_HW²): 도달 가능 자유 공간의 픽셀 래스터.

    픽셀마다 렌더 depth로 3D 지점을 복원해(추론 때 모델이 하는 리프트와 동일
    규약) 그 지점이 바닥이고 reach 안이며 HEAT_MAX_M 이내면 양성. 벽·가구를
    맞은 픽셀은 지점이 reach 밖이라 자연히 음성이 된다. 양·음 경계 1픽셀은
    HEAT_IGNORE — 픽셀 양자화가 만드는 라벨 잡음을 학습에서 뺀다.

    바닥 높이는 y=0 가정이라 계단 등반 프레임은 전부 음성이 된다(계단실은
    free_mask에서 제외돼 reach도 비어 있으므로 결과가 일관됨).
    """
    k = RES // HEAT_HW
    ij = np.arange(HEAT_HW)
    # 히트맵 픽셀 중심에 해당하는 렌더 픽셀 인덱스
    px = (ij * k + k // 2).astype(np.int32)
    uu, vv = np.meshgrid(px, px)  # (S, S)
    d = np.asarray(depth, dtype=np.float32)[vv, uu]
    ry = math.radians(yaw)
    right = (uu + 0.5 - RES / 2) * d / FX
    rel_y = -(vv + 0.5 - RES / 2) * d / FX
    wx = x + d * math.sin(ry) + right * math.cos(ry)
    wz = z + d * math.cos(ry) - right * math.sin(ry)
    on_ground = np.abs(rel_y + cam_y) <= HEAT_GROUND_TOL
    near = (d > 0.2) & (np.hypot(wx - x, wz - z) <= HEAT_MAX_M)
    iz = np.clip(((wz - geom.z0) / GRID).astype(np.int32), 0, geom.nz - 1)
    ix = np.clip(((wx - geom.x0) / GRID).astype(np.int32), 0, geom.nx - 1)
    pos = on_ground & near & reach[iz, ix]
    out = pos.astype(np.uint8)
    # 경계 1픽셀 = 팽창과 침식이 갈리는 곳
    p8 = pos.astype(np.uint8)
    kern = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(p8, kern) != cv2.erode(p8, kern)
    out[edge] = HEAT_IGNORE
    return out


def draw_heading(img, x, z, yaw, cam_y, wx, wz, wh_deg, arrow_m=0.8,
                 color=(0, 255, 255)):
    """waypoint에서 도착 heading 방향으로 화살표 (렌더 해상도 픽셀 기준).

    도착 방향은 후보의 정체성이 아니라 보조 출력이지만, 카드에서는 "여기로
    가서 저쪽을 향한다"가 한눈에 보여야 검수가 된다.
    """
    a = project((wx, 0.0, wz), (x, cam_y, z), yaw)
    tx = wx + arrow_m * math.sin(math.radians(wh_deg))
    tz = wz + arrow_m * math.cos(math.radians(wh_deg))
    b = project((tx, 0.0, tz), (x, cam_y, z), yaw)
    if a is None or b is None or (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < 9:
        return False
    cv2.arrowedLine(img, a, b, color, 2, cv2.LINE_AA, tipLength=0.35)
    return True


def heat_pixel(x, z, yaw, cam_y, wx, wz, wy=0.0):
    """월드 바닥 점 → 히트맵 픽셀 (u, v). 시야 밖이면 None."""
    uv = project((wx, wy, wz), (x, cam_y, z), yaw)
    if uv is None:
        return None
    k = RES // HEAT_HW
    return uv[0] // k, uv[1] // k




GSEL = {"forward": 0, "node": 1, "stop": 2}


def global_head_gt(pose, t):
    """홉 t의 도착 heading 정답 = pose[t+1]의 실제 heading (deg). 없으면 None.

    heading 예측은 선택기 몫이고(제안기 dense heading은 폐기), 정답은 expert가
    그 홉 끝에서 실제로 향한 방향 그대로다 — 기하 대리물도, 안 가본 점의
    "정답 방향"이라는 억지 개념도 필요 없다.
    """
    if t >= len(pose) - 1:
        return None
    return float(pose[t + 1][3])


def global_gt(pose, t, heat):
    """홉 t의 전역 선택 정답 = (종류, 히트맵 픽셀).

    forward — 다음 포즈가 현재 뷰에서 도달 가능 픽셀로 잡히는 홉. 그 픽셀에서
              뜬 고스트 후보가 정답이 된다. 경계 무시 픽셀(HEAT_IGNORE)도
              인정한다 — expert가 실제로 간 지점이라는 독립 증거가 있는데
              양자화 경계에 걸렸다고 방향 전환으로 라벨하면 GT가 서로 어긋난다.
    node    — 화면 밖이거나 층이 바뀌는 홉(방향 전환·백트래킹·층 전환).
              정답 노드는 학습 시 포즈열과 기억 그래프로 해석한다.
    stop    — 마지막 프레임(도달·포기 공통). 도착 여부 구분은 레지스트리 검증.

    정답의 월드 좌표는 pose[t+1]이 그대로 갖고 있으므로 별도 저장하지 않는다.
    """
    if t >= len(pose) - 1:
        return GSEL["stop"], (-1, -1)
    fl, x, z, yaw, cam_y = (float(pose[t][0]), float(pose[t][1]),
                            float(pose[t][2]), float(pose[t][3]),
                            float(pose[t][4]))
    nxt = pose[t + 1]
    if int(nxt[0]) != int(fl):
        return GSEL["node"], (-1, -1)
    uv = heat_pixel(x, z, yaw, cam_y, float(nxt[1]), float(nxt[2]))
    if uv is None or heat[uv[1], uv[0]] == 0:
        return GSEL["node"], (-1, -1)
    return GSEL["forward"], uv


def launch_controller(width=1024, height=1024, render=False, **kwargs):
    """MANSION 씬 로딩 AI2-THOR 기동. 층 로드는 load_floor_scene(CreateHouse).

    두 모드 (2026-07-25 검증):
      render=True  — **기본 CloudRendering 빌드**(Vulkan GPU) + 에셋 훅.
        렌더 전용: 카메라 갱신 ~350ms/장(패치 빌드 대비 40배), 가구·계단·엘베
        전부 렌더됨(훅이 patch 에셋 small_stair 등도 등록). perspective
        AddThirdPartyCamera/Update 정상 — 이전 타임아웃은 orthographic 한정,
        단 카메라 3개 이상 추가는 패치 빌드에서 불안정했으므로 1개를 Update로.
      render=False — 패치 빌드(LOCAL_AI2THOR_PATH). 스킬 API(UseStairs/
        UseElevator)·엘베 문 개폐 등 상호작용용. Xvfb OpenGL(llvmpipe)라
        스텝당 2초+로 느림 — 렌더 용도 금지.
    공통: 훅 없는 로딩은 "Asset not in Database"로 가구 없는 빈 방이 됨.
    잔존 프로세스 정리는 pkill -x thor-Linux64-local / -f "[t]hor-CloudRendering".
    """
    from ai2thor.controller import Controller
    from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
    from mansion_api.config import LOCAL_AI2THOR_PATH, OBJATHOR_ASSETS_DIR

    opts = dict(
        scene="Procedural", width=width, height=height,
        makeAgentsVisible=False, visibilityScheme="Distance",
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=OBJATHOR_ASSETS_DIR, asset_symlink=True,
            verbose=False))
    if render:
        from ai2thor.platform import CloudRendering

        opts.update(platform=CloudRendering, fastActionEmit=True,
                    renderDepthImage=True,
                    # Vulkan 렌더 GPU 지정 — 없으면 전 렌더러가 GPU 0에 몰림
                    # (실측: MANSION_GPU는 torch 플래너에만 적용되던 구멍)
                    gpu_device=int(os.environ.get("MANSION_GPU", 0)))
    else:
        opts.update(local_executable_path=LOCAL_AI2THOR_PATH)
    opts.update(kwargs)
    return Controller(**opts)


ELEV_DOOR_ASSET = "elevator_doors_custom"


def _ensure_elevator_asset():
    """브러시드 메탈 엘리베이터 문 런타임 에셋 생성 (없으면 1회).

    MANSION의 엘베 "문"은 손잡이 달린 주택용 더블도어라 시각적으로 엘베로
    인식 불가(사용자 지적, THOR 내장 문 10종 스윕 결과 전부 주택풍).
    objathor 런타임 에셋 형식(json+obj+PBR 텍스처)으로 슬라이딩 도어 외형을
    만들어 에셋 디렉토리에 추가한다 — 계단(small_stair)과 같은 로딩 경로.
    """
    from mansion_api.config import OBJATHOR_ASSETS_DIR

    adir = os.path.join(OBJATHOR_ASSETS_DIR, ELEV_DOOR_ASSET)
    if os.path.exists(os.path.join(adir, f"{ELEV_DOOR_ASSET}.json")):
        return
    os.makedirs(adir, exist_ok=True)
    # 텍스처: 브러시드 스틸 albedo(세로 스트릭 노이즈), 평평한 normal,
    # 무발광 emission, 고 metallic
    rng = np.random.default_rng(0)
    streaks = rng.normal(0, 6, (1, 512)).repeat(512, 0)
    albedo = np.clip(205 + streaks + rng.normal(0, 2, (512, 512)), 180, 230)
    cv2.imwrite(os.path.join(adir, "albedo.jpg"),
                cv2.merge([albedo.astype(np.uint8)] * 3))
    cv2.imwrite(os.path.join(adir, "normal.jpg"),
                np.full((64, 64, 3), (255, 128, 128), np.uint8))
    # 문 오목부는 조명 사각이라 무발광이면 앰비언트 남색으로 묻힘(실제 발생)
    # — 약한 자체발광으로 스틸 톤 유지
    cv2.imwrite(os.path.join(adir, "emission.jpg"),
                np.full((64, 64, 3), 55, np.uint8))
    # 고 metallic은 환경 반사가 없는 실내에서 검게 렌더됨(실제 발생) — 저 metallic
    cv2.imwrite(os.path.join(adir, "metallic_smoothness.jpg"),
                np.full((64, 64, 3), 60, np.uint8))

    def box(cx, cy, cz, sx, sy, sz):
        """중심·크기 → (정점, 삼각형, 법선, uv) 완전 전개."""
        x0, x1 = cx - sx / 2, cx + sx / 2
        y0, y1 = cy - sy / 2, cy + sy / 2
        z0, z1 = cz - sz / 2, cz + sz / 2
        faces = [  # (4코너, 법선)
            ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1)),
            ([(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)], (0, 0, -1)),
            ([(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)], (-1, 0, 0)),
            ([(x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)], (1, 0, 0)),
            ([(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)], (0, 1, 0)),
            ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0)),
        ]
        V, T, N, UV = [], [], [], []
        for corners, nrm in faces:
            base = len(V)
            for k, (px, py, pz) in enumerate(corners):
                V.append({"x": px, "y": py, "z": pz})
                N.append({"x": nrm[0], "y": nrm[1], "z": nrm[2]})
                UV.append({"x": (0.0, 1.0, 1.0, 0.0)[k],
                           "y": (0.0, 0.0, 1.0, 1.0)[k]})
            # 양면 방출 — 감김 방향이 면마다 엇갈리면 일부 면이 뒷면만
            # 보여 검게 렌더됨(실제 발생). 문짝은 tri 수가 작아 부담 없음
            T += [base, base + 1, base + 2, base, base + 2, base + 3,
                  base, base + 2, base + 1, base, base + 3, base + 2]
        return V, T, N, UV

    # 두 문짝(1.42×2.05×0.06, 중앙 2cm 틈) + 상단 인방 — 원점=중심
    V, T, N, UV = [], [], [], []
    for cx, sx, sy, cy in ((-0.36, 0.69, 2.05, 0.0), (0.36, 0.69, 2.05, 0.0),
                           (0.0, 1.46, 0.12, 1.085)):
        v, t, n, uv = box(cx, cy, 0.0, sx, sy, 0.06)
        T += [i + len(V) for i in t]
        V += v
        N += n
        UV += uv
    col_v, col_t, _, _ = box(0, 0, 0, 1.46, 2.29, 0.06)
    asset = {
        "action": "CreateObjectPrefab", "name": ELEV_DOOR_ASSET,
        "receptacleCandidate": False,
        "albedoTexturePath": "albedo.jpg", "normalTexturePath": "normal.jpg",
        "emissionTexturePath": "emission.jpg",
        "vertices": V, "triangles": T, "normals": N, "uvs": UV,
        "visibilityPoints": [{"x": x, "y": y, "z": 0.04}
                             for x in (-0.5, 0, 0.5) for y in (-0.8, 0, 0.8)],
        "physicalProperties": {"mass": 1.0, "drag": 0, "angularDrag": 0.05,
                               "useGravity": True, "isKinematic": False},
        "yRotOffset": 0.0,
        "colliders": [{"vertices": col_v,
                       "triangles": col_t}],
        # annotations는 검증된 패널 에셋 값 그대로 — "Static" 등 임의 enum은
        # 프리팹 생성이 조용히 실패해 비주얼만 빠짐(실제 발생 의심 지점)
        "annotations": {"objectType": "Undefined",
                        "primaryProperty": "CanPickup",
                        "secondaryProperties": ["Receptacle"]},
    }
    with open(os.path.join(adir, f"{ELEV_DOOR_ASSET}.json"), "w") as fh:
        json.dump(asset, fh)
    print(f"엘리베이터 문 에셋 생성: {adir}")


def load_floor_scene(controller, scene):
    """기동된 컨트롤러에 층 씬(JSON dict) 로드.

    openable 문은 openness=1로 열어서 로드 — 기하 파이프라인이 문을 전부
    개구부로 취급하므로 렌더도 열려 있어야 관측-기하가 정합한다(사용자 지적:
    닫힌 문짝을 통과하는 관측이 생김). exterior의 열 수 없는 문(3개 조사됨)은
    바깥이라 경로와 무관 — 그대로 둔다. 원본 scene dict는 변형하지 않는다.
    """
    _ensure_elevator_asset()
    house = dict(scene)
    doors, extra_objs = [], []
    for d in scene.get("doors", []):
        rooms = (d.get("room0", "") + d.get("room1", "")).lower()
        if "elev" in rooms and d.get("doorSegment"):
            # 엘베 문: 주택용 문짝 → 개방 프레임 + 커스텀 메탈 슬라이딩 도어
            # (닫힌 주택 문은 엘베로 인식 불가 — 사용자 지적)
            (ax, az), (bx, bz) = d["doorSegment"]
            mx, mz = (ax + bx) / 2, (az + bz) / 2
            yaw = math.degrees(math.atan2(bx - ax, bz - az)) + 90.0
            doors.append(dict(d, assetId="Doorframe_Double_9"))
            fp = [[(mx - 0.75) * 100, (mz - 0.05) * 100],
                  [(mx + 0.75) * 100, (mz - 0.05) * 100],
                  [(mx + 0.75) * 100, (mz + 0.05) * 100],
                  [(mx - 0.75) * 100, (mz + 0.05) * 100]]
            extra_objs.append({
                "assetId": ELEV_DOOR_ASSET,
                "id": f"elevator_doors|{d['id']}",
                "object_name": f"elevator_doors_{d['id']}",
                "kinematic": True, "layer": "Procedural0",
                "position": {"x": mx, "y": 1.145, "z": mz},
                "rotation": {"x": 0, "y": yaw, "z": 0},
                "roomId": d.get("room1", ""), "vertices": fp,
            })
        elif d.get("openable"):
            # 일반 문은 문짝 자체를 제거(프레임으로 교체) — openness=1로 열면
            # 젖혀진 문짝이 문 옆 공간을 차지해 궤적이 문짝을 뚫는 모순이 생김
            # (기하 파이프라인에는 문짝이 없음, 사용자 지적). Doorway_*와
            # Doorframe_*는 1~10 대응 변형이 있어 크기·스타일 유지됨.
            doors.append(dict(d, assetId=d["assetId"].replace("Doorway",
                                                              "Doorframe")))
        else:
            doors.append(d)  # exterior 잠긴 문 등은 그대로
    house["doors"] = doors
    house["objects"] = list(scene.get("objects", [])) + extra_objs
    controller.reset(scene="Procedural")
    controller.step(action="CreateHouse", house=house, raise_for_failure=True)
    controller.step(action="Pass", raise_for_failure=True)


def _render_topviews(building_dir, floors_missing, size=1024):
    """빠진 층의 정사영 탑뷰 + 카메라 메타를 TOPVIEW_DIR에 렌더 (lazy)."""
    os.makedirs(TOPVIEW_DIR, exist_ok=True)
    bname = os.path.basename(building_dir.rstrip("/")).split("#")[0]
    controller = launch_controller(width=size, height=size)
    try:
        for no in floors_missing:
            with open(os.path.join(building_dir, f"floor_{no}.json")) as fh:
                scene = json.load(fh)
            load_floor_scene(controller, scene)
            ev = controller.step(action="GetMapViewCameraProperties")
            cam = ev.metadata["actionReturn"]
            # 패치 빌드는 ThirdPartyCameraTemplate이 깨져(AddThirdPartyCamera
            # 타임아웃) 메인 카메라를 맵뷰로 전환해 캡처한다
            ev = controller.step(action="ToggleMapView")
            frame = ev.frame
            controller.step(action="ToggleMapView")
            base = os.path.join(TOPVIEW_DIR, f"{bname}_F{no}")
            cv2.imwrite(base + ".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            with open(base + ".json", "w") as fh:
                json.dump({"position": cam["position"],
                           "orthographicSize": cam["orthographicSize"],
                           "width": size, "height": size}, fh)
            print(f"탑뷰 렌더: {base}.png")
    finally:
        controller.stop()


def sim_view(geom, building_dir):
    """시뮬레이션 탑뷰 배경 + 월드→픽셀 변환 (BGR 이미지, w2p).

    정사영 렌더(+카메라 메타)를 사용한다 — 카메라 위치와 orthographicSize를
    알므로 변환이 정확하다. 배포 렌더(floor_N.png)는 원근 카메라라 가장자리에서
    바닥 좌표가 방사형으로 밀려 선형 매핑으로는 오버레이가 안 맞는다(실제 발생).
    렌더가 없으면 그 자리에서 만든다(빌딩당 1회, 이후 캐시).
    """
    bname = os.path.basename(building_dir.rstrip("/")).split("#")[0]
    base = os.path.join(TOPVIEW_DIR, f"{bname}_F{geom.floor_no}")
    if not os.path.exists(base + ".png"):
        _render_topviews(building_dir, [geom.floor_no])
    img = cv2.imread(base + ".png")
    with open(base + ".json") as fh:
        cam = json.load(fh)
    px, pz = cam["position"]["x"], cam["position"]["z"]
    H, W = img.shape[:2]
    scale = H / (2.0 * cam["orthographicSize"])

    def w2p(x, z):
        return (int((x - px) * scale + W / 2),
                int(H / 2 - (z - pz) * scale))

    # 플래너가 쓰는 벽 래스터(개구부 포함)를 명시 오버레이 — 정사영 뷰에서는
    # 벽 단면이 몇 px라 안 보여서, 경로가 벽을 지나는지 카드로 검증 불가하다는
    # 문제(사용자 지적)를 해결. 흰 선 + 검은 테두리로 배경 무관하게 보이게.
    iz, ix = np.where(geom.wall)
    wu = ((geom.x0 + (ix + 0.5) * GRID - px) * scale + W / 2).astype(np.int64)
    wv = (H / 2 - (geom.z0 + (iz + 0.5) * GRID - pz) * scale).astype(np.int64)
    ok = (wu >= 0) & (wu < W) & (wv >= 0) & (wv < H)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[wv[ok], wu[ok]] = 1
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    border = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    img[border > 0] = (0, 0, 0)
    img[mask > 0] = (255, 255, 255)

    return img, w2p


def cells_to_px(geom, cells, w2p):
    return [w2p(*geom.to_world(iz, ix)) for iz, ix in cells]


def draw_climb(img, geom, w2p, transitions, color):
    """이 층의 계단 등반 구간을 높이 그라데이션(어두움→밝음)+화살표로 그림.

    왕복 계단은 2D 투영이 고리처럼 보여 오독됨(사용자 지적) — 높이·방향
    부호화로 '올라가는 중'임을 표시한다.
    """
    for t in transitions:
        climb = t.get("climb")
        if not climb or min(t["floor"], t["to"]) != geom.floor_no:
            continue
        ys = [c[2] for c in climb]
        y0, y1 = min(ys), max(ys)
        pts = [w2p(*geom.to_world(c[0], c[1])) for c in climb]
        for (a, b, ya) in zip(pts[:-1], pts[1:], ys[:-1]):
            f = (ya - y0) / max(y1 - y0, 1e-6)  # 0=바닥, 1=꼭대기
            col = tuple(int(ch * (0.35 + 0.65 * f) + 200 * f * 0.3)
                        for ch in color)
            cv2.line(img, a, b, tuple(min(255, c) for c in col), 4,
                     cv2.LINE_AA)
        for k in (len(pts) // 3, 2 * len(pts) // 3):
            if 0 < k < len(pts):
                cv2.arrowedLine(img, pts[k - 1], pts[k], (255, 255, 255), 2,
                                tipLength=2.0)


def draw_transitions(img, geom, w2p, transitions, color):
    """이 층에서의 전환 표시: 문 중앙 통과 선분 + S(계단)/E(엘베) 마커."""
    for t in transitions:
        letter = "S" if t["mode"] == "stairs" else "E"
        for ckey, dkey in (("from_cell", "from_door"), ("to_cell", "to_door")):
            fl = t["floor"] if ckey == "from_cell" else t["to"]
            cell = t.get(ckey)
            if fl != geom.floor_no or not cell:
                continue
            u, v = w2p(*geom.to_world(*cell))
            door = t.get(dkey)
            if door:
                du, dv = w2p(*door)
                cv2.line(img, (u, v), (du, dv), color, 3)  # 문 중앙 통과
                u, v = du, dv
            cv2.circle(img, (u, v), 12, color, 2)
            cv2.putText(img, letter, (u - 6, v + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def topdown(geom, h=H_MIN):
    """층 기하 탑다운 (BGR). 회색=자유, 남색=바닥 장애물, 주황=낮은 통로."""
    img = np.zeros((geom.nz, geom.nx, 3), dtype=np.uint8)
    img[geom.inside] = (70, 70, 70)
    free = geom.free_mask(h)
    img[free] = (120, 120, 120)
    img[geom.inside & ~free & ~geom.wall] = (90, 45, 35)
    low = free & (geom.clearance < 1.6)
    img[low] = (30, 140, 200)
    img[geom.wall & geom.inside] = (25, 25, 25)
    img[geom.stair_mask] = (90, 190, 90)
    img[geom.elev_mask] = (180, 130, 60)
    return img


def draw_path(img, cells, color):
    for (iz, ix), (jz, jx) in zip(cells[:-1], cells[1:]):
        cv2.line(img, (ix, iz), (jx, jz), color, 2)
    return img


def flip_save(path, img, scale=1):
    """z축이 위로 가게 뒤집어 저장."""
    out = cv2.flip(img, 0)
    if scale != 1:
        out = cv2.resize(out, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, out)


# ---------------------------------------------------------------- 갈림 판정
# (본 생성 에피소드 생성기의 내부 단계용 — 독립 채굴 스크립트는 두지 않는다.
#  파일럿 검증 결론은 code.md에 기록: 호텔 1채 11/12 에피소드 갈림)

DIVERGENCE_IOU = 0.70  # 경로 겹침율 기준 (plan 확정 초기값)
_COARSE = 12           # 겹침율 격자 = 12셀 × 2.5 cm = 0.3 m


def route_modes(res):
    return tuple(t["mode"] for t in res.get("transitions", []))


def _coarse_cells(res):
    out = set()
    for fl, cells in res.get("segments", []):
        for iz, ix in cells:
            out.add((fl, iz // _COARSE, ix // _COARSE))
    return out


def divergent(ra, rb):
    """두 expert 결과의 갈림 원인 ('reach'|'mode'|'overlap'|None)."""
    if ra.get("success") != rb.get("success"):
        return "reach"
    if not ra.get("success"):
        return None
    if route_modes(ra) != route_modes(rb):
        return "mode"
    A, B = _coarse_cells(ra), _coarse_cells(rb)
    if len(A & B) / max(1, len(A | B)) < DIVERGENCE_IOU:
        return "overlap"
    return None


def representative_buckets(robots, nav, n=12):
    """대표 버킷 표집: (w,h,stairs_ok) 버킷을 계단 가부별 폭 스펙트럼 균등
    + 양극단으로 n개 선택. 전 로봇 전수 프로브는 불필요(사용자 확정) —
    프로브로 에피소드를 선별하고, 선별분만 전체 페어를 생성한다."""
    from collections import defaultdict

    by_bucket = defaultdict(list)
    for r in robots:
        by_bucket[(*nav._bucket(r), r["stairs_ok"])].append(r)
    if n and n < len(by_bucket):
        picked = set()
        for sok in (True, False):
            ks = sorted(k for k in by_bucket if k[2] == sok)
            take = sorted({int(round(i)) for i in
                           np.linspace(0, len(ks) - 1, max(2, n // 2))})
            picked |= {ks[i] for i in take}
        by_bucket = {k: v for k, v in by_bucket.items() if k in picked}
    return dict(by_bucket)

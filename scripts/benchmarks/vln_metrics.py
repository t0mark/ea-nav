"""VLN 표준 지표 — SR/SPL/NE/TL/nDTW (다층 규약 포함, 벤치마크 공용).

다층 규약(문서화된 확장 — MP3D류 단층 지표의 자연 확장):
  · 점 거리 d(p,q) = 유클리드(동일층) / 유클리드 + FLOOR_PEN×층차(이층)
  · SR = 최종 위치가 goal과 동일층 & 3m 이내
  · nDTW = exp(-DTW/(|ref|·3m)) (Ilharco et al. 규약, 경로는 0.5m 재표집)
"""
import math

import numpy as np

SUCCESS_M = 3.0
FLOOR_PEN = 10.0


def pdist(p, q):
    """p,q = (floor, x, z)."""
    d = math.hypot(p[1] - q[1], p[2] - q[2])
    return d + FLOOR_PEN * abs(int(p[0]) - int(q[0]))


def path_length(path):
    return sum(pdist(path[i], path[i + 1]) for i in range(len(path) - 1))


def resample(path, step=0.5):
    """폴리라인 0.5m 등간격 재표집 (층 전환 지점은 그대로 유지)."""
    if len(path) < 2:
        return list(path)
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        if int(a[0]) != int(b[0]):
            out.append(b)
            continue
        seg = math.hypot(b[1] - a[1], b[2] - a[2])
        n = max(1, int(seg / step))
        for k in range(1, n + 1):
            t = k / n
            out.append((a[0], a[1] + (b[1] - a[1]) * t,
                        a[2] + (b[2] - a[2]) * t))
    return out


def dtw(pred, ref):
    P, R = len(pred), len(ref)
    D = np.full((P + 1, R + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, P + 1):
        for j in range(1, R + 1):
            c = pdist(pred[i - 1], ref[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[P, R])


def episode_metrics(pred_path, ref_path, goal):
    """pred/ref = [(floor,x,z), ...], goal = (floor,x,z) → 지표 dict.

    SPL = S·L_ref/max(L_ref, L_pred) (Anderson et al.),
    L_ref = 참조 경로 길이(geodesic 대용 — expert 경로가 곧 GT 최적).
    """
    pred = resample(pred_path)
    ref = resample(ref_path)
    ne = pdist(pred[-1], goal)
    sr = float(int(pred[-1][0]) == int(goal[0])
               and math.hypot(pred[-1][1] - goal[1],
                              pred[-1][2] - goal[2]) <= SUCCESS_M)
    tl = path_length(pred_path)
    lref = max(path_length(ref_path), 1e-6)
    spl = sr * lref / max(lref, tl)
    nd = math.exp(-dtw(pred, ref) / (len(ref) * SUCCESS_M))
    return {"SR": sr, "SPL": spl, "NE": ne, "TL": tl, "nDTW": nd,
            "L_ref": lref}


class Agg:
    def __init__(self):
        self.rows = []

    def add(self, m, **tags):
        self.rows.append({**m, **tags})

    def table(self, by=None):
        keys = ("SR", "SPL", "NE", "TL", "nDTW")
        groups = {}
        for r in self.rows:
            g = r.get(by, "전체") if by else "전체"
            groups.setdefault(g, []).append(r)
        out = {}
        for g, rs in sorted(groups.items()):
            # nanmean — 참조 경로가 없는 케이스(결합형)는 SPL·nDTW가 nan이다.
            # 0으로 채우면 없는 성능을 만든 셈이 되므로 평균에서 뺀다.
            out[g] = {}
            for k in keys:
                v = np.array([r.get(k, np.nan) for r in rs], dtype=float)
                out[g][k] = (float(np.nanmean(v))
                             if np.isfinite(v).any() else float("nan"))
            out[g]["n"] = len(rs)
        return out

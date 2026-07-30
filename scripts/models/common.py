"""학습·평가 스크립트 공용 유틸 — 경로 주입, 결정적 스플릿, 동결, 손실·지표.

스플릿 규약(학습·평가 스크립트가 반드시 공유):
  · 게이트: md5(f"{tag}:{gate_idx}") 해시 < val_frac → 검증 (통과성)
  · 궤적:  md5(f"{tag}:{traj_npz명}") 해시 < val_frac → 검증 (제안기·전역)
  해시 기반이라 스크립트·세션 간 재현 — 셔플 시드와 무관.

손실 규약:
  · 제안기 = heat_loss (히트맵 BCE, IGNORE 픽셀 제외) 단일
  · 전역 선택 = 후보 CE + heading_loss(1 − cos Δθ)
"""
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "models"))


def pick_gpu(gpu):
    """GPU 0~3만 사용(프로젝트 규칙). CUDA_VISIBLE_DEVICES로 고정."""
    assert 0 <= int(gpu) <= 3, "GPU는 0~3만"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def _hash01(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def is_val_gate(tag, gate_idx, val_frac=0.1):
    return _hash01(f"{tag}:g{gate_idx}") < val_frac


def is_val_traj(tag, traj_name, val_frac=0.1):
    return _hash01(f"{tag}:{traj_name}") < val_frac


def freeze(module, on=True):
    for p in module.parameters():
        p.requires_grad_(not on)


def count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# 히트맵 라벨 도메인 — 데이터 생성부(common.HEAT_IGNORE)와 같은 값 유지
HEAT_IGNORE = 2


def heat_loss(logits, target, pos_weight=None):
    """제안기 히트맵 BCE — IGNORE(경계 1픽셀) 제외.

    logits (B,1,S,S) 또는 (B,S,S), target (B,S,S) uint8 {0,1,HEAT_IGNORE}.
    양성이 5~20%라 클래스 불균형이 크지 않지만, pos_weight를 주면 그대로
    BCE에 전달한다. 유효 픽셀이 없으면 0을 반환(기울기 없음).
    """
    import torch
    lo = logits.squeeze(1) if logits.dim() == 4 else logits
    tgt = target.to(lo.dtype)
    keep = target != HEAT_IGNORE
    if not bool(keep.any()):
        return lo.sum() * 0.0
    pw = (torch.as_tensor(pos_weight, device=lo.device, dtype=lo.dtype)
          if pos_weight is not None else None)
    per = torch.nn.functional.binary_cross_entropy_with_logits(
        lo, tgt.clamp(max=1.0), reduction="none", pos_weight=pw)
    return (per * keep).sum() / keep.sum()


def heat_metrics(logits, target, thresh=0.5):
    """히트맵 지표 — IGNORE 제외 픽셀에서 **프레임별로 계산해 평균**.

    프레임 평균으로 고정한 이유는 eval.py 보고와 같은 정의를 쓰기 위해서다
    (배치 풀링과 프레임 평균은 값이 달라 같은 이름으로 섞으면 안 된다).
    반환 dict: prec·recall·iou(임계 thresh), auc(rank 기반),
    pos_rate(GT 양성 비율), peak_hit(최대 로짓 픽셀이 양성인 비율).
    """
    import torch
    lo = logits.squeeze(1) if logits.dim() == 4 else logits
    # 지표별로 분모를 따로 센다 — AUC는 단일 클래스 프레임(계단처럼 전부
    # 음성)에서 정의되지 않아 건너뛰는데, 분모를 전체 프레임으로 쓰면
    # 그만큼 희석돼 실제보다 낮게 보고된다
    acc, cnt = {}, {}
    for i in range(lo.shape[0]):
        keep = target[i] != HEAT_IGNORE
        if not bool(keep.any()):
            continue
        s = lo[i][keep].float()
        y = (target[i][keep] > 0).float()
        p = (s > 0 if thresh == 0.5
             else torch.sigmoid(s) > thresh).float()
        tp = float((p * y).sum())
        fp = float((p * (1 - y)).sum())
        fn = float(((1 - p) * y).sum())
        peak = int(lo[i].flatten().argmax())
        m = {"prec": tp / max(tp + fp, 1e-6),
             "recall": tp / max(tp + fn, 1e-6),
             "iou": tp / max(tp + fp + fn, 1e-6),
             "auc": auc(s, y),
             "pos_rate": float(y.mean()),
             "peak_hit": float(target[i].flatten()[peak] == 1)}
        for k, v in m.items():
            if v == v:                       # NaN(단일 클래스 AUC) 제외
                acc[k] = acc.get(k, 0.0) + v
                cnt[k] = cnt.get(k, 0) + 1
    return {k: v / cnt[k] for k, v in acc.items()}


def heading_loss(pred_sincos, gt_deg):
    """도착 heading 손실 = 1 − cos(Δθ).

    pred_sincos (N,2) 단위벡터(sin, cos), gt_deg (N,) 월드 heading(도).
    선택기가 정답 후보 위치에서 낸 값에만 건다.
    """
    import math

    import torch
    r = torch.as_tensor(gt_deg, dtype=pred_sincos.dtype,
                        device=pred_sincos.device) * (math.pi / 180.0)
    return (1.0 - (pred_sincos[:, 0] * torch.sin(r)
                   + pred_sincos[:, 1] * torch.cos(r))).mean()


def auc(scores, labels):
    """이진 ROC-AUC (rank 기반, 의존성 없음). labels: 0/1 텐서."""
    import torch
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(len(scores), dtype=torch.float,
                                device=scores.device)
    pos = labels.bool()
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos - 1) / 2)
                 / (n_pos * n_neg))


class Meter:
    def __init__(self):
        self.s, self.n = 0.0, 0

    def add(self, v, k=1):
        self.s += float(v) * k
        self.n += k

    @property
    def avg(self):
        return self.s / max(self.n, 1)


def save_ckpt(path, **modules):
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: (v.state_dict() if hasattr(v, "state_dict") else v)
                for k, v in modules.items()}, path)


def load_into(path, **modules):
    import torch

    import weights as lp
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k, m in modules.items():
        # Phase 7 zero-init 신규 키만 구 ckpt 결손 허용, 그 외 불일치는 오류
        lp.load_compat(m, ck[k])
    return ck

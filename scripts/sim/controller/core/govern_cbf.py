from __future__ import annotations

import torch

def project_interval(nominal: torch.Tensor, lo: torch.Tensor,
                     hi: torch.Tensor) -> torch.Tensor:
    """스칼라 barrier 구간 제약의 닫힌 해 CBF-QP 투영.

    min (u - nominal)^2  s.t.  lo <= u <= hi 의 해는 clamp 그 자체 (KKT를 눈으로 풀면
    제약이 위반될 때만 경계로 옮기는 것과 동일) — 이 문제 스케일(barrier 1-2개)에서는
    cvxpy 등 범용 QP 솔버 없이 batched torch 클램프로 충분하다.
    """
    return torch.clamp(nominal, lo, hi)

def project_halfplane(u_ref: torch.Tensor, a: torch.Tensor,
                      b: torch.Tensor) -> torch.Tensor:
    """barrier가 제어에 대해 아핀(affine)인 일반형 CBF-QP의 닫힌 해.

    제약: a . u + b >= 0 (a: (..., nu), b: (...,)).
    min ||u - u_ref||^2 s.t. a.u + b >= 0 의 해는, 위반 시에만 제약 경계로의
    직교 투영이다 — u_ref가 이미 안전하면 그대로 반환한다.
    """
    margin = (a * u_ref).sum(dim=-1) + b
    violated = margin < 0.0
    a_sq = (a * a).sum(dim=-1).clamp_min(1e-9)
    correction = (-margin / a_sq).unsqueeze(-1) * a
    return torch.where(violated.unsqueeze(-1), u_ref + correction, u_ref)

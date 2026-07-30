"""통과성 스코어 헤드 g(feature, z)와 FiLM 변조.

TraversabilityHead: 관측 feature 하나를 통과성 로짓 스칼라로 사상한다.
한 인스턴스를 네 곳이 공유한다 — 입력 feature 차원만 같으면(768) 후보의
종류는 가리지 않는다:
  ① 기억 스냅샷 랭킹·불가능 판정 (GraphPlanning.rank_snapshots)
  ② 제안기 픽셀 후보 바이어스 (WaypointNet, feature 맵 (B,S,S,768))
  ③ 전역 선택기의 고스트·노드 후보 차단/바이어스
     (GraphPlanning.forward_navigation, (B,N,768))
  ④ 통과성 라벨로 직접 학습 (사전학습 후 위 셋에 그대로 합류)
feat 단독 사영 경로를 유지해, z 기여가 0이어도 기하만의 판단은 학습된다.
후보 로짓에 얹는 공통 규약은 bias_logits()로 통일한다.

FiLM: z 조건 아핀 변조(zero-init 시작). 주입 방식 ablation용.
"""

import torch
import torch.nn as nn


class TraversabilityHead(nn.Module):
    def __init__(self, feat_dim=768, z_dim=128, hidden=256):
        super().__init__()
        self.f_proj = nn.Linear(feat_dim, hidden)
        self.z_proj = nn.Linear(z_dim, hidden)
        self.mlp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat, z):
        """feat (..., feat_dim), z (B, z_dim) → 통과성 로짓 (...,).

        z는 feat의 배치 이후 중간 차원에 브로드캐스트된다.
        """
        f = self.f_proj(feat)
        zz = self.z_proj(z)
        shape = [z.shape[0]] + [1] * (feat.dim() - 2) + [zz.shape[-1]]
        zz = zz.view(shape).expand(*f.shape[:-1], -1)
        return self.mlp(torch.cat([f, zz], dim=-1)).squeeze(-1)

    def bias_logits(self, logits, feat, z, tau=None, mask=None):
        """후보 로짓에 통과성을 얹는 공통 규약.

        logits·feat의 후보 차원은 같아야 한다(픽셀 맵이면 (B,S,S),
        노드/고스트면 (B,N)). tau를 주면 통과성 로짓이 tau 미만인 후보를
        −inf로 차단한다(하드 게이팅). mask(bool)가 있으면 True인 후보에만
        바이어스·차단을 적용한다 — 고스트에만 걸고 방문 노드는 두는 용도.
        """
        s = self(feat, z)
        add = s if mask is None else s * mask
        out = logits + add
        if tau is not None:
            blocked = s < tau if mask is None else (s < tau) & mask
            out = out.masked_fill(blocked, -float("inf"))
        return out


class FiLM(nn.Module):
    def __init__(self, feat_dim, z_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, z):
        """x (..., feat_dim), z (B, z_dim) → x·(1+γ(z)) + β(z).

        γ·β는 x의 배치 이후 중간 차원에 브로드캐스트된다.
        """
        gamma, beta = self.net(z).chunk(2, dim=-1)
        shape = [z.shape[0]] + [1] * (x.dim() - 2) + [gamma.shape[-1]]
        gamma = gamma.view(shape)
        beta = beta.view(shape)
        return x * (1.0 + gamma) + beta

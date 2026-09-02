from __future__ import annotations

import torch
import torch.nn as nn

def _mlp(input_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    """Linear+ReLU를 hidden_dims만큼 쌓은 MLP trunk를 만든다."""

    layers = []
    prev = input_dim
    for size in hidden_dims:
        layers.append(nn.Linear(prev, size))
        layers.append(nn.ReLU())
        prev = size
    return nn.Sequential(*layers)

class Actor(nn.Module):
    """상태를 받아 [-1, 1] 범위의 parameter action을 낸다.

    Daffan/ros_jackal의 rl_algos/td3.py::Actor와 같은 구조(MLP trunk + tanh 출력)다.
    원본은 이미지·시계열 관측을 위해 encoder/head를 분리하지만, wheeled parameter obs는
    이미 평평한 벡터라 trunk 하나로 합쳤다.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]):

        super().__init__()
        self.trunk = _mlp(obs_dim, hidden_dims)
        self.out = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """관측을 [-1, 1] 범위의 action으로 변환한다."""

        return torch.tanh(self.out(self.trunk(obs)))

class Critic(nn.Module):
    """(상태, action) 쌍의 가치를 twin Q-network로 추정한다.

    TD3의 overestimation bias 방지 기법 그대로다 — 서로 다른 초기화의 Q1/Q2를 동시에 학습해
    타깃 계산에서 더 작은 값을 쓴다(td3.py의 train_rl 참고).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]):

        super().__init__()
        self.trunk1 = _mlp(obs_dim + action_dim, hidden_dims)
        self.out1 = nn.Linear(hidden_dims[-1], 1)
        self.trunk2 = _mlp(obs_dim + action_dim, hidden_dims)
        self.out2 = nn.Linear(hidden_dims[-1], 1)

    def forward(self, obs: torch.Tensor,
               action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(상태, action)에 대한 Q1, Q2 추정값을 반환한다."""

        sa = torch.cat([obs, action], dim=1)
        return self.out1(self.trunk1(sa)), self.out2(self.trunk2(sa))

    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """actor 학습에 쓰는 Q1만 반환한다."""

        return self.out1(self.trunk1(torch.cat([obs, action], dim=1)))

from __future__ import annotations

import torch

class ReplayBuffer:
    """K*M개 env가 매 decision 동시에 만드는 transition을 저장하는 n-step replay buffer.

    Daffan/ros_jackal의 rl_algos/base_rl_algo.py::ReplayBuffer와 같은 역할(circular buffer +
    n-step return)이다. 원본은 단일 env라 buffer가 1차원(시간)이고 n-step lookahead를 Python
    for문으로 env 하나씩 돈다. 우리는 env가 이미 GPU에서 벡터로 동시에 진행되므로 buffer를
    (시간, env) 2차원으로 두고, 같은 시간축을 env 전체가 공유한다 — sample()의 n-step
    lookahead도 배치 전체를 텐서 연산으로 한 번에 처리한다 (agent.md의 GPU 우선 원칙).
    next_state는 따로 저장하지 않는다 — slot t+1의 obs가 곧 next_obs다. done인 transition은
    TD3 타깃에서 not_done=0으로 곱해 사라지므로, slot t+1이 다음 episode의 reset obs여도
    문제없다.
    """

    def __init__(self, obs_dim: int, action_dim: int, num_envs: int,
                capacity_steps: int, device: str):

        self.num_envs = num_envs
        self.device = device
        self._capacity = capacity_steps
        self._obs = torch.zeros(capacity_steps, num_envs, obs_dim, device=device)
        self._action = torch.zeros(capacity_steps, num_envs, action_dim, device=device)
        self._reward = torch.zeros(capacity_steps, num_envs, device=device)
        self._done = torch.zeros(capacity_steps, num_envs, dtype=torch.bool, device=device)
        self._ptr = 0
        self._size = 0

    def add(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
           done: torch.Tensor):
        """이번 decision의 (obs, action, reward, done)을 모든 env에 대해 한 slot에 저장한다."""

        self._obs[self._ptr] = obs
        self._action[self._ptr] = action
        self._reward[self._ptr] = reward
        self._done[self._ptr] = done
        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def ready(self, batch_size: int) -> bool:
        """batch_size만큼 샘플링 가능한 크기가 쌓였는지 반환한다."""

        return self._size >= batch_size

    def sample(self, batch_size: int, n_step: int,
              gamma: float) -> tuple[torch.Tensor, ...]:
        """n-step TD 표준 공식으로 (obs, action, next_obs, reward, not_done, gamma^k)를 뽑는다.

        각 샘플은 done을 만날 때까지(최대 n_step) 보상을 gamma^0..gamma^(k-1)로 누적하고,
        중간에 done을 만나지 않았으면 그 k번째 다음 상태로 bootstrap한다(not_done=1,
        할인 gamma^k). done을 만났으면 그 지점에서 누적을 멈추고 not_done=0으로 bootstrap을
        끈다 — 표준 TD3/n-step DQN 공식과 동일하다.
        """

        t_idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        env_idx = torch.randint(0, self.num_envs, (batch_size,), device=self.device)

        obs = self._obs[t_idx, env_idx]
        action = self._action[t_idx, env_idx]

        reward = torch.zeros(batch_size, device=self.device)
        discount = torch.ones(batch_size, device=self.device)
        alive = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        final_idx = t_idx.clone()
        for k in range(n_step):
            idx_k = (t_idx + k) % self._size
            step_done = self._done[idx_k, env_idx]
            reward = reward + torch.where(alive, discount * self._reward[idx_k, env_idx],
                                          torch.zeros_like(reward))
            final_idx = torch.where(alive, idx_k, final_idx)
            discount = torch.where(alive, discount * gamma, discount)
            alive = alive & ~step_done
            if not bool(alive.any()):
                break

        next_idx = (final_idx + 1) % self._size
        next_obs = self._obs[next_idx, env_idx]
        return obs, action, next_obs, reward, alive.float(), discount

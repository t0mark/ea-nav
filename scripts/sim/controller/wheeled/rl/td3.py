from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .buffer import ReplayBuffer
from .nets import Actor, Critic

class TD3:
    """Twin Delayed DDPG. Daffan/ros_jackal의 rl_algos/td3.py::TD3를 그대로 포팅한다.

    target policy smoothing(다음 action에 clip된 노이즈), twin Q 중 작은 값으로 타깃 계산,
    critic보다 update_actor_freq배 느리게 actor를 갱신하고 그때만 target network를 soft
    update하는 세 가지 핵심 기법이 원본과 동일하다. action은 우리 쪽 설계상 [-1, 1]을
    그대로 유지한다 — controller parameter 실제 범위로의 변환은 WheeledRlAdapter가
    env 밖에서 하므로, 원본의 action bias/scale 변환이 여기서는 필요 없다.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int], device: str,
                gamma: float, tau: float, policy_noise: float, noise_clip: float,
                n_step: int, update_actor_freq: int, actor_lr: float, critic_lr: float):

        self.actor = Actor(obs_dim, action_dim, hidden_dims).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(obs_dim, action_dim, hidden_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.n_step = n_step
        self.update_actor_freq = update_actor_freq
        self._train_it = 0

    def select_action(self, obs: torch.Tensor, exploration_noise: float) -> torch.Tensor:
        """탐색용 action: actor 출력에 Gaussian noise를 더하고 [-1, 1]로 자른다."""

        with torch.no_grad():
            action = self.actor(obs)
            if exploration_noise > 0.0:
                action = action + torch.randn_like(action) * exploration_noise
            return torch.clamp(action, -1.0, 1.0)

    def train_step(self, buffer: ReplayBuffer, batch_size: int) -> dict[str, float | None]:
        """replay buffer에서 batch_size개를 뽑아 critic 1회, (지연) actor 1회를 갱신한다."""

        obs, action, next_obs, reward, not_done, gammas = buffer.sample(
            batch_size, self.n_step, self.gamma)

        # target policy smoothing: 다음 action에 clip된 노이즈를 더해 critic이 뾰족한
        # 극값에 과적합하지 못하게 한다
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = torch.clamp(self.actor_target(next_obs) + noise, -1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = reward.unsqueeze(1) + not_done.unsqueeze(1) * gammas.unsqueeze(1) * torch.min(target_q1, target_q2)

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = None
        self._train_it += 1
        # delayed policy update: critic이 충분히 안정된 다음에만 actor·target을 갱신한다
        if self._train_it % self.update_actor_freq == 0:
            actor_loss = -self.critic.q1(obs, self.actor(obs)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {"critic_loss": float(critic_loss.item()),
               "actor_loss": float(actor_loss.item()) if actor_loss is not None else None}

from __future__ import annotations

from typing import Callable

import torch

DynamicsFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
CostFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]

class MPPI:
    """샘플링 기반 로컬 궤적 최적화기 (Model Predictive Path Integral).

    동역학·비용 함수를 밖에서 주입받는 범용 옵티마이저 — 어떤 로봇 모델인지 모른다
    (Williams et al. 2017 정식화: 명목 제어열 주변에 노이즈를 뿌려 롤아웃한 뒤,
    비용의 softmax 가중 평균으로 명목열을 갱신한다).
    """

    def __init__(self, num_envs: int, nu: int, horizon: int, num_samples: int,
                 noise_std: torch.Tensor, lambda_: float,
                 u_min: torch.Tensor, u_max: torch.Tensor, device: str):

        self._num_envs = num_envs
        self._nu = nu
        self._horizon = horizon
        self._num_samples = num_samples
        self._noise_std = noise_std
        self._lambda = lambda_
        self._u_min = u_min
        self._u_max = u_max
        self._device = device
        self._nominal = torch.zeros(num_envs, horizon, nu, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):

        if env_ids is None:
            self._nominal.zero_()
        else:
            self._nominal[env_ids] = 0.0

    def command(self, x0: torch.Tensor, dynamics_fn: DynamicsFn,
               cost_fn: CostFn) -> tuple[torch.Tensor, torch.Tensor]:
        """x0: (num_envs, nx). dynamics_fn(x,u)->다음 상태, cost_fn(x,u,step)->스텝 비용은
        (num_envs*num_samples, ...) 배치에서 동작해야 한다.

        반환: (u0, x_traj) — 첫 스텝에 적용할 제어 (num_envs, nu)와, 갱신된 명목 제어열을
        노이즈 없이 롤아웃한 궤적 (num_envs, horizon, nx).
        """
        E, K, T, nu = self._num_envs, self._num_samples, self._horizon, self._nu

        noise = torch.randn(E, K, T, nu, device=self._device) * self._noise_std
        u_seq = torch.clamp(self._nominal.unsqueeze(1) + noise, self._u_min, self._u_max)

        x = x0.unsqueeze(1).expand(E, K, x0.shape[-1]).reshape(E * K, -1)
        cost = torch.zeros(E * K, device=self._device)
        for t in range(T):
            u = u_seq[:, :, t, :].reshape(E * K, nu)
            cost = cost + cost_fn(x, u, t)
            x = dynamics_fn(x, u)

        cost = cost.reshape(E, K)
        # 수치 안정화: 최소 비용을 빼고 지수화 (softmax와 동일, overflow 방지)
        beta = torch.amin(cost, dim=1, keepdim=True)
        weights = torch.softmax(-(cost - beta) / self._lambda, dim=1)

        nominal = torch.clamp(
            (weights.unsqueeze(-1).unsqueeze(-1) * u_seq).sum(dim=1),
            self._u_min, self._u_max)

        x_traj = torch.zeros(E, T, x0.shape[-1], device=self._device)
        x = x0.clone()
        for t in range(T):
            x = dynamics_fn(x, nominal[:, t, :])
            x_traj[:, t, :] = x

        u0 = nominal[:, 0, :].clone()
        # 수용 지평(receding horizon): 다음 호출을 위해 한 스텝 당겨 warm-start,
        # 마지막 스텝은 반복해 시퀀스 길이를 유지
        self._nominal = torch.cat([nominal[:, 1:, :], nominal[:, -1:, :]], dim=1)
        return u0, x_traj

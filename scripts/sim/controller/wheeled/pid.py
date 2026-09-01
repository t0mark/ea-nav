from __future__ import annotations

import torch

class PID:

    def __init__(self, num_envs: int, device: str, *, kp: float, ki: float,
                 kd: float, out_limit: float, i_limit: float):

        self._kp, self._ki, self._kd = kp, ki, kd
        self._out_limit = out_limit
        self._i_limit = i_limit
        self._integral = torch.zeros(num_envs, device=device)
        self._prev_err = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):

        if env_ids is None:
            self._integral.zero_()
            self._prev_err.zero_()
        else:
            self._integral[env_ids] = 0.0
            self._prev_err[env_ids] = 0.0

    def update(self, setpoint: torch.Tensor, measured: torch.Tensor,
               dt: float) -> torch.Tensor:

        err = setpoint - measured
        self._integral = torch.clamp(self._integral + err * dt,
                                     -self._i_limit, self._i_limit)
        deriv = (err - self._prev_err) / dt
        self._prev_err = err
        out = self._kp * err + self._ki * self._integral + self._kd * deriv
        return torch.clamp(setpoint + out, -self._out_limit, self._out_limit)

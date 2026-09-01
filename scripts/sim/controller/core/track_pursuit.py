from __future__ import annotations

import torch

class PurePursuit:

    def __init__(self, model: str, bounds: torch.Tensor, pp_cfg: dict,
                 num_envs: int, device: str, wheelbase: float = 0.0,
                 min_turn_radius: float = 0.0, decel: float = 1.5,
                 lat_accel: float = 2.0, pivot_creep: float = 0.0):

        self._model = model
        self._bounds = bounds
        self._cfg = pp_cfg
        self._num_envs = num_envs
        self._device = device
        self._wheelbase = wheelbase
        self._min_turn_radius = min_turn_radius
        self._decel = decel
        self._lat_accel = lat_accel
        self._pivot_creep = pivot_creep
        self._seg_start = torch.zeros(num_envs, 2, device=device)
        self._last_goal = torch.full((num_envs, 2), float("nan"), device=device)
        self._last_v = torch.zeros(num_envs, device=device)
        self._reversing = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):

        if env_ids is None:
            self._last_goal.fill_(float("nan"))
            self._last_v.zero_()
            self._reversing.fill_(False)
        else:
            self._last_goal[env_ids] = float("nan")
            self._last_v[env_ids] = 0.0
            self._reversing[env_ids] = False

    def _speed_limit(self, dist_goal: torch.Tensor,
                     kappa: torch.Tensor) -> torch.Tensor:

        cfg = self._cfg
        v_max = self._bounds[0, 1]
        v_stop = torch.sqrt(2.0 * cfg["decel_margin"] * self._decel
                            * torch.clamp(dist_goal - cfg["stop_dist"], min=0.0))
        v_curv = torch.sqrt(self._lat_accel
                            / torch.clamp(torch.abs(kappa), min=1e-6))
        return torch.minimum(torch.minimum(v_stop, v_curv),
                             torch.full_like(v_stop, float(v_max)))

    def plan(self, pos_xy: torch.Tensor, yaw: torch.Tensor,
             goal_xy: torch.Tensor) -> torch.Tensor:

        cfg = self._cfg

        moved = torch.norm(goal_xy - self._last_goal, dim=1)            > float(cfg["goal_change_eps"])
        new_seg = moved | torch.isnan(self._last_goal[:, 0])
        self._seg_start = torch.where(new_seg.unsqueeze(1), pos_xy,
                                      self._seg_start)
        self._last_goal = goal_xy.clone()

        seg = goal_xy - self._seg_start
        seg_len = torch.clamp(torch.norm(seg, dim=1), min=1e-6)
        seg_dir = seg / seg_len.unsqueeze(1)

        proj = ((pos_xy - self._seg_start) * seg_dir).sum(dim=1)
        lookahead = torch.minimum(
            torch.maximum(torch.full_like(proj, cfg["lookahead_min"]),
                          self._last_v * cfg["lookahead_time"]),
            seg_len)
        s = torch.minimum(torch.clamp(proj + lookahead, min=0.0), seg_len)
        look = self._seg_start + s.unsqueeze(1) * seg_dir

        c, sn = torch.cos(yaw), torch.sin(yaw)
        dl = look - pos_xy
        x_l = c * dl[:, 0] + sn * dl[:, 1]
        y_l = -sn * dl[:, 0] + c * dl[:, 1]
        L = torch.clamp(torch.sqrt(x_l ** 2 + y_l ** 2), min=1e-6)
        kappa = 2.0 * (y_l / L) / lookahead
        dg = goal_xy - pos_xy
        x_g = c * dg[:, 0] + sn * dg[:, 1]
        y_g = -sn * dg[:, 0] + c * dg[:, 1]
        dist_goal = torch.norm(dg, dim=1)

        v = self._speed_limit(dist_goal, kappa)

        parked = dist_goal < cfg["stop_dist"]
        v = torch.where(parked, torch.zeros_like(v), v)

        if self._model == "bicycle":
            u = self._bicycle(v, kappa, x_g, y_g, parked)
        elif self._model == "holonomic":
            u = self._holonomic(v, x_l, y_l, parked)
        else:
            u = self._unicycle(v, kappa, x_g, y_g, parked)

        if self._model == "holonomic":
            self._last_v = torch.hypot(u[:, 0], u[:, 1])
        else:
            self._last_v = torch.abs(u[:, 0])
        return u

    def _unicycle(self, v, kappa, x_g, y_g, parked):

        cfg = self._cfg
        v_max, w_max = self._bounds[0, 1], self._bounds[1, 1]
        bearing = torch.atan2(y_g, x_g)
        turn_first = (torch.abs(bearing) > cfg["pivot_bearing"]) & ~parked

        v = torch.minimum(v, w_max / torch.clamp(torch.abs(kappa), min=1e-6))
        w = v * kappa

        v = torch.where(turn_first, self._pivot_creep * v_max
                        * torch.ones_like(v), v)
        w = torch.where(turn_first,
                        torch.sign(bearing) * cfg["turn_w_ratio"] * w_max, w)
        w = torch.where(parked, torch.zeros_like(w), w)
        return torch.stack([v, w], dim=1)

    def _bicycle(self, v, kappa, x_g, y_g, parked):

        cfg = self._cfg
        v_max = self._bounds[0, 1]
        delta_max = self._bounds[1, 1]
        R = self._min_turn_radius
        inside_circle = (x_g ** 2 + (torch.abs(y_g) - R) ** 2) < R ** 2

        bearing = torch.atan2(y_g, x_g)
        enter = (torch.abs(bearing) > float(cfg["reverse_enter_bearing"]))            | inside_circle
        leave = (torch.abs(bearing) < float(cfg["reverse_exit_bearing"]))            & ~inside_circle
        self._reversing = (self._reversing | enter) & ~leave
        reverse = self._reversing & ~parked

        delta = torch.clamp(torch.atan(self._wheelbase * kappa),
                            -delta_max, delta_max)

        delta_rev = -torch.sign(y_g) * cfg["turn_w_ratio"] * delta_max
        v = torch.where(reverse, -cfg["reverse_ratio"] * v_max
                        * torch.ones_like(v), v)
        delta = torch.where(reverse, delta_rev, delta)
        delta = torch.where(parked, torch.zeros_like(delta), delta)
        return torch.stack([v, delta], dim=1)

    def _holonomic(self, v, x_l, y_l, parked):

        cfg = self._cfg
        w_max = self._bounds[2, 1]
        L = torch.clamp(torch.sqrt(x_l ** 2 + y_l ** 2), min=1e-6)
        vx, vy = v * x_l / L, v * y_l / L
        w = torch.clamp(cfg["yaw_kp"] * torch.atan2(y_l, x_l), -w_max, w_max)
        w = torch.where(parked, torch.zeros_like(w), w)
        return torch.stack([vx, vy, w], dim=1)

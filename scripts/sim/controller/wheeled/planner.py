from __future__ import annotations

import torch

from ..core.plan_mppi import MPPI
from ..core.scan_terrain import TerrainScan

class LocalPlanner:
    """MPPI 로컬 플래너 — 지형 경사·목표 진행을 함께 고려해 PP가 추종할 근거리 조준점을 낸다.

    core/plan_mppi.py(범용 샘플러)에 이 로봇의 롤아웃 동역학과 비용을 주입하는 wrapper.
    출력은 전체 로컬 경로가 아니라, 최적화된 명목 궤적 위 lookahead_steps 지점 하나 —
    core/track_pursuit.py의 PurePursuit는 원래 인터페이스(단일 goal_xy)를 그대로 쓴다.

    dt는 실제 제어 주기(control period)가 아니라 MPPI 자체 롤아웃 한 스텝의 시간이다 —
    실측(diff_0000, flat)으로 확인된 실패 모드: dt를 제어 주기(0.02s)와 같게 두면
    horizon(12스텝)을 다 더해도 0.24초밖에 못 내다봐서 lookahead 지점이 로봇 바로
    앞(~0.1-0.15m)에 계속 붙고, PP의 감속 로직(dist_goal 기준 v_stop)이 "곧 도착"으로
    오판해 속도를 계속 낮게 눌러버리는 자기강화 루프에 빠진다 — planner는 제어 루프보다
    훨씬 낮은 주기로만 재계획하고(controller.py의 replan_decimation), 그 사이는 PP가
    자기 원래 고주기 추종 로직으로 채우는 구조라 dt를 제어 주기에 묶을 이유가 없다.
    """

    def __init__(self, model: str, bounds: torch.Tensor, wheelbase: float,
                dt: float, mppi_cfg: dict, num_envs: int, device: str):

        self._model = model
        self._wheelbase = wheelbase
        self._dt = dt
        self._device = device
        self._num_envs = num_envs
        self._num_samples = int(mppi_cfg["num_samples"])
        self._lookahead = min(int(mppi_cfg["lookahead_steps"]),
                              int(mppi_cfg["horizon"]) - 1)
        self._goal_weight = float(mppi_cfg["goal_weight"])
        self._slope_weight = float(mppi_cfg["slope_weight"])
        self._slope_cap = float(mppi_cfg["slope_cap"])
        self._effort_weight = float(mppi_cfg["smooth_weight"])

        nu = bounds.shape[0]
        u_min, u_max = bounds[:, 0], bounds[:, 1]
        noise_std = float(mppi_cfg["noise_ratio"]) * (u_max - u_min)
        self._mppi = MPPI(num_envs, nu, int(mppi_cfg["horizon"]),
                          self._num_samples, noise_std, float(mppi_cfg["lambda_"]),
                          u_min, u_max, device)

        self._env_idx_k = torch.arange(num_envs, device=device)            .repeat_interleave(self._num_samples)

    def reset(self, env_ids: torch.Tensor | None = None):

        self._mppi.reset(env_ids)

    def _dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """x=(x,y,yaw) 상태를 dt만큼 전진한다 — track_pursuit.py의 같은 세 모델을
        추종 목적이 아니라 예측 목적으로 재구현한 것 (기하 관계는 동일)."""

        px, py, yaw = x[:, 0], x[:, 1], x[:, 2]
        c, sn = torch.cos(yaw), torch.sin(yaw)
        if self._model == "bicycle":
            v, delta = u[:, 0], u[:, 1]
            yaw_rate = v * torch.tan(delta) / self._wheelbase
            nx, ny = px + v * c * self._dt, py + v * sn * self._dt
        elif self._model == "holonomic":
            vx, vy, yaw_rate = u[:, 0], u[:, 1], u[:, 2]
            nx = px + (vx * c - vy * sn) * self._dt
            ny = py + (vx * sn + vy * c) * self._dt
        else:
            v, yaw_rate = u[:, 0], u[:, 1]
            nx, ny = px + v * c * self._dt, py + v * sn * self._dt
        return torch.stack([nx, ny, yaw + yaw_rate * self._dt], dim=1)

    def _cost(self, x: torch.Tensor, u: torch.Tensor, goal_k: torch.Tensor,
             terrain_scan: TerrainScan | None) -> torch.Tensor:

        cost = self._goal_weight * torch.norm(x[:, :2] - goal_k, dim=1)
        cost = cost + self._effort_weight * (u * u).sum(dim=1)
        if terrain_scan is not None:
            slope = terrain_scan.sample_slope(x[:, :2], self._env_idx_k)
            cost = cost + self._slope_weight * torch.clamp(slope, max=self._slope_cap)
        return cost

    def plan(self, pos_xy: torch.Tensor, yaw: torch.Tensor, far_goal_xy: torch.Tensor,
             terrain_scan: TerrainScan | None) -> torch.Tensor:
        """far_goal_xy: 실제 웨이포인트. 반환: PP에 넘길 근거리 조준점 (num_envs, 2)."""

        x0 = torch.stack([pos_xy[:, 0], pos_xy[:, 1], yaw], dim=1)
        goal_k = far_goal_xy.repeat_interleave(self._num_samples, dim=0)

        def dynamics_fn(x, u):
            return self._dynamics(x, u)

        def cost_fn(x, u, _t):
            return self._cost(x, u, goal_k, terrain_scan)

        _, x_traj = self._mppi.command(x0, dynamics_fn, cost_fn)
        return x_traj[:, self._lookahead, :2]

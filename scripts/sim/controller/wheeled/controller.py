from __future__ import annotations

import math

import torch

from ..core.base import (BaseController, ControlObs, JointTargets,
                         RobotCtrlParams)
from ..core.track_pursuit import PurePursuit
from . import ik
from .governor import Governor
from .pid import PID
from .planner import LocalPlanner

def _yaw_rate_limit(A: torch.Tensor, limits: torch.Tensor) -> float:

    arm = torch.abs(A[:, 2])
    return float(torch.min(limits / torch.clamp(arm, min=1e-6)))

class WheeledRobotController(BaseController):
    """파이프라인: planner(MPPI) -> pursuit(PP) -> governor(CBF) -> ik.

    각 단계는 별 파일(planner.py/track_pursuit.py/governor.py/ik.py)에 있고, 이 클래스는
    그 인스턴스를 만들고 compute()에서 순서대로 호출하는 오케스트레이션만 담당한다.
    """

    def __init__(self, params: RobotCtrlParams, joint_names: list[str],
                 default_pose: torch.Tensor, num_envs: int, device: str,
                 cfg: dict, physics_dt: float):

        super().__init__(params, joint_names, default_pose, num_envs, device)
        self.decimation = int(cfg["ctrl"]["decimation"])

        self._wheel_names, self._A, self._limits = ik.build_wheel_matrix(
            params, device, skid_yaw_scale=cfg["ctrl"]["skid_yaw_scale"])
        self._wheel_idx = self._index_of(self._wheel_names)

        self._front_wheels = []
        if params.base_tag == "ackermann":
            for i, wf in enumerate(params.wheels):
                if wf.pos[0] > 0:

                    side = max(params.steer_y,
                               key=lambda n: params.steer_y[n] * float(wf.pos[1]))
                    self._front_wheels.append((i, side))

        margin = float(cfg["ctrl"]["limit_margin"])
        v_max = params.max_lin_vel * margin
        w_max = min(_yaw_rate_limit(self._A, self._limits) * margin,
                    float(cfg["ctrl"]["yaw_rate_cap"]))

        # 전복 안전 여유는 더 이상 여기서 속도 상한을 깎아 사전 예방하지 않는다 — governor가
        # 매 스텝 실제 자세(gravity_b)를 보고 필요할 때만 개입한다 (평지에서는 전혀 안 깎임)
        lin_accel = float(cfg["ctrl"]["lin_accel"])
        lat_accel = float(cfg["pp"]["lat_accel"])
        self._lat_accel = lat_accel

        if params.base_tag == "ackermann":

            tan_l = math.tan(params.steer_range)
            y_in = max(abs(y) for y in params.steer_y.values())
            kappa_max = tan_l / (params.wheelbase + y_in * tan_l)
            delta_max = math.atan(params.wheelbase * kappa_max)
            bounds = [[-v_max, v_max], [-delta_max, delta_max]]
            model = "bicycle"

            r_turn = 1.0 / kappa_max
        elif params.holonomic:

            bounds = [[-v_max, v_max], [-v_max, v_max], [-w_max, w_max]]
            model = "holonomic"
            r_turn = v_max / w_max
        else:
            bounds = [[-v_max, v_max], [-w_max, w_max]]
            model = "unicycle"
            r_turn = v_max / w_max

        self.nav_limits = {"v_max": v_max, "w_max": w_max, "r_turn": r_turn}

        creep = float(cfg["pp"]["creep_ratio"])
        bounds_t = torch.tensor(bounds, dtype=torch.float32, device=device)

        pp_cfg = dict(cfg["pp"])
        pp_cfg["lookahead_min"] = max(
            float(pp_cfg["lookahead_min"]),
            float(pp_cfg["lookahead_turn_ratio"]) * r_turn)
        self._pursuit = PurePursuit(
            model, bounds_t, pp_cfg, num_envs, device, wheelbase=params.wheelbase,
            min_turn_radius=r_turn if model == "bicycle" else 0.0,
            decel=lin_accel, lat_accel=lat_accel, pivot_creep=creep)

        self._period = self.decimation * physics_dt
        # planner의 dt는 제어 주기가 아니라 MPPI 자체 계획 스텝 시간 — planner.py 참고
        self._planner = LocalPlanner(model, bounds_t, params.wheelbase,
                                     float(cfg["mppi"]["plan_dt"]), cfg["mppi"],
                                     num_envs, device)
        self._replan_decimation = int(cfg["mppi"]["replan_decimation"])
        self._plan_step = 0
        self._local_goal = None
        self._governor = Governor(params, float(cfg["governor"]["margin"]), v_max)

        lin_step = lin_accel * self._period
        yaw_step = float(cfg["ctrl"]["yaw_accel"]) * self._period
        if params.base_tag == "ackermann":

            du = [lin_step, params.steer_vel_limit * self._period]
        elif params.holonomic:
            du = [lin_step, lin_step, yaw_step]
        else:
            du = [lin_step, yaw_step]
        self._du = torch.tensor(du, dtype=torch.float32, device=device)
        self._last_u = torch.zeros(num_envs, len(du), device=device)

        if model == "unicycle":
            pid_kw = dict(kp=float(cfg["ctrl"]["pid_kp"]),
                         ki=float(cfg["ctrl"]["pid_ki"]),
                         kd=float(cfg["ctrl"]["pid_kd"]),
                         i_limit=float(cfg["ctrl"]["pid_i_limit"]))
            self._lin_pid = PID(num_envs, device, out_limit=v_max, **pid_kw)
            self._yaw_pid = PID(num_envs, device, out_limit=w_max, **pid_kw)
        else:
            self._lin_pid = None
            self._yaw_pid = None

    def reset(self, env_ids: torch.Tensor | None = None):

        self._planner.reset(env_ids)
        self._pursuit.reset(env_ids)
        if self._lin_pid is not None:
            self._lin_pid.reset(env_ids)
            self._yaw_pid.reset(env_ids)
        if env_ids is None:
            self._last_u.zero_()
        else:
            self._last_u[env_ids] = 0.0
        # 리셋된 env가 일부여도 다음 compute()에서 전체 재계획을 강제한다 — env별로
        # 정확히 쪼개는 대신 약간의 여분 계산으로 단순함을 유지 (correctness는 유지됨)
        self._plan_step = 0
        self._local_goal = None

    def compute(self, obs: ControlObs, goal_xy: torch.Tensor) -> JointTargets:

        # 1. planner(MPPI) — replan_decimation 주기로만 재계획, 그 사이는 직전 조준점 재사용
        #    (MPPI 자체 계획 주기 << 제어 주기로 두면 lookahead 지점이 로봇 근처에 묶여
        #    pursuit의 감속 로직이 항상 "곧 도착"으로 오판하는 정체가 실측으로 확인됨)
        if self._local_goal is None or self._plan_step % self._replan_decimation == 0:
            self._local_goal = self._planner.plan(obs.pos_xy, obs.yaw, goal_xy,
                                                   obs.terrain_scan)
        self._plan_step += 1
        # 2. pursuit(PP) — 조준점을 명목 body 명령으로 변환 (로직 자체는 기존과 동일)
        u = self._pursuit.plan(obs.pos_xy, obs.yaw, self._local_goal)
        u = torch.clamp(u, self._last_u - self._du, self._last_u + self._du)
        self._last_u = u

        if self._params.base_tag == "ackermann":
            v, delta = u[:, 0], u[:, 1]
            omega = v * torch.tan(delta) / self._params.wheelbase
            cmd = torch.stack([v, torch.zeros_like(v), omega], dim=1)
        elif self._params.holonomic:
            cmd = u
        else:
            v0, w = u[:, 0], u[:, 1]
            if self._lin_pid is not None:
                v0 = self._lin_pid.update(v0, obs.vel_b[:, 0], self._period)
                w = self._yaw_pid.update(w, obs.ang_b[:, 2], self._period)

                cap = self._lat_accel / torch.clamp(torch.abs(v0), min=0.1)
                w = torch.clamp(w, -cap, cap)
            cmd = torch.stack([v0, torch.zeros_like(v0), w], dim=1)

        # 3. governor(CBF) — 실시간 자세 기준 전복 방지 barrier로 cmd 투영
        cmd = self._governor.filter(cmd, obs)

        pos = self._default_pose.clone()
        steer = {}
        if self._params.base_tag == "ackermann":
            # governor가 w를 바꿨을 수 있어, 조향각을 필터링된 cmd에서 다시 유도한다
            # (조향각 자체가 물리적으로 회전율을 만드는 기구이므로 독립적으로 못 바꿈)
            v_safe = torch.where(cmd[:, 0].abs() > 1e-3, cmd[:, 0],
                                 torch.ones_like(cmd[:, 0]) * 1e-3)
            delta = torch.atan(self._params.wheelbase * cmd[:, 2] / v_safe)
            steer = ik.ackermann_steer(delta, self._params)
            for name, angle in steer.items():
                pos[:, self._joint_index[name]] = angle

        # 4. ik — 액추에이터 배분·실현가능성 (기존 로직 그대로)
        cmd = ik.feasible_scale(self._A, self._limits, cmd)
        cmd_model = cmd.clone()

        speeds = ik.wheel_speeds(self._A, cmd)

        for col, sname in self._front_wheels:
            scale = 1.0 / torch.clamp(torch.cos(steer[sname]), min=0.5)
            speeds[:, col] = torch.clamp(speeds[:, col] * scale,
                                         -self._limits[col], self._limits[col])
        vel = torch.zeros_like(self._default_pose)
        vel[:, self._wheel_idx] = speeds
        return JointTargets(pos=pos, vel=vel, effort=None, cmd=cmd_model)

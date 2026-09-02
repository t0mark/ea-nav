from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Mapping

import numpy as np
import torch

from ..core.base import BaseController, ControlObs, JointTargets, RobotCtrlParams
from ..core.pure_pursuit import PurePursuit
from .rl import WheeledRlAdapter

logger = logging.getLogger(__name__)

def _merged_section(section: dict, base_tag: str) -> dict:
    """전역 기본값에 로봇 타입별 override를 얹은 설정 섹션을 만든다."""

    merged = {k: v for k, v in section.items() if not isinstance(v, Mapping)}
    merged.update(section.get(base_tag, {}))
    return merged

def controller_cfg_for(cfg: dict, base_tag: str) -> dict:
    """wheeled controller가 사용할 타입별 유효 설정을 만든다."""

    out = dict(cfg)
    out["ctrl"] = _merged_section(cfg["ctrl"], base_tag)
    out["pp"] = _merged_section(cfg["pp"], base_tag)
    return out

def yaw_rate_limit(A: torch.Tensor, limits: torch.Tensor) -> float:
    """휠 속도 한계가 허용하는 body yaw-rate 상한을 계산한다."""

    arm = torch.abs(A[:, 2])
    return float(torch.min(limits / torch.clamp(arm, min=1e-6)))

# --- wheeled 기구학 (legged와 공유되지 않는 wheeled 전용 로직이라 여기 둔다) ---

def build_wheel_matrix(params: RobotCtrlParams, device: str,
                       yaw_scale: float = 1.0) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """body velocity (vx, vy, wz)를 각 구동 wheel angular velocity로 바꾸는 행렬을 만든다."""

    rows, names, limits = [], [], []
    radius = max(float(params.wheel_radius), 1e-6)
    for wheel in params.wheels:
        x, y = float(wheel.pos[0]), float(wheel.pos[1])
        if wheel.joint in params.mecanum_sign:
            sign = float(params.mecanum_sign[wheel.joint])
            rows.append([1.0 / radius, sign / radius,
                         yaw_scale * (sign * x - y) / radius])
        else:
            tangent = np.array([wheel.axis[1], -wheel.axis[0]], dtype=float)
            tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
            rows.append([tangent[0] / radius, tangent[1] / radius,
                         yaw_scale * (tangent[1] * x - tangent[0] * y) / radius])
        names.append(wheel.joint)
        limits.append(float(params.wheel_vel_limit[wheel.joint]))
    return (names,
            torch.tensor(rows, dtype=torch.float32, device=device),
            torch.tensor(limits, dtype=torch.float32, device=device))

def wheel_speeds(A: torch.Tensor, cmd: torch.Tensor,
                 yaw_scale: torch.Tensor | None = None) -> torch.Tensor:
    """body command batch를 wheel angular velocity batch로 변환한다."""

    if yaw_scale is None:
        return cmd @ A.T
    cmd_for_wheels = cmd.clone()
    cmd_for_wheels[:, 2] = cmd_for_wheels[:, 2] * yaw_scale
    return cmd_for_wheels @ A.T

def scale_to_limits(speeds: torch.Tensor, limits: torch.Tensor,
                    cmd: torch.Tensor) -> torch.Tensor:
    """휠 속도가 한계를 넘으면 body command 전체를 같은 비율로 줄인다.

    세 축을 같은 배율로 줄이므로 경로 곡률은 보존되고 속도만 낮아진다. 조향 관절이 있는
    타입은 조향각을 반영한 speeds를 넘겨 같은 규칙을 그대로 쓴다.
    """

    worst = torch.amax(torch.abs(speeds) / limits.unsqueeze(0).clamp_min(1e-6), dim=1)
    scale = torch.clamp(1.0 / torch.clamp(worst, min=1e-9), max=1.0)
    return cmd * scale.unsqueeze(1)

def feasible_scale(A: torch.Tensor, limits: torch.Tensor,
                   cmd: torch.Tensor,
                   yaw_scale: torch.Tensor | None = None) -> torch.Tensor:
    """고정축 휠만 있는 타입의 body command를 휠 속도 한계 안으로 줄인다."""

    return scale_to_limits(wheel_speeds(A, cmd, yaw_scale), limits, cmd)

def ackermann_steer(delta: torch.Tensor, params: RobotCtrlParams) -> dict[str, torch.Tensor]:
    """자전거 모델 조향각을 좌우 ackermann steering joint 각도로 변환한다."""

    kappa = torch.tan(delta) / max(float(params.wheelbase), 1e-6)
    out = {}
    for name in params.steer_joints:
        y = float(params.steer_y[name])
        angle = torch.atan2(float(params.wheelbase) * kappa, 1.0 - y * kappa)
        out[name] = torch.clamp(angle, -float(params.steer_range),
                                float(params.steer_range))
    return out

# --- wheeled controller base ---

class WheeledControllerBase(BaseController):
    """wheeled controller 공통 골격.

    공개 ROS mobile-base controller와 같은 단순 계층을 쓴다:
      1. waypoint tracker가 body command를 만든다.
      2. RL parameter adapter가 이번 tick 관측으로 controller parameter 배율을 갱신한다.
      3. 속도·가속도 한계를 적용한다.
      4. 타입별 기구학이 wheel velocity와 steering position으로 변환한다.

    경사로는 별도 회피 비용으로 취급하지 않는다. 같은 body command를 평지와 경사에 주고,
    실제 주행 실패는 wheel target, wheel feedback, torque, contact trace로 분리해 판단한다.

    하위 클래스는 tracker 운동 모델(MODEL)과 입력 경계·명령 변환 두 메서드만 채운다.
    """

    # tracker가 사용할 운동 모델 이름 (unicycle / bicycle / holonomic)
    MODEL = "unicycle"

    def __init__(self, params: RobotCtrlParams, joint_names: list[str],
                 default_pose: torch.Tensor, num_envs: int, device: str,
                 cfg: dict, physics_dt: float):

        super().__init__(params, joint_names, default_pose, num_envs, device)
        cfg = controller_cfg_for(cfg, params.base_tag)
        self.decimation = int(cfg["ctrl"]["decimation"])
        self._period = self.decimation * physics_dt
        # wheel_yaw_scale은 고정축 휠이 옆미끄럼으로 회전하는 타입(skid)의 실효 선회반경
        # 보정이다. ROS diff-drive의 wheel_separation_multiplier와 같은 성격이라 타입별
        # 설정으로 두고, 보정이 필요 없는 타입은 1.0을 쓴다.
        self._wheel_names, self._A, self._limits = build_wheel_matrix(
            params, device, yaw_scale=float(cfg["ctrl"]["wheel_yaw_scale"]))
        self._wheel_idx = self._index_of(self._wheel_names)
        margin = float(cfg["ctrl"]["limit_margin"])
        self._v_max = float(params.max_lin_vel) * margin
        self._w_max = min(yaw_rate_limit(self._A, self._limits) * margin,
                          float(cfg["ctrl"]["yaw_rate_cap"]))
        bounds, r_turn = self._control_bounds()
        self._bounds = torch.tensor(bounds, dtype=torch.float32, device=device)
        self.nav_limits = {"v_max": self._v_max, "w_max": self._w_max,
                           "r_turn": r_turn}
        # lookahead 대역은 로봇 크기·선회 능력에 맞춰 여기서 한 번 정한다. RL은 이 대역
        # 전체에 배율 하나(lookahead_scale)만 걸어 로봇별로 늘리거나 줄인다
        pp_cfg = dict(cfg["pp"])
        pp_cfg["lookahead_min"] = max(float(pp_cfg["lookahead_min"]),
                                      float(pp_cfg["lookahead_turn_ratio"]) * r_turn)
        pp_cfg["lookahead_max"] = max(float(pp_cfg["lookahead_max"]),
                                      pp_cfg["lookahead_min"])
        self._tracker = PurePursuit(
            self.MODEL, self._bounds, pp_cfg, num_envs, device,
            wheelbase=params.wheelbase,
            turn_radius=r_turn,
            decel=float(cfg["ctrl"]["lin_accel"]),
            pivot_creep=float(pp_cfg["creep_ratio"]))
        self._body_du = torch.tensor(
            [float(cfg["ctrl"]["lin_accel"]) * self._period,
             float(cfg["ctrl"]["lin_accel"]) * self._period,
             float(cfg["ctrl"]["yaw_accel"]) * self._period],
            dtype=torch.float32, device=device)
        self._last_cmd = torch.zeros(num_envs, 3, device=device)
        self._rl_adapter = WheeledRlAdapter(params, cfg.get("wheeled_rl", {}),
                                            num_envs, device, self._period)
        logger.info("%s wheeled controller: %s, v %.2f m/s, w %.2f rad/s, "
                    "lookahead %.2f-%.2f m", params.name, self.MODEL,
                    self._v_max, self._w_max, pp_cfg["lookahead_min"],
                    pp_cfg["lookahead_max"])

    @property
    def rl_adapter(self) -> WheeledRlAdapter:
        """controller parameter를 들고 있는 RL adapter를 반환한다."""

        return self._rl_adapter

    def reset(self, env_ids: torch.Tensor | None = None):
        """tracker, rate limiter, RL adapter 상태를 초기화한다."""

        self._tracker.reset(env_ids)
        self._rl_adapter.reset(env_ids)
        if env_ids is None:
            self._last_cmd.zero_()
        else:
            self._last_cmd[env_ids] = 0.0

    def compute(self, obs: ControlObs, goal_xy: torch.Tensor) -> JointTargets:
        """이번 tick 관측으로 RL parameter를 갱신하고 joint target을 계산한다."""

        self._rl_adapter.update(obs, goal_xy)
        cmd = self._rl_adapter.adapt(self.nominal_body_cmd(obs, goal_xy))
        return self._targets_from_body_cmd(cmd, obs)

    def compute_with_action(self, obs: ControlObs, goal_xy: torch.Tensor,
                            action: torch.Tensor) -> JointTargets:
        """학습용 외부 action을 controller parameter로 적용해 joint target을 계산한다."""

        self._rl_adapter.set_action(action)
        cmd = self._rl_adapter.adapt(self.nominal_body_cmd(obs, goal_xy))
        return self._targets_from_body_cmd(cmd, obs)

    def nominal_body_cmd(self, obs: ControlObs, goal_xy: torch.Tensor) -> torch.Tensor:
        """Pure Pursuit가 만든 scale 전 body command를 반환한다."""

        return self._to_body_cmd(self._tracker.plan(
            obs.pos_xy, obs.yaw, goal_xy, self._rl_adapter.pp_params()))

    def _targets_from_body_cmd(self, cmd: torch.Tensor,
                               obs: ControlObs) -> JointTargets:
        """body command 제한과 타입별 IK를 적용한다."""

        targets = self._joint_targets(self._limit_body_cmd(cmd), obs)
        self._last_cmd = targets.cmd if targets.cmd is not None else cmd
        return targets

    def _limit_body_cmd(self, cmd: torch.Tensor) -> torch.Tensor:
        """body command에 속도와 control-period당 변화량 제한을 적용한다."""

        # 가속 한계 배율은 RL parameter로 로봇마다 조정된다 (기본 1.0)
        accel_scale = torch.stack([
            self._rl_adapter.lin_accel_scale,
            self._rl_adapter.lin_accel_scale,
            self._rl_adapter.yaw_accel_scale,
        ], dim=1)
        du = self._body_du.unsqueeze(0) * accel_scale
        limited = torch.clamp(cmd, self._last_cmd - du, self._last_cmd + du)
        vx_max = self._v_max * self._rl_adapter.drive_speed_scale
        vy_max = self._v_max * self._rl_adapter.lateral_speed_scale
        wz_max = self._w_max * self._rl_adapter.yaw_speed_scale
        limited[:, 0] = torch.clamp(limited[:, 0], -vx_max, vx_max)
        limited[:, 1] = torch.clamp(limited[:, 1], -vy_max, vy_max)
        limited[:, 2] = torch.clamp(limited[:, 2], -wz_max, wz_max)
        return limited

    def _joint_targets(self, cmd: torch.Tensor, obs: ControlObs) -> JointTargets:
        """조향 관절이 없는 기본 wheel velocity target을 만든다."""

        cmd_model = feasible_scale(
            self._A, self._limits, cmd, self._rl_adapter.wheel_yaw_scale)
        vel = torch.zeros_like(self._default_pose)
        vel[:, self._wheel_idx] = wheel_speeds(
            self._A, cmd_model, self._rl_adapter.wheel_yaw_scale)
        return JointTargets(pos=self._default_pose.clone(), vel=vel,
                            effort=None, cmd=cmd_model)

    def _unicycle_bounds(self) -> tuple[list, float]:
        """(v, w) 입력 경계와 명목 선회반경을 반환한다."""

        r_turn = self._v_max / max(self._w_max, 1e-6)
        return [[-self._v_max, self._v_max], [-self._w_max, self._w_max]], r_turn

    @abstractmethod
    def _control_bounds(self) -> tuple[list, float]:
        pass

    @abstractmethod
    def _to_body_cmd(self, u: torch.Tensor) -> torch.Tensor:
        pass

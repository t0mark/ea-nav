from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ...core.base import ControlObs, RobotCtrlParams

_TYPES = ("diff", "skid", "ackermann", "omni")
_MORPH_DIM = 10
# goal(2, body frame) + vel_b(2) + ang_b(1, wz) + gravity_b(3)
_STATE_DIM = 8

@dataclass(frozen=True)
class ObsSpec:
    """controller parameter policy가 보는 관측 구성.

    Daffan/ros_jackal의 obs(laser scan + 로컬 목표 + 직전 action)와 같은 원칙이지만, 우리는
    wheeled 로봇에 라이다가 없어 대신 실시간 상태(목표·속도·기울기)를 쓰고, 여러 morphology를
    한 정책이 같이 배우므로 type/morphology 블록이 추가된다.
    """

    include_type: bool = True
    include_morph: bool = True
    include_state: bool = True

    def __post_init__(self):
        """관측이 비지 않는지 확인한다."""

        if not (self.include_type or self.include_morph or self.include_state):
            raise ValueError("wheeled_rl.obs는 include_type / include_morph / "
                             "include_state 중 적어도 하나가 참이어야 한다")

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "ObsSpec":
        """설정 dict에서 관측 구성을 만든다 (없으면 기본값 = 셋 다 사용)."""

        cfg = cfg or {}
        return cls(include_type=bool(cfg.get("include_type", True)),
                   include_morph=bool(cfg.get("include_morph", True)),
                   include_state=bool(cfg.get("include_state", True)))

    def static_dim(self) -> int:
        """type + morph처럼 로봇 하나에 고정인 블록의 차원을 반환한다."""

        return (len(_TYPES) if self.include_type else 0) + (
            _MORPH_DIM if self.include_morph else 0)

    def dim(self, num_actions: int) -> int:
        """이 구성과 action 차원 수로 정해지는 전체 관측 차원을 반환한다."""

        return self.static_dim() + (_STATE_DIM + num_actions if self.include_state else 0)

    def build_static(self, params: RobotCtrlParams, morph: torch.Tensor,
                     num_envs: int) -> torch.Tensor:
        """로봇 하나의 정적(type+morph) 관측 행을 만들어 env 수만큼 펼친다."""

        parts = []
        if self.include_type:
            parts.append(type_one_hot(params.base_tag, morph.device))
        if self.include_morph:
            parts.append(morph)
        if not parts:
            return torch.zeros(num_envs, 0, device=morph.device)
        single = torch.cat(parts, dim=0)
        return single.unsqueeze(0).expand(num_envs, -1)

    def assemble(self, static_row: torch.Tensor, ctrl_obs: ControlObs,
                goal_xy: torch.Tensor, prev_action: torch.Tensor,
                state_cfg: dict | None = None) -> torch.Tensor:
        """정적 블록에 이번 tick의 실시간 상태와 직전 action을 이어 붙인다."""

        if not self.include_state:
            return static_row
        state = live_state(ctrl_obs, goal_xy, state_cfg)
        return torch.cat([static_row, state, prev_action], dim=1)

def type_one_hot(base_tag: str, device: str) -> torch.Tensor:
    """wheeled base type을 one-hot tensor로 변환한다."""

    values = torch.zeros(len(_TYPES), device=device)
    if base_tag in _TYPES:
        values[_TYPES.index(base_tag)] = 1.0
    return values

def morphology_vector(params: RobotCtrlParams, device: str,
                      morph_cfg: dict | None = None) -> torch.Tensor:
    """URDF/meta에서 추출한 morphology를 policy 입력용 크기로 정규화한다."""

    cfg = morph_cfg or {}
    len_scale = max(float(cfg.get("len_scale", 1.0)), 1e-6)
    vel_scale = max(float(cfg.get("vel_scale", 3.0)), 1e-6)
    effort_log_scale = max(float(cfg.get("effort_log_scale", 5.0)), 1e-6)
    track = _track_width(params)
    effort = min(params.wheel_effort_limit.values()) if params.wheel_effort_limit else 0.0
    velocity = min(params.wheel_vel_limit.values()) if params.wheel_vel_limit else 0.0
    wheel_count = max(len(params.wheels), 1)
    values = [
        math.log10(max(float(params.wheel_radius), 1e-4)),
        float(params.wheelbase) / len_scale,
        track / len_scale,
        float(params.max_lin_vel) / vel_scale,
        float(params.steer_range),
        float(params.tip_accel) / 9.81,
        math.log1p(max(float(effort), 0.0)) / effort_log_scale,
        max(float(velocity), 0.0) / 30.0,
        wheel_count / 6.0,
        1.0 if params.holonomic else 0.0,
    ]
    return torch.tensor(values, dtype=torch.float32, device=device)

def live_state(obs: ControlObs, goal_xy: torch.Tensor,
               state_cfg: dict | None = None) -> torch.Tensor:
    """이번 tick의 목표 방향·속도·기울기를 정책 입력 크기로 정규화한다.

    목표 방향은 body frame으로 투영한다 — core.pure_pursuit이 carrot을 계산할 때 쓰는
    것과 같은 투영이다.
    """

    cfg = state_cfg or {}
    goal_scale = max(float(cfg.get("goal_scale", 3.0)), 1e-6)
    vel_scale = max(float(cfg.get("vel_scale", 2.0)), 1e-6)
    ang_scale = max(float(cfg.get("ang_scale", 2.0)), 1e-6)

    dx = goal_xy[:, 0] - obs.pos_xy[:, 0]
    dy = goal_xy[:, 1] - obs.pos_xy[:, 1]
    c, s = torch.cos(obs.yaw), torch.sin(obs.yaw)
    x_b = torch.clamp((c * dx + s * dy) / goal_scale, -3.0, 3.0)
    y_b = torch.clamp((-s * dx + c * dy) / goal_scale, -3.0, 3.0)

    vel_xy = obs.vel_b[:, :2] / vel_scale
    ang_z = (obs.ang_b[:, 2] / ang_scale).unsqueeze(1)
    gravity = (obs.gravity_b if obs.gravity_b is not None
              else torch.zeros_like(obs.vel_b))

    return torch.cat([x_b.unsqueeze(1), y_b.unsqueeze(1), vel_xy, ang_z, gravity], dim=1)

def _track_width(params: RobotCtrlParams) -> float:
    """wheel frame 위치에서 대표 track width를 계산한다."""

    if not params.wheels:
        return 0.0
    ys = [abs(float(w.pos[1])) for w in params.wheels]
    return 2.0 * max(ys)

from __future__ import annotations

import math

import torch

from ..core.base import ControlObs, JointTargets
from . import controller as kinematics
from .controller import WheeledControllerBase

class AckermannController(WheeledControllerBase):
    """자동차형 wheeled 로봇의 경로 추종 controller.

    ROS ackermann controller와 같은 역할 분리를 쓴다. tracker가 자전거 모델 명령을 만들고,
    조향 관절에는 좌우 ackermann 각도를, 구동 휠에는 구름 속도를 준다. 지형 비용이나
    샘플링 기반 제어는 여기에 관여하지 않는다.
    """

    MODEL = "bicycle"

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # 조향축 위에 얹힌 구동 휠(전륜 구동)은 조향각만큼 돌아간 접선 방향으로 구르므로,
        # 어느 조향 관절을 따라가는지 미리 짝지어 둔다. 후륜 구동이면 빈 목록이 된다.
        self._steered_drive_wheels: list[tuple[int, str]] = []
        if self._params.steer_joints and self._params.steer_y:
            for col, wheel in enumerate(self._params.wheels):
                if float(wheel.pos[0]) <= 0.0:
                    continue
                side = max(self._params.steer_y,
                           key=lambda name: self._params.steer_y[name] * float(wheel.pos[1]))
                self._steered_drive_wheels.append((col, side))

    def _control_bounds(self) -> tuple[list, float]:
        """조향 한계에서 유도한 (v, delta) 경계와 최소 선회반경을 반환한다."""

        steer_limit = max(float(self._params.steer_range), 1e-6)
        wheelbase = max(float(self._params.wheelbase), 1e-6)
        tan_limit = math.tan(steer_limit)
        y_inner = max((abs(float(y)) for y in self._params.steer_y.values()),
                      default=0.0)
        # 안쪽 휠이 먼저 조향 한계에 닿으므로, 그 물리 한계에서 자전거 모델 등가 조향각을 얻는다
        kappa_max = tan_limit / max(wheelbase + y_inner * tan_limit, 1e-6)
        delta_max = math.atan(wheelbase * kappa_max)
        return ([[-self._v_max, self._v_max], [-delta_max, delta_max]],
                1.0 / max(kappa_max, 1e-6))

    def _to_body_cmd(self, u: torch.Tensor) -> torch.Tensor:
        """tracker 출력 (v, delta)를 body command (vx, 0, wz)로 변환한다."""

        v, delta = u[:, 0], u[:, 1]
        w = v * torch.tan(delta) / max(float(self._params.wheelbase), 1e-6)
        return torch.stack([v, torch.zeros_like(v), w], dim=1)

    def _joint_targets(self, cmd: torch.Tensor, obs: ControlObs) -> JointTargets:
        """조향 관절 각도와 구동 휠 속도를 함께 만든다."""

        # 휠 속도 한계로 command를 줄이면 곡률은 보존되지만 속도가 바뀌므로,
        # 조향각은 최종 body command 기준으로 한 번 더 계산한다
        steer = self._steering_angles(cmd)
        limited = kinematics.scale_to_limits(self._wheel_speeds_for(cmd, steer),
                                             self._limits, cmd)
        steer = self._steering_angles(limited)
        limited = self._slow_until_steered(limited, steer, obs.joint_pos)
        speeds = self._wheel_speeds_for(limited, steer)

        pos = self._default_pose.clone()
        for name, angle in steer.items():
            pos[:, self._joint_index[name]] = angle
        vel = torch.zeros_like(self._default_pose)
        vel[:, self._wheel_idx] = speeds
        return JointTargets(pos=pos, vel=vel, effort=None, cmd=limited)

    def _steering_angles(self, cmd: torch.Tensor) -> dict[str, torch.Tensor]:
        """body command (vx, ., wz)에서 좌우 조향 관절 목표각을 계산한다."""

        v, w = cmd[:, 0], cmd[:, 2]
        wheelbase = max(float(self._params.wheelbase), 1e-6)
        # delta = atan(L * w / v)는 v가 0에 가까우면 발산하므로 부호를 유지한 하한을 둔다
        eps = torch.full_like(v, 1e-3)
        v_safe = torch.where(v.abs() > 1e-3, v, torch.where(v >= 0.0, eps, -eps))
        delta = torch.clamp(torch.atan(wheelbase * w / v_safe),
                            -self._params.steer_range, self._params.steer_range)
        return kinematics.ackermann_steer(delta, self._params)

    def _slow_until_steered(self, cmd: torch.Tensor,
                            steer: dict[str, torch.Tensor],
                            joint_pos: torch.Tensor | None) -> torch.Tensor:
        """조향각이 목표에 닿을 때까지 전진·회전 명령을 줄인다."""

        if joint_pos is None or not steer:
            return cmd
        err = torch.zeros(cmd.shape[0], device=cmd.device)
        for name, target in steer.items():
            err = torch.maximum(err, torch.abs(target - joint_pos[:, self._joint_index[name]]))
        tolerance = max(0.15 * float(self._params.steer_range), 1e-3)
        scale = torch.clamp(1.0 - err / tolerance, min=0.2, max=1.0)
        out = cmd.clone()
        out[:, 0] = out[:, 0] * scale
        out[:, 2] = out[:, 2] * scale
        return out

    def _wheel_speeds_for(self, cmd: torch.Tensor,
                          steer: dict[str, torch.Tensor]) -> torch.Tensor:
        """조향각을 반영한 구동 휠 각속도를 계산한다."""

        speeds = kinematics.wheel_speeds(self._A, cmd)
        if not self._steered_drive_wheels:
            return speeds

        # 휠 반지름은 개별 WheelFrame이 아니라 로봇 파라미터가 한 값으로 들고 있다
        # (build_wheel_matrix도 같은 값을 쓴다). 0 나눗셈 방지 하한도 동일하게 맞춘다.
        radius = max(float(self._params.wheel_radius), 1e-6)
        speeds = speeds.clone()
        for col, side in self._steered_drive_wheels:
            wheel = self._params.wheels[col]
            angle = steer.get(side)
            if angle is None:
                continue

            # 조향 전 접지 접선 방향을 조향각만큼 회전시켜, 그 방향의 접지점 속도를 구른다
            tx0, ty0 = float(wheel.axis[1]), float(-wheel.axis[0])
            norm = math.hypot(tx0, ty0)
            if norm < 1e-9:
                continue
            tx0, ty0 = tx0 / norm, ty0 / norm
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            tx = cos_a * tx0 - sin_a * ty0
            ty = sin_a * tx0 + cos_a * ty0

            # 접지점 속도 = body 속도 + 각속도 x 위치 (평면 강체 운동)
            x, y = float(wheel.pos[0]), float(wheel.pos[1])
            vx = cmd[:, 0] - cmd[:, 2] * y
            vy = cmd[:, 1] + cmd[:, 2] * x
            speeds[:, col] = (vx * tx + vy * ty) / radius
        return speeds

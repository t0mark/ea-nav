from __future__ import annotations

import numpy as np
import torch

from ..core.base import RobotCtrlParams

def build_wheel_matrix(params: RobotCtrlParams, device: str,
                       skid_yaw_scale: float = 1.0)        -> tuple[list[str], torch.Tensor, torch.Tensor]:

    names, rows, limits = [], [], []
    r = params.wheel_radius
    yaw_scale = skid_yaw_scale if params.base_tag == "skid" else 1.0
    for w in params.wheels:
        x, y = float(w.pos[0]), float(w.pos[1])
        if w.joint in params.mecanum_sign:

            if float(w.axis[1]) < 0.99:
                raise ValueError(f"{w.joint}: 매커넘 바퀴 축이 몸체 +y가 아님 "
                                 f"(axis={w.axis}) — 배분 행 전제 위반")
            s = params.mecanum_sign[w.joint]
            rows.append([1.0 / r, s / r, (s * x - y) / r])
        else:

            t = np.array([w.axis[1], -w.axis[0]])
            t /= max(np.linalg.norm(t), 1e-9)
            rows.append([t[0] / r, t[1] / r,
                         yaw_scale * (t[1] * x - t[0] * y) / r])
        names.append(w.joint)
        limits.append(params.wheel_vel_limit[w.joint])
    return (names,
            torch.tensor(rows, dtype=torch.float32, device=device),
            torch.tensor(limits, dtype=torch.float32, device=device))

def wheel_speeds(A: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:

    return cmd @ A.T

def feasible_scale(A: torch.Tensor, limits: torch.Tensor,
                   cmd: torch.Tensor) -> torch.Tensor:

    speeds = torch.abs(cmd @ A.T)

    worst = torch.amax(speeds / limits.unsqueeze(0), dim=1)
    scale = torch.clamp(1.0 / torch.clamp(worst, min=1e-9), max=1.0)
    return cmd * scale.unsqueeze(1)

def ackermann_steer(delta: torch.Tensor, params: RobotCtrlParams)        -> dict[str, torch.Tensor]:

    kappa = torch.tan(delta) / params.wheelbase
    out = {}
    for name in params.steer_joints:
        d = params.steer_y[name]
        angle = torch.atan2(params.wheelbase * kappa, 1.0 - d * kappa)
        out[name] = torch.clamp(angle, -params.steer_range, params.steer_range)
    return out

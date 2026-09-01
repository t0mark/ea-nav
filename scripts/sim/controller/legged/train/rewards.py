from __future__ import annotations

import torch

def track_lin_vel_exp(vel_b: torch.Tensor, cmd: torch.Tensor,
                      sigma: float) -> torch.Tensor:

    err = torch.sum((cmd[:, :2] - vel_b[:, :2]) ** 2, dim=1)
    return torch.exp(-err / sigma)

def track_ang_vel_exp(ang_b: torch.Tensor, cmd: torch.Tensor,
                      sigma: float) -> torch.Tensor:

    err = (cmd[:, 2] - ang_b[:, 2]) ** 2
    return torch.exp(-err / sigma)

def lin_vel_z_l2(vel_b: torch.Tensor) -> torch.Tensor:

    return vel_b[:, 2] ** 2

def ang_vel_xy_l2(ang_b: torch.Tensor) -> torch.Tensor:

    return torch.sum(ang_b[:, :2] ** 2, dim=1)

def flat_orientation_l2(gravity_b: torch.Tensor) -> torch.Tensor:

    return torch.sum(gravity_b[:, :2] ** 2, dim=1)

def torque_ratio_l2(tau: torch.Tensor, effort: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:

    ratio = tau / torch.clamp(effort, min=1e-6) * mask
    return torch.sum(ratio ** 2, dim=1)

def dof_acc_l2(qd: torch.Tensor, qd_prev: torch.Tensor,
               ctrl_dt: float) -> torch.Tensor:

    return torch.sum(((qd - qd_prev) / ctrl_dt) ** 2, dim=1)

def action_rate_l2(action: torch.Tensor,
                   prev_action: torch.Tensor) -> torch.Tensor:

    return torch.sum((action - prev_action) ** 2, dim=1)

def feet_air_time(last_air_time: torch.Tensor, first_contact: torch.Tensor,
                  cmd: torch.Tensor, threshold: float,
                  deadband: float) -> torch.Tensor:

    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward * (torch.norm(cmd[:, :2], dim=1) > deadband)

def feet_air_time_biped(air_time: torch.Tensor, contact_time: torch.Tensor,
                        cmd: torch.Tensor, threshold: float,
                        deadband: float) -> torch.Tensor:

    in_contact = contact_time > 0.0
    in_mode = torch.where(in_contact, contact_time, air_time)
    single_stance = in_contact.int().sum(dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode,
                                   torch.zeros_like(in_mode)), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    return reward * (torch.norm(cmd[:, :2], dim=1) > deadband)

def undesired_contacts(forces: torch.Tensor, threshold: float) -> torch.Tensor:

    return (torch.norm(forces, dim=-1) > threshold).float().sum(dim=1)

def feet_slide(feet_vel_xy: torch.Tensor,
               in_contact: torch.Tensor) -> torch.Tensor:

    return (torch.norm(feet_vel_xy, dim=-1) * in_contact.float()).sum(dim=1)

def joint_deviation_l1(q_err: torch.Tensor,
                       dev_mask: torch.Tensor) -> torch.Tensor:

    return torch.sum(torch.abs(q_err) * dev_mask, dim=1)

def dof_pos_limits(q: torch.Tensor, soft_lower: torch.Tensor,
                   soft_upper: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:

    low = torch.clamp(soft_lower - q, min=0.0)
    high = torch.clamp(q - soft_upper, min=0.0)
    return torch.sum((low + high) * mask, dim=1)

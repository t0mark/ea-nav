from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from ..core.base import (BaseController, ControlObs, JointTargets,
                         RobotCtrlParams)
from ..core.pure_pursuit import PurePursuit
from .rl import bundle as low_rl

logger = logging.getLogger(__name__)

class LeggedRobotController(BaseController):

    def __init__(self, params: RobotCtrlParams, joint_names: list[str],
                 default_pose: torch.Tensor, num_envs: int, device: str,
                 cfg: dict, physics_dt: float, urdf_path: Path,
                 usd_dir: Path, policy_dir: Path):

        super().__init__(params, joint_names, default_pose, num_envs, device)
        self._bundle = low_rl.PolicyBundle.load(policy_dir, device)
        meta = self._bundle.meta

        self._slots = low_rl.build_slots(urdf_path, usd_dir, meta["morph"])
        if self._slots.form != meta["form"]:
            raise ValueError(f"정책 form 불일치: 번들 {meta['form']} vs "
                             f"로봇 {self._slots.form}")
        self._num_rays = low_rl.scan_rays(meta["scan"])
        if low_rl.obs_dim(self._slots.form, self._num_rays,
                          meta["obs"]) != int(meta["obs_dim"]):
            raise ValueError("관측 차원 불일치 — 번들과 코드의 규약 버전이 다름")

        self.decimation = int(meta["decimation"])
        if abs(float(meta["physics_dt"]) - physics_dt) > 1e-9:
            logger.warning("물리 스텝 불일치: 학습 %.4fs vs 실행 %.4fs — 정책 유효성 저하",
                           float(meta["physics_dt"]), physics_dt)

        S = self._slots.num_slots
        self._slot_idx = low_rl.slot_index_tensor(self._slots, self._joint_index,
                                                  device)
        self._mask = torch.tensor(self._slots.mask, dtype=torch.float32,
                                  device=device)
        self._slot_default = torch.tensor(self._slots.default,
                                          dtype=torch.float32, device=device)
        self._morph = torch.tensor(self._slots.morph, dtype=torch.float32,
                                   device=device).unsqueeze(0).expand(num_envs, -1)
        self._valid = self._mask > 0
        self._valid_idx = self._slot_idx[self._valid]
        self._action_scale = float(meta["action_scale"])
        self._action_clip = float(meta["action_clip"])
        self._prev_action = torch.zeros(num_envs, S, device=device)

        with open(Path(usd_dir) / "meta.json") as f:
            robot_meta = json.load(f)
        lim = low_rl.cmd_limits(robot_meta["params"], meta["cmd"])
        ratio = float(meta["cmd"].get("deploy_ratio", 1.0))
        v_max = ratio * lim["v_max"]
        w_max = ratio * lim["wz_max"]

        train_info = meta.get("train", {})
        reached_v = train_info.get("reached_v", {}).get(robot_meta["name"])
        if reached_v is not None:

            v_new = ratio * float(reached_v)
            if v_new < v_max:
                logger.warning("배포 상한 %.2f -> %.2f m/s (%.2f x 로봇 도달 %.2f)",
                               v_max, v_new, ratio, float(reached_v))
            v_max = v_new
        else:
            global_ratio = train_info.get("cmd_reached_ratio")
            if global_ratio is not None and float(global_ratio) < ratio:
                logger.warning("명령 커리큘럼 도달률 %.2f < deploy_ratio %.2f — "
                               "배포 상한을 도달률로 제한", float(global_ratio), ratio)
                v_max = float(global_ratio) * lim["v_max"]
                w_max = float(global_ratio) * lim["wz_max"]
        self._cmd_low = torch.tensor([-v_max, 0.0, -w_max], device=device)
        self._cmd_high = torch.tensor([v_max, 0.0, w_max], device=device)

        self.nav_limits = {"v_max": v_max, "w_max": w_max,
                           "r_turn": v_max / max(w_max, 1e-6)}

        pp_cfg = dict(cfg["pp"])
        bounds = torch.tensor([[-v_max, v_max], [-w_max, w_max]], device=device)
        self._pp = PurePursuit("unicycle", bounds, pp_cfg, num_envs, device,
                               decel=float(cfg["ctrl"]["lin_accel"]),
                               pivot_creep=0.0)

    def reset(self, env_ids: torch.Tensor | None = None):

        self._pp.reset(env_ids)
        if env_ids is None:
            self._prev_action.zero_()
        else:
            self._prev_action[env_ids] = 0.0

    def compute(self, obs: ControlObs, goal_xy: torch.Tensor) -> JointTargets:

        u = self._pp.plan(obs.pos_xy, obs.yaw, goal_xy)
        cmd = torch.stack([u[:, 0], torch.zeros_like(u[:, 0]), u[:, 1]], dim=1)
        cmd = torch.clamp(cmd, self._cmd_low, self._cmd_high)

        q_err = low_rl.slot_gather(obs.joint_pos, self._slot_idx, self._mask)            - self._slot_default * self._mask
        qd = low_rl.slot_gather(obs.joint_vel, self._slot_idx, self._mask)
        scan = obs.height_scan if obs.height_scan is not None else            torch.zeros(self._num_envs, self._num_rays, device=self._device)
        obs_vec = low_rl.assemble_obs(obs.vel_b, obs.ang_b, obs.gravity_b, cmd,
                                      q_err, qd, self._prev_action, self._morph,
                                      scan, self._bundle.meta["obs"])

        action = torch.clamp(self._bundle.act(obs_vec),
                             -self._action_clip, self._action_clip) * self._mask

        self._prev_action = action

        targets = self._default_pose.clone()
        slot_target = self._slot_default + self._action_scale * action
        targets[:, self._valid_idx] = slot_target[:, self._valid]
        return JointTargets(pos=targets, vel=None, effort=None, cmd=cmd)

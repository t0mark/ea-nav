from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from isaaclab.sensors import (ContactSensor, ContactSensorCfg, RayCaster,
                              RayCasterCfg, patterns)

from scripts.sim.controller.legged import low_rl
from scripts.sim.controller.legged.train import rewards
from scripts.sim.utils.environment import SimEnvironment

logger = logging.getLogger(__name__)

_ENV_NS = "/World/envs"

_PENALTY_TERMS = ("lin_vel_z", "ang_vel_xy", "orientation", "torque",
                  "dof_acc", "action_rate", "undesired", "dof_limits",
                  "joint_dev", "feet_slide")

class LeggedTrainEnv(VecEnv):

    def __init__(self, form: str, robot_dirs: list[tuple[Path, Path]],
                 rl_cfg: dict, sim_cfg: dict, envs_per_robot: int,
                 device: str, seed: int, terrain_cfg: dict | None):

        env_cfg = rl_cfg["env"]
        self._rl_cfg = rl_cfg
        self._form = form
        self._physics_dt = float(env_cfg["physics_dt"])
        self._decimation = int(env_cfg["decimation"])
        self._ctrl_dt = self._physics_dt * self._decimation
        self._action_scale = float(env_cfg["action_scale"])
        self._action_clip = float(env_cfg["action_clip"])
        self._rough = terrain_cfg is not None

        self._curr_up = 0.5 * float(terrain_cfg["size"][0]) if self._rough else 0.0
        self._num_rays = low_rl.scan_rays(rl_cfg["scan"])
        self._scan_clip = float(rl_cfg["scan"]["clip"])

        K, M = len(robot_dirs), envs_per_robot
        self.num_envs = K * M
        self.num_actions = low_rl.FORM_SLOTS[form][0] * low_rl.FORM_SLOTS[form][1]
        self.max_episode_length = int(round(float(env_cfg["episode_len_s"])
                                            / self._ctrl_dt))
        self.device = device
        self.cfg = rl_cfg
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long,
                                              device=device)

        self._robot_names = [Path(d).name for _, d in robot_dirs]
        self._slots = [low_rl.build_slots(u, d, rl_cfg["morph"])
                       for u, d in robot_dirs]
        overrides = [low_rl.loco_gain_overrides(s, rl_cfg["gains"])
                     for s in self._slots]

        self._env = SimEnvironment(self._physics_dt, device)
        if self._rough:
            self._env.add_terrain(terrain_cfg, num_envs=self.num_envs)
            g_idx = torch.arange(self.num_envs, device=device) // M
            k_idx = torch.arange(self.num_envs, device=device) % M
            self._perm = k_idx * K + g_idx
            spawn_origins = self._env.terrain.env_origins[self._perm]
        else:
            spacing = float(env_cfg["spacing"])
            half = 0.5 * spacing * math.ceil(math.sqrt(self.num_envs)) + spacing
            self._env.add_ground(size=max(20.0, half),
                                 friction=sim_cfg["contact"]["ground_friction"])
            self._perm = None
            spawn_origins = None

        self._env.spawn_robot_groups(
            [d for _, d in robot_dirs], sim_cfg["drive"], envs_per_robot,
            spacing=float(env_cfg["spacing"]),
            spawn_margin=sim_cfg["sim_run"]["spawn_margin"],

            self_collision=bool(env_cfg["self_collision"]),
            gain_overrides_list=overrides, origins=spawn_origins,
            activate_contact_sensors=True,
            friction_links_list=[list(s.contact_links) for s in self._slots],
            friction_links_mu=float(sim_cfg["contact"]["foot_friction"]),
            solver_iters=tuple(env_cfg["solver_iters"]),

            actuator_model=str(env_cfg["actuator_model"]))

        self._make_sensors(rl_cfg)
        self._env.reset()

        S = self.num_actions
        soft = float(rl_cfg["rewards"]["dof_soft_ratio"])
        self._slot_idx, self._masks, self._defaults = [], [], []
        self._valid, self._valid_idx, self._efforts, self._morphs = [], [], [], []
        self._base_heights, self._feet_body_ids = [], []
        self._soft_lower, self._soft_upper, self._dev_masks = [], [], []
        self._air_thresh = []
        for g, ((urdf, usd_dir), slots) in enumerate(zip(robot_dirs, self._slots)):
            art = self._env.robots[g]

            ids, _ = art.find_bodies([f"^{n}$" for n in slots.contact_links])
            self._feet_body_ids.append(torch.tensor(ids, dtype=torch.long,
                                                    device=device))
            jmap = {n: i for i, n in enumerate(art.joint_names)}
            idx = low_rl.slot_index_tensor(slots, jmap, device)
            mask = torch.tensor(slots.mask, dtype=torch.float32, device=device)
            self._slot_idx.append(idx)
            self._masks.append(mask)
            self._defaults.append(torch.tensor(slots.default, dtype=torch.float32,
                                               device=device))
            self._valid.append(mask > 0)
            self._valid_idx.append(idx[mask > 0])
            self._efforts.append(torch.tensor(slots.effort, dtype=torch.float32,
                                              device=device))
            self._morphs.append(torch.tensor(slots.morph, dtype=torch.float32,
                                             device=device).unsqueeze(0).expand(M, -1))

            lower = torch.tensor(slots.lower, dtype=torch.float32, device=device)
            upper = torch.tensor(slots.upper, dtype=torch.float32, device=device)
            center, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
            self._soft_lower.append(center - soft * half)
            self._soft_upper.append(center + soft * half)

            dev = [1.0 if (n and "hip" in n and ("yaw" in n or "roll" in n))
                   else 0.0 for n in slots.names]

            if float(rl_cfg["rewards"]["joint_deviation"]) != 0.0                    and sum(dev) == 0:
                logger.warning("그룹 %d: joint_deviation 대상 슬롯 0개 — "
                               "고관절 이름 규약 확인 필요", g)
            self._dev_masks.append(torch.tensor(dev, dtype=torch.float32,
                                                device=device))
            with open(Path(usd_dir) / "meta.json") as f:
                meta = json.load(f)
            self._base_heights.append(float(meta["metrics"]["base_height"]))

            w = rl_cfg["rewards"]
            thresh = float(w["feet_air_thresh_scale"])                * math.sqrt(float(meta["params"]["stance_height"]) / 9.81)
            self._air_thresh.append(min(max(thresh, float(w["feet_air_thresh_min"])),
                                        float(w["feet_air_thresh_max"])))

        v_max = torch.zeros(self.num_envs, device=device)
        for g, (_, usd_dir) in enumerate(robot_dirs):
            with open(Path(usd_dir) / "meta.json") as f:
                params = json.load(f)["params"]
            lim = low_rl.cmd_limits(params, rl_cfg["cmd"])
            v_max[self._env.group_slice(g)] = lim["v_max"]
        self._v_max = v_max
        self._wz_max = float(rl_cfg["cmd"]["wz_max"])
        self._heading_mode = bool(rl_cfg["cmd"].get("heading_command", False))
        self._heading_kp = float(rl_cfg["cmd"].get("heading_kp", 0.5))

        self._v_lim = torch.minimum(
            torch.full_like(v_max, float(rl_cfg["cmd"]["curriculum_v_start"])),
            v_max)

        self._track_sum = torch.zeros(self.num_envs, device=device)
        self._track_steps = torch.zeros(self.num_envs, dtype=torch.long,
                                        device=device)

        self._walked_sum = [0.0] * K
        self._walked_n = [0] * K

        self._obs_buf = torch.zeros(self.num_envs,
                                    low_rl.obs_dim(form, self._num_rays,
                                                   rl_cfg["obs"]),
                                    device=device)
        self._commands = torch.zeros(self.num_envs, 3, device=device)
        self._prev_action = torch.zeros(self.num_envs, S, device=device)
        self._prev_qd = torch.zeros(self.num_envs, S, device=device)

        self._prev_root_xy = torch.zeros(self.num_envs, 2, device=device)

        self._heading = torch.zeros(self.num_envs, device=device)
        self._standing = torch.zeros(self.num_envs, dtype=torch.bool,
                                     device=device)
        self._cur_return = torch.zeros(self.num_envs, device=device)
        self._return_hist = deque(maxlen=200)
        self._len_hist = deque(maxlen=200)
        self._term_count = 0

        self._since_reset = torch.zeros(self.num_envs, dtype=torch.long,
                                        device=device)

        self._global_step = 0
        self._rng = torch.Generator(device=device)
        self._rng.manual_seed(seed)

        self._tilt_cos = math.cos(math.radians(float(env_cfg["max_tilt_deg"])))
        self._contact_thresh = float(env_cfg["contact_force_thresh"])

        self._term_height_ratio = float(env_cfg["term_height_ratio"])
        self._only_positive = bool(rl_cfg["rewards"]["only_positive_rewards"])

        self._track_avg_vel = bool(rl_cfg["rewards"]["track_avg_vel"])

        self._reset_envs(torch.arange(self.num_envs, device=device))
        self._compute_obs()
        logger.info("학습 환경 구성: form=%s 로봇 %d종 x env %d = %d envs, "
                    "obs %d (스캔 %d) / act %d, 제어 %.0fHz, 지형 %s",
                    form, K, M, self.num_envs, self._obs_buf.shape[1],
                    self._num_rays, S, 1.0 / self._ctrl_dt,
                    "커리큘럼" if self._rough else "평지")

    def _make_sensors(self, rl_cfg: dict):

        scan_cfg = rl_cfg["scan"]
        self._scanners: list[RayCaster | None] = []
        self._feet_sensors: list[ContactSensor] = []
        self._base_sensors: list[ContactSensor] = []
        self._mid_sensors: list[ContactSensor] = []
        for g, slots in enumerate(self._slots):
            robot = f"{_ENV_NS}/env_.*/Robot_g{g:03d}"
            if self._rough:

                ray_cfg = RayCasterCfg(
                    prim_path=f"{robot}/{slots.root_link}",
                    mesh_prim_paths=["/World/ground"],
                    ray_alignment="yaw",
                    pattern_cfg=patterns.GridPatternCfg(
                        resolution=float(scan_cfg["resolution"]),
                        size=tuple(scan_cfg["size"])),
                    offset=RayCasterCfg.OffsetCfg(
                        pos=(0.0, 0.0, float(scan_cfg["offset_z"]))))
                self._scanners.append(RayCaster(ray_cfg))
            else:
                self._scanners.append(None)
            feet = "|".join(slots.contact_links)
            self._feet_sensors.append(ContactSensor(ContactSensorCfg(
                prim_path=f"{robot}/({feet})", track_air_time=True)))

            self._base_sensors.append(ContactSensor(ContactSensorCfg(
                prim_path=f"{robot}/{slots.root_link}", history_length=4)))

            mid = "|".join(slots.undesired_links)
            self._mid_sensors.append(ContactSensor(ContactSensorCfg(
                prim_path=f"{robot}/({mid})", history_length=3)))

    def get_observations(self) -> TensorDict:

        return TensorDict({"policy": self._obs_buf}, batch_size=[self.num_envs],
                          device=self.device)

    def step(self, actions: torch.Tensor):

        actions = torch.clamp(actions.to(self.device),
                              -self._action_clip, self._action_clip)

        for g in range(len(self._env.robots)):
            sl = self._env.group_slice(g)
            actions[sl] *= self._masks[g]

        targets = []
        for g, art in enumerate(self._env.robots):
            sl = self._env.group_slice(g)
            slot_target = self._defaults[g] + self._action_scale * actions[sl]
            full = art.data.default_joint_pos.clone()
            full[:, self._valid_idx[g]] = slot_target[:, self._valid[g]]
            targets.append(full)

        for g, art in enumerate(self._env.robots):
            self._prev_root_xy[self._env.group_slice(g)] =                art.data.root_pos_w[:, :2]

        if self._heading_mode:
            self._update_heading_cmd()

        push_every = max(1, int(round(float(self._rl_cfg["env"]["push_interval_s"])
                                      / self._ctrl_dt)))
        if self._global_step % push_every == 0 and self._global_step > 0:
            pv = float(self._rl_cfg["env"]["push_max_vel"])
            for art in self._env.robots:
                vel = art.data.root_vel_w.clone()
                vel[:, :2] += (torch.rand(vel.shape[0], 2, generator=self._rng,
                                          device=self.device) * 2.0 - 1.0) * pv
                art.write_root_velocity_to_sim(vel)

        for i in range(self._decimation):
            self._env.step_multi(targets if i == 0 else None)

            for sensor in self._base_sensors:
                sensor.update(self._physics_dt)

        for sensor in (self._feet_sensors + self._mid_sensors):
            sensor.update(self._ctrl_dt)
        for scanner in self._scanners:
            if scanner is not None:
                scanner.update(self._ctrl_dt)
        self.episode_length_buf += 1
        self._since_reset += 1
        self._global_step += 1

        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for g, art in enumerate(self._env.robots):
            terminated[self._env.group_slice(g)] = self._group_termination(g, art)
        grace = int(self._rl_cfg["env"]["termination_grace_steps"])
        if grace > 0:
            terminated &= self._since_reset > grace
        timeout = self.episode_length_buf >= self.max_episode_length
        dones = terminated | timeout

        rew = torch.zeros(self.num_envs, device=self.device)
        log_terms: dict[str, torch.Tensor] = {}
        for g, art in enumerate(self._env.robots):
            sl = self._env.group_slice(g)
            rew[sl], terms = self._group_reward(g, art, actions[sl],
                                                terminated[sl])

            for k, v in terms.items():
                log_terms[k] = log_terms.get(k, 0.0) + v / len(self._env.robots)

        self._cur_return += rew
        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            self._return_hist.extend(self._cur_return[done_ids].tolist())
            self._len_hist.extend(self.episode_length_buf[done_ids].tolist())
            self._term_count += int(terminated.sum())
            self._cur_return[done_ids] = 0.0
            self._reset_envs(done_ids, timed_out=timeout[done_ids])

        resample_every = max(1, int(round(float(self._rl_cfg["env"]["resample_cmd_s"])
                                          / self._ctrl_dt)))
        tick = (self.episode_length_buf % resample_every == 0) & ~dones
        if tick.any():
            self._sample_commands(tick.nonzero(as_tuple=False).squeeze(-1))

        self._prev_action = actions.clone()
        self._prev_action[dones] = 0.0
        self._compute_obs()

        extras = {"time_outs": timeout,
                  "log": {f"/reward/{k}": v for k, v in log_terms.items()}}
        extras["log"]["/curriculum/cmd_v_lim"] = self._v_lim.mean()
        if self._rough:
            extras["log"]["/curriculum/terrain_level"] =                self._env.terrain.terrain_levels.float().mean()
        return (self.get_observations(), rew, dones.to(torch.long), extras)

    def reset(self):

        self._reset_envs(torch.arange(self.num_envs, device=self.device))
        self._compute_obs()
        return self.get_observations(), {}

    def _penalty_ramp(self) -> float:

        ramp_steps = float(self._rl_cfg["rewards"]["penalty_ramp_iters"])            * float(self._rl_cfg["ppo"]["num_steps_per_env"])
        return min(self._global_step / max(ramp_steps, 1.0), 1.0)

    def _group_reward(self, g: int, art, actions_g: torch.Tensor,
                      terminated_g: torch.Tensor):

        w = self._rl_cfg["rewards"]
        ramp = self._penalty_ramp()
        sl = self._env.group_slice(g)
        cmd = self._commands[sl]
        vel_b = art.data.root_lin_vel_b

        if self._track_avg_vel:
            dxy = (art.data.root_pos_w[:, :2] - self._prev_root_xy[sl])                / self._ctrl_dt
            yaw_q = art.data.root_quat_w
            w_, x_, y_, z_ = yaw_q[:, 0], yaw_q[:, 1], yaw_q[:, 2], yaw_q[:, 3]
            yaw = torch.atan2(2.0 * (w_ * z_ + x_ * y_),
                              1.0 - 2.0 * (y_ * y_ + z_ * z_))
            c, s = torch.cos(yaw), torch.sin(yaw)
            vel_track = torch.stack([c * dxy[:, 0] + s * dxy[:, 1],
                                     -s * dxy[:, 0] + c * dxy[:, 1],
                                     torch.zeros_like(yaw)], dim=1)
        else:
            vel_track = vel_b
        ang_b = art.data.root_ang_vel_b
        grav = art.data.projected_gravity_b
        q = low_rl.slot_gather(art.data.joint_pos, self._slot_idx[g],
                               self._masks[g])
        q_err = q - self._defaults[g] * self._masks[g]
        qd = low_rl.slot_gather(art.data.joint_vel, self._slot_idx[g],
                                self._masks[g])
        tau = low_rl.slot_gather(art.data.applied_torque, self._slot_idx[g],
                                 self._masks[g])
        feet = self._feet_sensors[g]

        track_lin = rewards.track_lin_vel_exp(vel_track, cmd,
                                              float(w["track_lin_sigma"]))

        score_min = float(self._rl_cfg["cmd"]["curriculum_score_min_cmd"])
        valid_cmd = torch.norm(cmd[:, :2], dim=1) > score_min
        self._track_sum[sl] += track_lin * valid_cmd
        self._track_steps[sl] += valid_cmd
        terms = {
            "track_lin": float(w["track_lin"]) * track_lin,
            "track_ang": float(w["track_ang"]) * rewards.track_ang_vel_exp(
                ang_b, cmd, float(w["track_ang_sigma"])),
            "lin_vel_z": float(w["lin_vel_z"]) * rewards.lin_vel_z_l2(vel_b),
            "ang_vel_xy": float(w["ang_vel_xy"]) * rewards.ang_vel_xy_l2(ang_b),
            "orientation": float(w["orientation"]) * rewards.flat_orientation_l2(grav),
            "torque": float(w["torque"]) * rewards.torque_ratio_l2(
                tau, self._efforts[g], self._masks[g]),
            "dof_acc": float(w["dof_acc"]) * rewards.dof_acc_l2(
                qd, self._prev_qd[sl], self._ctrl_dt),
            "action_rate": float(w["action_rate"]) * rewards.action_rate_l2(
                actions_g, self._prev_action[sl]),

            "undesired": float(w["undesired_contact"]) * rewards.undesired_contacts(
                self._mid_sensors[g].data.net_forces_w, self._contact_thresh),
        }

        if float(w["dof_pos_limits"]) != 0.0:
            terms["dof_limits"] = float(w["dof_pos_limits"]) * rewards.dof_pos_limits(
                q, self._soft_lower[g], self._soft_upper[g], self._masks[g])

        if float(w["joint_deviation"]) != 0.0:
            terms["joint_dev"] = float(w["joint_deviation"])                * rewards.joint_deviation_l1(q_err, self._dev_masks[g])
        if self._form == "humanoid":

            terms["feet_air"] = float(w["feet_air_time_biped"])                * rewards.feet_air_time_biped(
                    feet.data.current_air_time, feet.data.current_contact_time,
                    cmd, float(w["feet_air_threshold_biped"]),
                    float(w["cmd_deadband"]))
            in_contact = torch.norm(feet.data.net_forces_w, dim=-1)                > self._contact_thresh
            feet_vel = art.data.body_lin_vel_w[:, self._feet_body_ids[g], :2]
            terms["feet_slide"] = float(w["feet_slide"]) * rewards.feet_slide(
                feet_vel, in_contact)
        else:
            first_contact = feet.compute_first_contact(self._ctrl_dt)
            terms["feet_air"] = float(w["feet_air_time"]) * rewards.feet_air_time(
                feet.data.last_air_time, first_contact, cmd,
                self._air_thresh[g], float(w["cmd_deadband"]))
        self._prev_qd[sl] = qd

        if ramp < 1.0:
            for k in _PENALTY_TERMS:
                if k in terms:
                    terms[k] = terms[k] * ramp
        total = sum(terms.values()) * self._ctrl_dt

        if self._only_positive:
            total = torch.clamp(total, min=0.0)

        term_pen = float(w["termination"]) * terminated_g.float()
        terms["termination"] = term_pen
        return total + term_pen, {k: v.mean() for k, v in terms.items()}

    def _group_termination(self, g: int, art) -> torch.Tensor:

        hist = self._base_sensors[g].data.net_forces_w_history[:, :, 0]
        base_force = torch.norm(hist, dim=-1).max(dim=1).values
        tilted = art.data.projected_gravity_b[:, 2] > -self._tilt_cos
        if self._rough and self._scanners[g] is not None:
            ground = self._scanners[g].data.ray_hits_w[:, :, 2]                .median(dim=1).values
        else:
            ground = 0.0
        collapsed = (art.data.root_pos_w[:, 2] - ground)            < self._term_height_ratio * self._base_heights[g]
        return (base_force > self._contact_thresh) | tilted | collapsed

    def _origins_of(self, env_ids: torch.Tensor) -> torch.Tensor:

        if self._rough:
            return self._env.terrain.env_origins[self._perm[env_ids]]
        return self._env.origins[env_ids]

    def _reset_envs(self, env_ids: torch.Tensor,
                    timed_out: torch.Tensor | None = None):

        env_cfg = self._rl_cfg["env"]
        cmd_cfg = self._rl_cfg["cmd"]
        js_lo, js_hi = (float(env_cfg["reset_joint_scale"][0]),
                        float(env_cfg["reset_joint_scale"][1]))
        xy_n = float(env_cfg["reset_pos_xy_noise"])
        yn = float(env_cfg["yaw_noise"])

        if timed_out is not None:
            steps = self._track_steps[env_ids]
            perf = self._track_sum[env_ids]                / torch.clamp(steps.float(), min=1.0)
            good = timed_out & (steps > 0)                & (perf > float(cmd_cfg["curriculum_track_threshold"]))
            grow = env_ids[good]
            self._v_lim[grow] = torch.minimum(
                self._v_lim[grow] + float(cmd_cfg["curriculum_v_step"]),
                self._v_max[grow])
        self._track_sum[env_ids] = 0.0
        self._track_steps[env_ids] = 0

        for g, art in enumerate(self._env.robots):
            sl = self._env.group_slice(g)
            in_group = (env_ids >= sl.start) & (env_ids < sl.stop)
            if not in_group.any():
                continue
            ids = env_ids[in_group]
            local = ids - sl.start

            walked = torch.norm(art.data.root_pos_w[local, :2]
                                - self._origins_of(ids)[:, :2], dim=1)
            self._walked_sum[g] += float(walked.sum())
            self._walked_n[g] += len(ids)

            if self._rough:
                required = torch.norm(self._commands[ids, :2], dim=1)                    * self.max_episode_length * self._ctrl_dt
                move_up = walked > self._curr_up
                move_down = (walked < 0.5 * required) & ~move_up
                self._env.terrain.update_env_origins(self._perm[ids],
                                                     move_up, move_down)

            root = art.data.default_root_state[local].clone()
            root[:, :3] += self._origins_of(ids)
            root[:, :2] += (torch.rand(len(local), 2, generator=self._rng,
                                       device=self.device) * 2.0 - 1.0) * xy_n
            yaw = (torch.rand(len(local), generator=self._rng,
                              device=self.device) * 2.0 - 1.0) * yn
            root[:, 3] = torch.cos(0.5 * yaw)
            root[:, 4:6] = 0.0
            root[:, 6] = torch.sin(0.5 * yaw)
            vn = float(env_cfg["reset_vel_noise"])
            root[:, 7:] = (torch.rand(len(local), 6, generator=self._rng,
                                      device=self.device) * 2.0 - 1.0) * vn
            art.write_root_pose_to_sim(root[:, :7], env_ids=local)
            art.write_root_velocity_to_sim(root[:, 7:], env_ids=local)

            q = art.data.default_joint_pos[local].clone()
            scale = torch.rand(len(local), len(self._valid_idx[g]),
                               generator=self._rng, device=self.device)                * (js_hi - js_lo) + js_lo
            v = self._valid[g]
            q[:, self._valid_idx[g]] = torch.clamp(
                q[:, self._valid_idx[g]] * scale,
                self._soft_lower[g][v], self._soft_upper[g][v])
            art.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=local)

            art.update(0.0)

            self._feet_sensors[g].reset(env_ids=local)
            self._base_sensors[g].reset(env_ids=local)
            self._mid_sensors[g].reset(env_ids=local)
            if self._scanners[g] is not None:
                self._scanners[g].reset(env_ids=local)

        self.episode_length_buf[env_ids] = 0
        self._since_reset[env_ids] = 0
        self._prev_action[env_ids] = 0.0
        self._prev_qd[env_ids] = 0.0
        self._sample_commands(env_ids)

    def _sample_commands(self, env_ids: torch.Tensor):

        env_cfg = self._rl_cfg["env"]
        n = len(env_ids)
        u = torch.rand(n, 5, generator=self._rng, device=self.device)

        v_max = self._v_lim[env_ids]
        back = float(env_cfg["back_ratio"])
        vy_r = float(env_cfg["vy_ratio"])
        cmd = torch.zeros(n, 3, device=self.device)
        cmd[:, 0] = (u[:, 0] * (1.0 + back) - back) * v_max
        cmd[:, 1] = (u[:, 1] * 2.0 - 1.0) * vy_r * v_max
        if not self._heading_mode:
            cmd[:, 2] = (u[:, 2] * 2.0 - 1.0) * self._wz_max
        standing = u[:, 3] < float(env_cfg["zero_cmd_prob"])
        cmd[standing] = 0.0

        small = torch.norm(cmd[:, :2], dim=1)            < float(self._rl_cfg["cmd"]["min_cmd_norm"])
        cmd[small, :2] = 0.0
        self._commands[env_ids] = cmd

        self._heading[env_ids] = (u[:, 4] * 2.0 - 1.0) * math.pi
        self._standing[env_ids] = standing

    def _update_heading_cmd(self):

        for g, art in enumerate(self._env.robots):
            sl = self._env.group_slice(g)
            q = art.data.root_quat_w
            yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                              1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            err = self._heading[sl] - yaw

            err = torch.atan2(torch.sin(err), torch.cos(err))
            wz = torch.clamp(self._heading_kp * err, -self._wz_max, self._wz_max)
            self._commands[sl, 2] = torch.where(self._standing[sl],
                                                torch.zeros_like(wz), wz)

    def _noisy(self, x: torch.Tensor, key: str) -> torch.Tensor:

        nc = self._rl_cfg["obs_noise"]
        if not nc["enabled"]:
            return x
        n = (torch.rand(x.shape, generator=self._rng, device=self.device)
             * 2.0 - 1.0) * float(nc[key])
        return x + n

    def _compute_obs(self):

        for g, art in enumerate(self._env.robots):
            sl = self._env.group_slice(g)
            q_err = self._noisy(
                low_rl.slot_gather(art.data.joint_pos, self._slot_idx[g],
                                   self._masks[g])
                - self._defaults[g] * self._masks[g], "joint_pos")                * self._masks[g]
            qd = self._noisy(
                low_rl.slot_gather(art.data.joint_vel, self._slot_idx[g],
                                   self._masks[g]), "joint_vel")                * self._masks[g]
            if self._scanners[g] is not None:
                scan = low_rl.scan_obs(art.data.root_pos_w[:, 2],
                                       self._scanners[g].data.ray_hits_w[:, :, 2],
                                       self._base_heights[g], self._scan_clip)
            else:

                scan = torch.zeros(sl.stop - sl.start, self._num_rays,
                                   device=self.device)
            self._obs_buf[sl] = low_rl.assemble_obs(
                self._noisy(art.data.root_lin_vel_b, "lin_vel"),
                self._noisy(art.data.root_ang_vel_b, "ang_vel"),
                self._noisy(art.data.projected_gravity_b, "gravity"),
                self._commands[sl], q_err, qd,
                self._prev_action[sl], self._morphs[g], scan,
                self._rl_cfg["obs"])

    def per_robot_reached(self) -> dict[str, float]:

        return {name: round(float(self._v_lim[self._env.group_slice(g)].mean()), 3)
                for g, name in enumerate(self._robot_names)}

    def pop_stats(self) -> dict:

        stats = {
            "episodes": len(self._return_hist),
            "mean_return": (sum(self._return_hist) / len(self._return_hist))
            if self._return_hist else 0.0,
            "mean_ep_len_s": (sum(self._len_hist) / len(self._len_hist)
                              * self._ctrl_dt) if self._len_hist else 0.0,
            "terminations": self._term_count,
        }

        stats["cmd_v_ratio"] = round(
            float((self._v_lim / self._v_max).mean()), 2)

        stats["walked_m"] = {name: round(self._walked_sum[g] / max(self._walked_n[g], 1), 2)
                             for g, name in enumerate(self._robot_names)}
        self._walked_sum = [0.0] * len(self._robot_names)
        self._walked_n = [0] * len(self._robot_names)
        if self._rough:
            stats["terrain_level"] = round(
                float(self._env.terrain.terrain_levels.float().mean()), 2)
        self._return_hist.clear()
        self._len_hist.clear()
        self._term_count = 0
        return stats

from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path

import torch

from isaaclab.sensors import ContactSensor, ContactSensorCfg

from scripts.sim.controller.core.base import ControlObs, make_controller
from scripts.sim.controller.wheeled.rl import episode
from scripts.sim.controller.wheeled.rl import obs as obs_lib
from scripts.sim.controller.wheeled.rl import rewards
from scripts.sim.controller.wheeled.rl.adapter import ACTION_PARAM_NAMES, validate_schema_cfg
from scripts.sim.utils.environment import SimEnvironment
from scripts.sim.utils.robot_spawn import standard_friction_links

logger = logging.getLogger(__name__)

_ENV_NS = "/World/envs"

class WheeledParameterTrainEnv:
    """wheeled controller parameter 정책을 학습하는 시뮬레이션 환경.

    env 1개 = 로봇(URDF) 1대의 실제 시뮬 인스턴스다. num_envs = K*M (K=morphology 종수,
    M=종당 동시 인스턴스 수). step() 1번은 decision 1번(기본 1초)이고, RL은 이 경계에서만
    parameter 배율을 다시 뽑는다 — Daffan/APPLR·Daffan/ros_jackal의 _take_action이 파라미터를
    한 번 설정한 뒤 time_step초 동안 그대로 두는 것과 같은 구조다. 종료·목표·reward의 순수
    계산은 episode.py/rewards.py에 있고, 이 클래스는 시뮬 상태를 들고 그 계산을 순서대로
    호출하는 오케스트레이션만 맡는다.
    """

    def __init__(self, robot_dirs: list[tuple[Path, Path]], rl_cfg: dict,
                 sim_cfg: dict, ctrl_cfg: dict, envs_per_robot: int,
                 device: str, seed: int, terrain_cfg: dict | None):

        env_cfg = rl_cfg["env"]
        self._rl_cfg = rl_cfg
        self._physics_dt = float(env_cfg["physics_dt"])
        self._decimation = int(env_cfg["decimation"])
        self._ctrl_dt = self._physics_dt * self._decimation
        self._decision_period_s = float(env_cfg.get("decision_period_s", 1.0))
        self._decision_ticks = max(1, round(self._decision_period_s / self._ctrl_dt))
        self._rough = terrain_cfg is not None
        self._terrain_cfg = terrain_cfg
        self.device = device

        K, M = len(robot_dirs), envs_per_robot
        self.num_envs = K * M
        self.num_actions = len(ACTION_PARAM_NAMES)
        validate_schema_cfg(env_cfg.get("param_schema"))
        self._obs_spec = obs_lib.ObsSpec.from_cfg(rl_cfg.get("obs"))
        self._state_cfg = dict(rl_cfg.get("state", {}))
        self.obs_dim = self._obs_spec.dim(self.num_actions)
        self.max_episode_length = max(1, round(float(env_cfg["episode_len_s"])
                                               / self._decision_period_s))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)

        # 지형: 랜덤이면 절차적 지형 그리드를 만들고, 같은 로봇의 M개 env를 서로 다른
        # 칸에 흩어 배치한다 (한 칸에 몰리면 지형 일반화를 학습 신호가 못 본다)
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
            self._env.add_ground(size=max(20.0, half), friction=sim_cfg["contact"]["ground_friction"])
            self._perm = None
            spawn_origins = None

        # 로봇 그룹 스폰과 base contact sensor 부착
        metas, joints = [], []
        for _, usd_dir in robot_dirs:
            with open(Path(usd_dir) / "meta.json") as f:
                metas.append(json.load(f))
            with open(Path(usd_dir) / "joints.json") as f:
                joints.append(json.load(f))
        self._env.spawn_robot_groups(
            [d for _, d in robot_dirs], sim_cfg["drive"], M,
            spacing=float(env_cfg["spacing"]), spawn_margin=sim_cfg["sim_run"]["spawn_margin"],
            self_collision=bool(env_cfg["self_collision"]), origins=spawn_origins,
            activate_contact_sensors=True, contact_cfg=sim_cfg["contact"],
            friction_links_list=[standard_friction_links(meta["contact_links"]) for meta in metas],
            friction_links_mu=float(sim_cfg["contact"]["foot_friction"]),
            solver_iters=tuple(env_cfg["solver_iters"]),
            actuator_model=str(env_cfg.get("actuator_model", "implicit")))
        self._base_sensors = [
            ContactSensor(ContactSensorCfg(
                prim_path=f"{_ENV_NS}/env_.*/Robot_g{g:03d}/{info['base_link']}",
                history_length=max(4, self._decimation)))
            for g, info in enumerate(joints)]
        self._env.reset()
        for g in range(K):
            if self._env.group_slice(g) != slice(g * M, (g + 1) * M):
                raise ValueError("로봇 그룹 env 구간이 연속이 아니다 — 그룹 텐서 결합 불가")

        # 로봇별 controller 생성. 학습 중엔 이 env가 매 decision 준 action으로 배율이
        # 채워지므로(compute_with_action) policy_path는 끄고 schema·범위만 넘긴다
        ctrl_train_cfg = copy.deepcopy(ctrl_cfg)
        ctrl_train_cfg.setdefault("wheeled_rl", {})
        ctrl_train_cfg["wheeled_rl"].update(
            policy_path=None,
            param_schema={k: list(v) for k, v in dict(env_cfg.get("param_schema", {})).items()},
            param_bounds={k: list(v) for k, v in dict(env_cfg["param_bounds"]).items()},
            obs=dict(rl_cfg.get("obs", {})), morph=dict(rl_cfg.get("morph", {})),
            state=dict(rl_cfg.get("state", {})))
        self._controllers = []
        static_rows = []
        for (urdf, usd_dir), art in zip(robot_dirs, self._env.robots):
            ctrl = make_controller(
                urdf, usd_dir, ctrl_train_cfg, joint_names=art.joint_names,
                default_pose=art.data.default_joint_pos.clone(), num_envs=M, device=device,
                physics_dt=self._physics_dt)
            self._controllers.append(ctrl)
            morph = obs_lib.morphology_vector(ctrl.params, device, rl_cfg.get("morph", {}))
            static_rows.append(self._obs_spec.build_static(ctrl.params, morph, M))
        # 정적(type+morph) 관측 행은 로봇 하나에 고정이라 여기서 한 번만 만든다
        self._static = torch.cat(static_rows, dim=0)

        # 정규화에 쓸 로봇별 schema mask(그룹 (1,N)을 env 수만큼 펼침)와 공통 parameter 범위
        self._active_mask = torch.cat(
            [ctrl.rl_adapter.active_mask.expand(M, -1) for ctrl in self._controllers], dim=0)
        self._param_min, self._param_max = self._controllers[0].rl_adapter.param_bounds

        self._prev_action = torch.zeros(self.num_envs, self.num_actions, device=device)
        self._prev_root_xy = torch.zeros(self.num_envs, 2, device=device)
        self._start_xy = torch.zeros(self.num_envs, 2, device=device)
        self._goals = torch.zeros(self.num_envs, 2, device=device)
        self._episode_reaches = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._stuck_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._since_reset = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._rng = torch.Generator(device=device)
        self._rng.manual_seed(seed)
        self._reset_stats()

        self._tilt_cos = math.cos(math.radians(float(env_cfg["max_tilt_deg"])))
        self._contact_thresh = float(env_cfg["contact_force_thresh"])
        self._termination_grace_ticks = int(env_cfg["termination_grace_steps"])
        self._stuck_speed_thresh = float(env_cfg["stuck_speed_thresh"])
        self._stuck_goal_scale = float(env_cfg["stuck_goal_scale"])
        self._stuck_tick_limit = max(1, round(float(env_cfg["stuck_time_s"]) / self._ctrl_dt))
        self._reach_radius = float(env_cfg.get("reach_radius", 0.35))
        terrain_size = self._terrain_cfg["size"] if self._rough else None
        self._goal_radius = episode.effective_goal_radius(
            env_cfg.get("goal_radius", [1.5, 3.0]), terrain_size, self._reach_radius,
            float(env_cfg["goal_margin"]))

        self._reset_envs(torch.arange(self.num_envs, device=device))
        self._obs_buf = self._compute_obs()
        logger.info("wheeled RL 학습 환경: 로봇 %d종 x env %d = %d envs, obs %d / act %d, "
                    "제어 %.0fHz / decision %.1fs(tick %d개), 지형 %s", K, M, self.num_envs,
                    self.obs_dim, self.num_actions, 1.0 / self._ctrl_dt,
                    self._decision_period_s, self._decision_ticks,
                    "mixed" if self._rough else "flat")

    def reset(self) -> torch.Tensor:
        """모든 env를 초기화하고 첫 관측을 반환한다."""

        self._reset_envs(torch.arange(self.num_envs, device=self.device))
        self._obs_buf = self._compute_obs()
        return self._obs_buf

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """decision 1번(control tick decision_ticks개)을 진행한다.

        action은 이 decision 동안 고정된 채 매 tick 그대로 쓰인다. env마다 decision 도중
        종료 시점이 다를 수 있어(전복 등), 이미 끝난 env는 active에서 빼고 남은 tick 동안
        정지 target만 받게 한다 — 벡터화된 배치 물리 스텝을 개별 env만 멈출 수 없어서다.
        """

        actions = torch.clamp(actions.to(self.device), -1.0, 1.0)
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError("wheeled RL action은 (env 수, parameter 수) 모양이어야 함")

        active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        final_terminated = torch.zeros_like(active)
        final_stuck = torch.zeros_like(active)
        env_return = torch.zeros(self.num_envs, device=self.device)
        log_sums: dict[str, torch.Tensor] = {}
        log_ticks = 0

        for _ in range(self._decision_ticks):
            if not active.any():
                break

            # 현재 action으로 target을 계산하고(비활성 env는 정지) 물리를 decimation만큼 진행
            pos_targets, vel_targets, effort_targets = [], [], []
            for g, art in enumerate(self._env.robots):
                sl = self._env.group_slice(g)
                self._prev_root_xy[sl] = art.data.root_pos_w[:, :2]
                ctrl_obs = ControlObs.from_articulation(art)
                targets = self._controllers[g].compute_with_action(ctrl_obs, self._goals[sl], actions[sl])
                inactive = ~active[sl]
                if inactive.any():
                    if targets.pos is not None:
                        targets.pos[inactive] = art.data.default_joint_pos[inactive]
                    if targets.vel is not None:
                        targets.vel[inactive] = 0.0
                    if targets.effort is not None:
                        targets.effort[inactive] = 0.0
                pos_targets.append(targets.pos)
                vel_targets.append(targets.vel)
                effort_targets.append(targets.effort)
            for i in range(self._decimation):
                self._env.step_multi(pos_targets if i == 0 else None,
                                     vel_targets if i == 0 else None,
                                     effort_targets if i == 0 else None)
                for sensor in self._base_sensors:
                    sensor.update(self._physics_dt)
            self._since_reset[active] += 1

            # 종료·stuck·목표 도달 판정
            terminated_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for g, art in enumerate(self._env.robots):
                terminated_now[self._env.group_slice(g)] = episode.group_termination(
                    self._base_sensors[g], art.data.projected_gravity_b[:, 2],
                    self._contact_thresh, self._tilt_cos)
            terminated_now &= self._since_reset > self._termination_grace_ticks
            dist = self._goal_dist()
            reached_now = dist < self._reach_radius
            vel_xy = self._root_field("root_lin_vel_b")[:, :2]
            stuck_now = episode.stuck_mask(dist, vel_xy, active, self._stuck_steps,
                                           self._stuck_goal_scale, self._reach_radius,
                                           self._stuck_speed_thresh, self._stuck_tick_limit)
            terminated_now |= stuck_now

            # reward 계산과 목표 도달 시 재샘플링
            reached_alive = active & reached_now & ~terminated_now
            pos_xy = self._root_field("root_pos_w")[:, :2]
            gravity_b = self._root_field("projected_gravity_b")
            base_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for g, sensor in enumerate(self._base_sensors):
                hist = sensor.data.net_forces_w_history[:, :, 0]
                base_hit[self._env.group_slice(g)] = (
                    torch.norm(hist, dim=-1).max(dim=1).values > self._contact_thresh)
            stall_dist = self._stuck_goal_scale * self._reach_radius
            rew_tick, terms_tick = rewards.step_reward(
                self._prev_root_xy, pos_xy, self._goals, vel_xy, gravity_b, base_hit,
                terminated_now & active, reached_alive, self._ctrl_dt, self._rl_cfg["rewards"],
                stall_dist, self._stuck_speed_thresh)
            env_return += torch.where(active, rew_tick, torch.zeros_like(rew_tick))
            for name, value in terms_tick.items():
                log_sums[name] = log_sums.get(name, torch.zeros((), device=self.device)) + value[active].mean()
            log_ticks += 1

            if reached_alive.any():
                ids = reached_alive.nonzero(as_tuple=False).squeeze(-1)
                self._episode_reaches[ids] += 1
                self._stuck_steps[ids] = 0
                self._goals[ids] = episode.sample_goals(self._origins(ids)[:, :2], self._goal_radius,
                                                         self._rng, self.device)

            newly_done = active & terminated_now
            final_terminated |= newly_done
            final_stuck |= newly_done & stuck_now
            active &= ~newly_done

        self.episode_length_buf += 1
        timeout = self.episode_length_buf >= self.max_episode_length
        dones = final_terminated | timeout

        # 배율 자체에 매기는 penalty라 decision 동안 값이 그대로다 — tick마다 더하면
        # progress 등 다른 항 대비 과도해지므로 decision당 1번만 더한다
        scales = torch.cat([ctrl.rl_adapter.effective_scales for ctrl in self._controllers], dim=0)
        reg = float(self._rl_cfg["rewards"]["parameter_regularization"]) * rewards.parameter_regularization(
            scales, self._active_mask, self._param_min, self._param_max)
        env_return += reg

        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            self._record_episode_stats(done_ids, env_return, final_terminated, final_stuck)
            self._reset_envs(done_ids, failed=final_terminated[done_ids])

        self._prev_action = actions.clone()
        self._prev_action[dones] = 0.0
        self._obs_buf = self._compute_obs()

        info = {f"reward/{k}": (v / max(log_ticks, 1)).item() for k, v in log_sums.items()}
        info["reward/parameter_regularization"] = reg.mean().item()
        info["timeout_rate"] = (timeout & ~final_terminated).float().mean().item()
        if self._rough:
            info["terrain_level"] = self._env.terrain.terrain_levels.float().mean().item()
        return self._obs_buf, env_return, dones, info

    def _reset_envs(self, env_ids: torch.Tensor, failed: torch.Tensor | None = None):
        """종료된 env의 로봇 자세·controller·목표·지형 커리큘럼을 초기화한다."""

        env_cfg = self._rl_cfg["env"]
        xy_noise = float(env_cfg.get("reset_pos_xy_noise", 0.1))
        yaw_noise = float(env_cfg.get("yaw_noise", math.pi))
        vel_noise = float(env_cfg.get("reset_vel_noise", 0.05))

        # 지형 커리큘럼: 이동 거리·도달 횟수·실패 여부로 다음 난이도를 정한다
        if self._rough and len(env_ids) > 0:
            was_running = self.episode_length_buf[env_ids] > 0
            if was_running.any():
                ids = env_ids[was_running]
                failed_running = (failed[was_running] if failed is not None
                                  else torch.zeros(len(ids), dtype=torch.bool, device=self.device))
                walked = torch.norm(self._root_field("root_pos_w")[ids, :2] - self._start_xy[ids], dim=1)
                move_up, move_down = episode.curriculum_move(
                    walked, self._episode_reaches[ids], failed_running,
                    int(env_cfg["curriculum_min_reaches"]), float(env_cfg["curriculum_walked_ratio"]),
                    float(env_cfg["curriculum_fail_walked_ratio"]), self._goal_radius[1])
                self._env.terrain.update_env_origins(self._perm[ids], move_up, move_down)

        for g, art in enumerate(self._env.robots):
            sl = self._env.group_slice(g)
            in_group = (env_ids >= sl.start) & (env_ids < sl.stop)
            if not in_group.any():
                continue
            ids = env_ids[in_group]
            local = ids - sl.start

            root = art.data.default_root_state[local].clone()
            root[:, :3] += self._origins(ids)
            root[:, :2] += (torch.rand(len(local), 2, generator=self._rng,
                                       device=self.device) * 2.0 - 1.0) * xy_noise
            yaw = (torch.rand(len(local), generator=self._rng,
                              device=self.device) * 2.0 - 1.0) * yaw_noise
            root[:, 3] = torch.cos(0.5 * yaw)
            root[:, 4:6] = 0.0
            root[:, 6] = torch.sin(0.5 * yaw)
            root[:, 7:] = (torch.rand(len(local), 6, generator=self._rng,
                                      device=self.device) * 2.0 - 1.0) * vel_noise
            art.write_root_pose_to_sim(root[:, :7], env_ids=local)
            art.write_root_velocity_to_sim(root[:, 7:], env_ids=local)
            art.write_joint_state_to_sim(art.data.default_joint_pos[local].clone(),
                                         torch.zeros_like(art.data.default_joint_vel[local]),
                                         env_ids=local)
            art.update(0.0)
            self._start_xy[ids] = root[:, :2]
            self._controllers[g].reset(local)
            self._base_sensors[g].reset(env_ids=local)

        self.episode_length_buf[env_ids] = 0
        self._since_reset[env_ids] = 0
        self._episode_reaches[env_ids] = 0
        self._stuck_steps[env_ids] = 0
        self._prev_action[env_ids] = 0.0
        self._goals[env_ids] = episode.sample_goals(self._origins(env_ids)[:, :2], self._goal_radius,
                                                     self._rng, self.device)

    def _origins(self, env_ids: torch.Tensor) -> torch.Tensor:
        """env별 스폰 원점(지형이 랜덤이면 배정된 칸, 아니면 격자 위치)을 반환한다."""

        if self._rough:
            return self._env.terrain.env_origins[self._perm[env_ids]]
        return self._env.origins[env_ids]

    def _goal_dist(self) -> torch.Tensor:
        """현재 목표까지의 거리를 반환한다."""

        return torch.norm(self._goals - self._root_field("root_pos_w")[:, :2], dim=1)

    def _root_field(self, name: str) -> torch.Tensor:
        """로봇 그룹별 root 상태를 env 순서대로 이어 붙인다."""

        return torch.cat([getattr(art.data, name) for art in self._env.robots])

    def _compute_obs(self) -> torch.Tensor:
        """정적(type+morph) 관측에 이번 decision의 실시간 상태와 직전 action을 이어 붙인다."""

        buf = torch.zeros(self.num_envs, self.obs_dim, device=self.device)
        for g, ctrl in enumerate(self._controllers):
            sl = self._env.group_slice(g)
            ctrl_obs = ControlObs.from_articulation(self._env.robots[g])
            buf[sl] = self._obs_spec.assemble(
                self._static[sl], ctrl_obs, self._goals[sl], self._prev_action[sl], self._state_cfg)
        return buf

    def _reset_stats(self):
        """episode 통계 누적값을 초기화한다."""

        self._return_sum = 0.0
        self._return_samples = 0
        self._len_sum = 0.0
        self._len_samples = 0
        self._episode_count = 0
        self._term_count = 0
        self._stuck_count = 0
        self._reach_count = 0
        self._reach_episode_count = 0

    def _record_episode_stats(self, done_ids: torch.Tensor, env_return: torch.Tensor,
                              terminated: torch.Tensor, stuck: torch.Tensor):
        """종료된 env의 episode 수익·길이·실패 통계를 누적한다."""

        self._return_sum += float(env_return[done_ids].sum())
        self._return_samples += int(done_ids.numel())
        self._len_sum += float(self.episode_length_buf[done_ids].sum())
        self._len_samples += int(done_ids.numel())
        self._episode_count += int(done_ids.numel())
        self._term_count += int(terminated[done_ids].sum())
        self._stuck_count += int(stuck[done_ids].sum())
        self._reach_count += int((self._episode_reaches[done_ids] > 0).sum())
        self._reach_episode_count += int((self._episode_reaches[done_ids] > 0).sum())

    def pop_stats(self) -> dict:
        """직전 구간의 episode 단위 통계를 반환하고 누적값을 비운다."""

        stats = {
            "episodes": self._episode_count,
            "mean_return": self._return_sum / max(self._return_samples, 1),
            "mean_ep_len_s": (self._len_sum / max(self._len_samples, 1)) * self._decision_period_s,
            "terminations": self._term_count,
            "stuck": self._stuck_count,
            "reached": self._reach_count,
            "reached_episodes": self._reach_episode_count,
        }
        if self._rough:
            stats["terrain_level"] = round(float(self._env.terrain.terrain_levels.float().mean()), 2)
        self._reset_stats()
        return stats

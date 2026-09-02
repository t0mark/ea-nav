from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch

from scripts.sim.controller.wheeled.rl import obs as obs_lib
from scripts.sim.controller.wheeled.rl.adapter import (ACTION_PARAM_NAMES,
                                                       schema_names,
                                                       validate_schema_cfg)
from scripts.sim.controller.wheeled.rl.bundle import PolicyBundle
from scripts.sim.controller.wheeled.rl.buffer import ReplayBuffer
from scripts.sim.controller.wheeled.rl.td3 import TD3
from scripts.sim.controller.wheeled.rl.train_env import WheeledParameterTrainEnv

logger = logging.getLogger(__name__)

def _robot_name(robot_dir: str | Path) -> str:
    """로봇 이름을 form/name 형식으로 정규화한다 (train/holdout 표기 통일)."""

    path = Path(robot_dir)
    return f"{path.parent.name}/{path.name}" if path.parent.name else path.name

def _exploration_noise(iteration: int, iterations: int, start: float, end: float) -> float:
    """Daffan/ros_jackal train.py와 같은 선형 감쇠로 이번 iteration의 탐색 noise를 정한다."""

    frac = min(iteration / max(iterations, 1), 1.0)
    return start - (start - end) * frac

def train(robot_dirs: list[tuple[Path, Path]], rl_cfg: dict, sim_cfg: dict,
          ctrl_cfg: dict, out_dir: Path, mode: str, device: str,
          holdout_names: list[str] | None = None) -> dict:
    """wheeled controller parameter TD3 정책 학습 진입점이다.

    Daffan/APPLR·Daffan/ros_jackal의 train.py와 같은 구조다: pre-collect로 buffer를 채운
    뒤, 매 iteration마다 (수집 -> TD3 업데이트 update_per_step회 -> 로그) 순서로 돈다.
    원본은 단일 env를 Python 루프로 도는 collector를 쓰지만, 여기서는 env가 이미 K*M개를
    GPU에서 벡터로 동시에 진행하므로 그 배치를 그대로 buffer에 밀어넣는다. 학습 대상은
    robot_dirs에 든 로봇뿐이고, holdout_names는 번들에만 기록한다.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = rl_cfg["train"]
    scale = train_cfg[mode]
    iterations = int(scale["iterations"])
    chunk = max(1, min(int(train_cfg["chunk_iters"]), iterations))
    seed = int(train_cfg["seed"])
    torch.manual_seed(seed)
    validate_schema_cfg(rl_cfg["env"].get("param_schema"))
    obs_spec = obs_lib.ObsSpec.from_cfg(rl_cfg.get("obs"))

    terrain_mode = scale.get("terrain_mode", rl_cfg["env"]["terrain"])
    terrain_cfg = ({**sim_cfg["terrain"], **scale.get("terrain", {})}
                   if terrain_mode == "mixed" else None)

    env = WheeledParameterTrainEnv(
        robot_dirs, rl_cfg, sim_cfg, ctrl_cfg,
        envs_per_robot=int(scale["envs_per_robot"]),
        device=device, seed=seed, terrain_cfg=terrain_cfg)

    td3_cfg = rl_cfg["td3"]
    policy = TD3(obs_dim=env.obs_dim, action_dim=env.num_actions,
                hidden_dims=list(td3_cfg["hidden_dims"]), device=device,
                gamma=float(td3_cfg["gamma"]), tau=float(td3_cfg["tau"]),
                policy_noise=float(td3_cfg["policy_noise"]),
                noise_clip=float(td3_cfg["noise_clip"]), n_step=int(td3_cfg["n_step"]),
                update_actor_freq=int(td3_cfg["update_actor_freq"]),
                actor_lr=float(td3_cfg["actor_lr"]), critic_lr=float(td3_cfg["critic_lr"]))
    buffer = ReplayBuffer(obs_dim=env.obs_dim, action_dim=env.num_actions,
                          num_envs=env.num_envs, capacity_steps=int(td3_cfg["buffer_decisions"]),
                          device=device)

    obs = env.reset()
    noise_start = float(td3_cfg["exploration_noise_start"])
    noise_end = float(td3_cfg["exploration_noise_end"])
    pre_collect = int(td3_cfg["pre_collect_decisions"])
    logger.info("wheeled TD3 pre-collect: decision %d회", pre_collect)
    for _ in range(pre_collect):
        action = policy.select_action(obs, noise_start)
        obs, _, dones, _ = env.step(action)
    env.pop_stats()

    curve, done_iters = [], 0
    t0 = time.time()
    update_per_step = int(td3_cfg["update_per_step"])
    batch_size = int(td3_cfg["batch_size"])
    failure: Exception | None = None
    try:
        while done_iters < iterations:
            n = min(chunk, iterations - done_iters)
            loss_sums = {"critic_loss": 0.0, "actor_loss": 0.0, "actor_updates": 0}
            for _ in range(n):
                noise = _exploration_noise(done_iters, iterations, noise_start, noise_end)
                action = policy.select_action(obs, noise)
                obs, reward, dones, _ = env.step(action)
                buffer.add(obs, action, reward, dones)
                if buffer.ready(batch_size):
                    for _ in range(update_per_step):
                        loss = policy.train_step(buffer, batch_size)
                        loss_sums["critic_loss"] += loss["critic_loss"]
                        if loss["actor_loss"] is not None:
                            loss_sums["actor_loss"] += loss["actor_loss"]
                            loss_sums["actor_updates"] += 1
                done_iters += 1

            stats = env.pop_stats()
            stats["iteration"] = done_iters
            stats["elapsed_s"] = round(time.time() - t0, 1)
            stats["exploration_noise"] = round(
                _exploration_noise(done_iters, iterations, noise_start, noise_end), 4)
            updates = max(n * update_per_step, 1)
            stats["critic_loss"] = loss_sums["critic_loss"] / updates
            stats["actor_loss"] = (loss_sums["actor_loss"] / loss_sums["actor_updates"]
                                   if loss_sums["actor_updates"] > 0 else None)
            curve.append(stats)
            PolicyBundle.export(out_dir, policy.actor, env.obs_dim,
                               _bundle_meta(env, obs_spec, rl_cfg, mode, done_iters,
                                          iterations, seed, terrain_mode, scale,
                                          robot_dirs, holdout_names))
            logger.info("[wheeled %d/%d] 평균 수익 %.2f, 에피소드 %.1fs x %d개, "
                        "도달 %d회/%d 에피소드, 전복 %d, stuck %d, noise %.3f, "
                        "critic_loss %.4f, 지형 레벨 %s, 경과 %.0fs", done_iters, iterations,
                        stats["mean_return"], stats["mean_ep_len_s"], stats["episodes"],
                        stats["reached"], stats["reached_episodes"], stats["terminations"],
                        stats["stuck"], stats["exploration_noise"], stats["critic_loss"],
                        stats.get("terrain_level", "-"), stats["elapsed_s"])
    except Exception as exc:
        failure = exc
        done_iters = curve[-1]["iteration"] if curve else 0
        logger.exception("wheeled 학습 중 예외 — 현재 정책을 export하고 중단한다")

    bundle_meta = _bundle_meta(env, obs_spec, rl_cfg, mode, done_iters, iterations, seed,
                               terrain_mode, scale, robot_dirs, holdout_names)
    PolicyBundle.export(out_dir, policy.actor, env.obs_dim, bundle_meta)
    with open(out_dir / "train_curve.json", "w") as f:
        json.dump(curve, f, indent=1, ensure_ascii=False)
    logger.info("wheeled 정책 export 완료: %s (학습 %d회, %.0fs)",
                out_dir, done_iters, time.time() - t0)
    if failure is not None:
        raise failure
    return {"curve": curve, "bundle": bundle_meta}

def _bundle_meta(env: WheeledParameterTrainEnv, obs_spec, rl_cfg: dict, mode: str,
                 done_iters: int, iterations: int, seed: int, terrain_mode: str,
                 scale: dict, robot_dirs: list[tuple[Path, Path]],
                 holdout_names: list[str] | None) -> dict:
    """배포 경로가 정책의 parameter 규약을 그대로 재구성할 수 있도록 metadata를 만든다."""

    return {
        "kind": "wheeled_controller_parameter_policy",
        "obs_dim": obs_spec.dim(len(ACTION_PARAM_NAMES)),
        "num_actions": env.num_actions,
        "action_param_names": list(ACTION_PARAM_NAMES),
        "param_schema": schema_names(),
        "param_bounds": {name: [float(v) for v in bounds] for name, bounds
                         in rl_cfg["env"]["param_bounds"].items()},
        "decision_period_s": float(rl_cfg["env"].get("decision_period_s", 1.0)),
        "decimation": int(rl_cfg["env"]["decimation"]),
        "physics_dt": float(rl_cfg["env"]["physics_dt"]),
        "obs": {"include_type": obs_spec.include_type,
                "include_morph": obs_spec.include_morph,
                "include_state": obs_spec.include_state},
        "morph": dict(rl_cfg["morph"]),
        "state": dict(rl_cfg.get("state", {})),
        "train": {
            "mode": mode,
            "iterations": done_iters,
            "max_iterations": iterations,
            "seed": seed,
            "terrain": terrain_mode,
            "envs_per_robot": int(scale["envs_per_robot"]),
            "goal_radius": list(rl_cfg["env"]["goal_radius"]),
            "reach_radius": float(rl_cfg["env"]["reach_radius"]),
            "stuck_time_s": float(rl_cfg["env"]["stuck_time_s"]),
            "stuck_speed_thresh": float(rl_cfg["env"]["stuck_speed_thresh"]),
            "curriculum_min_reaches": int(rl_cfg["env"]["curriculum_min_reaches"]),
            "terrain_override": dict(scale.get("terrain", {})),
            "num_robots": len(robot_dirs),
            "robots": [_robot_name(usd_dir) for _, usd_dir in robot_dirs],
            "holdout_robots": [_robot_name(name) for name in (holdout_names or [])],
        },
    }

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import torch

from rsl_rl.runners import OnPolicyRunner

from scripts.sim.controller.legged.rl import bundle as low_rl
from scripts.sim.controller.legged.rl.train_env import LeggedTrainEnv

logger = logging.getLogger(__name__)

def _merge_form_cfg(rl_cfg: dict, form: str) -> dict:

    def merge(base, over):
        out = dict(base)
        for k, v in over.items():
            out[k] = merge(base[k], v) if isinstance(v, dict)                and isinstance(base.get(k), dict) else v
        return out

    return merge(rl_cfg, rl_cfg.get("forms", {}).get(form, {}))

def _runner_cfg(ppo_cfg: dict) -> dict:

    return {
        "num_steps_per_env": int(ppo_cfg["num_steps_per_env"]),
        "save_interval": 10 ** 9,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": list(ppo_cfg["hidden_dims"]),
            "critic_hidden_dims": list(ppo_cfg["hidden_dims"]),
            "activation": ppo_cfg["activation"],
            "init_noise_std": float(ppo_cfg["init_noise_std"]),

            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": int(ppo_cfg["num_learning_epochs"]),
            "num_mini_batches": int(ppo_cfg["num_mini_batches"]),
            "clip_param": float(ppo_cfg["clip_param"]),
            "gamma": float(ppo_cfg["gamma"]),
            "lam": float(ppo_cfg["lam"]),
            "value_loss_coef": float(ppo_cfg["value_loss_coef"]),
            "entropy_coef": float(ppo_cfg["entropy_coef"]),
            "learning_rate": float(ppo_cfg["learning_rate"]),
            "max_grad_norm": float(ppo_cfg["max_grad_norm"]),
            "schedule": ppo_cfg["schedule"],
            "desired_kl": float(ppo_cfg["desired_kl"]),
        },
    }

def train_form(form: str, robot_dirs: list[tuple[Path, Path]], rl_cfg: dict,
               sim_cfg: dict, out_dir: Path, mode: str, device: str) -> dict:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rl_cfg = _merge_form_cfg(rl_cfg, form)
    train_cfg = rl_cfg["train"]
    scale = train_cfg[mode]
    iterations = int(scale["iterations"])
    chunk = max(1, min(int(train_cfg["chunk_iters"]), iterations))
    seed = int(train_cfg["seed"])
    torch.manual_seed(seed)

    terrain_mode = scale.get("terrain_mode", rl_cfg["env"]["terrain"])
    if terrain_mode == "rough":
        terrain_cfg = {**sim_cfg["terrain"], **scale.get("terrain", {})}
    else:
        terrain_cfg = None
    env = LeggedTrainEnv(form, robot_dirs, rl_cfg, sim_cfg,
                         envs_per_robot=int(scale["envs_per_robot"]),
                         device=device, seed=seed, terrain_cfg=terrain_cfg)

    runner = OnPolicyRunner(env, _runner_cfg(rl_cfg["ppo"]),
                            log_dir=str(out_dir / "rsl_log"), device=device)

    curve, done_iters = [], 0
    t0 = time.time()
    init_std = float(rl_cfg["ppo"]["init_noise_std"])
    anneal_to = rl_cfg["ppo"].get("std_anneal_to")
    while done_iters < iterations:
        n = min(chunk, iterations - done_iters)
        runner.learn(n, init_at_random_ep_len=(done_iters == 0))
        done_iters += n

        if anneal_to is not None:
            cap = init_std + (float(anneal_to) - init_std)                * min(done_iters / iterations, 1.0)
            with torch.no_grad():
                runner.alg.policy.std.data.clamp_(max=cap)
        stats = env.pop_stats()
        stats["iteration"] = done_iters
        stats["elapsed_s"] = round(time.time() - t0, 1)
        curve.append(stats)
        runner.save(str(out_dir / "checkpoint.pt"))
        logger.info("[%s %d/%d] 평균 수익 %.2f, 에피소드 %.1fs x %d개 (조기 종료 %d), "
                    "지형 레벨 %s, 명령 커리큘럼 %s, 보행 %s, 경과 %.0fs",
                    form, done_iters, iterations, stats["mean_return"],
                    stats["mean_ep_len_s"], stats["episodes"],
                    stats["terminations"], stats.get("terrain_level", "-"),
                    stats.get("cmd_v_ratio", "-"), stats.get("walked_m", "-"),
                    stats["elapsed_s"])

    policy = runner.alg.policy
    bundle_meta = {
        "form": form,
        "obs_dim": low_rl.obs_dim(form, low_rl.scan_rays(rl_cfg["scan"]),
                                  rl_cfg["obs"]),
        "num_actions": env.num_actions,
        "action_scale": float(rl_cfg["env"]["action_scale"]),
        "action_clip": float(rl_cfg["env"]["action_clip"]),
        "decimation": int(rl_cfg["env"]["decimation"]),
        "physics_dt": float(rl_cfg["env"]["physics_dt"]),
        "obs": dict(rl_cfg["obs"]),
        "morph": dict(rl_cfg["morph"]),

        "actuator_model": str(rl_cfg["env"]["actuator_model"]),
        "cmd": dict(rl_cfg["cmd"]),
        "gains": dict(rl_cfg["gains"]),

        "scan": dict(rl_cfg["scan"]),
        "train": {"mode": mode, "iterations": iterations, "seed": seed,
                  "terrain": rl_cfg["env"]["terrain"],

                  "cmd_reached_ratio": (curve[-1].get("cmd_v_ratio")
                                        if curve else None),

                  "reached_v": env.per_robot_reached(),
                  "num_robots": len(robot_dirs),
                  "robots": [str(Path(u).parent.name) for u, _ in robot_dirs]},
    }
    low_rl.PolicyBundle.export(out_dir,
                               copy.deepcopy(policy.actor_obs_normalizer),
                               copy.deepcopy(policy.actor), bundle_meta)
    with open(out_dir / "train_curve.json", "w") as f:
        json.dump(curve, f, indent=1, ensure_ascii=False)
    logger.info("정책 export 완료: %s (학습 %d회, %.0fs)", out_dir, iterations,
                time.time() - t0)
    return {"curve": curve, "bundle": bundle_meta}

"""MLP 액터-크리틱 정책 - 지금까지의 기본 방식([512,256,128], ELU)."""

from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg


def build(agent_cfg: dict) -> RslRlPpoActorCriticCfg:
    """agent_cfg의 hidden_dims/init_noise_std로 액터-크리틱 MLP를 만든다."""
    hidden_dims = agent_cfg.get("hidden_dims", [512, 256, 128])
    return RslRlPpoActorCriticCfg(
        init_noise_std=agent_cfg.get("init_noise_std", 1.0),
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,
        activation="elu",
    )

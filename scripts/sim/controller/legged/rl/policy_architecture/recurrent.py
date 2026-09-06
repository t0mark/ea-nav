"""LSTM 기반 재귀 액터-크리틱 정책 - Unitree 공식(unitree_rl_gym) g1/h1/h1_2가 실제로 쓰는 구조.

Isaac Lab의 rsl_rl 래퍼(isaaclab_rl.rsl_rl)가 제공하는 RslRlPpoActorCriticRecurrentCfg를 그대로
쓴다 - RslRlPpoActorCriticCfg에 rnn_type/rnn_hidden_dim/rnn_num_layers가 추가된 형태다.

주의: 이 클래스명·필드명은 Isaac Lab 문서 기준으로 작성했고, 이 저장소에 실제로 설치된 isaaclab_rl
버전에서 직접 import해서 확인하지는 못했다(이 환경에 Isaac Sim이 없음) - 처음 이 axis를 쓰는 로봇을
학습시킬 때 ImportError/필드명 불일치가 나면, 설치된 isaaclab_rl.rsl_rl의 실제 클래스 정의를 확인해
바로잡아야 한다.
"""

from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg


def build(agent_cfg: dict) -> RslRlPpoActorCriticRecurrentCfg:
    """agent_cfg의 hidden_dims/rnn_hidden_dim 등으로 LSTM 액터-크리틱을 만든다."""
    hidden_dims = agent_cfg.get("hidden_dims", [32])
    return RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=agent_cfg.get("init_noise_std", 0.8),
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,
        activation="elu",
        rnn_type=agent_cfg.get("rnn_type", "lstm"),
        rnn_hidden_dim=agent_cfg.get("rnn_hidden_dim", 64),
        rnn_num_layers=agent_cfg.get("rnn_num_layers", 1),
    )

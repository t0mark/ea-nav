"""PPO 알고리즘 하이퍼파라미터 - Isaac Lab 표준 legged locomotion PPO 값을 기본으로 하되, 로봇 yaml의
agent 블록에서 로봇마다 다르게 오버라이드할 수 있다(공식 리포마다 learning_rate가 1e-5~1e-3까지 갈렸음)."""

from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


def build(agent_cfg: dict) -> RslRlPpoAlgorithmCfg:
    """agent_cfg에서 PPO 하이퍼파라미터를 읽어 RslRlPpoAlgorithmCfg를 만든다."""
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=agent_cfg.get("value_loss_coef", 1.0),
        use_clipped_value_loss=True,
        clip_param=agent_cfg.get("clip_param", 0.2),
        entropy_coef=agent_cfg.get("entropy_coef", 0.01),
        num_learning_epochs=agent_cfg.get("num_learning_epochs", 5),
        num_mini_batches=agent_cfg.get("num_mini_batches", 4),
        learning_rate=agent_cfg.get("learning_rate", 1.0e-3),
        schedule="adaptive",
        gamma=agent_cfg.get("gamma", 0.99),
        lam=agent_cfg.get("lam", 0.95),
        desired_kl=agent_cfg.get("desired_kl", 0.01),
        max_grad_norm=agent_cfg.get("max_grad_norm", 1.0),
    )

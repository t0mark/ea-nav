"""preset.algorithm(+ robot yaml의 agent 오버라이드)으로 RslRlOnPolicyRunnerCfg를 조립.

정책 구조와 알고리즘은 preset 한 장이 정한다 - 4개 로봇이 같은 preset을 가리키면 embodiment 비교가
성립한다(학습 설계 동일). robot yaml에 agent: 블록이 있으면 그 키만 preset.algorithm 위에 덮어쓴다
(로봇별로 꼭 필요한 경우만 - 예: anymal 계열 entropy_coef).

device는 반드시 호출부(rl_trainer.py의 CurriculumSession)가 --device로 받은 값을 그대로 넘겨야 한다 -
Isaac Lab의 RslRlBaseRunnerCfg.device 기본값이 "cuda:0"으로 고정돼 있어, 여기서 넘기지 않으면
env(PhysX·렌더링)만 지정한 GPU를 쓰고 학습 러너(정책망·PPO 옵티마이저)는 cuda:0으로 고정돼
멀티 GPU 장비에서 GPU 지정이 반쪽만 먹는다.

max_iterations는 RslRlOnPolicyRunnerCfg의 필수 필드라 값을 넣긴 하지만(커리큘럼의 단계 상한을 그대로
재사용), 실제 학습량은 rl_trainer.py의 CurriculumTrainer가 runner.learn(num_learning_iterations=...)로
매 구간마다 명시적으로 넘기므로 이 값에 의존하지 않는다 - 전역 이터레이션 종료 로직은 없다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

if TYPE_CHECKING:
    from .robot_profile import RLPreset, RobotProfile


def build_agent_cfg(
    profile: RobotProfile, preset: RLPreset, experiment_name: str, device: str
) -> RslRlOnPolicyRunnerCfg:
    """preset.algorithm에 robot yaml의 agent 오버라이드를 병합해 MLP 액터-크리틱 + PPO 러너 설정을 만든다."""
    hp = {**preset.algorithm, **(profile.agent or {})}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=hp.get("init_noise_std", 1.0),
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=list(hp["hidden_dims"]),
        critic_hidden_dims=list(hp["hidden_dims"]),
        activation=hp.get("activation", "elu"),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=hp.get("value_loss_coef", 1.0),
        use_clipped_value_loss=True,
        clip_param=hp.get("clip_param", 0.2),
        entropy_coef=hp.get("entropy_coef", 0.01),
        num_learning_epochs=hp.get("num_learning_epochs", 5),
        num_mini_batches=hp.get("num_mini_batches", 4),
        learning_rate=hp.get("learning_rate", 1.0e-3),
        schedule=hp.get("schedule", "adaptive"),
        gamma=hp.get("gamma", 0.99),
        lam=hp.get("lam", 0.95),
        desired_kl=hp.get("desired_kl", 0.01),
        max_grad_norm=hp.get("max_grad_norm", 1.0),
    )
    return RslRlOnPolicyRunnerCfg(
        device=device,
        num_steps_per_env=hp.get("num_steps_per_env", 24),
        # 필수 필드라 채우지만 학습량 결정에는 안 쓰인다(모듈 docstring 참고) - 커리큘럼 단계 상한을 재사용.
        max_iterations=preset.curriculum["per_stage"]["max_iterations"],
        save_interval=hp.get("save_interval", 50),
        experiment_name=experiment_name,
        policy=policy,
        algorithm=algorithm,
    )

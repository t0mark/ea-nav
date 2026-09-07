"""RobotProfile의 policy_architecture/algorithm 축 + agent 하이퍼파라미터로 RslRlOnPolicyRunnerCfg를 조립.

env cfg(actuator/action/rewards/termination/domain_randomization)와 agent cfg(policy_architecture/
algorithm)를 분리하는 이유: 전자는 시뮬레이션 환경(ManagerBasedRLEnv) 소관이고, 후자는 학습 러너
(OnPolicyRunner) 소관이라 책임이 다르다.

device는 반드시 호출부(tools/03_controller_rl.py)가 --device로 받은 값을 그대로 넘겨야 한다 -
Isaac Lab의 RslRlBaseRunnerCfg.device 기본값이 "cuda:0"으로 고정돼 있어서(재클론해 확인: isaaclab_rl/
rsl_rl/rl_cfg.py), 여기서 넘기지 않으면 env(PhysX·렌더링)는 --device로 지정한 GPU를 쓰는데 학습
러너(정책망·PPO 옵티마이저)만 항상 cuda:0으로 가버린다 - 멀티 GPU 장비에서 GPU를 지정해도 실제로는
반쪽만 적용되는 버그였다.
"""

from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from . import algorithm as algorithm_axis
from . import policy_architecture as policy_architecture_axis
from .robot_profile import RobotProfile


def build_agent_cfg(profile: RobotProfile, experiment_name: str, device: str) -> RslRlOnPolicyRunnerCfg:
    """profile.policy_architecture/algorithm으로 액터-크리틱+알고리즘 설정을 만들고 러너 설정으로 묶는다.

    device를 env cfg와 같은 값으로 명시하지 않으면 RslRlBaseRunnerCfg 기본값(cuda:0)으로 고정돼,
    시뮬레이션은 지정한 GPU에서 돌면서 정책망만 항상 GPU 0에서 도는 불일치가 생긴다 - 여러 GPU에
    동시에 학습을 나눠 돌릴 때 모든 정책망 연산이 GPU 0으로 몰리는 원인이 된다.
    """
    return RslRlOnPolicyRunnerCfg(
        device=device,
        num_steps_per_env=profile.agent.get("num_steps_per_env", 24),
        max_iterations=profile.agent["max_iterations"],
        save_interval=profile.agent.get("save_interval", 50),
        experiment_name=experiment_name,
        policy=policy_architecture_axis.build(profile.policy_architecture, profile.agent),
        algorithm=algorithm_axis.build(profile.algorithm, profile.agent),
    )

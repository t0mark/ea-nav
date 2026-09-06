"""RobotProfile의 policy_architecture/algorithm 축 + agent 하이퍼파라미터로 RslRlOnPolicyRunnerCfg를 조립.

env cfg(actuator/action/rewards/termination/domain_randomization)와 agent cfg(policy_architecture/
algorithm)를 분리하는 이유: 전자는 시뮬레이션 환경(ManagerBasedRLEnv) 소관이고, 후자는 학습 러너
(OnPolicyRunner) 소관이라 책임이 다르다.
"""

from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from . import algorithm as algorithm_axis
from . import policy_architecture as policy_architecture_axis
from .robot_profile import RobotProfile


def build_agent_cfg(profile: RobotProfile, experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    """profile.policy_architecture/algorithm으로 액터-크리틱+알고리즘 설정을 만들고 러너 설정으로 묶는다."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=profile.agent.get("num_steps_per_env", 24),
        max_iterations=profile.agent["max_iterations"],
        save_interval=profile.agent.get("save_interval", 50),
        experiment_name=experiment_name,
        policy=policy_architecture_axis.build(profile.policy_architecture, profile.agent),
        algorithm=algorithm_axis.build(profile.algorithm, profile.agent),
    )

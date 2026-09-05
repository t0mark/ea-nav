"""학습된 legged RL 정책을 로드해 실행하는 러너."""

from __future__ import annotations

from pathlib import Path

import torch

from isaaclab.envs import ManagerBasedRLEnv

from scripts.sim.controller.base import RobotController
from scripts.sim.controller.legged.rl.loco_rl_env import build_loco_rl_env_cfg

_REPO_ROOT = Path(__file__).resolve().parents[4]
_POLICY_ROOT = _REPO_ROOT / "data" / "sim" / "policies" / "legged"


class LocoRunner(RobotController):
    """robot_id에 맞는 RL 환경을 만들고, 학습된 정책 체크포인트로 관절 목표값을 낸다.

    관측 구성(고유 감각 + height scan)은 학습 때와 완전히 같은 순서·정규화로 나와야 정책이 제대로
    동작하므로, 직접 관측 벡터를 조립하지 않고 학습에 썼던 환경(ManagerBasedRLEnv)을 그대로 재사용해
    관측·행동 적용까지 env.step()에 맡긴다.
    """

    def __init__(self, category: str, robot_id: str, num_envs: int = 1, device: str = "cuda:0") -> None:
        """robot_id의 env cfg를 조립하고, data/sim/policies/legged/{category}/{robot_id}/policy.pt를 로드한다."""
        env_cfg = build_loco_rl_env_cfg(category, robot_id, num_envs=num_envs)
        env_cfg.sim.device = device
        self._env = ManagerBasedRLEnv(cfg=env_cfg)

        policy_path = _POLICY_ROOT / category / robot_id / "policy.pt"
        self._policy = torch.jit.load(str(policy_path), map_location=device)
        self._policy.eval()
        self._observation: torch.Tensor | None = None

    def reset(self) -> None:
        """환경을 리셋하고 첫 관측을 받아 둔다."""
        observation_dict, _ = self._env.reset()
        self._observation = observation_dict["policy"]

    def compute_joint_targets(
        self, command: torch.Tensor, observation: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """command(목표 속도 [vx, vy, wz])를 명령 매니저에 주입한 뒤, 정책 추론 -> env.step()까지 수행한다.

        env.step()이 액션(관절 위치 목표)을 관절에 실제로 적용하는 것까지 끝내므로, 반환값은 다시
        관절에 적용할 필요 없이 RobotController 인터페이스 형식만 맞춘 것이다.
        """
        self._env.command_manager.get_term("base_velocity").command[:] = command
        with torch.inference_mode():
            action = self._policy(self._observation)
        observation_dict, _, _, _, _ = self._env.step(action)
        self._observation = observation_dict["policy"]
        return {"position": action}

    @property
    def env(self) -> ManagerBasedRLEnv:
        """tools/04_controller_test.py가 카메라 시점 갱신에 쓸 수 있도록 내부 env를 노출한다."""
        return self._env

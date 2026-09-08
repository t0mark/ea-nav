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
    관측·행동 적용까지 env.step()에 맡긴다. ObservationsCfg가 관측 항목을 "policy" 그룹 하나로만
    묶어 두므로(concatenate_terms=True), env가 돌려주는 관측 딕셔너리도 이미 "policy" 키 하나에 대한
    평평한 텐서다 - 그룹을 이어붙이는 별도 순서 조립이 필요 없다.
    """

    def __init__(self, category: str, robot_id: str, num_envs: int = 1, device: str = "cuda:0", stage: int = 0) -> None:
        """robot_id의 env cfg를 조립하고, data/sim/policies/legged/{category}/{robot_id}/policy.pt를 로드한다.

        stage 기본값은 0(평지)이다 - LocoRunner는 학습이 아니라 구동 확인/테스트용이고, 그 용도는
        보통 wheeled와 동일한 조건(평지)에서 이동 명령 추종만 보는 것이라 지형 난이도가 필요 없다.
        category는 build_loco_rl_env_cfg에는 쓰이지 않는다(RobotProfile.load가 robot_id만으로 로봇
        yaml을 찾는다) - 정책 저장 경로(_POLICY_ROOT/{category}/{robot_id})를 rl/rl_trainer.py의
        StageSession이 policy.pt를 내보내는 경로와 맞추는 데만 쓴다.
        """
        env_cfg = build_loco_rl_env_cfg(robot_id, stage=stage, num_envs=num_envs)
        env_cfg.sim.device = device
        self._env = ManagerBasedRLEnv(cfg=env_cfg)

        policy_path = _POLICY_ROOT / category / robot_id / "policy.pt"
        self._policy = torch.jit.load(str(policy_path), map_location=device)
        self._policy.eval()
        self._observation: torch.Tensor | None = None

    def _concat_policy_observation(self, observation_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """env가 돌려준 "policy" 그룹의 평평한 관측 텐서를 그대로 꺼낸다."""
        return observation_dict["policy"]

    def reset(self) -> None:
        """환경을 리셋하고 첫 관측을 받아 둔다."""
        observation_dict, _ = self._env.reset()
        self._observation = self._concat_policy_observation(observation_dict)

    def compute_joint_targets(
        self, command: torch.Tensor, observation: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """command(목표 속도 [vx, vy, wz])를 명령 매니저에 주입한 뒤, 정책 추론 -> env.step()까지 수행한다.

        env.step()이 액션(관절 위치 목표)을 관절에 실제로 적용하는 것까지 끝내므로, 반환값은 다시
        관절에 적용할 필요 없이 RobotController 인터페이스 형식만 맞춘 것이다.

        base_velocity의 heading_command=True(학습 다양성 목적)는 CommandManager.compute()가
        env.step() 안에서 매 resampling 주기마다 wz 성분을 무작위 heading 추종값으로, 그리고
        rel_standing_envs 확률로 명령 전체를 0으로 되돌린다(isaaclab velocity_command.py의
        _update_command) - 배포 시에는 여기 넣은 command가 그대로 정책에 들어가야 하므로, 그 두
        오버라이드 플래그를 매 스텝 꺼서 무력화한다.
        """
        base_velocity = self._env.command_manager.get_term("base_velocity")
        base_velocity.is_heading_env[:] = False
        base_velocity.is_standing_env[:] = False
        base_velocity.command[:] = command
        with torch.inference_mode():
            action = self._policy(self._observation)
        observation_dict, _, _, _, _ = self._env.step(action)
        self._observation = self._concat_policy_observation(observation_dict)
        return {"position": action}

    @property
    def env(self) -> ManagerBasedRLEnv:
        """tools/04_controller_test.py가 카메라 시점 갱신에 쓸 수 있도록 내부 env를 노출한다."""
        return self._env

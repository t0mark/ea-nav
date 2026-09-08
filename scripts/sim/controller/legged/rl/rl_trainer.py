"""커리큘럼 한 단계를 학습하는 워커 - 진입점이 --stage로 재실행될 때 실제 학습을 도는 쪽.

"단계 하나 = 프로세스 하나"인 이유는 scripts/.../rl/curriculum_driver.py 모듈 docstring에 있다.
이 모듈은 "받은 stage 하나를 학습하고, 결과를 stage_outcome.yaml로 남기고, 프로세스를 끝낸다"만
책임진다 - 다음 단계로 올릴지 말지는 부모(CurriculumDriver)가 정한다.

isaaclab에 의존하므로 AppLauncher 기동 이후에만 import할 수 있다.

이터레이션 카운트 주의: rsl_rl OnPolicyRunner.learn()은 마지막 루프 인덱스를 그대로
current_learning_iteration에 넣기 때문에(on_policy_runner.py:153) learn(N) 한 번이 인덱스를 N-1만
전진시키고 경계 이터레이션 하나를 다시 돈다. 또 load()가 체크포인트의 iter로 카운터를 복원하므로
(on_policy_runner.py:323) 이전 단계를 이어받으면 카운터가 그 단계 번호부터 이어진다 - 단계 안의
model_*.pt 번호도 이전 단계에 이어서 붙는다. 그래서 단계 상한은 반드시 "단계 시작 시점 대비
상대값"으로 센다.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit
from rsl_rl.runners import OnPolicyRunner

from scripts.sim.env.curriculum.stage_env import stage_terrain_summary

from .agent_cfg import build_agent_cfg
from .curriculum_driver import STAGE_OUTCOME_FILENAME, experiment_name, policy_output_dir, stage_log_dir
from .loco_rl_env import build_loco_rl_env_cfg
from .robot_profile import RLPreset, RobotProfile


def _first_matching_metric(metrics: dict[str, float], needle: str) -> float | None:
    """'Episode_Reward/track_lin_vel_xy_exp'처럼 prefix가 붙은 키 중 needle을 포함하는 첫 값."""
    for key, value in metrics.items():
        if needle in key:
            return value
    return None


class EpisodeMetricRunner(OnPolicyRunner):
    """rsl_rl OnPolicyRunner에 "직전 학습 구간의 평균 에피소드 지표"를 노출하는 훅만 더한 것.

    learn()이 매 이터레이션 끝에 log(locals())를 부르는데, 그 locals의 ep_infos(이번 구간에 끝난
    에피소드들의 로그 dict 목록)를 항목별 평균으로 접어 들고 있는다. StageTrainer가 이 값에서
    track_lin_vel_xy_exp / track_ang_vel_z_exp를 읽어 수렴을 판정한다.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._latest_ep_metrics: dict[str, float] = {}

    @property
    def latest_ep_metrics(self) -> dict[str, float]:
        """직전 학습 구간에서 끝난 에피소드들의 항목별 평균 지표."""
        return self._latest_ep_metrics

    def log(self, locs, width: int = 80, pad: int = 35) -> None:  # noqa: D102 - 상위 시그니처 유지
        ep_infos = locs.get("ep_infos") if isinstance(locs, dict) else None
        if ep_infos:
            averaged: dict[str, float] = {}
            for key in ep_infos[0].keys():
                try:
                    values = torch.cat(
                        [torch.as_tensor(e[key], dtype=torch.float32).flatten() for e in ep_infos if key in e]
                    )
                except Exception:  # noqa: BLE001 - 지표 하나 못 읽어도 학습은 계속돼야 한다
                    continue
                if values.numel():
                    averaged[key] = float(values.mean())
            if averaged:
                self._latest_ep_metrics = averaged
        try:
            super().log(locs, width=width, pad=pad)
        except Exception:  # noqa: BLE001 - 콘솔 로그 실패가 학습을 멈추면 안 된다
            pass


@dataclass
class StageOutcome:
    """커리큘럼 한 단계 학습의 결과 - 그대로 stage_outcome.yaml이 된다."""

    status: str  # "converged" | "plateaued" | "max_iterations" | "diverged"
    iterations: int
    tracking_lin: float  # 정규화(가중치 제거) 평균 추종 점수 [0,1] 근방
    tracking_ang: float
    best_score: float
    checkpoint: str | None
    summary: dict


class StageTrainer:
    """한 단계를 수렴/정체/단계상한/발산 중 하나가 될 때까지 eval_interval_iters씩 끊어 학습한다.

    수렴 판정에 쓰는 점수는 "가중치를 제거한 추종 보상"이다. Isaac Lab이 남기는
    Episode_Reward/<term>은 Σ(weight·value·dt) / max_episode_length_s 이므로 weight로 나누면
    mean(value) × (생존 스텝 / 최대 스텝)이 된다 - 추종 정확도와 생존률의 곱이라, 조기 종료가
    잦으면 점수가 threshold에 도달하지 못한다.
    """

    def __init__(self, curriculum_cfg: dict, reward_weights: dict[str, float]) -> None:
        per_stage = curriculum_cfg["per_stage"]
        self._eval_interval = int(per_stage["eval_interval_iters"])
        self._max_iterations = int(per_stage["max_iterations"])
        convergence = curriculum_cfg["convergence"]
        self._threshold = float(convergence["threshold"])
        self._patience_evals = int(convergence["patience_evals"])
        plateau = curriculum_cfg["plateau"]
        self._plateau_window = int(plateau["window_evals"])
        self._min_delta = float(plateau["min_delta"])
        self._lin_weight = reward_weights["track_lin_vel_xy_exp"]
        self._ang_weight = reward_weights["track_ang_vel_z_exp"]
        self._started_at = 0.0

    def run(self, runner: EpisodeMetricRunner, log_dir: str, summary: dict) -> StageOutcome:
        """수렴/정체/단계상한/발산 중 하나가 나올 때까지 학습하고 결과를 돌려준다."""
        # 이 시각 이후에 쓰인 체크포인트만 "이번 실행이 남긴 것"으로 인정한다(_latest_checkpoint 참고)
        self._started_at = time.time()
        start_iteration = runner.current_learning_iteration
        evals_above_threshold = 0
        evals_without_progress = 0
        best_score = float("-inf")
        iterations_done = 0
        tracking_lin = tracking_ang = 0.0
        is_first_chunk = True

        while True:
            try:
                runner.learn(num_learning_iterations=self._eval_interval, init_at_random_ep_len=is_first_chunk)
            except RuntimeError as exc:
                # PPO 발산(보상 폭주 -> 가치함수 발산 -> 정책 표준편차 NaN)은 rsl_rl이 RuntimeError로 던진다
                print(f"[stage] ! PPO 발산 - {exc}", flush=True)
                return self._outcome(
                    "diverged", runner, log_dir, summary, iterations_done, tracking_lin, tracking_ang, best_score
                )
            is_first_chunk = False
            # learn()은 인덱스를 eval_interval-1만 전진시키므로 실제 진행량은 카운터에서 직접 읽는다
            iterations_done = runner.current_learning_iteration - start_iteration

            metrics = runner.latest_ep_metrics
            lin_reward = _first_matching_metric(metrics, "track_lin_vel_xy_exp")
            ang_reward = _first_matching_metric(metrics, "track_ang_vel_z_exp")
            tracking_lin = (lin_reward / self._lin_weight) if lin_reward is not None else 0.0
            tracking_ang = (ang_reward / self._ang_weight) if ang_reward is not None else 0.0
            score = min(tracking_lin, tracking_ang)
            print(
                f"[stage]   iter {iterations_done:>5} | track_lin={tracking_lin:.3f} track_ang={tracking_ang:.3f} "
                f"| above={evals_above_threshold}/{self._patience_evals} "
                f"stale={evals_without_progress}/{self._plateau_window} best={max(best_score, score):.3f}",
                flush=True,
            )

            # NaN/inf는 정체 판정을 plateau_window만큼 헛돌게 하므로 그 자리에서 발산으로 끊는다
            if not math.isfinite(score):
                print("[stage] ! 추종 점수가 유한하지 않다 - 발산으로 판정", flush=True)
                return self._outcome(
                    "diverged", runner, log_dir, summary, iterations_done, tracking_lin, tracking_ang, best_score
                )

            # 수렴: 두 추종 점수가 threshold 이상인 상태가 patience_evals회 연속
            if tracking_lin >= self._threshold and tracking_ang >= self._threshold:
                evals_above_threshold += 1
            else:
                evals_above_threshold = 0
            if evals_above_threshold >= self._patience_evals:
                converged_path = os.path.join(log_dir, "converged.pt")
                runner.save(converged_path)
                return StageOutcome(
                    status="converged",
                    iterations=iterations_done,
                    tracking_lin=tracking_lin,
                    tracking_ang=tracking_ang,
                    best_score=max(best_score, score),
                    checkpoint=converged_path,
                    summary=summary,
                )

            # 정체: best가 plateau_window 동안 min_delta 이상 개선되지 않음
            if score > best_score + self._min_delta:
                best_score = score
                evals_without_progress = 0
            else:
                evals_without_progress += 1
            if evals_without_progress >= self._plateau_window:
                return self._outcome(
                    "plateaued", runner, log_dir, summary, iterations_done, tracking_lin, tracking_ang, best_score
                )
            if iterations_done >= self._max_iterations:
                return self._outcome(
                    "max_iterations", runner, log_dir, summary, iterations_done, tracking_lin, tracking_ang, best_score
                )

    def _outcome(
        self,
        status: str,
        runner: EpisodeMetricRunner,
        log_dir: str,
        summary: dict,
        iterations_done: int,
        tracking_lin: float,
        tracking_ang: float,
        best_score: float,
    ) -> StageOutcome:
        """수렴하지 못한 단계의 결과를 만든다 - 체크포인트는 이번 실행이 남긴 마지막 것."""
        return StageOutcome(
            status=status,
            iterations=iterations_done,
            tracking_lin=tracking_lin,
            tracking_ang=tracking_ang,
            best_score=best_score,
            checkpoint=self._latest_checkpoint(runner, log_dir),
            summary=summary,
        )

    def _latest_checkpoint(self, runner: EpisodeMetricRunner, log_dir: str) -> str | None:
        """이번 실행이 실제로 남긴 마지막 체크포인트 경로.

        learn()이 정상적으로 끝나면 model_{current_learning_iteration}.pt를 저장하므로
        (on_policy_runner.py:175) 경로가 결정된다. 발산으로 learn()이 중간에 끊기면 그 파일이 없으니,
        학습 시작 이후에 쓰인 model_*.pt 중 가장 최근 것으로 물러선다 - mtime 조건은 같은
        디렉터리에 남아 있는 이전 실행의 체크포인트를 집지 않기 위한 것이다.
        """
        expected = Path(log_dir) / f"model_{runner.current_learning_iteration}.pt"
        if expected.exists():
            return str(expected)
        written_by_this_run = [
            path for path in Path(log_dir).glob("model_*.pt") if path.stat().st_mtime >= self._started_at
        ]
        if not written_by_this_run:
            return None
        return str(max(written_by_this_run, key=lambda path: path.stat().st_mtime))


class StageSession:
    """단계 하나를 학습하는 프로세스의 자원 수명과 산출물을 관리한다.

    env·러너를 조립해 StageTrainer에 넘기고, 수렴하면 policy.pt를 내보내고, 결과를
    stage_outcome.yaml로 남긴다. with 문으로 써서 어떤 경로로 빠져나가든 env가 닫히게 한다.
    """

    def __init__(
        self,
        robot_id: str,
        stage: int,
        preset_name: str | None = None,
        num_envs: int | None = None,
        device: str = "cuda:0",
        resume_checkpoint: str | None = None,
    ) -> None:
        self._robot_id = robot_id
        self._stage = stage
        self._profile = RobotProfile.load(robot_id)
        self._preset_name = preset_name or self._profile.rl_preset
        self._preset = RLPreset.load(self._preset_name)
        self._num_envs = num_envs or self._preset.num_envs
        self._device = device
        self._resume_checkpoint = resume_checkpoint
        self._stage_log_dir = stage_log_dir(robot_id, stage)
        self._output_dir = policy_output_dir(robot_id)
        self._env: ManagerBasedRLEnv | None = None

    def __enter__(self) -> StageSession:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> bool:
        if self._env is not None:
            self._env.close()
            self._env = None
        return False

    def run(self) -> StageOutcome:
        """이 단계를 학습하고 결과를 파일로 남긴 뒤 돌려준다."""
        summary = stage_terrain_summary(self._stage, self._preset.curriculum["stage"])
        print(
            f"[stage] {self._robot_id} | stage={self._stage} | preset={self._preset_name} "
            f"| num_envs={self._num_envs} | device={self._device}",
            flush=True,
        )
        print(f"[stage] 지형 {summary}", flush=True)

        env_cfg = build_loco_rl_env_cfg(
            self._robot_id, stage=self._stage, num_envs=self._num_envs, preset=self._preset_name
        )
        env_cfg.sim.device = self._device
        self._env = ManagerBasedRLEnv(cfg=env_cfg)

        agent_cfg = build_agent_cfg(self._profile, self._preset, experiment_name(self._robot_id), device=self._device)
        vec_env = RslRlVecEnvWrapper(self._env, clip_actions=agent_cfg.clip_actions)
        runner = EpisodeMetricRunner(
            vec_env, agent_cfg.to_dict(), log_dir=str(self._stage_log_dir), device=agent_cfg.device
        )
        if self._resume_checkpoint is not None:
            print(f"[stage] 가중치를 {self._resume_checkpoint} 에서 이어받음", flush=True)
            runner.load(self._resume_checkpoint)

        trainer = StageTrainer(self._preset.curriculum, self._profile.reward_weights)
        outcome = trainer.run(runner, str(self._stage_log_dir), summary)
        print(f"[stage] stage {self._stage} 결과: {outcome.status} ({outcome.iterations} iters)", flush=True)

        # 수렴한 단계의 정책만 배포용으로 내보낸다 - policy.pt는 항상 "마지막으로 깬 난이도"의 정책이다
        if outcome.status == "converged":
            self._export_policy(runner, agent_cfg)
        self._write_outcome(outcome)
        return outcome

    def _export_policy(self, runner: EpisodeMetricRunner, agent_cfg) -> None:
        """현재 러너의 정책을 jit로 내보내 04_controller_test.py의 LocoRunner가 바로 로드하게 한다."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        policy_nn = runner.alg.policy
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(self._output_dir), filename="policy.pt")
        dump_yaml(str(self._output_dir / "train_config.yaml"), agent_cfg)
        print(f"[stage] policy.pt 갱신 (stage {self._stage} 정책)", flush=True)

    def _write_outcome(self, outcome: StageOutcome) -> None:
        """부모(CurriculumDriver)가 읽는 단계 결과 파일을 남긴다."""
        self._stage_log_dir.mkdir(parents=True, exist_ok=True)
        outcome_path = self._stage_log_dir / STAGE_OUTCOME_FILENAME
        with open(outcome_path, "w") as f:
            yaml.safe_dump(asdict(outcome), f, allow_unicode=True, sort_keys=False)
        print(f"[stage] 결과 저장: {outcome_path}", flush=True)

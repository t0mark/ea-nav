"""한 로봇의 커리큘럼 전체를 한 프로세스에서 학습한다.

승급을 프로세스 재시작으로 처리하지 않는다. 지형은 전 난이도 행을 한 번에 만들어 두고
(stage_env.build_curriculum_terrain_importer_cfg), 승급은 그 안에서 env를 더 어려운 행으로 옮겨서
한다 - TerrainImporter.terrain_levels / env_origins를 바꾸는 텐서 연산이고, Isaac Lab의 공식
terrain_levels_vel 커리큘럼이 쓰는 것과 같은 경로다. 그래서 지형을 다시 만들 일도, env를 다시
만들 일도 없다(한 프로세스에서 ManagerBasedRLEnv는 한 번만 만들 수 있다).

축(오르막 단차/내리막 단차/오르막 경사/내리막 경사)은 지형의 열로 표현된다 - 계단·parkour 열에
있는 env의 행이 곧 단차 레벨, 경사면 열에 있는 env의 행이 경사 레벨이고, 오르내림은 스폰 지점이
꼭대기인지 바닥인지로 갈린다(stage_env 참조). 그래서 축별 독립 승급이 열/행 인덱스 조작만으로 된다.

isaaclab에 의존하므로 AppLauncher 기동 이후에만 import할 수 있다.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit
from rsl_rl.runners import OnPolicyRunner

from scripts.sim.env.curriculum.stage_env import (
    DIFFICULTY_AXES,
    FLAT_AXIS,
    NUM_LEVELS,
    build_curriculum_terrain_importer_cfg,
    flat_terrain_column,
    stage_command_speed_limits,
    stage_terrain_summary,
    terrain_column_axes,
)

from .agent_cfg import build_agent_cfg
from .curriculum_driver import curriculum_log_dir, experiment_name, policy_output_dir, write_curriculum_result
from .loco_rl_env import build_loco_rl_env_cfg
from .robot_profile import RLPreset, RobotProfile

# 가치함수(크리틱) 손실이 이 값을 넘으면 폭주가 시작된 것으로 본다. 정상 학습에서는 단계 전환
# 직후를 포함해도 1.3을 넘지 않는데(같은 커리큘럼을 완주한 로봇 5종의 전 구간 최대), 폭주가
# 시작되면 크리틱이 자기 예측을 부트스트랩으로 증폭시켜 이터레이션마다 10^4배씩 커진다. 폭주를
# 끝까지 두면 손실이 inf가 되고, inf - inf가 만드는 NaN이 clip_grad_norm_을 타고 전 파라미터로
# 퍼져(NaN 노름 -> NaN 스케일) 정책 표준편차까지 NaN이 되어 복구가 불가능해진다. 그래서 정상
# 최대치의 8배 지점에서 잡아 되돌린다.
_VALUE_LOSS_RUNAWAY = 10.0
# 폭주를 만났을 때 직전 정상 스냅샷으로 되돌려 재시도하는 횟수.
_RECOVERY_ATTEMPTS = 3
# 재시도할 때마다 크리틱 손실 가중치를 이 비율로 줄인다 - 폭주하는 항의 그래디언트 기여를 직접
# 낮추는 것이라, adaptive 스케줄이 곧 되돌려버리는 학습률 조정보다 효과가 지속된다.
_VALUE_LOSS_COEF_DECAY = 0.5
# 되돌아갈 기준점 - 평가를 통과한(=지표가 유한한) 마지막 가중치를 이 이름으로 덮어쓴다.
_STABLE_CHECKPOINT_FILENAME = "stable.pt"
# 승급 직후 가중치 - "마지막으로 깬 난이도"의 정책이라 배포용으로 내보낸다.
_CLEARED_CHECKPOINT_FILENAME = "cleared.pt"
# env를 상한 근처(프런티어)에 두는 비율과 그 창의 폭. 나머지는 [0, 상한] 전 구간에 흩어 둔다.
# 전 구간 균등만 쓰면 상한이 높아질수록 상한에 서는 env가 1/(상한+1)까지 줄어(상한 20에서 5%)
# 가장 어려운 난이도를 연습하는 표본과 승급 판정 표본이 함께 얇아진다. 반대로 상한에만 몰면 과제가
# 균일하게 어려워져 정체 구간에서 정책 경사 신호가 끊긴다. 둘을 섞어 프런티어를 두껍게 하면서
# 쉬운 난이도도 항상 남긴다.
_FRONTIER_FRACTION = 0.75
_FRONTIER_WINDOW = 3
# 상한 레벨에 닿은 축에 남길 env 배분 몫(미완 축 대비 비율). 배분이 고정이면 이미 최고 난이도까지
# 올라간 축이 끝까지 같은 몫의 env를 차지해, 아직 못 올라간 축의 연습량과 승급 판정 표본이 그만큼
# 줄어든다. 0으로 두지 않는 이유는 그 축에서 이미 얻은 능력이 남은 학습 동안 무너지지 않는지
# 계속 봐야 하기 때문이다.
_SATURATED_AXIS_ENV_WEIGHT = 0.25


class ValueFunctionRunaway(RuntimeError):
    """크리틱 손실 폭주 - NaN이 파라미터로 퍼지기 전에 학습 구간을 중단시키는 신호."""


class EpisodeMetricRunner(OnPolicyRunner):
    """rsl_rl OnPolicyRunner에 "직전 학습 구간의 평균 에피소드 지표"와 폭주 감지를 더한 것.

    learn()이 매 이터레이션 끝에 log(locals())를 부르는데, 그 locals에는 이번 이터레이션의
    ep_infos(끝난 에피소드들의 로그 dict 목록)와 loss_dict가 들어 있다. ep_infos는 항목별 평균으로
    접어 들고 있고(진행 상황 확인용), loss_dict의 가치함수 손실은 폭주 여부를 보는 데 쓴다 -
    폭주면 예외를 던져 learn()을 그 자리에서 끊는다.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._latest_ep_metrics: dict[str, float] = {}

    @property
    def latest_ep_metrics(self) -> dict[str, float]:
        """직전 학습 구간에서 끝난 에피소드들의 항목별 평균 지표."""
        return self._latest_ep_metrics

    def log(self, locs, width: int = 80, pad: int = 35) -> None:  # noqa: D102 - 상위 시그니처 유지
        # 폭주 감지가 가장 먼저다 - 콘솔 로깅에서 예외가 나도 이 판정은 건너뛰지 않는다
        value_loss = None
        if isinstance(locs, dict):
            loss_dict = locs.get("loss_dict")
            if isinstance(loss_dict, dict):
                value_loss = loss_dict.get("value_function")
        if value_loss is not None:
            value_loss = float(value_loss)
            if not math.isfinite(value_loss) or value_loss > _VALUE_LOSS_RUNAWAY:
                raise ValueFunctionRunaway(f"가치함수 손실 {value_loss:.4g} > {_VALUE_LOSS_RUNAWAY}")
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


# 축별 추종 점수를 재는 평가 롤아웃 길이(스텝)와, 점수에 넣지 않고 버리는 앞부분 길이.
# 리셋 직후에는 로봇이 정지 상태에서 명령 속도까지 가속하는 과도구간이라 어떤 정책이든 추종이
# 나쁘게 나온다 - 그 구간을 그대로 세면 모든 로봇의 점수가 체계적으로 낮아져 임계 통과가
# 부당하게 어려워진다. 그래서 앞부분을 워밍업으로 버린다.
#
# 창 길이는 "로봇이 채점 구간 안에 지형에 닿는가"가 정한다. 로봇은 타일 중심의 평지 플랫폼에서
# 리셋되므로, 이동거리 = 평균 명령 속력 x 창 길이가 플랫폼 반폭을 넘어야 계단·경사를 밟는다.
# 명령이 축별 균등분포에서 뽑히므로 평균 속력은 상한보다 낮고, 상한은 난이도가 올라갈수록
# lin_vel_max_floor_mps까지 낮아진다 - 그 하한에서도 지형에 닿도록 창을 잡는다. 짧게 잡으면 로봇이
# 플랫폼을 벗어나기 전에 채점이 끝나 "평지 보행 점수"로 승급하게 되고, 난이도가 오를수록 상한이
# 낮아지므로 그 오판이 승급마다 심해진다.
_AXIS_EVAL_WARMUP_STEPS = 50
_AXIS_EVAL_STEPS = 250
# 추종 보상의 커널 폭 - 공식 velocity_env_cfg가 쓰는 std^2 = 0.25 그대로다.
_TRACKING_STD_SQ = 0.25


def _axis_tracking_scores(vec_env, axis_of_env: torch.Tensor | None, policy) -> tuple[dict, float]:
    """축별로 (선속도 추종, 각속도 추종)을 잰다 - env가 밟고 있는 지형의 축으로 나눠 평균한다.

    학습 로그의 Episode_Reward는 전체 env 평균이라 어느 축에서 막혔는지 구분할 수 없다. 그래서
    평가 전용 롤아웃을 짧게 돌려 축별로 따로 잰다. 모든 env를 리셋하고 워밍업 구간을 버린 뒤 같은
    길이의 창을 보고, 도중에 종료된 env는 그 시점부터 0으로 세므로 값은 "추종 정확도 x 생존률"이 된다.

    관측은 정책이 기대하는 형식(그룹별 TensorDict)이어야 하므로 학습에 쓰는 래퍼를 그대로 통과시킨다.
    리셋까지 포함해 전 구간을 inference_mode 안에서 돌린다 - rsl_rl의 learn()이 롤아웃을 그 모드로
    돌면서 env 내부 버퍼가 inference 텐서가 되므로, 밖에서 리셋하면 그 버퍼 쓰기가 거부된다.

    axis_of_env가 -1인 env는 채점에서 뺀다 - 축 상한보다 쉬운 행에 있는 env가 여기 들어온다.
    승급은 "지금 연 가장 어려운 난이도를 해내는가"로 판정해야 하므로, 쉬운 행의 점수를 평균에
    섞으면 상한을 못 넘는 정책도 통과하게 된다.

    axis_of_env가 None이면 평지라 축 구분이 없다 - 모든 축이 같은 표본(전체 env)을 쓴다. 평지를
    통과하면 전 축이 함께 레벨 1로 올라가고, 그때부터 축별로 갈라진다.
    """
    env = vec_env.unwrapped
    with torch.inference_mode():
        obs, _ = vec_env.reset()
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        lin_sum = torch.zeros(env.num_envs, device=env.device)
        ang_sum = torch.zeros(env.num_envs, device=env.device)
        for step in range(_AXIS_EVAL_WARMUP_STEPS + _AXIS_EVAL_STEPS):
            obs, _, dones, _ = vec_env.step(policy(obs))
            # 워밍업 중 넘어진 env는 계속 죽은 것으로 둔다 - 가속 구간의 추종만 점수에서 빼는 것이지
            # 그 사이에 쓰러진 사실까지 없던 일로 하면 생존률이 점수에 반영되지 않는다
            if step >= _AXIS_EVAL_WARMUP_STEPS:
                command = env.command_manager.get_command("base_velocity")
                root = env.scene["robot"].data
                lin_error = torch.sum(torch.square(command[:, :2] - root.root_lin_vel_b[:, :2]), dim=1)
                ang_error = torch.square(command[:, 2] - root.root_ang_vel_b[:, 2])
                lin_sum += torch.exp(-lin_error / _TRACKING_STD_SQ) * alive
                ang_sum += torch.exp(-ang_error / _TRACKING_STD_SQ) * alive
            alive &= ~dones.bool()
        lin_mean = (lin_sum / _AXIS_EVAL_STEPS).clone()
        ang_mean = (ang_sum / _AXIS_EVAL_STEPS).clone()
        scored_alive = alive if axis_of_env is None else alive[axis_of_env >= 0]
        survival = float(scored_alive.float().mean()) if scored_alive.numel() else 0.0
    scores = {}
    for axis in DIFFICULTY_AXES:
        if axis_of_env is None:
            scores[axis] = (float(lin_mean.mean()), float(ang_mean.mean()))
            continue
        mask = axis_of_env == DIFFICULTY_AXES.index(axis)
        if not bool(mask.any()):
            continue
        scores[axis] = (float(lin_mean[mask].mean()), float(ang_mean[mask].mean()))
    return scores, survival


@dataclass
class CurriculumOutcome:
    """커리큘럼 한 번의 결과 - 그대로 curriculum_result.yaml의 요약이 된다."""

    status: str  # "plateaued" | "max_iterations" | "ceiling" | "diverged"
    iterations: int
    # 축 이름 -> {"lin": .., "ang": .., "score": .., "mean": .., "best": ..}
    axis_scores: dict
    # 축 이름 -> 마지막으로 깬 난이도 레벨(1부터. 0이면 첫 레벨도 못 깼다는 뜻)
    levels: dict
    checkpoint: str | None


class CurriculumTrainer:
    """축별 난이도를 올려 가며 한 로봇의 커리큘럼 전체를 한 프로세스에서 학습한다.

    레벨 0은 평지다. 랜덤 정책을 처음부터 메시 지형에 세우면 매 스텝 순보상이 음수라 "빨리 넘어져
    벌점 누적을 끊는" 국소 최적이 먼저 잡히고(실제로 추종 보상이 약한 로봇에서 에피소드 길이가
    600에서 33으로 무너진다), 거기서는 걷기를 배우지 못한다. 그래서 지형 안에 평지 열을 하나 두고
    레벨 0에서는 그 열에 세운다 - 지형을 다시 만들지도, 프로세스를 다시 띄우지도 않는다.

    승급 판정은 축마다 따로 한다. 지형에는 다섯 축(오르막·내리막 x 단차·경사, 그리고 파쿠르 단차)이
    함께 들어 있고 학습도 함께 하지만, 평가는 각 env가 밟고 있는 타일의 축으로 나눠 집계한다 - 계단은
    오르는데 내려오지 못하는 로봇, 경사에서만 미끄러지는 로봇이 있으므로 막힌 축만 남기고 나머지는
    올리는 것이 맞다.

    다섯 축을 한 지형에서 함께 학습하는 것이 순차 학습(오르막을 끝까지 -> 그다음 내리막)보다 낫다.
    순차로 하면 뒤 과제를 배우는 동안 앞 과제의 정책이 무너져(catastrophic forgetting) 마지막에
    섞어 되돌려야 하는데, 그 되돌리기가 두 능력을 모두 중간값으로 끌어내린다. Rudin et al. 이후의
    지형 커리큘럼이 전부 "종류는 처음부터 섞고 난이도만 따로 올리는" 형태인 이유다.

    축의 레벨은 "지금까지 연 난이도의 상한"이고, 그 축의 env는 0(평지)부터 상한까지에 균등 배정된다 -
    쉬운 난이도가 항상 섞여야 학습 신호가 끊기지 않기 때문이다(_apply_levels 참조). 승급 판정은
    상한에 있는 env만 채점한다.

    승급은 지형을 다시 만들지 않는다. 지형에는 이미 전 난이도 행이 들어 있으므로, terrain_types와
    terrain_levels를 다시 배정하고 env_origins를 계산하면 다음 리셋에서 로봇이 그 타일에 놓인다 -
    Isaac Lab의 terrain_levels_vel 커리큘럼과 같은 경로다. 승급할 때 축별 env 배분도 다시 나눈다 -
    상한에 닿은 축의 몫을 줄여 아직 못 올라간 축의 연습량과 판정 표본을 늘린다.

    명령 속도 상한은 env마다 자기 레벨에서 뽑는다 - 공통 상한을 쓰면 한 축의 승급이 다른 축의 과제까지
    바꿔 그 축의 점수 이력이 서로 다른 과제의 혼합이 된다.

    점수는 축별 (선속도 추종, 각속도 추종) 중 작은 쪽이다. 판정은 최근 patience_evals회 점수의
    평균으로 한다 - eval 사이 점수는 명령 샘플링과 지형 배정이 매번 달라 ±0.05 정도로 흔들리므로,
    "매회 threshold 이상"을 요구하면 평균이 threshold를 웃도는 정책도 통과 확률이 낮아진다.
    승급한 축은 난이도가 달라지므로 그 축의 점수 이력을 비운다.

    정체는 모든 축이 plateau_window 동안 개선되지 않았을 때만 선언한다 - 한 축이라도 아직
    나아지고 있으면 그 축의 승급 가능성이 남아 있다.
    """

    def __init__(self, curriculum_cfg: dict, log_dir: str, start_levels: dict | None = None) -> None:
        per_stage = curriculum_cfg["per_stage"]
        self._eval_interval = int(per_stage["eval_interval_iters"])
        self._max_iterations_per_level = int(per_stage["max_iterations"])
        convergence = curriculum_cfg["convergence"]
        self._threshold = float(convergence["threshold"])
        self._patience_evals = int(convergence["patience_evals"])
        plateau = curriculum_cfg["plateau"]
        self._plateau_window = int(plateau["window_evals"])
        self._min_delta = float(plateau["min_delta"])
        self._command_cfg = curriculum_cfg["command"]
        self._log_dir = log_dir
        # 축별 현재 난이도 상한. 0은 평지, L>=1은 지형의 행 L-1(= 사다리의 레벨 L)이다.
        start_levels = start_levels or {}
        self._levels = {
            axis: min(max(int(start_levels.get(axis, 0)), 0), NUM_LEVELS) for axis in DIFFICULTY_AXES
        }
        self._home_columns: torch.Tensor | None = None
        self._axis_of_env: torch.Tensor | None = None
        self._env_levels: torch.Tensor | None = None
        self._level_speed_scales: torch.Tensor | None = None
        self._flat_column = 0
        self._stable_checkpoint: str | None = None
        self._recoveries_left = _RECOVERY_ATTEMPTS

    def run(self, runner: EpisodeMetricRunner, vec_env) -> CurriculumOutcome:
        """모든 축이 정체하거나 상한에 닿을 때까지 학습하고 결과를 돌려준다."""
        env = vec_env.unwrapped
        terrain = env.scene.terrain
        if getattr(terrain, "terrain_types", None) is None:
            raise RuntimeError("타일 격자가 없는 지형이다 - 커리큘럼 학습은 generator 지형에서만 된다")
        self._flat_column = flat_terrain_column()
        self._env_levels = torch.zeros_like(terrain.terrain_levels)
        self._assign_home_columns(env)
        self._apply_levels(env)

        recent = {axis: deque(maxlen=self._patience_evals) for axis in DIFFICULTY_AXES}
        best = {axis: float("-inf") for axis in DIFFICULTY_AXES}
        stale = {axis: 0 for axis in DIFFICULTY_AXES}
        latest = {axis: {"lin": 0.0, "ang": 0.0, "score": 0.0, "mean": 0.0, "best": 0.0} for axis in DIFFICULTY_AXES}
        start_iteration = runner.current_learning_iteration
        iterations_done = 0
        iterations_at_promotion = 0
        is_first_chunk = True

        while True:
            try:
                runner.learn(num_learning_iterations=self._eval_interval, init_at_random_ep_len=is_first_chunk)
            except (ValueFunctionRunaway, RuntimeError) as exc:
                if not self._recover(runner, str(exc), start_iteration):
                    return self._outcome("diverged", iterations_done, latest)
                is_first_chunk = False
                continue
            is_first_chunk = False
            # learn()은 인덱스를 eval_interval-1만 전진시키므로 실제 진행량은 카운터에서 직접 읽는다
            iterations_done = runner.current_learning_iteration - start_iteration

            scores, survival = _axis_tracking_scores(
                vec_env, self._ceiling_axis_mask(), runner.get_inference_policy(device=env.device)
            )
            if any(not math.isfinite(v) for axis_score in scores.values() for v in axis_score):
                if not self._recover(runner, "추종 점수가 유한하지 않다", start_iteration):
                    return self._outcome("diverged", iterations_done, latest)
                continue
            # 여기까지 왔으면 가중치가 멀쩡하다 - 다음 폭주 때 되돌아올 지점으로 갱신한다
            self._save_stable(runner)

            promoted = []
            report = []
            for axis in DIFFICULTY_AXES:
                lin, ang = scores.get(axis, (0.0, 0.0))
                score = min(lin, ang)
                recent[axis].append(score)
                mean = sum(recent[axis]) / len(recent[axis])
                if score > best[axis] + self._min_delta:
                    best[axis] = score
                    stale[axis] = 0
                else:
                    stale[axis] += 1
                if len(recent[axis]) == self._patience_evals and mean >= self._threshold:
                    promoted.append(axis)
                latest[axis] = {"lin": lin, "ang": ang, "score": score, "mean": mean, "best": best[axis]}
                report.append(f"{axis} L{self._levels[axis]} {score:.3f}(mean{mean:.3f} stale{stale[axis]})")
            # 생존률을 함께 찍는다 - 평가창을 다 못 버티면 점수가 0이라, 그것만으로는 "조금 걷는다"와
            # "즉시 넘어진다"를 구분할 수 없다
            print(f"[curriculum] iter {iterations_done:>5} | 생존 {survival:.2f} | " + " | ".join(report), flush=True)

            if promoted:
                for axis in promoted:
                    self._levels[axis] = min(self._levels[axis] + 1, NUM_LEVELS)
                    # 난이도가 달라졌으니 이전 난이도의 점수와 섞이면 안 된다
                    recent[axis].clear()
                    best[axis] = float("-inf")
                    stale[axis] = 0
                self._assign_home_columns(env)
                self._apply_levels(env)
                iterations_at_promotion = iterations_done
                self._save_cleared(runner)
                print(
                    f"[curriculum] 승급 {promoted} -> "
                    + " ".join(f"{axis} L{self._levels[axis]}" for axis in DIFFICULTY_AXES),
                    flush=True,
                )
                continue

            # 상한에 닿은 축은 더 올라갈 데가 없으므로 정체와 같게 본다 - 그렇게 보지 않으면 상한 축의
            # 점수가 계속 갱신되며 stale이 리셋돼, 나머지 축이 다 굳어도 학습이 끝나지 않는다
            if all(
                stale[axis] >= self._plateau_window or self._levels[axis] >= NUM_LEVELS
                for axis in DIFFICULTY_AXES
            ):
                status = "ceiling" if all(self._levels[a] >= NUM_LEVELS for a in DIFFICULTY_AXES) else "plateaued"
                return self._outcome(status, iterations_done, latest)
            if iterations_done - iterations_at_promotion >= self._max_iterations_per_level:
                return self._outcome("max_iterations", iterations_done, latest)

    def _assign_home_columns(self, env) -> None:
        """env마다 담당할 축 열과 그 열의 축 인덱스(-1은 중립)를 정한다.

        TerrainImporter의 무작위 배정을 쓰지 않고 직접 나누는 이유는 둘이다. 평지 열은 레벨 0 전용이라
        누구의 담당 열도 되면 안 되고, 축별 표본 수가 어긋나면 표본이 적은 축의 승급 판정이 더 크게
        흔들려 축끼리 비교가 성립하지 않는다.

        몫은 축의 진행 상태에 따라 승급마다 다시 나눈다 - 상한에 닿은 축은 _SATURATED_AXIS_ENV_WEIGHT
        만큼만 남기고 나머지를 미완 축에 넘긴다. 열별로 정수 개수를 정해 순서대로 나눠 주므로 배분은
        결정적이고, 내림으로 남은 env는 몫이 큰 열부터 하나씩 채운다.
        """
        column_axes = terrain_column_axes()
        # 열별 가중치 - 평지 열은 담당에서 빼고(레벨 0 전용), 축 열은 그 축이 상한에 닿았는지를 따른다
        weights = torch.zeros(len(column_axes), dtype=torch.float64)
        for column, name in enumerate(column_axes):
            if name == FLAT_AXIS:
                continue
            saturated = name in DIFFICULTY_AXES and self._levels[name] >= NUM_LEVELS
            weights[column] = _SATURATED_AXIS_ENV_WEIGHT if saturated else 1.0
        counts = torch.floor(weights / weights.sum() * env.num_envs).to(torch.long)
        remainder = env.num_envs - int(counts.sum())
        if remainder > 0:
            counts[torch.argsort(weights, descending=True)[:remainder]] += 1
        home = torch.repeat_interleave(torch.arange(len(column_axes)), counts).to(env.device)
        lookup = torch.tensor(
            [DIFFICULTY_AXES.index(name) if name in DIFFICULTY_AXES else -1 for name in column_axes],
            device=env.device,
        )
        self._home_columns = home
        self._axis_of_env = lookup[home]

    def _apply_levels(self, env) -> None:
        """축별 상한 아래로 env를 흩뿌린다 - 레벨 0은 평지 열, 레벨 L은 담당 열의 행 L-1이다.

        전 env를 상한 하나에 몰면 모든 과제가 균일하게 어려워져, 정체 구간에서 정책 경사 신호가
        끊기고 엔트로피 항이 정책 표준편차를 키우는 쪽으로 지배한다(그 노이즈가 다시 action_rate
        벌점이 되어 보상이 무너지는 되먹임이 생긴다). legged_gym과 Isaac Lab이 env마다 다른 레벨을
        주는(max_init_terrain_level) 이유가 이것이고, 여기서도 상한을 올리되 그 아래로 흩어 둔다 -
        대부분을 상한 근처에 두어 프런티어 표본을 두껍게 하면서, 나머지를 [0, 상한] 전 구간에 퍼뜨려
        쉬운 난이도와 평지 env가 항상 섞이게 한다(_sample_levels).
        """
        terrain = env.scene.terrain
        for index, axis in enumerate(DIFFICULTY_AXES):
            self._sample_levels(self._axis_of_env == index, self._levels[axis])
        # 중립(random_rough) 열은 어느 축도 아니라서 축들 중 가장 높은 상한을 따라간다
        self._sample_levels(self._axis_of_env < 0, max(self._levels.values()))
        terrain.terrain_types[:] = torch.where(self._env_levels == 0, self._flat_column, self._home_columns)
        terrain.terrain_levels[:] = (self._env_levels - 1).clamp(min=0)
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        self._apply_command_speed(env)

    def _sample_levels(self, mask: torch.Tensor, ceiling: int) -> None:
        """mask에 걸린 env의 레벨을 배정한다 - 대부분 상한 근처에, 나머지는 [0, ceiling] 전 구간에."""
        count = int(mask.sum())
        if not count:
            return
        device, dtype = self._env_levels.device, self._env_levels.dtype
        spread = torch.randint(0, ceiling + 1, (count,), device=device, dtype=dtype)
        frontier = torch.randint(max(0, ceiling - _FRONTIER_WINDOW), ceiling + 1, (count,), device=device, dtype=dtype)
        self._env_levels[mask] = torch.where(torch.rand(count, device=device) < _FRONTIER_FRACTION, frontier, spread)

    def _ceiling_axis_mask(self) -> torch.Tensor:
        """상한 레벨에 있는 env만 자기 축 인덱스를 갖고 나머지는 -1인 텐서 - 승급 판정의 표본이다.

        env가 상한 아래 여러 레벨에 흩어져 있으므로, 그중 "지금 열려 있는 가장 어려운 레벨"에 선
        env만 채점해야 승급이 그 난이도를 실제로 해낸다는 뜻이 된다.
        """
        scored = torch.full_like(self._axis_of_env, -1)
        for index, axis in enumerate(DIFFICULTY_AXES):
            scored[(self._axis_of_env == index) & (self._env_levels == self._levels[axis])] = index
        return scored

    def _apply_command_speed(self, env) -> None:
        """env마다 자기 레벨에 맞는 명령 속도 배율을 준다.

        지형이 험해질수록 같은 속도를 유지하는 것이 물리적으로 불가능해지므로 상한을 낮춰야 하는데,
        그 상한을 전 env 공통으로 두면 한 축의 승급이 다른 축의 과제까지 바꾼다 - 난이도가 그대로인
        축의 명령이 느려지므로, 승급 판정에 쓰는 점수 이력이 서로 다른 과제에서 얻은 점수의 혼합이
        되어 그 축의 정체·승급 판정이 모두 흐트러진다. env의 레벨은 이미 축별로 따로 정해져 있으므로
        배율도 그 레벨에서 뽑는다.
        """
        if self._level_speed_scales is None:
            # 레벨 -> 기준 상한 대비 배율. 레벨 수가 고정이라 한 번만 만들어 두고 인덱싱으로 쓴다
            scales = [stage_command_speed_limits(level, self._command_cfg)[1] for level in range(NUM_LEVELS + 1)]
            self._level_speed_scales = torch.tensor(scales, device=self._env_levels.device, dtype=torch.float32)
        command_term = env.command_manager.get_term("base_velocity")
        command_term.speed_scale[:] = self._level_speed_scales[self._env_levels]

    def _save_stable(self, runner: EpisodeMetricRunner) -> None:
        """폭주 시 되돌아갈 기준 스냅샷을 갱신한다 - 옵티마이저 상태까지 같이 저장된다."""
        path = os.path.join(self._log_dir, _STABLE_CHECKPOINT_FILENAME)
        runner.save(path)
        self._stable_checkpoint = path

    def _save_cleared(self, runner: EpisodeMetricRunner) -> None:
        """승급 직후의 가중치 - "마지막으로 깬 난이도"의 정책이라 배포용으로 쓴다."""
        runner.save(os.path.join(self._log_dir, _CLEARED_CHECKPOINT_FILENAME))

    def _recover(self, runner: EpisodeMetricRunner, reason: str, start_iteration: int) -> bool:
        """폭주 지점을 버리고 직전 정상 스냅샷으로 되돌린다 - 되돌릴 수 없으면 False.

        되돌릴 곳이 없는 경우는 첫 평가에 닿기도 전에 폭주한 때뿐이다(스냅샷은 평가를 통과할 때만
        갱신된다). 그때는 쓸 만한 가중치가 애초에 없으므로 발산으로 넘긴다.
        """
        if self._stable_checkpoint is None or self._recoveries_left <= 0:
            print(f"[curriculum] ! 발산 - {reason} (되돌릴 스냅샷 없음 또는 복구 횟수 소진)", flush=True)
            return False
        self._recoveries_left -= 1
        runner.load(self._stable_checkpoint)
        runner.alg.value_loss_coef *= _VALUE_LOSS_COEF_DECAY
        print(
            f"[curriculum] ! 폭주 감지 - {reason} | iter "
            f"{runner.current_learning_iteration - start_iteration}로 되돌리고 "
            f"value_loss_coef={runner.alg.value_loss_coef:.4g}로 재개 "
            f"(남은 복구 {self._recoveries_left}회)",
            flush=True,
        )
        return True

    def _outcome(self, status: str, iterations: int, axis_scores: dict) -> CurriculumOutcome:
        """마지막으로 깬 난이도 레벨과 함께 결과를 만든다.

        상한은 "지금 도전 중인 레벨"이라 아직 깬 것이 아니다. 그래서 깬 레벨은 상한에서 하나를 뺀
        값이고, 평지(레벨 0)도 못 깬 경우는 0으로 둔다.
        """
        return CurriculumOutcome(
            status=status,
            iterations=iterations,
            axis_scores=axis_scores,
            levels={axis: max(level - 1, 0) for axis, level in self._levels.items()},
            checkpoint=self._cleared_or_stable_checkpoint(),
        )

    def _cleared_or_stable_checkpoint(self) -> str | None:
        """배포용 가중치 경로 - 승급 시 저장한 것이 있으면 그것, 없으면 마지막 정상 스냅샷."""
        cleared = Path(self._log_dir) / _CLEARED_CHECKPOINT_FILENAME
        if cleared.exists():
            return str(cleared)
        return self._stable_checkpoint


class CurriculumSession:
    """한 로봇의 커리큘럼 학습에 필요한 자원 수명과 산출물을 관리한다.

    env·러너를 한 번만 조립해 CurriculumTrainer에 넘기고, 끝나면 policy.pt와
    curriculum_result.yaml을 남긴다. with 문으로 써서 어떤 경로로 빠져나가든 env가 닫히게 한다.
    """

    def __init__(
        self,
        robot_id: str,
        preset_name: str | None = None,
        num_envs: int | None = None,
        device: str = "cuda:0",
        resume_checkpoint: str | None = None,
        start_levels: dict | None = None,
    ) -> None:
        self._robot_id = robot_id
        self._start_levels = start_levels
        self._profile = RobotProfile.load(robot_id)
        self._preset_name = preset_name or self._profile.rl_preset
        self._preset = RLPreset.load(self._preset_name)
        self._num_envs = num_envs or self._preset.num_envs
        self._device = device
        self._resume_checkpoint = resume_checkpoint
        self._log_dir = curriculum_log_dir(robot_id)
        self._output_dir = policy_output_dir(robot_id)
        self._env: ManagerBasedRLEnv | None = None

    def __enter__(self) -> CurriculumSession:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> bool:
        if self._env is not None:
            self._env.close()
            self._env = None
        return False

    def run(self) -> CurriculumOutcome:
        """커리큘럼을 끝까지 학습하고 결과를 파일로 남긴 뒤 돌려준다."""
        stage_cfg = self._preset.curriculum["stage"]
        print(
            f"[curriculum] {self._robot_id} | preset={self._preset_name} | num_envs={self._num_envs} "
            f"| device={self._device} | 축 {'/'.join(DIFFICULTY_AXES)} | 난이도 레벨 1-{NUM_LEVELS}",
            flush=True,
        )
        print(f"[curriculum] 진행 상황: tensorboard --logdir {self._log_dir}", flush=True)

        env_cfg = build_loco_rl_env_cfg(
            self._robot_id,
            num_envs=self._num_envs,
            preset=self._preset_name,
            terrain_cfg=build_curriculum_terrain_importer_cfg(stage_cfg),
        )
        env_cfg.sim.device = self._device
        self._env = ManagerBasedRLEnv(cfg=env_cfg)

        agent_cfg = build_agent_cfg(self._profile, self._preset, experiment_name(self._robot_id), device=self._device)
        vec_env = RslRlVecEnvWrapper(self._env, clip_actions=agent_cfg.clip_actions)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        runner = EpisodeMetricRunner(vec_env, agent_cfg.to_dict(), log_dir=str(self._log_dir), device=agent_cfg.device)
        if self._resume_checkpoint is not None:
            print(f"[curriculum] 가중치를 {self._resume_checkpoint} 에서 이어받음", flush=True)
            runner.load(self._resume_checkpoint)

        trainer = CurriculumTrainer(self._preset.curriculum, str(self._log_dir), self._start_levels)
        outcome = trainer.run(runner, vec_env)
        summary = stage_terrain_summary(outcome.levels, stage_cfg)
        print(
            "[curriculum] 완료 - "
            + " ".join(f"{axis} 레벨 {outcome.levels[axis]}" for axis in DIFFICULTY_AXES)
            + f" | status={outcome.status} ({outcome.iterations} iters)",
            flush=True,
        )
        self._export_policy(runner, agent_cfg)
        write_curriculum_result(
            robot_id=self._robot_id,
            preset_name=self._preset_name,
            outcome=outcome,
            terrain_summary=summary,
        )
        return outcome

    def _export_policy(self, runner: EpisodeMetricRunner, agent_cfg) -> None:
        """현재 러너의 정책을 jit로 내보내 04_controller_test.py의 LocoRunner가 바로 로드하게 한다."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        policy_nn = runner.alg.policy
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(self._output_dir), filename="policy.pt")
        dump_yaml(str(self._output_dir / "train_config.yaml"), agent_cfg)

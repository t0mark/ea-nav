"""form 정책 PPO 학습 오케스트레이션 라이브러리 (진입점 아님 — tools가 호출).

Isaac 앱 기동 후에만 임포트할 수 있다 (train_env가 Isaac 의존).

rsl-rl OnPolicyRunner를 청크 단위로 돌린다: 청크마다 완료 에피소드 통계를
로그·곡선에 기록하고 체크포인트를 저장한다 (시간 기준 모니터링 규약 —
rsl-rl 내부 로거 대신 우리 로그로 통일). 학습 종료 시 PolicyBundle로
배포 포맷(policy.pt + bundle.json)을 export한다.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import torch

from rsl_rl.runners import OnPolicyRunner

from scripts.sim.controller.legged import low_rl
from scripts.sim.controller.legged.train.train_env import LeggedTrainEnv

logger = logging.getLogger(__name__)


def _merge_form_cfg(rl_cfg: dict, form: str) -> dict:
    """rl.yaml 기본값 위에 forms.{form} 오버라이드를 깊은 병합한다.

    form마다 다른 값(humanoid의 종료 감점·명령 범위·게인 등)을 한 파일에서
    관리하기 위한 구조 — 병합 결과가 bundle.json에 저장되므로 배포도 form별
    값을 그대로 재현한다. 독립 검증 지적("quad 실측으로 humanoid 값을
    완화하는 인과 뒤집힘")의 구조적 처방.
    """
    def merge(base, over):
        out = dict(base)
        for k, v in over.items():
            out[k] = merge(base[k], v) if isinstance(v, dict) \
                and isinstance(base.get(k), dict) else v
        return out

    return merge(rl_cfg, rl_cfg.get("forms", {}).get(form, {}))


def _runner_cfg(ppo_cfg: dict) -> dict:
    """configs/rl.yaml ppo 섹션 -> rsl-rl OnPolicyRunner train_cfg 변환.

    관측 그룹은 "policy" 하나 (critic 동일 관측 — 특권 관측 없음).
    관측 정규화는 actor·critic 양쪽 활성 (배포 시 actor 정규화가 번들에
    포함된다 — low_rl._DeployPolicy).
    """
    return {
        "num_steps_per_env": int(ppo_cfg["num_steps_per_env"]),
        "save_interval": 10 ** 9,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": list(ppo_cfg["hidden_dims"]),
            "critic_hidden_dims": list(ppo_cfg["hidden_dims"]),
            "activation": ppo_cfg["activation"],
            "init_noise_std": float(ppo_cfg["init_noise_std"]),
            # 스톡 Isaac velocity cfg와 정렬: 정규화 OFF. ON(경험 통계 영구
            # 갱신)은 입력 표현 비정상성 + 저분산 차원 eps 증폭 자가강화로
            # "평균 정책 기립 수렴 + std 고정"을 단일 기전으로 만드는 1위
            # 결함 후보 (스톡/우리 전 필드 diff — code.md). OFF면 export
            # 래퍼의 normalizer는 rsl-rl이 Identity로 채워 배포 경로 무수정
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": int(ppo_cfg["num_learning_epochs"]),
            "num_mini_batches": int(ppo_cfg["num_mini_batches"]),
            "clip_param": float(ppo_cfg["clip_param"]),
            "gamma": float(ppo_cfg["gamma"]),
            "lam": float(ppo_cfg["lam"]),
            "value_loss_coef": float(ppo_cfg["value_loss_coef"]),
            "entropy_coef": float(ppo_cfg["entropy_coef"]),
            "learning_rate": float(ppo_cfg["learning_rate"]),
            "max_grad_norm": float(ppo_cfg["max_grad_norm"]),
            "schedule": ppo_cfg["schedule"],
            "desired_kl": float(ppo_cfg["desired_kl"]),
        },
    }


def train_form(form: str, robot_dirs: list[tuple[Path, Path]], rl_cfg: dict,
               sim_cfg: dict, out_dir: Path, mode: str, device: str) -> dict:
    """form 정책 1개를 학습하고 배포 번들을 out_dir에 export한다.

    robot_dirs = [(robot.urdf, usd 폴더)] 학습 로봇 셋 (호출측이 시드
    결정적으로 선택). mode = pilot|full (configs/rl.yaml train 섹션의 규모
    분기 — 파일럿은 코드 실행 검증용 소규모). 반환: 학습 요약 (곡선 포함).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # form별 오버라이드 병합 — 이후 모든 참조(환경·번들)는 병합본 기준
    rl_cfg = _merge_form_cfg(rl_cfg, form)
    train_cfg = rl_cfg["train"]
    scale = train_cfg[mode]
    iterations = int(scale["iterations"])
    chunk = max(1, min(int(train_cfg["chunk_iters"]), iterations))
    seed = int(train_cfg["seed"])
    torch.manual_seed(seed)

    # 지형 설정 = sim.yaml terrain(서브지형 종류·치수) + rl.yaml 규모 오버레이.
    # 규모별 terrain_mode가 기본(env.terrain)을 덮는다 (보행 사다리 —
    # 단계 0은 평지에서 보행 존재 증명부터)
    terrain_mode = scale.get("terrain_mode", rl_cfg["env"]["terrain"])
    if terrain_mode == "rough":
        terrain_cfg = {**sim_cfg["terrain"], **scale.get("terrain", {})}
    else:
        terrain_cfg = None
    env = LeggedTrainEnv(form, robot_dirs, rl_cfg, sim_cfg,
                         envs_per_robot=int(scale["envs_per_robot"]),
                         device=device, seed=seed, terrain_cfg=terrain_cfg)
    # log_dir는 필수 — rsl-rl 3.0.1은 learn()의 코드 상태 저장이 log_dir
    # None을 처리하지 못한다 (store_code_state 무가드). tensorboard 이벤트도
    # 여기 쌓인다 (우리 청크 로그·곡선과 별개의 세부 곡선)
    runner = OnPolicyRunner(env, _runner_cfg(rl_cfg["ppo"]),
                            log_dir=str(out_dir / "rsl_log"), device=device)

    # 청크 학습 루프: 진행 로그 + 곡선 기록 + 체크포인트 (실행 중 모니터링)
    curve, done_iters = [], 0
    t0 = time.time()
    init_std = float(rl_cfg["ppo"]["init_noise_std"])
    anneal_to = rl_cfg["ppo"].get("std_anneal_to")
    while done_iters < iterations:
        n = min(chunk, iterations - done_iters)
        runner.learn(n, init_at_random_ep_len=(done_iters == 0))
        done_iters += n
        # 탐색 std 어닐링 (상한 선형 스케줄): std가 entropy 하향 후에도
        # 0.74-0.8에 구조적으로 고정되는 것 실측 (노이즈 디더링이 보행을
        # 대행 -> 축소 압력 소멸) — 상한을 내려 평균 정책이 보행을
        # 인수하게 강제한다 (탐색 어닐링 표준 기법)
        if anneal_to is not None:
            cap = init_std + (float(anneal_to) - init_std) \
                * min(done_iters / iterations, 1.0)
            with torch.no_grad():
                runner.alg.policy.std.data.clamp_(max=cap)
        stats = env.pop_stats()
        stats["iteration"] = done_iters
        stats["elapsed_s"] = round(time.time() - t0, 1)
        curve.append(stats)
        runner.save(str(out_dir / "checkpoint.pt"))
        logger.info("[%s %d/%d] 평균 수익 %.2f, 에피소드 %.1fs x %d개 (조기 종료 %d), "
                    "지형 레벨 %s, 명령 커리큘럼 %s, 보행 %s, 경과 %.0fs",
                    form, done_iters, iterations, stats["mean_return"],
                    stats["mean_ep_len_s"], stats["episodes"],
                    stats["terminations"], stats.get("terrain_level", "-"),
                    stats.get("cmd_v_ratio", "-"), stats.get("walked_m", "-"),
                    stats["elapsed_s"])

    # 배포 번들 export: 정책·정규화는 deepcopy로 굳힌다 (runner 상태 불변)
    policy = runner.alg.policy
    bundle_meta = {
        "form": form,
        "obs_dim": low_rl.obs_dim(form, low_rl.scan_rays(rl_cfg["scan"]),
                                  rl_cfg["obs"]),
        "num_actions": env.num_actions,
        "action_scale": float(rl_cfg["env"]["action_scale"]),
        "action_clip": float(rl_cfg["env"]["action_clip"]),
        "decimation": int(rl_cfg["env"]["decimation"]),
        "physics_dt": float(rl_cfg["env"]["physics_dt"]),
        "obs": dict(rl_cfg["obs"]),
        "morph": dict(rl_cfg["morph"]),
        # 평가 스폰이 학습과 같은 액추에이터 모델을 재현하기 위한 규약
        "actuator_model": str(rl_cfg["env"]["actuator_model"]),
        "cmd": dict(rl_cfg["cmd"]),
        "gains": dict(rl_cfg["gains"]),
        # 3단계 지형 롤아웃이 같은 스캔 격자로 RayCaster를 만들기 위한 규약
        "scan": dict(rl_cfg["scan"]),
        "train": {"mode": mode, "iterations": iterations, "seed": seed,
                  "terrain": rl_cfg["env"]["terrain"],
                  # 명령 커리큘럼 도달률 (1 미만이면 학습이 로봇 상한 속도까지
                  # 못 갔다는 뜻 — deploy_ratio 재산정 근거 자료)
                  "cmd_reached_ratio": (curve[-1].get("cmd_v_ratio")
                                        if curve else None),
                  # 로봇별 도달 상한 (m/s) — 배포 명령 클램프 (전 로봇 평균은
                  # v_max 큰 로봇에 과대 명령 -> 전도 실측)
                  "reached_v": env.per_robot_reached(),
                  "num_robots": len(robot_dirs),
                  "robots": [str(Path(u).parent.name) for u, _ in robot_dirs]},
    }
    low_rl.PolicyBundle.export(out_dir,
                               copy.deepcopy(policy.actor_obs_normalizer),
                               copy.deepcopy(policy.actor), bundle_meta)
    with open(out_dir / "train_curve.json", "w") as f:
        json.dump(curve, f, indent=1, ensure_ascii=False)
    logger.info("정책 export 완료: %s (학습 %d회, %.0fs)", out_dir, iterations,
                time.time() - t0)
    return {"curve": curve, "bundle": bundle_meta}

"""legged 로봇의 RL 보행 정책을 학습하는 진입점 - 로봇 하나를 커리큘럼으로 끝까지 학습한다.

configs/robots/legged/multi-legged/{robot_id}.yaml 한 장이 로봇의 물리적 사실(usd 경로·바디/관절
이름·액추에이터 수치·reward_weights·reward_type·도메인 랜덤화)을 담고, `rl_preset:`으로
configs/rl/legged/presets/{preset}.yaml을 가리킨다. preset이 학습 설계(관측/보상/종료/이벤트/액션
로직 선택 + PPO 하이퍼파라미터 + 커리큘럼 파라미터)를 정한다 - 4개 로봇이 같은 preset을 가리키면
embodiment 비교가 성립한다.

이 파일은 두 가지 모드로 동작하고, 자기 자신을 단계마다 다시 실행한다:
  --stage 없음 = 지휘 모드. 시뮬레이터를 띄우지 않고, 커리큘럼 단계마다 이 파일을 --stage N으로
    자식 프로세스에 다시 띄운다(scripts/.../rl/curriculum_driver.py).
  --stage N 있음 = 단계 모드. AppLauncher로 Isaac Sim을 띄우고 그 단계 하나만 학습한 뒤 결과를
    stage_outcome.yaml에 남기고 프로세스를 끝낸다(scripts/.../rl/rl_trainer.py).
단계를 프로세스로 자르는 이유는 Isaac Lab이 한 프로세스에서 ManagerBasedRLEnv를 두 번 만들지
못하기 때문이다 - env.close()가 USD stage의 prim을 지우지 않아 두 번째 env 생성이 "A prim already
exists at path"로 실패한다(자세한 근거는 curriculum_driver.py 모듈 docstring).

커리큘럼은 자동이다(scripts/sim/env/curriculum/stage_env.py + CurriculumDriver):
  stage 0(평지)부터 학습 -> 정규화 추종 정확도(track_lin_vel_xy_exp / track_ang_vel_z_exp)가
  preset의 convergence.threshold 이상인 상태가 patience_evals회 연속 -> 그 단계 "수렴" 판정 ->
  stage += 1 로 지형 난이도를 올려 다시 학습 -> 반복.
  한 단계가 수렴하지 못하면(정규화 점수 best가 plateau.window_evals 동안 min_delta 미만 개선 =
  "정체", 또는 per_stage.max_iterations 도달, 또는 PPO 발산) 커리큘럼을 종료한다. 자식 프로세스가
  비정상 종료하면 부모가 종료 코드로 알아채고 이력을 남긴 뒤 끝낸다.
전역 이터레이션 종료 로직은 없다 - 학습이 어디까지 가는지는 로봇이 지형을 못 깰 때 결정된다.

학습이 끝나면:
- data/sim/policies/legged/multi-legged/{robot_id}/policy.pt = 마지막으로 "수렴한" 단계의 정책(jit)
- .../curriculum_result.yaml = 단계별 이력 + 최종 클리어 단차/경사 등 메타데이터
- _train_logs/multi-legged_{robot_id}/stage{N}/ = 단계별 tensorboard 로그 + stage_outcome.yaml

학습 중에는 렌더링을 절대 켜지 않는다 - 병렬 env가 수천 개(PhysX)인 상태에서 카메라 렌더링까지
동시에 돌리면 GPU 순간 전력이 튀어 PSU 보호회로가 시스템 전체를 강제 종료시킬 수 있다(VRAM/RAM
부족이 아니라 순간 전력 스파이크 문제). 영상으로 실제 걷는 모습을 보고 싶으면 학습이 끝난 뒤
tools/04_controller_test.py를 별도 프로세스로 실행한다.

멀티 GPU 장비에서 특정 GPU 하나에 학습을 몰아넣고 싶으면 --device cuda:N을 쓴다 - 부모가 그 값을
자식에게 그대로 넘겨 env(PhysX)와 rsl_rl 학습 러너(정책망·PPO 옵티마이저) 둘 다 그 GPU로 맞춘다.

사용법:
    # 처음부터(stage 0, 평지) 커리큘럼 시작
    /workspace/isaaclab/isaaclab.sh -p tools/03_controller_rl.py --robot-id unitree_go2

    # 커리큘럼이 중간에 끊겼을 때(발산·정전 등) 특정 단계 체크포인트에서 이어서
    /workspace/isaaclab/isaaclab.sh -p tools/03_controller_rl.py --robot-id unitree_go2 \
        --start-stage 3 --resume-from data/sim/policies/legged/_train_logs/multi-legged_unitree_go2/stage2/converged.pt

    # 실험용 다른 preset으로
    /workspace/isaaclab/isaaclab.sh -p tools/03_controller_rl.py --robot-id unitree_go2 --preset my_variant

    # 단계 하나만 따로 학습(부모가 내부적으로 쓰는 형태 - 디버깅용으로 직접 써도 된다)
    /workspace/isaaclab/isaaclab.sh -p tools/03_controller_rl.py --robot-id unitree_go2 --stage 1
"""

import argparse
import sys
import traceback
from pathlib import Path

# 워크스페이스 루트를 sys.path에 추가해 scripts/, configs/ 를 패키지로 임포트할 수 있게 한다
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_ENTRY_POINT = Path(__file__).resolve()
_KEYBOARD_INTERRUPT_EXIT_CODE = 130


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    """지휘 모드와 단계 모드가 함께 쓰는 인자."""
    parser.add_argument(
        "--robot-id", type=str, required=True, help="configs/robots/legged/multi-legged/ 안의 robot_id"
    )
    parser.add_argument("--preset", type=str, default=None, help="RL 설계 번들 이름 (생략 시 robot yaml의 rl_preset)")
    parser.add_argument("--num-envs", type=int, default=None, help="병렬 env 수 (생략 시 preset의 num_envs)")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="시작 단계를 이 체크포인트(stage{N}/model_*.pt 또는 converged.pt)의 가중치에서 이어 시작한다.",
    )


def _is_stage_worker(argv: list[str]) -> bool:
    """--stage가 주어졌는지 - 주어졌으면 단계 모드(시뮬레이터 기동)다."""
    return any(arg == "--stage" or arg.startswith("--stage=") for arg in argv)


def _run_curriculum() -> None:
    """지휘 모드 - 시뮬레이터 없이 단계마다 이 진입점을 자식 프로세스로 다시 띄운다."""
    parser = argparse.ArgumentParser(description="legged 로봇 RL 보행 정책 학습 (자동 커리큘럼)")
    _add_shared_args(parser)
    parser.add_argument("--start-stage", type=int, default=0, help="커리큘럼을 이 단계부터 시작 (재개용, 기본 0)")
    # 단계 모드에서는 AppLauncher가 같은 이름의 인자를 정의하므로, 여기서만 직접 선언한다
    parser.add_argument("--device", type=str, default="cuda:0", help="env·학습 러너가 함께 쓸 장치 (예: cuda:1)")
    args, _ = parser.parse_known_args()

    from scripts.sim.controller.legged.rl.curriculum_driver import CurriculumDriver

    CurriculumDriver(
        entry_point=_ENTRY_POINT,
        robot_id=args.robot_id,
        preset_name=args.preset,
        num_envs=args.num_envs,
        device=args.device,
        start_stage=args.start_stage,
        resume_checkpoint=args.resume_from,
    ).run()


def _run_stage() -> None:
    """단계 모드 - Isaac Sim을 띄우고 --stage 하나만 학습한 뒤 종료 코드로 성패를 알린다."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="legged 로봇 RL 커리큘럼 단계 하나 학습")
    _add_shared_args(parser)
    parser.add_argument("--stage", type=int, required=True, help="학습할 커리큘럼 단계 (0=평지)")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    args.headless = True
    args.enable_cameras = False  # 학습 중에는 렌더링을 절대 켜지 않는다(모듈 docstring 참고)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    exit_code = 0
    try:
        # AppLauncher 기동 이후에만 isaaclab 의존 모듈을 임포트할 수 있다
        from scripts.sim.controller.legged.rl.rl_trainer import StageSession

        with StageSession(
            robot_id=args.robot_id,
            stage=args.stage,
            preset_name=args.preset,
            num_envs=args.num_envs,
            device=args.device,
            resume_checkpoint=args.resume_from,
        ) as session:
            session.run()
    except KeyboardInterrupt:
        print(f"\n[stage] 사용자 중단 - stage {args.stage} 종료", flush=True)
        exit_code = _KEYBOARD_INTERRUPT_EXIT_CODE
    except Exception:  # noqa: BLE001 - 어떤 실패든 앱을 닫고 종료 코드로 부모에게 알려야 한다
        traceback.print_exc()
        exit_code = 1
    finally:
        # 이 호출을 건너뛰면 Kit의 스레드가 살아남아 프로세스가 GPU를 잡은 채 끝나지 않는다
        simulation_app.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    if _is_stage_worker(sys.argv[1:]):
        _run_stage()
    else:
        _run_curriculum()

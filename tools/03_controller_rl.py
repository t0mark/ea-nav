"""legged 로봇 RL 보행 정책 학습 - 커리큘럼 전체를 한 프로세스로 돈다.

난이도 승급은 지형을 다시 만드는 일이 아니다. 지형은 시작할 때 전 난이도 행을 한 번에 만들어
두고(scripts/sim/env/curriculum/stage_env.py), 승급은 그 안에서 env를 더 어려운 행으로 옮기는
텐서 연산으로 처리한다(TerrainImporter.terrain_levels / env_origins) - Isaac Lab의 공식
terrain_levels_vel 커리큘럼이 쓰는 것과 같은 경로다. 그래서 단계마다 프로세스를 다시 띄우거나
env를 다시 만들 필요가 없다(한 프로세스에서 ManagerBasedRLEnv는 한 번만 만들 수 있다).

커리큘럼 구성:
  - 난이도 축은 오르막 단차·내리막 단차·오르막 경사·내리막 경사·파쿠르 단차 다섯이고 각자 레벨을
    가진다. 지형의 열이 곧 축이다 - Isaac Lab 서브지형은 스폰 지점이 타일 중심인데 pyramid_stairs는
    중심이 꼭대기라 내리막, inverted는 구덩이 바닥이라 오르막이 된다(경사면도 같은 구조).
    random_rough 열은 어느 축도 아닌 중립이다.
  - 다섯 축을 처음부터 한 지형에 섞어 학습한다. 오르막을 끝까지 올린 뒤 내리막을 학습하는 순차
    방식은 앞 과제의 정책이 무너지므로(catastrophic forgetting) 쓰지 않는다.
  - 명령 속도 상한은 env마다 자기 레벨에서 뽑는다 - 공통 상한을 쓰면 한 축의 승급이 다른 축의
    과제까지 바꿔 그 축의 점수 이력이 서로 다른 과제의 혼합이 된다.
  - eval_interval_iters마다 축별 추종 정확도를 재고, convergence.threshold를 넘긴 축만 한 레벨
    올린다. 올린 축은 난이도가 달라졌으므로 그 축의 점수 이력을 비운다.
  - 모든 축이 plateau.window_evals 동안 개선되지 않거나, 마지막 승급 이후
    per_stage.max_iterations를 넘기거나, 모든 축이 최고 레벨에 닿으면 끝난다.

산출물:
  - data/sim/policies/legged/multi-legged/{robot_id}/policy.pt          = 배포용 정책
  - data/sim/policies/legged/multi-legged/{robot_id}/curriculum_result.yaml = 축별 최종 난이도
  - data/sim/policies/legged/_train_logs/multi-legged_{robot_id}/       = tensorboard 로그·체크포인트

사용법:
    /workspace/isaaclab/isaaclab.sh -p tools/03_controller_rl.py --robot-id unitree_go2

    # 다른 GPU에서, 설계 번들을 바꿔서
    /workspace/isaaclab/isaaclab.sh -p tools/03_controller_rl.py --robot-id unitree_go2 \
        --preset my_variant --device cuda:1

    # 이전 학습 가중치와 난이도에서 이어서 (적지 않은 축은 레벨 0에서 시작)
    /workspace/isaaclab/isaaclab.sh -p tools/03_controller_rl.py --robot-id unitree_go2 \
        --start-levels step_up=7,step_down=4,slope_up=9,slope_down=5 \
        --resume-from data/sim/policies/legged/_train_logs/multi-legged_unitree_go2/cleared.pt
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

_KEYBOARD_INTERRUPT_EXIT_CODE = 130


def select_visible_gpu() -> str:
    """--device가 가리키는 GPU 하나만 CUDA에 노출하고, 프로세스 안에서 쓸 디바이스 이름을 돌려준다.

    Isaac Sim은 기동할 때 보이는 GPU를 전부 초기화해 장당 수백 MiB의 컨텍스트를 남긴다. --device는
    "어디서 계산할지"만 정하고 "어느 GPU를 열거할지"는 막지 못하므로, 노출 자체를 좁혀야 지정한
    GPU 밖으로 나가지 않는다. CUDA는 컨텍스트가 만들어진 뒤에는 노출 목록을 되돌릴 수 없어,
    isaaclab을 임포트하기 전에 정한다.

    노출된 GPU가 하나뿐이면 그 GPU의 논리 인덱스는 항상 0이므로 앱에는 cuda:0을 넘긴다.
    """
    device_parser = argparse.ArgumentParser(add_help=False)
    device_parser.add_argument("--device", type=str, default="cuda:0")
    device_arg = device_parser.parse_known_args()[0].device
    if "cuda" not in device_arg:
        return device_arg
    ordinal = device_arg.split(":")[1] if ":" in device_arg else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = ordinal
    return "cuda:0"


# 노출 GPU를 좁힌 뒤에야 isaaclab을 임포트할 수 있다 - 임포트가 CUDA를 먼저 초기화하면 늦는다
_REQUESTED_DEVICE = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0"
_SESSION_DEVICE = select_visible_gpu()

from isaaclab.app import AppLauncher  # noqa: E402

# 워크스페이스 루트를 sys.path에 추가해 scripts/, configs/ 를 패키지로 임포트할 수 있게 한다
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# 진행 로그를 줄 단위로 내보낸다 - 출력을 파일로 리다이렉트하면 stdout이 블록 버퍼링이 되는데,
# Isaac Sim은 simulation_app.close()에서 프로세스를 즉시 끊어 버퍼에 남은 줄이 사라진다
# (rsl_rl의 이터레이션 로그가 여기 해당한다. 이 프로젝트의 print는 전부 flush=True다).
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="legged 로봇 RL 보행 정책 학습 (자동 커리큘럼)")
parser.add_argument("--robot-id", type=str, required=True, help="configs/robots/legged/multi-legged/ 안의 robot_id")
parser.add_argument("--preset", type=str, default=None, help="RL 설계 번들 이름 (생략 시 robot yaml의 rl_preset)")
parser.add_argument("--num-envs", type=int, default=None, help="병렬 env 수 (생략 시 preset의 num_envs)")
parser.add_argument(
    "--start-levels",
    type=str,
    default="",
    help="재개용 축별 시작 레벨 - 'step_up=7,slope_down=3' 형식 (적지 않은 축은 0)",
)
parser.add_argument("--resume-from", type=str, default=None, help="이 체크포인트의 가중치에서 이어 시작한다")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
# 학습 중에는 렌더링을 절대 켜지 않는다 - 카메라를 켜면 GPU 전력 스파이크로 학습이 크게 느려진다
args_cli.enable_cameras = False
# 노출을 좁힌 뒤의 논리 인덱스로 덮어쓴다 - 명령줄의 cuda:N은 노출 전 인덱스라 그대로 쓰면 어긋난다
args_cli.device = _SESSION_DEVICE

print(f"[curriculum] 요청 GPU {_REQUESTED_DEVICE} -> 노출 후 {_SESSION_DEVICE}", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# AppLauncher 기동 이후에만 isaaclab 의존 모듈을 임포트할 수 있다
from scripts.sim.controller.legged.rl.rl_trainer import CurriculumSession  # noqa: E402
from scripts.sim.env.curriculum.stage_env import DIFFICULTY_AXES  # noqa: E402


def parse_start_levels(spec: str) -> dict:
    """'축=레벨' 목록을 축 이름 -> 레벨 dict로 바꾼다 - 축 이름을 코드에 박지 않기 위한 형식이다."""
    levels = {}
    for item in (piece.strip() for piece in spec.split(",") if piece.strip()):
        axis, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"'{item}' 은 '축=레벨' 형식이 아니다")
        axis = axis.strip()
        if axis not in DIFFICULTY_AXES:
            raise ValueError(f"알 수 없는 난이도 축 '{axis}' - 가능한 값: {', '.join(DIFFICULTY_AXES)}")
        levels[axis] = int(value)
    return levels


def main() -> int:
    """커리큘럼을 끝까지 학습하고 종료 코드를 돌려준다."""
    with CurriculumSession(
        robot_id=args_cli.robot_id,
        preset_name=args_cli.preset,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        resume_checkpoint=args_cli.resume_from,
        start_levels=parse_start_levels(args_cli.start_levels),
    ) as session:
        session.run()
    return 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n[curriculum] 사용자 중단", flush=True)
        exit_code = _KEYBOARD_INTERRUPT_EXIT_CODE
    except Exception:  # noqa: BLE001 - 어떤 실패든 앱을 닫고 종료 코드로 알려야 한다
        traceback.print_exc()
        exit_code = 1
    finally:
        # 이 호출을 건너뛰면 Kit의 스레드가 살아남아 프로세스가 GPU를 잡은 채 끝나지 않는다
        simulation_app.close()
    sys.exit(exit_code)

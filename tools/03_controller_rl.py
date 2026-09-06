"""legged 로봇의 RL 보행 정책을 학습하는 진입점 - 로봇 하나를 지정하면 그대로 본 학습을 돌린다.

robot_id가 multi-legged/humanoid 어느 쪽인지는 usd 파일이 실제로 있는 폴더를 뒤져 자동으로 정한다 -
robot_id 하나로 이미 어떤 usd를 볼지 정해지는데 카테고리까지 따로 입력받으면 같은 정보를 두 번
받는 셈이라 굳이 인자로 두지 않는다.

configs/robots/legged/{robot_type}/{robot_id}.yaml이 없으면 scripts/sim/utils/usd_export_config.py로
자동 생성한 뒤 바로 학습을 이어간다. 이 yaml 한 장이 로봇의 물리적 사실(usd 경로·base/foot 링크 이름)뿐
아니라 학습 방식 축(actuator/action/rewards/termination/policy_architecture/algorithm/
domain_randomization) 선택과 agent 하이퍼파라미터까지 전부 담고 있어, 로봇 종류에 따른 기본값 분기가
이 진입점 코드 안에는 전혀 없다 - scripts/sim/controller/legged/rl/robot_profile.py가 그 yaml을 파싱하고,
loco_rl_env.py/agent_cfg.py가 각 축 레지스트리를 조회해 조립한다.

학습 중에는 렌더링을 절대 켜지 않는다 - 병렬 env가 수천 개(PhysX)인 상태에서 카메라 렌더링까지
동시에 돌리면 GPU 순간 전력이 튀어 PSU 보호회로가 시스템 전체를 강제 종료시키는 것을 실제로
확인했다(관측됨: enable_cameras=True + 학습 내내 도는 RecordVideo 조합에서 재현, VRAM/RAM 부족이
아니라 전력 스파이크 문제). 그래서 학습은 완전히 헤드리스로만 돌리고, 진행 상황은 rsl_rl이 남기는
tensorboard 로그(숫자 지표라 렌더링이 전혀 필요 없다)로 확인한다 - `tensorboard --logdir
data/sim/policies/legged/_train_logs/{experiment_name}`. 영상으로 실제 걷는 모습을 보고 싶으면
학습이 끝난 뒤 tools/04_controller_test.py를 별도 프로세스로 실행해 확인한다 - 그때는 로봇 1대
(num_envs=1)만 돌아가서 학습 때와 같은 동시 부하가 생기지 않는다.

학습 종료는 max_iterations에 도달하면 끝나는 고정 스텝 방식이다 - rsl_rl의 OnPolicyRunner는 보행이
"충분히 학습됐는지"를 자동으로 판정하는 기능이 없다(PPO 학습은 통상 이렇다).

멀티 GPU 장비에서 특정 GPU 하나에 학습을 몰아넣고 싶으면 --device cuda:N을 쓴다(AppLauncher 표준
인자, 이 저장소의 다른 tools/ 스크립트와 동일한 방식) - env(PhysX·렌더링)와 rsl_rl 학습 러너(정책망·
PPO 옵티마이저) 둘 다 그 GPU로 맞춘다. rsl_rl의 RslRlOnPolicyRunnerCfg.device 기본값이 "cuda:0"으로
고정돼 있어서(재클론해 확인: isaaclab_rl/rsl_rl/rl_cfg.py) --device만으로는 env만 옮겨가고 러너는
그대로 cuda:0에 남는 문제가 있었다 - agent_cfg.build_agent_cfg()에 device를 명시적으로 넘겨 고쳤다.

사용법:
    /workspace/isaaclab/isaaclab.sh -p tools/03_controller_rl.py --robot-id unitree_go2
    /workspace/isaaclab/isaaclab.sh -p tools/03_controller_rl.py --robot-id unitree_g1
    /workspace/isaaclab/isaaclab.sh -p tools/03_controller_rl.py --robot-id unitree_g1 --device cuda:1
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# 워크스페이스 루트를 sys.path에 추가해 scripts/, configs/ 를 패키지로 임포트할 수 있게 한다
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

parser = argparse.ArgumentParser(description="legged 로봇 RL 보행 정책 학습")
parser.add_argument("--robot-id", type=str, required=True, help="data/sim/usd/real_robot/legged/ 안의 robot_id")
parser.add_argument("--num-envs", type=int, default=None, help="병렬 env 수 (생략 시 로봇 yaml의 agent.num_envs)")
parser.add_argument("--max-iterations", type=int, default=None, help="학습 반복 횟수 (생략 시 로봇 yaml의 agent.max_iterations)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False  # 학습 중에는 렌더링을 절대 켜지 않는다(모듈 docstring 참고)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# AppLauncher 기동 이후에만 isaaclab 의존 모듈(및 pxr를 쓰는 usd_export_config)을 임포트할 수 있다
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from scripts.sim.controller.legged.rl.agent_cfg import build_agent_cfg  # noqa: E402
from scripts.sim.controller.legged.rl.loco_rl_env import build_loco_rl_env_cfg  # noqa: E402
from scripts.sim.controller.legged.rl.robot_profile import RobotProfile  # noqa: E402
from scripts.sim.utils.usd_export_config import export_legged_robot_config  # noqa: E402

_USD_ROOT = _REPO_ROOT / "data" / "sim" / "usd" / "real_robot" / "legged"
_ROBOT_CONFIG_ROOT = _REPO_ROOT / "configs" / "robots" / "legged"
_POLICY_ROOT = _REPO_ROOT / "data" / "sim" / "policies" / "legged"


def _resolve_robot_type(robot_id: str) -> str:
    """robot_id의 usd가 실제로 있는 폴더(multi-legged/humanoid)를 찾아 그대로 로봇 종류로 쓴다."""
    for robot_type in ("multi-legged", "humanoid"):
        if (_USD_ROOT / robot_type / f"{robot_id}.usd").exists():
            return robot_type
    raise FileNotFoundError(
        f"{robot_id}.usd를 data/sim/usd/real_robot/legged/{{multi-legged,humanoid}}/ 어디서도 찾지 못했습니다."
    )


def _ensure_robot_config(robot_type: str, robot_id: str) -> None:
    """config yaml이 없으면 usd 구조 분석만으로 자동 생성한다 - 스폰·물리 시뮬레이션이 필요 없다."""
    config_path = _ROBOT_CONFIG_ROOT / robot_type / f"{robot_id}.yaml"
    if config_path.exists():
        return

    print(f"[controller_rl] {robot_id} config 없음 - usd에서 자동 추출")
    usd_path = _USD_ROOT / robot_type / f"{robot_id}.usd"
    export_legged_robot_config(usd_path, robot_type, config_path)


def main() -> None:
    """robot_id의 종류를 판별하고 config를 준비해 env·agent cfg를 조립한 뒤, 헤드리스로 학습을 돌린다."""
    robot_type = _resolve_robot_type(args_cli.robot_id)
    _ensure_robot_config(robot_type, args_cli.robot_id)

    # 로봇 yaml 하나(RobotProfile)가 env cfg와 agent cfg 양쪽에 필요한 축 선언을 전부 담고 있다
    profile = RobotProfile.load(robot_type, args_cli.robot_id)
    num_envs = args_cli.num_envs or profile.agent.get("num_envs", 4096)
    if args_cli.max_iterations is not None:
        profile.agent["max_iterations"] = args_cli.max_iterations

    # 로봇별 env cfg 조립 - 지형·로봇 USD·보상 세트가 여기서 전부 하나로 묶인다
    print(f"[controller_rl] {robot_type}/{args_cli.robot_id} env 조립 중 (num_envs={num_envs})...")
    env_cfg = build_loco_rl_env_cfg(robot_type, args_cli.robot_id, num_envs=num_envs)
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    experiment_name = f"{robot_type}_{args_cli.robot_id}"
    agent_cfg = build_agent_cfg(profile, experiment_name, args_cli.device)

    # PPO 러너 조립 - 로그는 정책 산출물과 분리해 별도 학습 로그 디렉터리에 남긴다
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    log_dir = str(_POLICY_ROOT / "_train_logs" / experiment_name)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    max_iterations = profile.agent["max_iterations"]
    print(f"[controller_rl] 학습 시작 - max_iterations={max_iterations} (헤드리스, 렌더링 없음)")
    print(f"[controller_rl] 진행 상황: tensorboard --logdir {log_dir}")
    try:
        runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
    except RuntimeError as exc:
        # PPO가 발산하면(보상 폭주 -> 가치함수 발산 -> 정책 표준편차 NaN) rsl_rl이 이 시점에서
        # RuntimeError를 던진다 - 정책을 내보내지 않고 즉시 멈춰 GPU 시간을 더 낭비하지 않는다.
        # save_interval마다 저장된 체크포인트(model_*.pt)는 log_dir에 그대로 남아있다.
        print(f"\n[controller_rl] 학습 발산으로 중단 (iteration {runner.current_learning_iteration} 근처): {exc}")
        print(f"[controller_rl] 직전 체크포인트: {log_dir}/model_*.pt - config·하이퍼파라미터를 재점검하세요.")
        env.close()
        raise

    # 학습된 정책을 jit로 내보내 04_controller_test.py의 LocoRunner가 바로 로드할 수 있게 한다
    print("[controller_rl] 학습 종료 - 정책 jit 변환 및 저장 중...")
    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    output_dir = _POLICY_ROOT / robot_type / args_cli.robot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(output_dir), filename="policy.pt")
    dump_yaml(str(output_dir / "train_config.yaml"), agent_cfg)
    print(f"[controller_rl] 저장 완료: {output_dir}")
    print("[controller_rl] 학습된 정책 구동 확인은 tools/04_controller_test.py를 별도로 실행하세요.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""공개 USD 40종을 Isaac Sim에 하나씩 스폰해서 문제없이 열리는지 확인하고, 전체 형상이 보이는
스크린샷을 남기는 진입점.

configs/robots/legged/multi-legged/{robot_id}.yaml(RobotProfile)이 있는 로봇은 USD가 열리는지만
보는 걸로는 부족하다 - 그 로봇의 실제 학습용 액추에이터 모델·기본 자세로 관절 제어 자체가 되는지도
확인해야 한다(보행 동작 테스트가 아니라 usd·액추에이터 설정의 정합성 확인). 그래서 프로필이 있는
로봇은 자동으로 관절 제어 검증(_verify_joint_control)까지 거치고, 프로필이 없는 로봇(wheeled 전체,
프로필 없는 legged)은 스폰만 확인한다 - 별도 모드 플래그 없이 로봇마다 자동으로 갈린다.

관절 제어 검증은 기본 자세를 유지시켜 정착하는지만 본다 - RL 정책 없이 순수 PD/DC모터로 고정
목표만 유지하는 사족보행은 능동 균형 보정이 없어 관절 하나만 살짝 흔들려도 자세가 무너질 수 있다.
이건 액추에이터 설정 오류가 아니라 "정책 없는 정적 자세는 원래 안정성 마진이 거의 없다"는 물리적
사실이므로, 오프셋을 줘서 추종 여부까지 보는 것은 이 검증의 목적(usd·액추에이터 설정 자체의 정합성
확인)에 맞지 않는다.

Kit(Isaac Sim) 부팅 자체가 무거워서(로봇 1종당 새로 부팅하면 5~10초씩 낭비), 부팅은 한 번만 하고
같은 씬에서 카메라는 고정해 둔 채 로봇만 하나씩 스폰 -> 검증 -> 촬영 -> 제거하며 순회한다. 로봇
하나가 실패해도 다음 로봇 처리에 영향이 없도록 로봇 단위로 감싼다.

사용법:
    # 40종 전체
    /workspace/isaaclab/isaaclab.sh -p tools/02_robot_test.py

    # 로봇 1종만 (실패한 로봇 재확인용)
    /workspace/isaaclab/isaaclab.sh -p tools/02_robot_test.py --robot-id unitree_aliengo
"""

import argparse
import sys
from pathlib import Path

# 실행 중 print()가 즉시 보이도록 줄 단위 버퍼링으로 바꾼다 - simulation_app.close(skip_cleanup=True)로
# 곧장 종료하면 버퍼링된 stdout이 플러시되지 않고 유실될 수 있다(tools/04_controller_test.py와 동일 조치).
sys.stdout.reconfigure(line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_USD_ROOT = _REPO_ROOT / "data" / "sim" / "usd" / "real_robot"
_CHECK_DIR = _REPO_ROOT / "check" / "02_robot_test"
_ROBOT_PRIM_PATH = "/World/Robot"

# 40종이 실제로 놓인 하위 폴더만 지정 - 같은 카테고리 밑의 URDF 자체 변환 결과물(플랫 경로)은
# 이 패턴에 안 걸려 섞이지 않는다.
_USD_GLOBS = (
    "wheeled/diff/*.usd",
    "wheeled/omni/*.usd",
    "wheeled/ackermann/*.usd",
    "legged/multi-legged/*.usd",
    "legged/humanoid/*.usd",
)

_HOLD_STEPS = 120  # 기본 자세를 유지시켜 정착 여부를 보는 구간 - 판정 기준이 아니라 입력값. sim dt=1/120이므로 약 1초

# Isaac Lab 공식 mdp.terminations.illegal_contact의 기본 threshold와 동일 - 우리 프로젝트 학습
# 종료조건(scripts/sim/controller/legged/rl/termination.py의 base_contact)도 이 값을 그대로 쓴다.
_BASE_CONTACT_FORCE_THRESHOLD_N = 1.0


class RobotDiscovery:
    """카테고리 하위 폴더를 뒤져 (robot_id, usd_path) 목록을 만드는 단일 책임 클래스."""

    def discover(self) -> list[tuple[str, Path]]:
        """40종 usd 전체를 (robot_id, 경로) 쌍으로 나열한다."""
        found = []
        for pattern in _USD_GLOBS:
            for usd_path in sorted(_USD_ROOT.glob(pattern)):
                found.append((usd_path.stem, usd_path))
        return found

    def find(self, robot_id: str) -> Path:
        """robot_id 하나에 해당하는 usd 경로를 찾는다. 없으면 예외."""
        for rid, path in self.discover():
            if rid == robot_id:
                return path
        raise FileNotFoundError(f"robot_id에 해당하는 usd를 찾을 수 없습니다: {robot_id}")


def _capture(sim, camera, robot_id: str) -> None:
    """로봇 중심이 이미지 정중앙에 오는 전체 뷰 스크린샷을 저장한다."""
    from scripts.sim.utils.capture import capture_scene_to_file, compute_diagonal_view_pose, compute_prim_world_bounds

    bbox_min, bbox_max = compute_prim_world_bounds(_ROBOT_PRIM_PATH)
    eye, target = compute_diagonal_view_pose(bbox_min, bbox_max)

    output_path = _CHECK_DIR / f"{robot_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture_scene_to_file(sim, camera, eye, target, str(output_path))


def _check_base_contact(contact_sensor) -> None:
    """Isaac Lab 공식 mdp.terminations.illegal_contact와 동일한 판정(접촉힘 노름이 threshold 초과) -
    base가 지면(또는 몸 다른 부위)에 닿을 만큼 쓰러졌으면 예외를 던진다."""
    force = contact_sensor.data.net_forces_w[0].norm(dim=-1).max().item()
    if force > _BASE_CONTACT_FORCE_THRESHOLD_N:
        raise RuntimeError(f"base 접촉력 {force:.2f}N > {_BASE_CONTACT_FORCE_THRESHOLD_N}N - 쓰러짐")


def _check_joint_limits(robot, joint_ids: list[int], joint_names: list[str]) -> None:
    """Isaac Lab 공식 mdp.terminations.joint_pos_out_of_limit과 동일한 판정 - 제어 대상 관절이
    soft_joint_pos_limits(로봇 실제 리밋의 90%, build_loco_rl_env_cfg의 soft_joint_pos_limit_factor와
    동일 계수)를 벗어나면 예외를 던진다."""
    positions = robot.data.joint_pos[0, joint_ids]
    limits = robot.data.soft_joint_pos_limits[0, joint_ids]
    out_of_limit = (positions < limits[:, 0]) | (positions > limits[:, 1])
    if out_of_limit.any():
        bad_joints = [joint_names[i] for i in out_of_limit.nonzero().flatten().tolist()]
        raise RuntimeError(f"관절 리밋 이탈: {bad_joints}")


def _verify_joint_control(sim, profile, usd_path: Path, robot_id: str) -> None:
    """로봇 프로필의 실제 액추에이터·기본 자세로 스폰해, 기본 자세를 유지시키는 관절 제어가 정상
    동작하는지 확인한다(보행 동작이 아니라 usd·액추에이터 설정 자체의 정합성 확인).

    쓰러짐·관절 리밋 이탈은 Isaac Lab 공식 판정 로직을 그대로 재현해 예외로 던진다.
    """
    from isaaclab.sensors import ContactSensor, ContactSensorCfg

    from scripts.sim.controller.legged.rl import actuator
    from scripts.sim.env.robot_spawn import (
        LOCO_ARTICULATION_PROPS,
        LOCO_RIGID_BODY_PROPS,
        LOCO_SOFT_JOINT_POS_LIMIT_FACTOR,
        resolve_spawn_height,
        spawn_robot_from_usd,
    )

    height = resolve_spawn_height(profile.spawn_height, usd_path)
    robot = spawn_robot_from_usd(
        str(usd_path),
        _ROBOT_PRIM_PATH,
        position=(0.0, 0.0, height),
        joint_pos=profile.build_joint_pos_cfg(),
        actuators=actuator.build(profile.actuator, profile.controlled_joint_names),
        activate_contact_sensors=True,
        rigid_props=LOCO_RIGID_BODY_PROPS,
        articulation_props=LOCO_ARTICULATION_PROPS,
        soft_joint_pos_limit_factor=LOCO_SOFT_JOINT_POS_LIMIT_FACTOR,
    )
    contact_sensor = ContactSensor(ContactSensorCfg(prim_path=f"{_ROBOT_PRIM_PATH}/{profile.base_body_name}"))
    sim.reset()
    # init_state.joint_pos는 data.default_joint_pos(기준값)만 채울 뿐 실제 물리 상태에는 반영되지
    # 않는다 - 학습 때는 EventManager의 reset_robot_joints(mode="reset")가 이 역할을 하지만, 여기서는
    # ManagerBasedRLEnv 없이 Articulation을 직접 쓰므로 같은 일을 하는 공식 API를 직접 호출해야 한다.
    # 이 호출이 없으면 스폰 직후 관절이 기본 자세가 아니라 USD 원본 자세 근처에서 시작해, 그 뒤
    # 목표 자세로 급격히 꺾이며 관절이 리밋을 순간적으로 넘는다.
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)

    joint_ids, joint_names = robot.find_joints(profile.controlled_joint_names, preserve_order=True)
    print(f"[{robot_id}] 제어 대상 관절 {len(joint_ids)}개: {joint_names}")

    default_targets = robot.data.default_joint_pos.clone()
    print(f"[{robot_id}] 기본 자세 유지 시작 (스폰 높이 {height:.3f}m)")
    for _ in range(_HOLD_STEPS):
        robot.set_joint_position_target(default_targets)
        robot.write_data_to_sim()
        sim.step()
        dt = sim.get_physics_dt()
        robot.update(dt)
        contact_sensor.update(dt, force_recompute=True)
        _check_base_contact(contact_sensor)
    _check_joint_limits(robot, joint_ids, joint_names)
    print(f"[{robot_id}] 기본 자세 정착 완료 - base 높이: {robot.data.root_pos_w[0, 2].item():.3f}m")


def _run_all(sim, camera, robots: list[tuple[str, Path]]) -> None:
    """로봇을 순서대로 스폰 -> (프로필이 있으면 관절 제어 검증까지) -> 촬영 -> 제거하며, 하나가
    실패해도 다음으로 계속 진행한다."""
    from scripts.sim.controller.legged.rl.robot_profile import RobotProfile
    from scripts.sim.env.robot_spawn import clear_prim, spawn_robot_safely

    results: dict[str, str] = {}
    for robot_id, usd_path in robots:
        print(f"[{robot_id}] 스폰 시작")
        try:
            try:
                profile = RobotProfile.load(robot_id)
            except FileNotFoundError:
                profile = None

            if profile is not None:
                _verify_joint_control(sim, profile, usd_path, robot_id)
            else:
                robot = spawn_robot_safely(sim, usd_path, _ROBOT_PRIM_PATH)
                robot.write_data_to_sim()
                for _ in range(5):
                    sim.step()
                    robot.update(sim.get_physics_dt())

            _capture(sim, camera, robot_id)
            results[robot_id] = "성공"
            print(f"[{robot_id}] 성공")
        except Exception as exc:  # noqa: BLE001 - 한 로봇의 실패가 나머지 로봇 처리에 번지면 안 됨
            results[robot_id] = f"실패: {exc}"
            print(f"[{robot_id}] 실패: {exc}")
        finally:
            clear_prim(_ROBOT_PRIM_PATH)

    failed = {rid: status for rid, status in results.items() if status != "성공"}
    print(f"\n=== 결과: 성공 {len(results) - len(failed)} / 실패 {len(failed)} ===")
    for rid, status in failed.items():
        print(f"  - {rid}: {status}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="공개 USD 로봇 스폰·관절 제어 검증 및 전체 뷰 스크린샷 저장")
    parser.add_argument("--robot-id", type=str, default=None, help="로봇 1종만 검증 (생략 시 40종 전체)")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, enable_cameras=True)
    return parser.parse_args()


def main() -> None:
    """Kit을 한 번만 띄우고, 지정한 로봇(들)을 순서대로 스폰 -> 검증 -> 촬영 -> 제거한다."""
    args_cli = _parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils

    from scripts.sim.utils.capture import spawn_capture_camera

    # SimulationCfg.device 기본값이 "cuda:0"으로 고정돼 있어(agent_cfg.py의 RslRlBaseRunnerCfg.device와
    # 같은 패턴), args_cli.device를 명시적으로 안 넘기면 --device로 다른 GPU를 지정해도 물리 텐서
    # 일부가 cuda:0에 만들어져 "Incompatible device" 에러가 난다.
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device))
    # 기본 바닥 색이 검정(0,0,0)이라 검은 바퀴·다리와 구분이 안 돼 중간 회색으로 바꾼다
    sim_utils.spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg(color=(0.5, 0.5, 0.5)))
    # 기본 씬에는 조명이 없어 렌더가 어둡게 나온다 - 01_sim_test.py와 같은 돔 라이트를 추가한다
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    camera = spawn_capture_camera("/World/Camera")  # 로봇마다 새로 만들지 않고 이 하나를 계속 재사용

    discovery = RobotDiscovery()
    robots = [(args_cli.robot_id, discovery.find(args_cli.robot_id))] if args_cli.robot_id else discovery.discover()
    _run_all(sim, camera, robots)

    # 카메라(render product)가 붙어있는 채로 close()의 기본 정리 단계(rep.orchestrator.stop() 등)를
    # 타면 멈추는 경우가 있다(01_sim_test.py와 동일 이슈). 스크린샷은 이미 동기 저장을 마쳤으므로
    # 그 정리 단계만 건너뛰는 공식 파라미터로 종료한다.
    simulation_app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()

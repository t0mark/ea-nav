"""wheeled/legged 컨트롤러 구동 테스트 진입점 - --robot-type으로 어느 쪽을 테스트할지 고른다.

wheeled는 config yaml이 없으면 scripts/sim/utils/usd_export_config.py로 자동 생성한 뒤 바로 테스트를
이어간다(로봇 자체의 물리 형상에서 뽑아낼 수 있는 값이라 사람이 미리 채워둘 필요가 없음). legged는
base/foot 링크를 형상만으로 자동 판별할 근거가 약해 자동 추출을 지원하지 않으므로, config가 없는
로봇은 애초에 테스트 대상에서 제외한다.

테스트 내용은 "평지에서 ㄱ자 웨이포인트 경로를 끝까지 따라가는가" 하나로 통일한다 - 직진만으로는
조향이 전혀 검증되지 않으므로, scripts/sim/nav의 pure pursuit 추종기로 코너가 있는 경로를 실제로
쫓아가게 한다. 파일럿(--mode pilot)은 그 결과를 영상(목표 지점 마커 포함)과 궤적 이미지로 남기고
서브카테고리별 대표 로봇만 돌리며, 본 실행(--mode full)은 그런 산출물 없이 판정만 내리고
(--robot-id 없으면) 카테고리 전체를 돌린다.

사용법:
    # wheeled 파일럿 - 서브카테고리(diff/ackermann/omni)별 대표 로봇만, 영상+궤적 이미지 저장
    /workspace/isaaclab/isaaclab.sh -p tools/04_controller_test.py --robot-type wheeled --mode pilot

    # wheeled 본 실행 - 전체 로봇, 판정만
    /workspace/isaaclab/isaaclab.sh -p tools/04_controller_test.py --robot-type wheeled --mode full

    # 로봇 1종 지정
    /workspace/isaaclab/isaaclab.sh -p tools/04_controller_test.py --robot-type wheeled --mode pilot --robot-id clearpath_jackal
"""

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# 워크스페이스 루트를 sys.path에 추가해 scripts/, configs/ 를 패키지로 임포트할 수 있게 한다
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

parser = argparse.ArgumentParser(description="wheeled/legged 컨트롤러 구동 테스트")
parser.add_argument("--robot-type", type=str, required=True, choices=["wheeled", "legged"], help="테스트할 제어기 계열")
parser.add_argument("--robot-id", type=str, default=None, help="로봇 1종만 테스트 (생략 시 --mode 기준으로 여러 종)")
parser.add_argument(
    "--mode", type=str, required=True, choices=["pilot", "full"], help="pilot=영상·궤적 이미지+대표 소수, full=판정만+전체"
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = args_cli.mode == "pilot"  # 영상이 필요한 파일럿일 때만 카메라 렌더링을 켠다

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# AppLauncher 기동 이후에만 isaaclab 의존 모듈(및 pxr를 쓰는 usd_export_config)을 임포트할 수 있다
import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

from scripts.sim.env.robot_spawn import clear_prim, spawn_robot_safely  # noqa: E402
from scripts.sim.nav.path import Path2D, generate_l_shaped_path  # noqa: E402
from scripts.sim.nav.pure_pursuit import PurePursuitTracker  # noqa: E402
from scripts.sim.utils.capture import (  # noqa: E402
    capture_camera_frame,
    compute_prim_world_bounds,
    record_frames_to_video,
    set_camera_view,
    spawn_capture_camera,
)
from scripts.sim.utils.goal_visualize import spawn_goal_marker, update_goal_marker  # noqa: E402
from scripts.sim.utils.trajectory_plot import save_trajectory_plot  # noqa: E402
from scripts.sim.utils.usd_export_config import export_wheeled_robot_config  # noqa: E402

_USD_ROOT = _REPO_ROOT / "data" / "sim" / "usd" / "real_robot"
_ROBOT_CONFIG_ROOT = _REPO_ROOT / "configs" / "robots"
_CHECK_DIR = _REPO_ROOT / "check" / "04_controller_test"
_ROBOT_PRIM_PATH = "/World/Robot"

_LEG_LENGTH_M = 1.5  # ㄱ자 경로 한 변의 길이
_NUM_STEPS = {"pilot": 3000, "full": 3600}


def _discover_wheeled_robots() -> list[tuple[str, str, Path]]:
    """data/sim/usd/real_robot/wheeled/ 밑의 usd 전체를 (서브카테고리, robot_id, usd 경로)로 나열한다."""
    found = []
    for sub_category in ("diff", "ackermann", "omni"):
        for usd_path in sorted((_USD_ROOT / "wheeled" / sub_category).glob("*.usd")):
            found.append((sub_category, usd_path.stem, usd_path))
    return found


def _discover_legged_robots() -> list[tuple[str, str]]:
    """configs/robots/legged/ 밑에 이미 config가 있는 로봇만 (서브카테고리, robot_id)로 나열한다.

    legged는 자동 추출을 지원하지 않으므로, config가 없는 로봇은 애초에 테스트 대상에 넣지 않는다.
    """
    found = []
    for sub_category in ("multi-legged", "humanoid"):
        for yaml_path in sorted((_ROBOT_CONFIG_ROOT / "legged" / sub_category).glob("*.yaml")):
            if yaml_path.stem != "_template":
                found.append((sub_category, yaml_path.stem))
    return found


def _select_robots(all_robots: list, robot_id: str | None, mode: str) -> list:
    """--robot-id가 있으면 그것만, 없으면 pilot=서브카테고리별 대표 1종, full=전체를 고른다."""
    if robot_id is not None:
        matches = [robot for robot in all_robots if robot[1] == robot_id]
        if not matches:
            raise ValueError(f"robot_id를 찾을 수 없습니다: {robot_id}")
        return matches
    if mode == "full":
        return all_robots

    # 파일럿은 형태별 커버리지가 목적이므로, 서브카테고리마다 처음 발견된 로봇 1종만 대표로 고른다
    seen_sub_categories = set()
    selected = []
    for robot in all_robots:
        if robot[0] not in seen_sub_categories:
            seen_sub_categories.add(robot[0])
            selected.append(robot)
    return selected


def _save_check_outputs(output_dir: Path, robot_id: str, frames: list, start, path: Path2D, position_history: list) -> None:
    """파일럿 모드에서 모은 프레임·궤적을 영상·이미지로 저장한다(프레임이 없으면 아무 것도 안 함)."""
    if not frames:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    record_frames_to_video(frames, str(output_dir / f"{robot_id}.mp4"))
    save_trajectory_plot([start, *path.waypoints], position_history, output_dir / f"{robot_id}_trajectory.png")


class _ChaseCamera:
    """로봇을 뒤쫓는 카메라 시점 - 로봇 크기에 맞춰 거리·높이를 한 번 계산해두고 매 스텝 pose만 갱신한다.

    고정된 거리를 쓰면 소형 로봇에는 맞아도 훨씬 큰 산업용 로봇에서는 카메라가 몸체 안에 들어가
    화면이 잘린다 - 로봇 대각선 길이에 비례해서 거리·높이를 정해야 크기와 무관하게 전신이 프레임
    안에 들어온다.
    """

    def __init__(self, prim_path: str) -> None:
        """prim_path 로봇의 월드 바운딩 박스로 카메라 거리·높이·시선 목표 높이를 계산해둔다."""
        bbox_min, bbox_max = compute_prim_world_bounds(prim_path)
        extent = tuple(bbox_max[i] - bbox_min[i] for i in range(3))
        diagonal = (extent[0] ** 2 + extent[1] ** 2 + extent[2] ** 2) ** 0.5
        self._back_distance = max(diagonal * 1.5, 1.5)
        self._height = max(extent[2], 0.3) + self._back_distance * 0.5
        self._target_height = bbox_min[2] + extent[2] * 0.5

    def pose(self, position: tuple[float, float], yaw: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """로봇의 현재 (x, y)·헤딩으로, 로봇 뒤에서 진행 방향을 바라보는 (eye, target) 월드 좌표를 만든다."""
        eye = (
            position[0] - self._back_distance * math.cos(yaw),
            position[1] - self._back_distance * math.sin(yaw),
            self._height,
        )
        target = (position[0], position[1], self._target_height)
        return eye, target


class _WheeledRobotTest:
    """wheeled 로봇 1종을 스폰해 ㄱ자 경로를 pure pursuit으로 추종시키고 도달 여부를 판정한다.

    camera/goal_marker는 씬 하나당 한 번만 만들어져 여러 로봇에 걸쳐 재사용된다 - 로봇마다 새로
    스폰하면 이전 것이 같은 prim 경로에 남아 내부 인덱스가 꼬인다.
    """

    def __init__(self, sim: sim_utils.SimulationContext, mode: str, camera, goal_marker) -> None:
        """공유 시뮬레이션 컨텍스트와, 파일럿 모드에서만 쓰는 카메라·목표 마커를 저장한다."""
        self._sim = sim
        self._mode = mode
        self._camera = camera
        self._goal_marker = goal_marker

    def run(self, sub_category: str, robot_id: str, usd_path: Path) -> bool:
        """config를 준비해 컨트롤러를 만들고, 경로를 끝까지 쫓아가는지 시뮬레이션한다."""
        config_path = self._ensure_config(sub_category, robot_id, usd_path)
        controller, resolved_usd_path, position_joint_names, velocity_joint_names, tracker = self._build_controller(
            sub_category, config_path
        )

        # spawn_robot_safely를 쓴다 - 관절 기본 자세가 0으로는 리밋을 벗어나는 로봇(예: 정지 위치가
        # 0이 아닌 서스펜션 쇼크 관절이 있는 nvidia_f1tenth)도 자동으로 재시도해 스폰한다
        robot = spawn_robot_safely(self._sim, Path(resolved_usd_path), _ROBOT_PRIM_PATH)
        # preserve_order=True 필수 - 기본값(False)이면 로봇 내부 관절 순서로 재정렬돼서, 컨트롤러가
        # 만든 command 텐서(좌/우 순서)와 관절 id 순서가 어긋난다
        position_ids = robot.find_joints(position_joint_names, preserve_order=True)[0] if position_joint_names else []
        velocity_ids = robot.find_joints(velocity_joint_names, preserve_order=True)[0] if velocity_joint_names else []
        chase_camera = _ChaseCamera(_ROBOT_PRIM_PATH) if self._camera is not None else None

        start = (0.0, 0.0)  # spawn_robot_safely가 항상 원점에 스폰하므로 경로 시작점도 원점으로 둔다
        path: Path2D = generate_l_shaped_path(start, _LEG_LENGTH_M)

        position_history: list[tuple[float, float]] = []
        frames = []
        for _ in range(_NUM_STEPS[self._mode]):
            position = (robot.data.root_pos_w[0, 0].item(), robot.data.root_pos_w[0, 1].item())
            _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
            linear_velocity, angular_velocity = tracker.compute_command(position, yaw.item(), path)
            command = self._unicycle_command(sub_category, linear_velocity, angular_velocity)

            joint_targets = controller.compute_joint_targets(command)
            if "position" in joint_targets:
                targets = joint_targets["position"].to(robot.device).unsqueeze(0)
                robot.set_joint_position_target(targets, joint_ids=position_ids)
            if "velocity" in joint_targets:
                targets = joint_targets["velocity"].to(robot.device).unsqueeze(0)
                robot.set_joint_velocity_target(targets, joint_ids=velocity_ids)
            robot.write_data_to_sim()
            self._sim.step()
            robot.update(self._sim.get_physics_dt())

            position_history.append(position)
            self._record_step(chase_camera, position, yaw.item(), path, frames)
            if path.is_finished:
                break

        success = path.is_finished
        print(f"[controller_test] {robot_id} ㄱ자 경로 도달 -> {'성공' if success else '실패'}")
        _save_check_outputs(_CHECK_DIR / "wheeled" / sub_category, robot_id, frames, start, path, position_history)
        clear_prim(_ROBOT_PRIM_PATH)
        return success

    def _record_step(self, chase_camera: _ChaseCamera | None, position, yaw: float, path: Path2D, frames: list) -> None:
        """목표 마커·카메라 시점을 갱신하고 프레임 하나를 모은다(파일럿 모드가 아니면 아무 것도 안 함)."""
        if self._goal_marker is not None:
            update_goal_marker(self._goal_marker, path.current_goal)
        if chase_camera is not None:
            eye, target = chase_camera.pose(position, yaw)
            set_camera_view(self._sim, self._camera, eye=eye, target=target)
            self._camera.update(dt=self._sim.get_physics_dt())
            frame = capture_camera_frame(self._camera)
            if frame is not None:
                frames.append(frame)

    def _ensure_config(self, sub_category: str, robot_id: str, usd_path: Path) -> Path:
        """configs/robots/wheeled/{sub_category}/{robot_id}.yaml이 없으면 usd에서 자동 생성한다."""
        config_path = _ROBOT_CONFIG_ROOT / "wheeled" / sub_category / f"{robot_id}.yaml"
        if not config_path.exists():
            print(f"[controller_test] {robot_id} config 없음 - usd에서 자동 추출")
            export_wheeled_robot_config(self._sim, usd_path, sub_category, config_path)
        return config_path

    @staticmethod
    def _build_controller(sub_category: str, config_path: Path):
        """config yaml을 읽어 기구학 컨트롤러·관절 이름·pure pursuit 추종기를 만든다."""
        from scripts.sim.controller.wheeled.ackermann import AckermannKinematics
        from scripts.sim.controller.wheeled.diff_drive import DifferentialDriveKinematics
        from scripts.sim.controller.wheeled.omni_drive import OmniDriveKinematics

        with open(config_path) as f:
            raw = yaml.safe_load(f)
        usd_path = _USD_ROOT / "wheeled" / sub_category / raw["usd_path"]

        if sub_category == "diff":
            controller = DifferentialDriveKinematics(
                wheel_radius=raw["wheel_radius"],
                wheel_base=raw["wheel_base"],
                num_left_wheels=len(raw["left_wheel_joint_names"]),
                num_right_wheels=len(raw["right_wheel_joint_names"]),
                direction_sign=raw.get("direction_sign", 1.0),
                rotation_sign=raw.get("rotation_sign", 1.0),
            )
            position_joint_names: list[str] = []
            velocity_joint_names = raw["left_wheel_joint_names"] + raw["right_wheel_joint_names"]
        elif sub_category == "ackermann":
            controller = AckermannKinematics(
                wheelbase=raw["wheelbase"],
                track_width=raw["track_width"],
                wheel_radius=raw["wheel_radius"],
                num_drive_wheels=len(raw["drive_wheel_joint_names"]),
                direction_sign=raw.get("direction_sign", 1.0),
                steering_sign=raw.get("steering_sign", 1.0),
            )
            position_joint_names = [raw["left_steering_joint_name"], raw["right_steering_joint_name"]]
            velocity_joint_names = raw["drive_wheel_joint_names"]
        elif sub_category == "omni":
            controller = OmniDriveKinematics(
                wheel_radius=raw["wheel_radius"],
                half_wheelbase=raw["half_wheelbase"],
                half_track_width=raw["half_track_width"],
                direction_sign=raw.get("direction_sign", 1.0),
                rotation_sign=raw.get("rotation_sign", 1.0),
            )
            position_joint_names = []
            velocity_joint_names = [
                raw["wheel_joint_names"]["front_left"],
                raw["wheel_joint_names"]["front_right"],
                raw["wheel_joint_names"]["rear_left"],
                raw["wheel_joint_names"]["rear_right"],
            ]
        else:
            raise ValueError(f"알 수 없는 wheeled 카테고리: {sub_category}")

        # ackermann처럼 아직 실측 보정을 안 하는 카테고리는 기존에 쓰던 고정값으로 대체한다
        tracker = PurePursuitTracker(
            linear_velocity=raw.get("nav_linear_velocity", 0.2),
            max_angular_velocity=raw.get("nav_max_angular_velocity", 4.5),
        )
        return controller, str(usd_path), position_joint_names, velocity_joint_names, tracker

    @staticmethod
    def _unicycle_command(sub_category: str, linear_velocity: float, angular_velocity: float) -> torch.Tensor:
        """(v, w) 유니사이클 명령을 서브카테고리별 command 텐서로 만든다.

        diff/ackermann은 (v, w)를 그대로 받고(각 컨트롤러 내부에서 자기 방식대로 처리), omni만
        [vx, vy, wz] 3원소가 필요하므로 옆이동(vy) 없이 앞으로만(0.0) 채워 넣는다.
        """
        if sub_category == "omni":
            return torch.tensor([linear_velocity, 0.0, angular_velocity])
        return torch.tensor([linear_velocity, angular_velocity])


class _LeggedRobotTest:
    """legged 로봇 1종을 학습된 정책으로 ㄱ자 경로를 추종시키고 도달 여부를 판정한다."""

    def __init__(self, mode: str) -> None:
        """파일럿/본 실행 여부를 저장한다 - legged는 로봇마다 독립된 env를 새로 만들어 재사용할 공유
        카메라·씬이 없다."""
        self._mode = mode

    def run(self, sub_category: str, robot_id: str) -> bool:
        """정책을 로드해 경로를 끝까지 쫓아가는지 시뮬레이션한다."""
        from scripts.sim.controller.legged.loco_runner import LocoRunner

        runner = LocoRunner(category=sub_category, robot_id=robot_id, num_envs=1)
        runner.reset()
        robot = runner.env.scene["robot"]

        camera = spawn_capture_camera("/World/TestCamera") if self._mode == "pilot" else None
        goal_marker = spawn_goal_marker() if self._mode == "pilot" else None
        if camera is not None:
            # 센서는 "다음 play 이벤트"에서 초기화되는데, LocoRunner 생성 시점(ManagerBasedRLEnv
            # 생성자 안의 sim.reset())에 그 이벤트를 이미 다 써버린 뒤라, 여기서 만든 카메라는
            # 그대로 두면 초기화가 안 된다(_ALL_INDICES 같은 내부 속성이 없다는 에러로 나타남) -
            # reset()을 한 번 더 호출해 새 play 이벤트를 만들어야 한다(usd_export_config.py의
            # ContactSensor에서 이미 겪은 것과 같은 문제).
            runner.env.sim.reset()
        # legged 로봇은 학습 안전을 위해 지면 위로 살짝 띄워 스폰된다(scripts/sim/env/robot_spawn.py의
        # ground_clearance) - 그 상태 그대로 카메라를 맞추면, 이후 중력으로 가라앉아 정착한 실제
        # 높이와 어긋나 로봇이 화면에서 잘려 보인다. 정책으로 제자리에서 몇 스텝 서 있게 해 정착시킨
        # 뒤에 그 자세를 기준으로 카메라를 맞춘다.
        settle_command = torch.zeros((1, 3), device=runner.env.device)
        for _ in range(60):
            runner.compute_joint_targets(settle_command)

        start = (robot.data.root_pos_w[0, 0].item(), robot.data.root_pos_w[0, 1].item())
        robot_prim_path = f"{runner.env.scene.env_prim_paths[0]}/Robot"
        chase_camera = _ChaseCamera(robot_prim_path) if camera is not None else None

        path: Path2D = generate_l_shaped_path(start, _LEG_LENGTH_M)
        tracker = PurePursuitTracker()

        position_history: list[tuple[float, float]] = []
        frames = []
        for _ in range(_NUM_STEPS[self._mode]):
            position = (robot.data.root_pos_w[0, 0].item(), robot.data.root_pos_w[0, 1].item())
            _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
            linear_velocity, angular_velocity = tracker.compute_command(position, yaw.item(), path)
            command = torch.tensor([[linear_velocity, 0.0, angular_velocity]], device=runner.env.device)
            runner.compute_joint_targets(command)

            position_history.append(position)
            if goal_marker is not None:
                update_goal_marker(goal_marker, path.current_goal)
            if chase_camera is not None:
                eye, target = chase_camera.pose(position, yaw.item())
                set_camera_view(runner.env.sim, camera, eye=eye, target=target)
                camera.update(dt=runner.env.sim.get_physics_dt())
                frame = capture_camera_frame(camera)
                if frame is not None:
                    frames.append(frame)
            if path.is_finished:
                break

        success = path.is_finished
        print(f"[controller_test] {robot_id} ㄱ자 경로 도달 -> {'성공' if success else '실패'}")
        _save_check_outputs(_CHECK_DIR / "legged" / sub_category, robot_id, frames, start, path, position_history)
        runner.env.close()
        return success


def main() -> None:
    """--robot-type/--mode에 따라 로봇을 고르고, 하나씩 테스트해 결과를 요약 출력한다."""
    results: dict[str, bool] = {}

    if args_cli.robot_type == "wheeled":
        robots = _select_robots(_discover_wheeled_robots(), args_cli.robot_id, args_cli.mode)

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60))
        sim_utils.spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # 씬 하나당 카메라·목표 마커를 하나씩만 만들어 로봇 전체 순회 동안 재사용한다
        camera = spawn_capture_camera("/World/TestCamera") if args_cli.mode == "pilot" else None
        goal_marker = spawn_goal_marker() if args_cli.mode == "pilot" else None
        test = _WheeledRobotTest(sim, args_cli.mode, camera, goal_marker)

        for sub_category, robot_id, usd_path in robots:
            try:
                results[robot_id] = test.run(sub_category, robot_id, usd_path)
            except Exception as exc:  # noqa: BLE001 - 로봇 1종 실패가 나머지 테스트에 번지면 안 됨
                results[robot_id] = False
                print(f"[controller_test] {robot_id} 테스트 실패: {exc}")
                clear_prim(_ROBOT_PRIM_PATH)
    else:
        robots = _select_robots(_discover_legged_robots(), args_cli.robot_id, args_cli.mode)
        test = _LeggedRobotTest(args_cli.mode)
        for sub_category, robot_id in robots:
            try:
                results[robot_id] = test.run(sub_category, robot_id)
            except Exception as exc:  # noqa: BLE001
                results[robot_id] = False
                print(f"[controller_test] {robot_id} 테스트 실패: {exc}")

    succeeded = sum(1 for ok in results.values() if ok)
    print(f"\n=== 결과: 성공 {succeeded} / 전체 {len(results)} ===")


if __name__ == "__main__":
    main()
    simulation_app.close(skip_cleanup=True)

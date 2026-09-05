"""공개 USD 40종을 Isaac Sim에 하나씩 스폰해서 문제없이 열리는지 확인하고, 전체 형상이 보이는
스크린샷을 남기는 진입점.

Kit(Isaac Sim) 부팅 자체가 무거워서(로봇 1종당 새로 부팅하면 5~10초씩 낭비), 부팅은 한 번만 하고
같은 씬에서 카메라는 고정해 둔 채 로봇만 하나씩 스폰 -> 촬영 -> 제거하며 순회한다. 스폰 자체의
안전장치(관절 기본 자세, 접지 높이 보정)는 scripts/sim/env/robot_spawn.py의 spawn_robot_safely가
맡고, 여기서는 어떤 로봇을 어떤 순서로 돌릴지와 화면 촬영만 다룬다. 로봇 하나가 실패해도 다음
로봇 처리에 영향이 없도록 로봇 단위로 감싼다.

사용법:
    # 40종 전체
    /workspace/isaaclab/isaaclab.sh -p tools/02_robot_test.py

    # 로봇 1종만 (실패한 로봇 재확인용)
    /workspace/isaaclab/isaaclab.sh -p tools/02_robot_test.py --robot-id unitree_aliengo
"""

import argparse
from pathlib import Path

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


def _run_all(sim, camera, robots: list[tuple[str, Path]]) -> None:
    """로봇을 순서대로 스폰 -> 촬영 -> 제거하며, 하나가 실패해도 다음으로 계속 진행한다."""
    from scripts.sim.env.robot_spawn import clear_prim, spawn_robot_safely

    results: dict[str, str] = {}
    for robot_id, usd_path in robots:
        print(f"[{robot_id}] 스폰 시작")
        try:
            robot = spawn_robot_safely(sim, usd_path, _ROBOT_PRIM_PATH)
            robot.write_data_to_sim()
            for _ in range(5):
                sim.step()
                robot.update(sim.get_physics_dt())
            _capture(sim, camera, robot_id)
            results[robot_id] = "성공"
            print(f"[{robot_id}] 성공")
        except Exception as exc:  # noqa: BLE001 - 한 로봇의 실패가 나머지 39종에 번지면 안 됨
            results[robot_id] = f"실패: {exc}"
            print(f"[{robot_id}] 실패: {exc}")
        finally:
            clear_prim(_ROBOT_PRIM_PATH)

    failed = {rid: status for rid, status in results.items() if status != "성공"}
    print(f"\n=== 결과: 성공 {len(results) - len(failed)} / 실패 {len(failed)} ===")
    for rid, status in failed.items():
        print(f"  - {rid}: {status}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="공개 USD 로봇 스폰 검증 및 전체 뷰 스크린샷 저장")
    parser.add_argument("--robot-id", type=str, default=None, help="로봇 1종만 검증 (생략 시 40종 전체)")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, enable_cameras=True)
    return parser.parse_args()


def main() -> None:
    """Kit을 한 번만 띄우고, 지정한 로봇(들)을 순서대로 스폰 -> 촬영 -> 제거한다."""
    args_cli = _parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils

    from scripts.sim.utils.capture import spawn_capture_camera

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
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

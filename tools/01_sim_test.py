"""TerrainGenerator 기반 sim 환경 세팅을 시각적으로 검증하는 진입점.

Isaac Sim은 AppLauncher가 Omniverse Kit 프로세스를 띄운 이후에만 isaaclab 모듈을 임포트할 수 있어,
isaaclab 의존 임포트와 씬 조립은 전부 AppLauncher 기동 아래에서 이 파일이 직접 한다.
env/, utils/ 에 있는 재사용 가능한 조각(지형 빌더, 카메라 캡처)만 가져다 쓴다.

사용법:
    /workspace/isaaclab/isaaclab.sh -p tools/01_sim_test.py
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# 워크스페이스 루트를 sys.path에 추가해 scripts/, configs/ 를 패키지로 임포트할 수 있게 한다
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

parser = argparse.ArgumentParser(description="TerrainGenerator 지형 생성 시각 검증")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, enable_cameras=True)  # 컨테이너 환경 기본값: 오프스크린 렌더링
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# AppLauncher 기동 이후에만 isaaclab 의존 모듈을 임포트할 수 있다
import isaaclab.sim as sim_utils  # noqa: E402

from scripts.sim.env.terrain import TerrainSceneBuilder, load_terrain_generator_cfg  # noqa: E402
from scripts.sim.utils.capture import capture_scene_to_file, spawn_capture_camera  # noqa: E402

output_dir = _REPO_ROOT / "check" / "01_sim_test"
output_dir.mkdir(parents=True, exist_ok=True)

# 지형·카메라를 임포트할 스테이지·물리 씬을 먼저 준비해야 한다
print("[sim_test] 시뮬레이션 컨텍스트 초기화 중...")
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))

# 지형 형상이 잘 보이도록 돔 라이트 배치
print("[sim_test] 조명 배치 중...")
light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
light_cfg.func("/World/Light", light_cfg)

# yaml 설정으로부터 지형을 생성해 스테이지에 임포트
print("[sim_test] configs/sim_terrain.yaml 로부터 지형 생성 중...")
terrain_gen_cfg = load_terrain_generator_cfg(_REPO_ROOT / "configs" / "sim_terrain.yaml")
terrain_builder = TerrainSceneBuilder(
    terrain_generator_cfg=terrain_gen_cfg,
    num_envs=terrain_gen_cfg.num_rows * terrain_gen_cfg.num_cols,
    env_spacing=terrain_gen_cfg.size[0] + 2.0,
)
terrain_builder.build(debug_vis=True)

# 캡처용 카메라는 씬 설계 단계(= sim.reset() 이전)에 스폰해야 한다
print("[sim_test] 캡처용 카메라 스폰 중...")
camera = spawn_capture_camera("/World/TestCamera")

sim.reset()

# 생성된 지형 전체가 한 화면에 들어오는 대각선 위 시점 계산
# TerrainGenerator는 격자를 (row -> x, col -> y) 축으로 원점(0, 0) 중심에 배치한다
terrain_extent_x = terrain_gen_cfg.size[0] * terrain_gen_cfg.num_rows
terrain_extent_y = terrain_gen_cfg.size[1] * terrain_gen_cfg.num_cols
camera_height = max(terrain_extent_x, terrain_extent_y) * 0.8
eye = (terrain_extent_x * 0.9, -terrain_extent_y * 0.6, camera_height)
target = (0.0, 0.0, 0.0)

# 카메라 렌더 대기 및 스크린샷 저장은 공용 유틸리티에 위임
print("[sim_test] 씬 안정화 및 카메라 렌더 대기 중...")
capture_scene_to_file(sim, camera, eye, target, str(output_dir / "terrain.png"))

print("[sim_test] 완료 - 스크린샷에서 계단/경사 지형이 올바르게 생성됐는지 시각적으로 확인할 것")

# --enable_cameras 로 실행하면 카메라 render product가 붙어있는 채로 close()가 시도하는
# rep.orchestrator.stop()/close_stage() 단계에서 멈출 수 있다. 캡처 결과는 위에서 이미 동기적으로
# 저장을 마쳤으므로, 이 정리 단계 자체를 건너뛰는 공식 파라미터로 즉시 종료한다.
simulation_app.close(skip_cleanup=True)

"""Isaac 앱 기동 공통 부트스트랩 (01_sim·02_controller·03_trav_gt 진입점이 공유).

Isaac 의존 모듈(isaaclab.*, isaacsim.*, pxr, omni.*)은 앱이 뜬 뒤에만 임포트할
수 있으므로, 각 진입점은 아래 순서를 지킨다:

    from tools.utils.sim import launch_app
    args, app = launch_app(parser, enable_cameras="pilot")
    from scripts.sim.utils.environment import SimEnvironment   # 앱 기동 후 임포트
    ...
    app.close()

실행 규약 (sim 컨테이너): /isaac-sim/python.sh /workspace/eatrav/tools/{스크립트}
H200(RTX 코어 없음)에서도 표준 AppLauncher 헤드리스 경로로 기동·렌더가 동작한다
(별도 우회 플래그 없음 — 오프스크린 렌더는 enable_cameras=True가 조건).
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def launch_app(parser: argparse.ArgumentParser, enable_cameras=False):
    """AppLauncher 인자를 붙여 파싱한 뒤 헤드리스 Isaac 앱을 기동한다.

    enable_cameras: 오프스크린 렌더 활성 여부. True/False 또는 "pilot"
    (--mode 인자가 pilot일 때만 활성 — 렌더 비활성이면 기동·스텝이 가벼워짐).
    반환: (파싱된 args, simulation_app). 앱 종료는 호출측이 app.close()로 수행.
    """
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # 서버 환경 강제값: 창 없는 헤드리스 + 필요할 때만 오프스크린 렌더
    args.headless = True
    if enable_cameras == "pilot":
        args.enable_cameras = getattr(args, "mode", None) == "pilot"
    else:
        args.enable_cameras = bool(enable_cameras)

    launcher = AppLauncher(args)
    return args, launcher.app


def close_app(app):
    """STOP 콜백 무한 렌더 루프를 막은 뒤 앱을 종료한다 (진입점 공용).

    Isaac Lab 2.3.0 실측: 종료 시 타임라인 STOP 콜백이 헤드리스에서 무한 렌더
    루프가 되므로, 마지막 SimulationContext에 방지 플래그를 켜고 닫아야 한다
    (reset()이 플래그를 매번 False로 되돌려서 종료 직전에 다시 켜야 함).
    """
    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    if sim is not None:
        sim._disable_app_control_on_stop_handle = True
    app.close()

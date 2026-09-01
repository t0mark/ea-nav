from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.utils.common import (DATA_ROOT, check_dir, init_logging, load_config,
                                mode_roots, resolve_robot_dirs, save_report_json)
from tools.utils.sim import close_app, launch_app

logger = logging.getLogger("01_sim")

def _preview_environment(cfg: dict, device: str, out_dir: Path) -> dict:

    from scripts.sim.utils.environment import SimEnvironment
    from scripts.sim.utils.render import (SceneCamera, structure_band_center,
                                          topview_camera)

    render_cfg = cfg["render"]
    env = SimEnvironment(cfg["sim_run"]["physics_dt"], device)
    camera = SceneCamera(render_cfg)
    size_x, size_y = env.add_terrain(cfg["terrain"])

    struct_cfg = cfg["structures"]
    band, center, items = (0.0, 0.0), (0.0, 0.0), []
    if struct_cfg.get("enabled", False):
        from scripts.sim.utils.structures import StructureBuilder

        band = StructureBuilder(struct_cfg).extent
        center = structure_band_center((size_x, size_y), band, struct_cfg["band_gap"])
        items = env.add_structures(struct_cfg, center)
    env.sim.reset()

    eye, target = topview_camera((size_x, size_y), band, center,
                                 render_cfg["topview_height_scale"])
    top_ok = camera.capture_rgb(out_dir / "terrain_topview.png", eye, target)
    report = {"terrain": {"rendered": top_ok, "size": [size_x, size_y]}}
    logger.info("지형 탑뷰 렌더 %s", "저장" if top_ok else "실패")

    struct_report = {"enabled": bool(items), "band_center": list(center),
                     "band_size": list(band), "items": items, "rendered": False}
    if items:
        span = max(band)
        dist = render_cfg["structures_cam_dist_scale"] * span
        eye_z = render_cfg["structures_cam_height_scale"] * span
        struct_report["rendered"] = camera.capture_rgb(
            out_dir / "structures_view.png",
            (center[0] - dist, center[1] - dist, eye_z), (center[0], center[1], 0.0))
        logger.info("구조물 사시도 렌더 %s (%d개)",
                    "저장" if struct_report["rendered"] else "실패", len(items))
    report["structures"] = struct_report
    return report

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                        help="pilot = 파일럿 셋 + 렌더 체크, full = 본 셋 + 시뮬 실행 확인")
    parser.add_argument("--robot", default=None, metavar="FORM/NAME",
                        help="해당 로봇 1대만 처리 (단건 테스트, 보고 저장 안 함)")
    parser.add_argument("--terrain-only", action="store_true",
                        help="지형·구조물 생성 + 렌더만 (단건 테스트, 보고 저장 안 함)")
    init_logging()

    cfg = load_config("sim")

    args, app = launch_app(parser, enable_cameras="pilot",
                           device_default=cfg["sim_run"]["device"])
    device = args.device

    if args.terrain_only and args.mode != "pilot":
        logger.error("--terrain-only는 --mode pilot에서만 사용할 수 있다")
        close_app(app)
        raise SystemExit(1)

    from scripts.sim.utils import robot_spawn, robot_state_check

    out_dir = check_dir("01_sim") if args.mode == "pilot" else DATA_ROOT / "sim"
    robots_root, usd_root = mode_roots(args.mode)
    single_test = args.robot is not None or args.terrain_only
    report = {"mode": args.mode, "device": device,
              "schema": "01_sim.standing_height_tilt.v1",
              "terrain": {}, "structures": {}, "convert": {}, "inspect": {},
              "sim_run": {}}

    if args.mode == "pilot" and not args.robot:
        report.update(_preview_environment(cfg, device, out_dir))
    if args.terrain_only:
        close_app(app)
        return

    robot_dirs = resolve_robot_dirs(robots_root, args.robot)
    if not robot_dirs:
        logger.error("처리 대상 없음 (URDF 루트: %s, --robot %s)", robots_root, args.robot)
        close_app(app)
        raise SystemExit(1)

    report["convert"] = robot_spawn.convert_batch(robot_dirs, robots_root, usd_root, cfg)
    report["inspect"] = robot_spawn.validate_batch(robot_dirs, robots_root, usd_root,
                                                    report["convert"])
    report["sim_run"] = robot_state_check.standing_batch(
        robot_dirs, robots_root, usd_root, cfg, device,
        report["convert"], report["inspect"],
        render_dir=(out_dir / "renders") if args.mode == "pilot" else None)

    saved = save_report_json(report, out_dir / f"report_{args.mode}.json", single_test)
    if saved:
        sim_results = list(report["sim_run"].values())
        n_pass = sum(r.get("status") == "pass" for r in sim_results)
        n_fail = sum(r.get("status") == "fail" for r in sim_results)
        logger.info("완료: 변환 %d/%d, 검증 %d/%d, 기립 통과 %d / 실패 %d -> %s",
                    sum(r["ok"] for r in report["convert"].values()), len(report["convert"]),
                    sum(r["pass"] for r in report["inspect"].values()), len(report["inspect"]),
                    n_pass, n_fail, saved)
    else:
        logger.info("단건 테스트 완료 (보고 저장 생략)")

    close_app(app)

if __name__ == "__main__":
    main()

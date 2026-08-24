"""sim 단계 진입점: URDF -> USD 변환 + USD 검증 + 지형 생성 + 스폰 + 시뮬·렌더 체크.

파라미터:
- --mode pilot|full (기본 pilot)
    pilot = check/00_urdf/robots 파일럿 셋 -> check/01_sim/usd 변환 후,
            지형 생성+탑뷰 렌더, 로봇별 스폰 -> 기립 홀드 시뮬 -> form 색 렌더 저장
    full  = /data/EA-Trav/urdf/synthesis 본 셋 -> /data/EA-Trav/sim/usd 변환 후,
            로봇별 스폰 -> 기립 홀드 시뮬 실행 확인 (렌더 없음 — 가볍게 기동)
- --robot {form}/{이름} : 해당 로봇 1대만 처리 (시점·색 확인용 단건 테스트.
            지형 단계 생략, 보고 json 저장 안 함 — 렌더 1장만 갱신)
- --terrain-only : 지형 생성 + 탑뷰 렌더만 (지형 파라미터 확인용 단건 테스트.
            보고 json 저장 안 함)
- AppLauncher 표준 인자(--device 등)도 그대로 받는다.

실행 (sim 컨테이너): /isaac-sim/python.sh /workspace/eatrav/tools/01_sim.py --mode pilot
경로 규약: 보고 json = {산출 루트}/report_{mode}.json, 설정 = configs/sim.yaml,
파일럿 렌더 = check/01_sim/renders/{form}__{이름}.png + terrain_topview.png
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

# 저장소 루트를 import 경로에 추가 (공통 유틸을 쓰기 위한 부트스트랩)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.utils.common import DATA_ROOT, check_dir, init_logging, load_config
from tools.utils.sim import close_app, launch_app

# 이 단계의 고정 경로 (모듈 docstring의 경로 규약)
_PILOT_ROBOTS = check_dir("00_urdf") / "robots"
_PILOT_USD = check_dir("01_sim") / "usd"
_RENDER_DIR = check_dir("01_sim") / "renders"
_FULL_ROBOTS = DATA_ROOT / "urdf/synthesis"
_FULL_USD = DATA_ROOT / "sim/usd"

logger = logging.getLogger("01_sim")


def _roots(mode: str) -> tuple[Path, Path]:
    """mode에 해당하는 (URDF 셋 루트, USD 산출 루트)를 반환한다."""
    if mode == "pilot":
        return _PILOT_ROBOTS, _PILOT_USD
    return _FULL_ROBOTS, _FULL_USD


def _tilt_deg(quat_wxyz) -> float:
    """루트 쿼터니언(w,x,y,z)에서 몸통 z축의 월드 z축 대비 기울기(deg)를 구한다.

    회전행렬 (z,z) 성분 = 1 - 2(x^2 + y^2) 이므로 tilt = acos(그 값).
    """
    zz = 1.0 - 2.0 * (float(quat_wxyz[1]) ** 2 + float(quat_wxyz[2]) ** 2)
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def main():
    """앱 기동 후 변환 -> 검증 -> 지형 체크 -> 시뮬 실행(+렌더) 순서로 처리한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                        help="pilot = 파일럿 셋 + 렌더 체크, full = 본 셋 + 시뮬 실행 확인")
    parser.add_argument("--robot", default=None, metavar="FORM/NAME",
                        help="해당 로봇 1대만 처리 (단건 테스트, 보고 저장 안 함)")
    parser.add_argument("--terrain-only", action="store_true",
                        help="지형 생성 + 탑뷰 렌더만 (단건 테스트, 보고 저장 안 함)")
    init_logging()
    # 렌더(오프스크린 카메라)는 pilot에서만 필요 — full은 가볍게 기동
    args, app = launch_app(parser, enable_cameras="pilot")
    # 지형 탑뷰는 렌더가 곧 산출물이라 pilot 전용 (조합 오류를 조기에 알린다).
    # app.close()가 내부에서 즉시 프로세스를 끝내므로 메시지를 먼저 남긴다
    if args.terrain_only and args.mode != "pilot":
        logger.error("--terrain-only는 --mode pilot에서만 사용할 수 있다")
        close_app(app)
        raise SystemExit(1)

    # Isaac 의존 모듈은 앱 기동 후에만 임포트 가능
    from scripts.sim.utils import robot_spawn
    from scripts.sim.utils.environment import SimEnvironment
    from scripts.sim.utils.render import SceneCamera

    cfg = load_config("sim")
    robots_root, usd_root = _roots(args.mode)
    run_cfg = cfg["sim_run"]
    render_cfg = cfg["render"]
    single_test = args.robot is not None or args.terrain_only
    report = {"mode": args.mode, "terrain": {}, "convert": {}, "inspect": {}, "sim_run": {}}

    # 0) 지형 생성 체크 + 탑뷰 렌더 (pilot 전용 — 렌더가 곧 검증 산출물)
    if args.mode == "pilot" and not args.robot:
        env = SimEnvironment(run_cfg["physics_dt"], run_cfg["device"])
        camera = SceneCamera(render_cfg)
        size_x, size_y = env.add_terrain(cfg["terrain"])
        env.sim.reset()
        # 탑뷰: 원점 상공 수직 내려보기 (-y 미세 오프셋 = 업벡터 퇴화 방지 + 지도 정렬)
        height = render_cfg["topview_height_scale"] * max(size_x, size_y)
        rendered = camera.capture_rgb(_RENDER_DIR.parent / "terrain_topview.png",
                                      (0.0, -0.02 * height, height), (0.0, 0.0, 0.0))
        report["terrain"] = {"rendered": rendered, "size": [size_x, size_y]}
        logger.info("지형 탑뷰 렌더 %s", "저장" if rendered else "실패")
    if args.terrain_only:
        close_app(app)
        return

    # 처리 대상 목록 (--robot 단건 테스트는 해당 로봇만)
    robot_dirs = robot_spawn.iter_robot_dirs(robots_root)
    if args.robot:
        robot_dirs = [d for d in robot_dirs
                      if str(d.relative_to(robots_root)) == args.robot]
        if not robot_dirs:
            logger.error("--robot 대상 없음: %s", args.robot)
            close_app(app)
            raise SystemExit(1)

    # 1) URDF -> USD 일괄 변환 (+ meta.json·joints.json 동봉)
    logger.info("변환 시작: %d대 (%s -> %s)", len(robot_dirs), robots_root, usd_root)
    for i, robot_dir in enumerate(robot_dirs):
        rel = str(robot_dir.relative_to(robots_root))
        try:
            robot_spawn.convert_robot(robot_dir, usd_root / rel, cfg["converter"])
            report["convert"][rel] = {"ok": True}
        except Exception as e:
            report["convert"][rel] = {"ok": False, "error": str(e)}
        logger.info("[%d/%d] 변환 %s %s", i + 1, len(robot_dirs), rel,
                    "OK" if report["convert"][rel]["ok"] else "FAIL")

    # 2) USD 정합성 검증 (가동 조인트 보존·게인 중립·articulation root)
    # 개체별 예외 격리: 한 대의 실패가 셋 전체 처리를 끊지 않게 한다 (full 모드 필수)
    for robot_dir in robot_dirs:
        rel = str(robot_dir.relative_to(robots_root))
        if not report["convert"][rel]["ok"]:
            continue
        try:
            result = robot_spawn.inspect_usd(usd_root / rel, robot_dir / "robot.urdf")
        except Exception as e:
            result = {"pass": False, "error": str(e)}
            logger.error("검증 예외 %s: %s", rel, e)
        report["inspect"][rel] = result
        if "error" not in result:
            logger.info("검증 %s %s (가동 %d, fixed 유지 %d, 강체 %d)", rel,
                        "OK" if result["pass"] else "FAIL", result["movable_joints"],
                        result["fixed_joints_kept"], result["rigid_bodies"])

    # 3) 로봇별 스폰 -> 기립 홀드 시뮬 실행 -> 상태 관찰 (+ pilot 색·렌더 저장)
    # 변환·검증을 모두 통과한 개체만 올리고, 개체별 예외를 격리한다
    steps = int(round(run_cfg["run_time"] / run_cfg["physics_dt"]))
    for robot_dir in robot_dirs:
        rel = str(robot_dir.relative_to(robots_root))
        if not report["convert"][rel]["ok"] or not report["inspect"].get(rel, {}).get("pass"):
            continue
        try:
            env = SimEnvironment(run_cfg["physics_dt"], run_cfg["device"])
            env.add_ground(run_cfg["ground_size"])
            # form 색은 pilot 렌더 체크에서만 입힌다 (full은 원본 회색)
            form = rel.split("/")[0]
            color = render_cfg["form_colors"].get(form, render_cfg["form_colors"]["default"]) \
                if args.mode == "pilot" else None
            env.spawn_robot(usd_root / rel, cfg["drive"], num_envs=1,
                            spawn_margin=run_cfg["spawn_margin"], color=color,
                            self_collision=run_cfg["self_collision"])
            # 카메라는 reset 전에 만들어야 센서 초기화가 물리 초기화에 묶인다
            camera = SceneCamera(render_cfg) if args.mode == "pilot" else None
            env.reset()
            for _ in range(steps):
                env.hold_step()

            # 시뮬 실행 결과 관찰값 기록 (판정은 NaN/발산 여부만 — 자세 평가는 제어기 단계)
            data = env.robot.data
            pos = data.root_pos_w[0].tolist()
            quat = data.root_quat_w[0].tolist()
            result = {
                "ok": all(math.isfinite(v) for v in pos + quat),
                "final_height": pos[2],
                "tilt_deg": _tilt_deg(quat),
                "rendered": False,
            }
            if camera is not None:
                # 전신이 잡히도록 최대 외형 치수 비례 거리 + 몸 중심 높이 조준
                with open(usd_root / rel / "meta.json") as f:
                    metrics = json.load(f)["metrics"]
                size = max(metrics["overall_length"], metrics["overall_width"],
                           metrics["overall_height"])
                dist = max(render_cfg["cam_dist_min"], render_cfg["cam_dist_scale"] * size)
                target = (pos[0], pos[1], 0.5 * metrics["overall_height"])
                eye = (target[0] + dist, target[1] + dist, target[2] + 0.6 * dist)
                render_path = _RENDER_DIR / f"{rel.replace('/', '__')}.png"
                result["rendered"] = camera.capture_rgb(render_path, eye, target)
            logger.info("시뮬 %s %s (높이 %.3fm, 기울기 %.1fdeg%s)", rel,
                        "OK" if result["ok"] else "FAIL", result["final_height"],
                        result["tilt_deg"], ", 렌더 저장" if result["rendered"] else "")
        except Exception as e:
            result = {"ok": False, "error": str(e), "rendered": False}
            logger.error("시뮬 예외 %s: %s", rel, e)
        report["sim_run"][rel] = result

    # 요약 보고 저장 — 단건 테스트는 저장하지 않는다 (본 보고 덮어쓰기 방지)
    if not single_test:
        report_path = (check_dir("01_sim") if args.mode == "pilot" else DATA_ROOT / "sim") \
            / f"report_{args.mode}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=1)
        n_conv = sum(r["ok"] for r in report["convert"].values())
        n_insp = sum(r["pass"] for r in report["inspect"].values())
        n_sim = sum(r["ok"] for r in report["sim_run"].values())
        logger.info("완료: 변환 %d/%d, 검증 %d/%d, 시뮬 %d/%d -> %s",
                    n_conv, len(report["convert"]), n_insp, len(report["inspect"]),
                    n_sim, len(report["sim_run"]), report_path)
    else:
        logger.info("단건 테스트 완료 (보고 저장 생략)")

    close_app(app)


if __name__ == "__main__":
    main()

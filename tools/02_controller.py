from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.utils.common import (DATA_ROOT, check_dir, init_logging, load_config,
                                mode_roots, resolve_robot_dirs, save_figure,
                                save_report_json)
from tools.utils.sim import close_app, launch_app

_OUT_ROOT = check_dir("02_controller")
_FULL_POLICY = DATA_ROOT / "sim/policies"

_WHEELED_FORMS = ("diff", "skid", "ackermann", "omni", "wheeled_humanoid")
_LEGGED_FORMS = ("quad", "hex", "humanoid")

logger = logging.getLogger("02_controller")

def _policy_dir(form: str, mode: str) -> Path | None:

    candidates = [_FULL_POLICY / form]
    if mode == "pilot":
        candidates.insert(0, _OUT_ROOT / "legged" / form)
    for d in candidates:
        if (d / "policy.pt").exists() and (d / "bundle.json").exists():
            return d
    return None

def _save_curve(path: Path, curve: list, title: str):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    its = [c["iteration"] for c in curve]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(its, [c["mean_return"] for c in curve], "-o", color="tab:blue")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("평균 에피소드 수익")
    ax1.grid(True, alpha=0.4)
    ax2.plot(its, [c["mean_ep_len_s"] for c in curve], "-o", color="tab:orange")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("평균 에피소드 길이 [s]")
    ax2.grid(True, alpha=0.4)
    fig.suptitle(title)
    save_figure(fig, path)

def _gpu_shard(device: str) -> tuple[int, int] | None:

    m = re.fullmatch(r"cuda:(\d+)", device or "")
    if not m:
        return None
    idx = int(m.group(1))
    return (idx, 4) if 0 <= idx < 4 else None

def run_eval(args, sim_cfg: dict, ctrl_cfg: dict, device: str):

    from scripts.sim.controller.rollout import evaluate_scene

    scn = ctrl_cfg["scenario"]
    pilot = args.mode == "pilot"
    robots_root, usd_root = mode_roots(args.mode)
    debugging = bool(getattr(args, "debug", False))

    out_root = _OUT_ROOT if pilot else DATA_ROOT / "sim"

    policy_dirs = {form: _policy_dir(form, args.mode) for form in _LEGGED_FORMS}
    for form, d in policy_dirs.items():
        if d is None:
            logger.warning("legged form %s: 정책 번들 없음 — 평가 제외 (train 선행 필요)", form)
        else:
            logger.info("legged form %s: 정책 %s", form, d)
    active_forms = _WHEELED_FORMS + tuple(f for f in _LEGGED_FORMS
                                          if policy_dirs[f] is not None)
    robot_dirs = resolve_robot_dirs(usd_root, args.robot, form_filter=active_forms)

    gpu_shard = _gpu_shard(device) if (debugging and args.robot is None) else None
    if gpu_shard is not None:
        idx, n = gpu_shard
        robot_dirs = robot_dirs[idx::n]

    if not robot_dirs:
        logger.error("처리 대상 없음 (usd 루트: %s, --robot %s, gpu_shard %s) — "
                     "01_sim/train 선행 여부 확인", usd_root, args.robot, gpu_shard)
        raise SystemExit(1)

    report = {"mode": args.mode, "scenario": {k: v for k, v in scn.items()
                                              if k != "video"}, "runs": {}}
    logger.info("제어기 시나리오 시작: %d대 (%s)", len(robot_dirs), usd_root)
    for i, usd_dir in enumerate(robot_dirs):
        rel = str(usd_dir.relative_to(usd_root))
        form = rel.split("/")[0]
        urdf_path = robots_root / rel / "robot.urdf"
        legged = form in _LEGGED_FORMS

        scenes = scn["scenes"]["legged" if legged else "wheeled"] if pilot            else ["flat"]
        for scene in scenes:
            key = f"{rel}::{scene}"
            try:
                result = evaluate_scene(usd_dir, urdf_path, rel, form, scene,
                                        sim_cfg, ctrl_cfg, device, policy_dirs,
                                        pilot, out_root, debugging=debugging)
            except Exception as e:
                result = {"ok": False, "error": str(e)}
                logger.error("시나리오 예외 %s (%s): %s", rel, scene, e)
            report["runs"][key] = result

            detail = result.get("error") if "error" in result else (
                f"도달 {result['reached']}/{result['n_waypoints']}, "
                f"잔여 {result['final_dist_m']}m, "
                f"{'조기종료 ' if result.get('fell') else ''}{result['sim_time_s']}s")
            logger.info("[%d/%d] %s (%s) %s (%s)", i + 1, len(robot_dirs), rel,
                        scene, "OK" if result["ok"] else "FAIL", detail)

    report_path = (_OUT_ROOT / "report_pilot.json" if pilot
                   else DATA_ROOT / "sim" / "report_controller_full.json")
    if gpu_shard is not None:
        idx, n = gpu_shard
        report_path = report_path.with_name(
            f"{report_path.stem}.gpu{idx}of{n}{report_path.suffix}")
    saved = save_report_json(report, report_path, single_test=args.robot is not None)
    if saved:
        n_ok = sum(r["ok"] for r in report["runs"].values())
        logger.info("완료: 시나리오 %d/%d 통과 -> %s", n_ok, len(report["runs"]), saved)
    else:
        logger.info("단건 테스트 완료 (보고 저장 생략)")

def run_train(args, sim_cfg: dict, ctrl_cfg: dict, rl_cfg: dict, device: str):

    from scripts.sim.controller.legged.train import ppo
    from scripts.sim.utils.robot_spawn import iter_robot_dirs

    robots_root, usd_root = mode_roots(args.mode)
    out_dir = (_OUT_ROOT / "legged" / args.form if args.mode == "pilot"
               else _FULL_POLICY / args.form)

    scale = rl_cfg["train"][args.mode]
    num_robots = int(scale["num_robots"])
    usd_dirs = [d for d in iter_robot_dirs(usd_root) if d.parent.name == args.form]
    names = scale.get("robot_names")
    if names:
        usd_dirs = [d for d in usd_dirs if d.name in names]
    usd_dirs = usd_dirs[:num_robots]
    if not usd_dirs:
        logger.error("학습 대상 없음 (%s/%s) — 01_sim 선행 여부 확인",
                     usd_root, args.form)
        raise SystemExit(1)
    robot_dirs = [(robots_root / d.relative_to(usd_root) / "robot.urdf", d)
                  for d in usd_dirs]
    logger.info("학습 시작: form=%s mode=%s 로봇 %d종 -> %s",
                args.form, args.mode, len(robot_dirs), out_dir)

    summary = ppo.train_form(args.form, robot_dirs, rl_cfg, sim_cfg, out_dir,
                             args.mode, device)
    _save_curve(out_dir / "train_curve.png", summary["curve"],
                f"{args.form} {args.mode} 학습 곡선")

    if args.mode == "pilot":
        args_eval = argparse.Namespace(mode="pilot",
                                       robot=str(usd_dirs[0].relative_to(usd_root)),
                                       debug=False)
        run_eval(args_eval, sim_cfg, ctrl_cfg, device)

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["eval", "train"],
                        help="train = RL 정책 학습, eval = 주행 시나리오 평가")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                        help="pilot = 파일럿 셋 + 시각화, full = 본 셋")
    parser.add_argument("--robot", default=None, metavar="FORM/NAME",
                        help="(eval) 해당 로봇 1대만 (단건 테스트, 보고 저장 안 함)")
    parser.add_argument("--debug", action="store_true",
                        help="(eval) 물리 디버깅 트레이스 저장 + 렌더링(카메라·"
                             "비디오) 생략 — {산출 루트}/{wheeled,legged}/"
                             "debugging_trace/. --robot 없이 쓰면 --device의"
                             " GPU 번호로 대상을 자동 4등분한다")
    parser.add_argument("--form", choices=list(_LEGGED_FORMS), default=None,
                        help="(train) 학습할 legged form")
    init_logging()
    args, app = launch_app(parser, enable_cameras="pilot")

    sim_cfg = load_config("sim")
    ctrl_cfg = load_config("controller")

    try:
        if args.command == "train":
            if args.form is None:
                logger.error("train은 --form이 필요하다 (quad|hex|humanoid)")
            else:
                run_train(args, sim_cfg, ctrl_cfg, load_config("rl"), args.device)
        else:
            run_eval(args, sim_cfg, ctrl_cfg, args.device)
    except SystemExit:
        pass
    except Exception:
        logger.exception("실행 실패")
    close_app(app)

if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import shutil
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

def _legged_policy_dir(form: str, mode: str) -> Path | None:

    candidates = [_FULL_POLICY / form]
    if mode == "pilot":
        candidates.insert(0, _OUT_ROOT / "legged" / form)
    for d in candidates:
        if (d / "policy.pt").exists() and (d / "bundle.json").exists():
            return d
    return None

def _wheeled_policy_dir(mode: str) -> Path | None:

    candidates = [_FULL_POLICY / "wheeled"]
    if mode == "pilot":
        candidates.insert(0, _OUT_ROOT / "wheeled" / "wheeled")
    for d in candidates:
        if (d / "policy.pt").exists() and (d / "bundle.json").exists():
            with open(d / "bundle.json") as f:
                meta = json.load(f)
            if meta.get("kind") == "wheeled_controller_parameter_policy":
                return d
            logger.warning("wheeled 정책 %s: 이전 규약(%s)이라 자동 적용 생략",
                           d, meta.get("kind"))
    return None

def _save_curve(path: Path, curve: list, title: str):

    if not curve:
        logger.warning("학습 곡선 데이터가 없어 %s 생성 생략", path)
        return

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

def _round_robin_by_form(usd_dirs: list[Path], forms: tuple[str, ...],
                         limit: int) -> list[Path]:
    """wheeled 학습 대상이 특정 타입 앞번호에 쏠리지 않도록 타입별로 번갈아 고른다."""

    buckets = {form: [] for form in forms}
    for path in usd_dirs:
        form = path.parent.name
        if form in buckets:
            buckets[form].append(path)
    selected = []
    depth = 0
    while len(selected) < limit:
        added = False
        for form in forms:
            if depth < len(buckets[form]):
                selected.append(buckets[form][depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected

def _split_train_holdout(usd_dirs: list[Path], forms: tuple[str, ...],
                         num_train: int,
                         num_holdout: int) -> tuple[list[Path], list[Path]]:
    """학습용과 holdout용 로봇을 타입이 고루 섞이도록 나눈다.

    라운드로빈 순서는 타입을 순환하므로, 뒤쪽 연속 구간을 holdout으로 떼면 타입 편향 없이
    "학습에 쓰지 않은 morphology"를 확보할 수 있다. 이 셋으로 평가하면 정책이 형태 분포를
    일반화했는지(= 처음 보는 URDF에도 쓸 만한 parameter set을 내는지) 확인할 수 있다.
    """

    picked = _round_robin_by_form(usd_dirs, forms, num_train + num_holdout)
    train, holdout = picked[:num_train], picked[num_train:]
    if num_holdout > 0 and len(holdout) < num_holdout:
        logger.warning("holdout 로봇 부족: 요청 %d대, 확보 %d대 (전체 %d대)",
                       num_holdout, len(holdout), len(usd_dirs))
    return train, holdout

def run_eval(args, sim_cfg: dict, ctrl_cfg: dict, device: str,
             robot_names: list[str] | None = None,
             holdout_names: set[str] | None = None):

    from scripts.sim.controller.rollout import evaluate_scene

    scn = ctrl_cfg["scenario"]
    pilot = args.mode == "pilot"
    robots_root, usd_root = mode_roots(args.mode)
    debugging = bool(getattr(args, "debug", False))

    out_root = _OUT_ROOT if pilot else DATA_ROOT / "sim"

    # 전체 실행은 이전 "평가" 산출물을 지우고 시작한다. 남겨두면 이번에 갱신되지 않은
    # 파일이 그대로 남아, 보고와 트레이스·플롯·영상이 서로 다른 시점의 코드를 가리키게 된다
    # (실측 사례: report_pilot.json은 수정 전, debugging_trace는 수정 후라 같은 로봇의
    # 결과가 정반대로 기록됨). 삭제 대상을 평가 산출물로 한정하는 이유는 같은 루트 아래
    # legged/{form}/policy.pt 로 학습 정책 번들이 저장되기 때문이다 — 루트를 통째로 지우면
    # eval이 train 결과를 날려 legged 평가가 불가능해진다.
    #
    # --debug는 삭제하지 않고 자기 shard 결과만 저장한다. debug는 --device의 GPU 번호로
    # 대상을 4등분해 여러 프로세스로 동시에 돌리는 모드라(아래 gpu_shard), 각 프로세스가
    # 시작하면서 같은 디렉터리를 지우면 먼저 끝난 shard의 산출물을 나중에 시작한 shard가
    # 지워버린다. 삭제는 병렬 작업 바깥에서만 한다.
    # 단건 테스트(--robot)는 보고를 쓰지 않으므로 건드리지 않는다.
    if args.robot is None and pilot and not debugging:
        stale = [_OUT_ROOT / "report_pilot.json"]
        for group in ("wheeled", "legged"):
            stale += [_OUT_ROOT / group / name
                      for name in ("plots", "videos", "debugging_trace")]
        removed = 0
        for path in stale:
            if path.is_dir():
                shutil.rmtree(path)
                removed += 1
            elif path.exists():
                path.unlink()
                removed += 1
        logger.info("이전 평가 산출물 %d개 삭제 (정책 번들은 보존)", removed)

    policy_dirs = {form: _legged_policy_dir(form, args.mode)
                   for form in _LEGGED_FORMS}
    for form, d in policy_dirs.items():
        if d is None:
            logger.warning("legged form %s: 정책 번들 없음 — 평가 제외 (train 선행 필요)", form)
        else:
            logger.info("legged form %s: 정책 %s", form, d)

    wheeled_policy = _wheeled_policy_dir(args.mode)
    if wheeled_policy is not None:
        ctrl_cfg = copy.deepcopy(ctrl_cfg)
        ctrl_cfg.setdefault("wheeled_rl", {})
        ctrl_cfg["wheeled_rl"]["policy_path"] = str(wheeled_policy)
        logger.info("wheeled RL parameter 정책 %s", wheeled_policy)
    active_forms = _WHEELED_FORMS + tuple(f for f in _LEGGED_FORMS
                                          if policy_dirs[f] is not None)
    robot_dirs = resolve_robot_dirs(usd_root, args.robot, form_filter=active_forms)

    # 학습 직후 평가처럼 대상 로봇이 지정된 경우, 그 목록만 본다 (train/holdout 분리 평가)
    if robot_names is not None:
        wanted = set(robot_names)
        robot_dirs = [d for d in robot_dirs
                      if str(d.relative_to(usd_root)) in wanted]

    gpu_shard = _gpu_shard(device) if (debugging and args.robot is None) else None
    if gpu_shard is not None:
        idx, n = gpu_shard
        robot_dirs = robot_dirs[idx::n]

    if not robot_dirs:
        logger.error("처리 대상 없음 (usd 루트: %s, --robot %s, gpu_shard %s) — "
                     "01_sim/train 선행 여부 확인", usd_root, args.robot, gpu_shard)
        raise SystemExit(1)

    holdout = holdout_names or set()
    report = {"mode": args.mode, "scenario": {k: v for k, v in scn.items()
                                              if k != "video"}, "runs": {}}
    if robot_names is not None:
        report["split"] = {
            "train": [n for n in robot_names if n not in holdout],
            "holdout": sorted(holdout),
        }
    logger.info("제어기 시나리오 시작: %d대 (%s), holdout %d대",
                len(robot_dirs), usd_root, len(holdout))
    for i, usd_dir in enumerate(robot_dirs):
        rel = str(usd_dir.relative_to(usd_root))
        form = rel.split("/")[0]
        urdf_path = robots_root / rel / "robot.urdf"
        legged = form in _LEGGED_FORMS

        # full 평가도 파일럿과 같은 지형 씬 목록을 본다. flat만 보면 "URDF별 고정
        # parameter set이 경사·계단에서도 통하는가"를 확인할 수 없어, 일반화 검증이라는
        # 평가 목적 자체가 성립하지 않는다. 씬 구성은 rollout.build_scene이 모드와 무관하게
        # 처리하고, pilot 여부는 카메라·장식 렌더링에만 쓰인다
        scenes = scn["scenes"]["legged" if legged else "wheeled"]
        for scene in scenes:
            key = f"{rel}::{scene}"
            try:
                result = evaluate_scene(usd_dir, urdf_path, rel, form, scene,
                                        sim_cfg, ctrl_cfg, device, policy_dirs,
                                        pilot, out_root, debugging=debugging)
            except Exception as e:
                result = {"ok": False, "error": str(e)}
                logger.error("시나리오 예외 %s (%s): %s", rel, scene, e)
            if robot_names is not None:
                result["split"] = "holdout" if rel in holdout else "train"
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
        _log_split_summary(report["runs"])
    else:
        logger.info("단건 테스트 완료 (보고 저장 생략)")

def _log_split_summary(runs: dict):
    """train/holdout 별 통과율을 따로 남겨 일반화 여부를 바로 보이게 한다."""

    buckets: dict[str, list] = {}
    for result in runs.values():
        split = result.get("split")
        if split is not None:
            buckets.setdefault(split, []).append(bool(result["ok"]))
    for split, oks in sorted(buckets.items()):
        logger.info("[%s] 시나리오 통과 %d/%d (%.1f%%)", split, sum(oks), len(oks),
                    100.0 * sum(oks) / max(len(oks), 1))

def run_train(args, sim_cfg: dict, ctrl_cfg: dict, rl_cfg: dict, device: str):

    from scripts.sim.controller.legged.rl import trainer as legged_trainer
    from scripts.sim.controller.wheeled.rl import trainer as wheeled_trainer
    from scripts.sim.utils.robot_spawn import iter_robot_dirs

    robots_root, usd_root = mode_roots(args.mode)

    scale = rl_cfg["train"][args.mode]
    num_robots = int(scale["num_robots"])
    holdout_dirs: list[Path] = []
    if args.form == "wheeled":
        forms = set(_WHEELED_FORMS)
        usd_dirs = [d for d in iter_robot_dirs(usd_root) if d.parent.name in forms]
        out_dir = (_OUT_ROOT / "wheeled" / "wheeled" if args.mode == "pilot"
                   else _FULL_POLICY / "wheeled")
    else:
        usd_dirs = [d for d in iter_robot_dirs(usd_root) if d.parent.name == args.form]
        group = "legged" if args.form in _LEGGED_FORMS else "wheeled"
        out_dir = (_OUT_ROOT / group / args.form if args.mode == "pilot"
                   else _FULL_POLICY / args.form)
    names = scale.get("robot_names")
    if names:
        usd_dirs = [d for d in usd_dirs if d.name in names]
    elif args.form == "wheeled":
        usd_dirs, holdout_dirs = _split_train_holdout(
            usd_dirs, _WHEELED_FORMS, num_robots,
            int(scale.get("holdout_robots", 0)))
    else:
        usd_dirs = usd_dirs[:num_robots]
    if not usd_dirs:
        logger.error("학습 대상 없음 (%s/%s) — 01_sim 선행 여부 확인",
                     usd_root, args.form)
        raise SystemExit(1)
    robot_dirs = [(robots_root / d.relative_to(usd_root) / "robot.urdf", d)
                  for d in usd_dirs]
    holdout_names = [str(d.relative_to(usd_root)) for d in holdout_dirs]
    logger.info("학습 시작: form=%s mode=%s 로봇 %d종 (holdout %d종) -> %s",
                args.form, args.mode, len(robot_dirs), len(holdout_names), out_dir)

    if args.form in _LEGGED_FORMS:
        summary = legged_trainer.train_form(args.form, robot_dirs, rl_cfg,
                                            sim_cfg, out_dir, args.mode, device)
    else:
        summary = wheeled_trainer.train(robot_dirs, rl_cfg, sim_cfg, ctrl_cfg,
                                        out_dir, args.mode, device,
                                        holdout_names=holdout_names)
    _save_curve(out_dir / "train_curve.png", summary["curve"],
                f"{args.form} {args.mode} 학습 곡선")

    if args.mode == "pilot":
        _pilot_eval_after_train(args, sim_cfg, ctrl_cfg, device, usd_root,
                                usd_dirs, holdout_dirs)

def _pilot_eval_after_train(args, sim_cfg: dict, ctrl_cfg: dict, device: str,
                            usd_root: Path, train_dirs: list[Path],
                            holdout_dirs: list[Path]):
    """학습 직후 파일럿 평가를 돌린다.

    wheeled는 학습에 쓴 로봇 전체 + holdout까지 평가해야 "URDF별 고정 parameter set"이
    일반화됐는지 볼 수 있으므로, 첫 로봇 1대만 보던 경로를 전체 셋으로 넓힌다. legged는
    form별 정책 구조가 달라 기존 단건 확인을 유지한다.
    """

    if args.form != "wheeled":
        single = argparse.Namespace(
            mode="pilot", robot=str(train_dirs[0].relative_to(usd_root)),
            debug=False)
        run_eval(single, sim_cfg, ctrl_cfg, device)
        return
    train_names = [str(d.relative_to(usd_root)) for d in train_dirs]
    holdout_names = [str(d.relative_to(usd_root)) for d in holdout_dirs]
    eval_args = argparse.Namespace(mode="pilot", robot=None, debug=False)
    run_eval(eval_args, sim_cfg, ctrl_cfg, device,
             robot_names=train_names + holdout_names,
             holdout_names=set(holdout_names))

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
    parser.add_argument("--form",
                        choices=list(_LEGGED_FORMS) + ["wheeled"],
                        default=None,
                        help="(train) 학습할 form. wheeled는 wheeled 타입 전체 공용 policy")
    init_logging()
    args, app = launch_app(parser, enable_cameras="pilot")

    sim_cfg = load_config("sim")
    ctrl_cfg = load_config("controller")

    try:
        if args.command == "train":
            if args.form is None:
                logger.error("train은 --form이 필요하다 (quad|hex|humanoid|wheeled)")
            else:
                cfg_name = "rl" if args.form in _LEGGED_FORMS else "wheeled_rl"
                run_train(args, sim_cfg, ctrl_cfg, load_config(cfg_name), args.device)
        else:
            run_eval(args, sim_cfg, ctrl_cfg, args.device)
    except SystemExit:
        pass
    except Exception:
        logger.exception("실행 실패")
    close_app(app)

if __name__ == "__main__":
    main()

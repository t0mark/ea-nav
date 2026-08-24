"""controller 단계 진입점: wheeled·legged 제어기 학습(train)과 주행 평가(eval).

사용법 (sim 컨테이너, /isaac-sim/python.sh /workspace/eatrav/tools/02_controller.py ...):

- train --form quad|hex|humanoid [--mode pilot|full]
    form 정책 1개를 PPO로 학습 -> 배포 번들(policy.pt + bundle.json) export.
    pilot = 걸음새 확인용 단기 학습 (로봇 2종 x env 512, 600 iter) + 보상 곡선
            플롯 + 배포 검증 롤아웃(비디오·궤적) -> check/02_controller/legged/{form}/
    full  = 본 학습 (configs/rl.yaml train.full 규모)
            -> /data/EA-Trav/sim/policies/{form}/
    학습 로봇 셋 = 해당 mode의 usd 루트 {form}에서 이름순 앞 num_robots종
    (시드 체계와 정합 — 같은 셋 재현).

- eval [--mode pilot|full] [--robot FORM/NAME]
    주행 시나리오 (직진 -> 좌회전 -> 우회전 웨이포인트) 평가.
    wheeled = pure pursuit + IK/LQR, legged = pure pursuit + RL 정책
    (해당 form 번들이 있어야 — 없으면 건너뛰고 로그).
    pilot = check 셋 x 씬(카테고리별: wheeled = 평지·경사, legged = 평지·
            경사·계단) — 궤적 플롯 + 주행 비디오(H.264 mp4, 목적지 마커·
            궤적 표식·격자 포함) -> check/02_controller/{wheeled,legged}/
    full  = 본 셋 동적 검사 (plan 정의 = 평지만, 비디오 없음)
            -> /data/EA-Trav/sim/report_controller_full.json
    --robot = 1대만 (단건 테스트, 보고 저장 안 함)

- AppLauncher 표준 인자(--device 등)도 그대로 받는다.

판정: 전 웨이포인트 도달(반경 안 체류 + 서행) + 전도 없음.
지표 = 도달 시간·최종 거리·최대 기울기 (configs/controller.yaml scenario).
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
_OUT_ROOT = check_dir("02_controller")
_FULL_ROBOTS = DATA_ROOT / "urdf/synthesis"
_FULL_USD = DATA_ROOT / "sim/usd"
_FULL_POLICY = DATA_ROOT / "sim/policies"

_WHEELED_FORMS = ("diff", "skid", "ackermann", "omni", "wheeled_humanoid")
_LEGGED_FORMS = ("quad", "hex", "humanoid")

logger = logging.getLogger("02_controller")


def _tilt_deg(quat_wxyz) -> float:
    """루트 쿼터니언(w,x,y,z)에서 몸통 z축의 월드 z축 대비 기울기(deg).

    회전행렬 (z,z) 성분 = 1 - 2(x^2 + y^2) 이므로 tilt = acos(그 값).
    """
    zz = 1.0 - 2.0 * (float(quat_wxyz[1]) ** 2 + float(quat_wxyz[2]) ** 2)
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def _save_plot(path: Path, traj: list, waypoints: list, reach_radius: float,
               title: str):
    """스폰 기준 상대 좌표로 궤적·웨이포인트·도달 반경을 그려 저장한다.

    시각화 체크 산출물: 명령 경로(웨이포인트)와 실제 궤적의 정합을 눈으로
    판정한다 (조향 부호·선회 방향 오류가 즉시 드러난다).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    xs, ys = [p[0] for p in traj], [p[1] for p in traj]
    ax.plot(xs, ys, "-", color="tab:blue", lw=1.5, label="궤적")
    ax.plot(0, 0, "o", color="black", ms=8, label="스폰")
    for i, (wx, wy) in enumerate(waypoints):
        ax.add_patch(plt.Circle((wx, wy), reach_radius, color="tab:green",
                                alpha=0.25))
        ax.plot(wx, wy, "*", color="tab:green", ms=14)
        ax.annotate(f"wp{i}", (wx, wy), textcoords="offset points",
                    xytext=(6, 6))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("x [m] (스폰 기준)")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_curve(path: Path, curve: list, title: str):
    """학습 곡선(iteration 대비 평균 수익·에피소드 길이)을 그려 저장한다.

    시각화 체크 산출물: 수익 상승·에피소드 연장 추세로 학습 루프의 정상
    동작을 눈으로 판정한다.
    """
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _timeout_s(ctrl_cfg: dict, controller, rel_wps: list) -> float:
    """로봇 물성 비례 제한 시간을 계산한다 (완주 보장은 제어기 책임이고,
    시계는 개체 속도·회전 반경에 공정해야 한다는 원칙).

    시간 = 여유율 x (경로 길이 / 유효 속도 + 웨이포인트당 선회 1바퀴 시간).
    속도·선회 반경은 제어기가 실제로 쓰는 명령 경계(nav_limits)를 그대로
    쓴다 — meta 기하값·전역 캡 근사는 전도 한계 등으로 깎인 실경계와
    어긋나 시계가 불공정해질 수 있다.
    """
    scn = ctrl_cfg["scenario"]
    nav = controller.nav_limits
    v = float(scn["timeout_v_ratio"]) * nav["v_max"]

    # 경로 길이 = 스폰 -> 웨이포인트 체인의 선분 합
    length, prev = 0.0, (0.0, 0.0)
    for wp in rel_wps:
        length += math.hypot(wp[0] - prev[0], wp[1] - prev[1])
        prev = tuple(wp)
    need = length / v + len(rel_wps) * 2.0 * math.pi * nav["r_turn"] / v
    return min(max(scn["timeout_margin"] * need, scn["timeout_min"]),
               scn["timeout_max"])


def _build_scene(env, scene: str, sim_cfg: dict, scn: dict,
                 decorate: bool) -> tuple:
    """평가 씬 1개를 만든다 (flat 평지 / slope·stairs 단일 셀 역피라미드).

    지형 씬은 기존 add_terrain을 그대로 재사용한다 — 해당 서브지형 비율만
    1로 두고 난이도 범위를 고정값으로 좁힌 1x1 격자 (중앙 플랫폼 스폰 ->
    바깥으로 갈수록 올라가는 등판 코스. plan 타겟 케이스 = 경사 각도·계단
    단차). decorate = 평지 격자선 (pilot 시각 확인 전용).
    반환: 스폰 원점 (x, y, z) — 지형 셀 원점의 z 포함.
    """
    from scripts.sim.utils import render as render_utils

    if scene == "flat":
        env.add_ground(sim_cfg["sim_run"]["ground_size"],
                       friction=sim_cfg["contact"]["ground_friction"])
        if decorate:
            render_utils.add_ground_grid(half_size=8.0)
        return (0.0, 0.0, 0.0)

    et = scn["eval_terrain"]
    tcfg = dict(sim_cfg["terrain"])
    tcfg.update(num_rows=1, num_cols=1, curriculum=False,
                size=list(et["size"]))
    for key in ("stairs_proportion", "stairs_inv_proportion",
                "boxes_proportion", "rough_proportion",
                "slope_proportion", "slope_inv_proportion"):
        tcfg[key] = 0.0
    if scene == "slope":
        tcfg["slope_inv_proportion"] = 1.0
        tcfg["slope_range"] = [et["slope_angle"], et["slope_angle"]]
    elif scene == "stairs":
        tcfg["stairs_inv_proportion"] = 1.0
        tcfg["step_height"] = [et["stair_height"], et["stair_height"]]
    else:
        raise ValueError(f"알 수 없는 씬: {scene}")
    # 무채색 + 음영 (높이 무지개색이 화면을 지배하는 것 피드백 — 탑뷰용
    # 높이 색상은 01_sim에서만 유지)
    env.add_terrain(tcfg, num_envs=1, color_scheme="none")
    return tuple(env.terrain.env_origins[0].tolist())


def _run_scenario(env, controller, ctrl_cfg: dict, physics_dt: float,
                  visual: dict | None = None,
                  scan_ctx: dict | None = None) -> tuple[dict, list]:
    """시나리오 1회 실행: 웨이포인트 순차 추종 + 전도·시간 판정.

    반환: (결과 dict, 스폰 기준 상대 궤적 [(x, y)]). 제어는 컨트롤러의
    decimation 주기로, 사이 물리 스텝은 직전 목표 유지(step 무인자)로 돈다.

    visual = pilot 시각 확인 컨텍스트 (None = 끔):
      recorder(VideoRecorder)·video_path·dist(추적 카메라 거리)·
      marker_colors·crumb_color·crumb_radius — 목적지 마커를 세우고,
      주행 중 추적 카메라 프레임과 궤적 표식을 남긴 뒤 mp4로 저장한다.
    scan_ctx = legged 지형 씬의 높이 스캔 (None = 평지 0 벡터):
      scanner(RayCaster)·base_height·clip — 제어 스텝마다 갱신해
      ControlObs.height_scan을 채운다 (학습 관측과 정합).
    """
    import torch

    from scripts.sim.controller.core.base import ControlObs
    from scripts.sim.controller.legged import low_rl
    from scripts.sim.utils import render as render_utils

    scn = ctrl_cfg["scenario"]
    balancing = controller.params.base_tag == "diff_balancing"
    rel_wps = scn["balancing_waypoints"] if balancing else scn["waypoints"]
    timeout = _timeout_s(ctrl_cfg, controller, rel_wps)
    device = env.origins.device
    origin = env.origins[0]
    origin_xy = origin[:2]
    goals = [origin_xy + torch.tensor(wp, dtype=torch.float32, device=device)
             for wp in rel_wps]

    # 목적지 비콘 (순수 시각 프림 — 반지름 = 도달 판정 반경)
    if visual is not None:
        render_utils.spawn_goal_markers([g.tolist() for g in goals],
                                        base_z=float(origin[2]),
                                        colors=scn["marker_colors"],
                                        radius=float(scn["reach_radius"]))

    # 정착: wheeled 일반 타입만 낙하·접지 안정 후 제어 시작. balancing은
    # 게인 0 바퀴로 버틸 수 없어 즉시 LQR. legged도 즉시 정책 제어 —
    # PD 홀드 정착 상태는 학습 리셋(상태 기록 직후 정책 제어)에 없는 관측
    # 분포라 정책이 폭주하는 것 실측 (settle 0s = 20s 생존, 0.5s = 0.26s
    # 전도. 학습 시작 조건과 일치시키는 것이 원칙)
    from scripts.sim.controller.core.base import LEGGED_TAGS
    legged_ctrl = controller.params.base_tag in LEGGED_TAGS
    if not balancing and not legged_ctrl:
        for _ in range(int(round(scn["settle_time"] / physics_dt))):
            env.hold_step()
    controller.reset()

    hold_steps = int(round(scn["balance_hold_time"] / physics_dt)) if balancing else 0
    record_every = max(1, int(round(scn["traj_interval"] / physics_dt)))
    total_steps = int(round(timeout / physics_dt))
    dwell_steps = max(1, int(round(scn["reach_dwell_time"] / physics_dt)))
    video_every = max(1, int(round(scn["video"]["capture_interval_s"] / physics_dt)))
    path_every = max(1, int(round(scn["video"]["path_interval_s"] / physics_dt)))

    traj, reach_times = [], [None] * len(goals)
    wp_i, max_tilt, fell, in_count = 0, 0.0, False, 0
    path_prev, n_segs = None, 0
    step = 0
    for step in range(total_steps):
        # 직립 유지 구간(balancing)은 목표 = 제자리, 이후 현재 웨이포인트
        goal = origin_xy if step < hold_steps else goals[wp_i]
        if step % controller.decimation == 0:
            obs = ControlObs.from_articulation(env.robot)
            # 지형 씬의 높이 스캔 (학습과 같은 정규화 — low_rl.scan_obs)
            if scan_ctx is not None:
                scan_ctx["scanner"].update(physics_dt * controller.decimation)
                obs.height_scan = low_rl.scan_obs(
                    env.robot.data.root_pos_w[:, 2],
                    scan_ctx["scanner"].data.ray_hits_w[:, :, 2],
                    scan_ctx["base_height"], scan_ctx["clip"])
            targets = controller.compute(obs, goal.unsqueeze(0))
            env.step(targets.pos, targets.vel, targets.effort)
        else:
            env.step()

        pos3 = env.robot.data.root_pos_w[0]
        pos = pos3[:2]
        if step % record_every == 0:
            rel = (pos - origin_xy).tolist()
            traj.append([round(rel[0], 4), round(rel[1], 4)])
        # 주행 비디오: 체이스 카메라 (로봇 헤딩 후방 상공 -> 전방 주시.
        # 고정 사선 시점은 진행 방향·동작이 안 읽히는 것 피드백) + 궤적 연속선
        if visual is not None:
            if step % video_every == 0:
                q = env.robot.data.root_quat_w[0]
                yaw = math.atan2(2.0 * float(q[0] * q[3] + q[1] * q[2]),
                                 1.0 - 2.0 * float(q[2] * q[2] + q[3] * q[3]))
                p = pos3.tolist()
                d = visual["dist"]
                dx, dy = math.cos(yaw), math.sin(yaw)
                eye = (p[0] - d * dx, p[1] - d * dy, p[2] + 0.55 * d)
                tgt = (p[0] + 0.4 * d * dx, p[1] + 0.4 * d * dy, p[2])
                # 시점 저역 필터 — 선회·후진 전환 시 카메라 홱 돌기 방지
                prev = visual.get("_cam")
                if prev is not None:
                    a = 0.25
                    eye = tuple(a * n + (1 - a) * o for n, o in zip(eye, prev[0]))
                    tgt = tuple(a * n + (1 - a) * o for n, o in zip(tgt, prev[1]))
                visual["_cam"] = (eye, tgt)
                visual["recorder"].add_frame(eye, tgt)
            if step % path_every == 0:
                # 선은 몸체 중심이 아니라 지면 높이에 깐다 (몸체 높이의
                # 공중 선은 떠 있는 전선처럼 보이는 것 실측)
                p3 = pos3.tolist()
                p = (p3[0], p3[1], p3[2] - visual["line_drop"])
                if path_prev is not None:
                    render_utils.spawn_path_segment(n_segs, path_prev, p,
                                                    visual["line_width"],
                                                    visual["line_color"])
                    n_segs += 1
                path_prev = p
        max_tilt = max(max_tilt, _tilt_deg(env.robot.data.root_quat_w[0]))
        if max_tilt > scn["fall_tilt_deg"]:
            fell = True
            break
        # 도달 판정 (유지 구간 종료 후): 반경 안 체류 + 서행 조건을 함께
        # 채워야 도달. 체류 시간만으로는 저속 개체의 무정지 관통을 못
        # 거른다 (반경 0.35 최대 현 0.7m / 0.3s = 2.33m/s 미만이면 통과
        # — 감사 수치 반증) -> 속도 상한이 실질적 "멈춰 섬"을 판정한다
        speed = float(torch.norm(env.robot.data.root_lin_vel_b[0, :2]))
        if step >= hold_steps and speed < scn["reach_speed_max"] \
                and float(torch.norm(pos - goals[wp_i])) < scn["reach_radius"]:
            in_count += 1
            if in_count >= dwell_steps:
                reach_times[wp_i] = round((step + 1) * physics_dt, 2)
                wp_i += 1
                in_count = 0
                if wp_i == len(goals):
                    break
        else:
            in_count = 0

    final_dist = float(torch.norm(env.robot.data.root_pos_w[0, :2] - goals[-1]))
    result = {
        "ok": wp_i == len(goals) and not fell,
        "control_tag": controller.params.control_tag,
        "reached": wp_i, "n_waypoints": len(goals),
        "reach_times_s": reach_times,
        "final_dist_m": round(final_dist, 3),
        "max_tilt_deg": round(max_tilt, 1),
        "fell": fell,
        "sim_time_s": round((step + 1) * physics_dt, 2),
        "timeout_s": round(timeout, 1),
    }
    if visual is not None:
        result["video"] = visual["recorder"].save(visual["video_path"])
    return result, traj


def _policy_dir(form: str, mode: str) -> Path | None:
    """form 정책 번들 폴더를 찾는다 (없으면 None — 해당 form 평가 불가).

    pilot은 파일럿 학습 산출물을 우선 쓰고 (배포 경로 검증 목적),
    없으면 본 정책으로 폴백한다. full은 본 정책만 쓴다.
    """
    candidates = [_FULL_POLICY / form]
    if mode == "pilot":
        candidates.insert(0, _OUT_ROOT / "legged" / form)
    for d in candidates:
        if (d / "policy.pt").exists() and (d / "bundle.json").exists():
            return d
    return None


def _legged_setup(urdf_path: Path, usd_dir: Path, policy_dir: Path):
    """legged 평가 준비물: (스폰 게인 덮어쓰기, 슬롯, 번들 메타).

    로코모션 관절은 번들의 RL 게인 규칙(학습 조건 재현), 슬롯은 스캔 센서
    부착(루트 링크 이름)에 쓴다.
    """
    from scripts.sim.controller.legged import low_rl

    with open(policy_dir / "bundle.json") as f:
        meta = json.load(f)
    slots = low_rl.build_slots(urdf_path, usd_dir, meta["morph"])
    return low_rl.loco_gain_overrides(slots, meta["gains"]), slots, meta


def _run_eval(args, sim_cfg: dict, ctrl_cfg: dict):
    """eval 커맨드: 대상 셋 x 씬에 주행 시나리오를 돌리고 보고를 남긴다."""
    from scripts.sim.utils import robot_spawn

    scn = ctrl_cfg["scenario"]
    pilot = args.mode == "pilot"
    robots_root = _PILOT_ROBOTS if pilot else _FULL_ROBOTS
    usd_root = _PILOT_USD if pilot else _FULL_USD

    # 처리 대상: wheeled 전체 + 정책 번들이 있는 legged form (없으면 건너뜀)
    policy_dirs = {form: _policy_dir(form, args.mode) for form in _LEGGED_FORMS}
    for form, d in policy_dirs.items():
        if d is None:
            logger.warning("legged form %s: 정책 번들 없음 — 평가 제외 (train 선행 필요)", form)
        else:
            logger.info("legged form %s: 정책 %s", form, d)
    active_forms = _WHEELED_FORMS + tuple(f for f in _LEGGED_FORMS
                                          if policy_dirs[f] is not None)
    robot_dirs = [d for d in robot_spawn.iter_robot_dirs(usd_root)
                  if d.parent.name in active_forms]
    if args.robot:
        robot_dirs = [d for d in robot_dirs
                      if str(d.relative_to(usd_root)) == args.robot]
    if not robot_dirs:
        logger.error("처리 대상 없음 (usd 루트: %s, --robot %s) — 01_sim/train 선행 여부 확인",
                     usd_root, args.robot)
        raise SystemExit(1)

    report = {"mode": args.mode, "scenario": {k: v for k, v in scn.items()
                                              if k != "video"}, "runs": {}}
    logger.info("제어기 시나리오 시작: %d대 (%s)", len(robot_dirs), usd_root)
    for i, usd_dir in enumerate(robot_dirs):
        rel = str(usd_dir.relative_to(usd_root))
        form = rel.split("/")[0]
        urdf_path = robots_root / rel / "robot.urdf"
        legged = form in _LEGGED_FORMS
        # 씬 목록: pilot = 카테고리별 시각 확인 코스, full = plan 정의(평지)
        scenes = scn["scenes"]["legged" if legged else "wheeled"] if pilot \
            else ["flat"]
        for scene in scenes:
            key = f"{rel}::{scene}"
            try:
                result = _run_one(args, sim_cfg, ctrl_cfg, usd_dir, urdf_path,
                                  rel, form, scene, policy_dirs, pilot)
            except Exception as e:
                result = {"ok": False, "error": str(e)}
                logger.error("시나리오 예외 %s (%s): %s", rel, scene, e)
            report["runs"][key] = result
            # 예외 개체는 지표 필드가 없으므로 메시지를 분기해 구성한다
            detail = result.get("error") if "error" in result else (
                f"도달 {result['reached']}/{result['n_waypoints']}, "
                f"잔여 {result['final_dist_m']}m, 기울기 {result['max_tilt_deg']}deg, "
                f"{result['sim_time_s']}s")
            logger.info("[%d/%d] %s (%s) %s (%s)", i + 1, len(robot_dirs), rel,
                        scene, "OK" if result["ok"] else "FAIL", detail)

    # 요약 보고 저장 — 단건 테스트는 저장하지 않는다 (본 보고 덮어쓰기 방지)
    if not args.robot:
        report_path = (_OUT_ROOT / "report_pilot.json" if pilot
                       else DATA_ROOT / "sim" / "report_controller_full.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
        n_ok = sum(r["ok"] for r in report["runs"].values())
        logger.info("완료: 시나리오 %d/%d 통과 -> %s", n_ok, len(report["runs"]),
                    report_path)
    else:
        logger.info("단건 테스트 완료 (보고 저장 생략)")


def _run_one(args, sim_cfg: dict, ctrl_cfg: dict, usd_dir: Path,
             urdf_path: Path, rel: str, form: str, scene: str,
             policy_dirs: dict, pilot: bool) -> dict:
    """로봇 1대 x 씬 1개 평가: 씬 구성 -> 스폰 -> 제어 -> 판정 (+ 시각 산출).

    _run_eval의 개체 루프 본문 — 씬마다 새 SimEnvironment를 만든다
    (이전 씬 정리는 생성자가 겸한다).
    """
    from isaaclab.sensors import RayCaster, RayCasterCfg, patterns

    from scripts.sim.controller.core.base import make_controller
    from scripts.sim.utils.environment import SimEnvironment
    from scripts.sim.utils.render import SceneCamera, VideoRecorder

    run_cfg = sim_cfg["sim_run"]
    render_cfg = sim_cfg["render"]
    scn = ctrl_cfg["scenario"]
    legged = form in _LEGGED_FORMS

    with open(usd_dir / "meta.json") as f:
        meta = json.load(f)
    with open(usd_dir / "joints.json") as f:
        groups = json.load(f)["groups"]

    # 스폰 게인 덮어쓰기: balancing = 바퀴 무력화 (LQR 토크 구동),
    # legged = 로코모션 관절 RL 게인 (학습 조건 재현)
    overrides, slots, bundle_meta = None, None, None
    if meta["control_tag"] == "diff_balancing":
        overrides = {n: (0.0, 0.0) for n in groups["velocity"]}
    elif legged:
        overrides, slots, bundle_meta = _legged_setup(urdf_path, usd_dir,
                                                      policy_dirs[form])

    env = SimEnvironment(run_cfg["physics_dt"], run_cfg["device"])
    origin = _build_scene(env, scene, sim_cfg, scn, decorate=pilot)
    color = render_cfg["form_colors"].get(form) if pilot else None
    env.spawn_robot(usd_dir, sim_cfg["drive"], num_envs=1,
                    spawn_margin=run_cfg["spawn_margin"], color=color,
                    # legged는 학습 스폰 물리와 동일 정렬 (셀프 충돌 끔·솔버
                    # 4/0·발 마찰 — 번들 학습 조건 재현). wheeled는 기존 유지
                    self_collision=(False if legged
                                    else run_cfg["self_collision"]),
                    gain_overrides=overrides,
                    contact_cfg=sim_cfg["contact"], origin=origin,
                    friction_links=(list(slots.contact_links) if slots else None),
                    friction_links_mu=float(sim_cfg["contact"]["foot_friction"]),
                    solver_iters=((4, 0) if legged else None),
                    # 학습과 동일 액추에이터 모델 (번들 기록 — DCMotor 포화
                    # 동역학 재현. 구 번들은 키가 없어 implicit 폴백)
                    actuator_model=(bundle_meta.get("actuator_model", "implicit")
                                    if bundle_meta else "implicit"))
    # 카메라·스캐너는 reset 전에 만들어야 센서 초기화가 물리 초기화에 묶인다
    camera = SceneCamera(render_cfg) if pilot else None
    scan_ctx = None
    if legged and scene != "flat":
        # 지형 씬의 높이 스캔 — 번들 규약(격자·클립)으로 학습 관측과 정합
        sc = bundle_meta["scan"]
        scanner = RayCaster(RayCasterCfg(
            prim_path=f"/World/envs/env_0/Robot/{slots.root_link}",
            mesh_prim_paths=["/World/ground"],
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=float(sc["resolution"]), size=tuple(sc["size"])),
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, float(sc["offset_z"])))))
        scan_ctx = {"scanner": scanner,
                    "base_height": float(meta["metrics"]["base_height"]),
                    "clip": float(sc["clip"])}
    env.reset()

    controller = make_controller(
        urdf_path, usd_dir, ctrl_cfg,
        joint_names=env.robot.joint_names,
        default_pose=env.robot.data.default_joint_pos.clone(),
        num_envs=1, device=run_cfg["device"],
        physics_dt=run_cfg["physics_dt"], on_terrain=(scene != "flat"),
        policy_dir=policy_dirs.get(form))

    # pilot 시각 확인 컨텍스트: 추적 카메라 비디오 + 마커·궤적 표식
    visual = None
    out_dir = _OUT_ROOT / ("legged" if legged else "wheeled")
    if pilot:
        metrics = meta["metrics"]
        size = max(metrics["overall_length"], metrics["overall_width"],
                   metrics["overall_height"])
        base = rel.replace("/", "__") + f"__{scene}"
        visual = {
            "recorder": VideoRecorder(camera, scn["video"]),
            "video_path": out_dir / "videos" / f"{base}.mp4",
            "dist": max(render_cfg["cam_dist_min"],
                        render_cfg["cam_dist_scale"] * size),
            "line_width": max(0.025, 0.03 * size),
            "line_color": tuple(0.35 * c for c in (color or (0.6, 0.6, 0.6))),
            # 지면까지의 낙하량 = 기립 루트 높이 (선이 바닥을 살짝 띄워 긁힘 방지)
            "line_drop": max(0.0, float(metrics["base_height"]) - 0.04),
        }

    result, traj = _run_scenario(env, controller, ctrl_cfg,
                                 run_cfg["physics_dt"], visual=visual,
                                 scan_ctx=scan_ctx)
    if pilot:
        # 시각화 체크: 씬별 궤적 플롯 (비디오는 _run_scenario가 저장)
        rel_wps = scn["balancing_waypoints"] \
            if result["control_tag"] == "diff_balancing" else scn["waypoints"]
        _save_plot(out_dir / "plots" / f"{rel.replace('/', '__')}__{scene}.png",
                   traj, rel_wps, scn["reach_radius"],
                   f"{rel} [{scene}] ({result['control_tag']}) "
                   f"{result['reached']}/{result['n_waypoints']} 도달")
    return result


def _run_train(args, sim_cfg: dict, ctrl_cfg: dict, rl_cfg: dict):
    """train 커맨드: form 정책 학습 + 곡선 플롯 (+ pilot 배포 검증 롤아웃)."""
    from scripts.sim.controller.legged.train import ppo
    from scripts.sim.utils import robot_spawn

    run_cfg = sim_cfg["sim_run"]
    robots_root = _PILOT_ROBOTS if args.mode == "pilot" else _FULL_ROBOTS
    usd_root = _PILOT_USD if args.mode == "pilot" else _FULL_USD
    out_dir = (_OUT_ROOT / "legged" / args.form if args.mode == "pilot"
               else _FULL_POLICY / args.form)

    # 학습 셋: {form} 폴더의 이름순 앞 num_robots종 (01_sim 변환 선행 전제).
    # robot_names가 있으면 이름 필터 우선 (보행 사다리 — 특정 개체 지정)
    scale = rl_cfg["train"][args.mode]
    num_robots = int(scale["num_robots"])
    usd_dirs = [d for d in robot_spawn.iter_robot_dirs(usd_root)
                if d.parent.name == args.form]
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
                             args.mode, run_cfg["device"])
    _save_curve(out_dir / "train_curve.png", summary["curve"],
                f"{args.form} {args.mode} 학습 곡선")

    # pilot 배포 검증: 방금 export한 번들로 make_controller -> 시나리오
    # (학습-배포 규약 정합의 통합 테스트 — 씬별 비디오 포함)
    if args.mode == "pilot":
        args_eval = argparse.Namespace(mode="pilot",
                                       robot=str(usd_dirs[0].relative_to(usd_root)))
        _run_eval(args_eval, sim_cfg, ctrl_cfg)


def main():
    """앱 기동 후 커맨드(train/eval)를 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["eval", "train"],
                        help="train = RL 정책 학습, eval = 주행 시나리오 평가")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                        help="pilot = 파일럿 셋 + 시각화, full = 본 셋")
    parser.add_argument("--robot", default=None, metavar="FORM/NAME",
                        help="(eval) 해당 로봇 1대만 (단건 테스트, 보고 저장 안 함)")
    parser.add_argument("--form", choices=list(_LEGGED_FORMS), default=None,
                        help="(train) 학습할 legged form")
    init_logging()
    args, app = launch_app(parser, enable_cameras="pilot")

    sim_cfg = load_config("sim")
    ctrl_cfg = load_config("controller")
    # app.close()는 fastShutdown이라 이후 코드(트레이스백 출력 포함)가 실행되지
    # 않는다 — 오류 사유를 close 전에 logger로 남긴다 (01_sim 가드 패턴)
    try:
        if args.command == "train":
            if args.form is None:
                logger.error("train은 --form이 필요하다 (quad|hex|humanoid)")
            else:
                _run_train(args, sim_cfg, ctrl_cfg, load_config("rl"))
        else:
            _run_eval(args, sim_cfg, ctrl_cfg)
    except SystemExit:
        pass
    except Exception:
        logger.exception("실행 실패")
    close_app(app)


if __name__ == "__main__":
    main()

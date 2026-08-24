"""로봇당 legged 정책 학습·재생 진입점 — 스톡 go2 rough 태스크에 로봇 cfg만 교체.

원칙 (사용자 지시): RL 설정(관측·보상·종료·커리큘럼·PPO)은 Isaac Lab 내장
Isaac-Velocity-Rough-Unitree-Go2-v0 구성(UnitreeGo2RoughEnvCfg)을 그대로 쓰고,
로봇 articulation과 로봇 종속 참조(베이스 링크 이름·발 링크 목록)만 바꾼다.
go2 대조군 대비 표준 오버라이드 3건(code.md "현행 표준 세팅")도 동일 적용:
feet_air_time.weight 0.25 / lin_vel_x [-0.5, 1.0] / iterations 기본 3000.

사용 (sim 컨테이너, GPU 선택은 CUDA_VISIBLE_DEVICES):
  /isaac-sim/python.sh tools/02_train_legged.py train --robot quad/quad_0000 --mode pilot
  /isaac-sim/python.sh tools/02_train_legged.py play  --robot quad/quad_0000 --mode pilot --scene fwd|lat|stairs

경로: usd 입력 = {check/01_sim | /data/EA-Trav/sim}/usd/{robot}/,
정책 출력 = /data/EA-Trav/sim/policies/{robot}/ (무누적 — 중간 체크포인트는
학습 후 최종본만 남기고 삭제), 비디오 = check/02_controller/legged/.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("cmd", choices=["train", "play"], help="train = 학습, play = 고정 명령 비디오")
parser.add_argument("--robot", required=True, help="{form}/{이름} (usd 루트 하위 경로)")
parser.add_argument("--mode", default="pilot", choices=["pilot", "full"], help="usd 입력 루트 선택")
parser.add_argument("--iterations", type=int, default=3000, help="학습 iteration (표준 3000)")
parser.add_argument("--num-envs", type=int, default=4096, help="병렬 env 수 (스톡 4096)")
parser.add_argument("--scene", default="fwd", choices=["fwd", "lat", "stairs"],
                    help="play 전용: 고정 명령 시각 확인 씬 (code.md visual check 표준)")
parser.add_argument("--video-length", type=int, default=500, help="play 캡처 스텝 수")
parser.add_argument("--arm", default="base", choices=["base", "gait", "energy"],
                    help="보행 품질 실험 팔: base = 현행 스톡 레시피, gait = Spot 걸음새 "
                         "보상 이식(위상 강제 계열), energy = 에너지 최소화 계열")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 비디오 캡처는 카메라 필요, 학습은 비활성이 가벼움 (01 단계 실측)
args.headless = True
args.enable_cameras = args.cmd == "play"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 앱 기동 후에만 Isaac 의존 임포트 가능 ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    UnitreeGo2RoughPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (  # noqa: E402
    UnitreeGo2RoughEnvCfg,
)
from isaaclab.managers import RewardTermCfg, SceneEntityCfg  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot import mdp as spot_mdp  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sim.utils.robot_spawn import _dcmotor_actuators  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
USD_ROOTS = {"pilot": REPO / "check/01_sim/usd", "full": Path("/data/EA-Trav/sim/usd")}
POLICY_ROOT = Path("/data/EA-Trav/sim/policies")
VIDEO_DIR = REPO / "check/02_controller/legged"

# 게인 규칙: kp = KP_PER_TAU x 관절별 스탠스 토크 요구 (meta stance_torque — 생성기가
# 기록한 지지 하중 x 모멘트 팔). 근거 = 실로봇 검산: go2 요구 3.7Nm x 5 = 18.5 (스톡
# kp 25), H1 무릎 60 x 5 = 300 (실값 200), ANYmal 30 x 5 = 150 (실값 80) — 전부 동일
# 자릿수. 질량 비례 kp = 2 x m는 소형(15-50kg)의 우연 — 대형 장다리(스탠스 요구
# 200Nm급)에서 평형 처짐 > 0.5 rad로 스폰 붕괴 실측 (에피소드 1s, 질량 순서 심각도).
# stance_torque 없는 관절(팔·구세대 meta)은 질량 비례 폴백.
# kd_ratio 0.025 확정: 0.05 상향 A/B는 기각 — DCMotor에서 감쇠 토크가 포화
# 예산을 잠식해 대조군 go2_proxy 보상 절반(12.5 vs 24.3)·quad_0001 하락(3.1 vs
# 13.7) 실측 (1400-1700 iter 시점 중단). 요동 억제는 감쇠가 아닌 다른 레버 필요
KP_PER_TAU, KP_PER_KG, KP_MIN, KD_RATIO = 5.0, 2.0, 20.0, 0.025


def build_robot_cfg(usd_dir: Path) -> tuple[ArticulationCfg, dict]:
    """변환 USD 폴더(meta.json + joints.json)에서 ArticulationCfg를 만든다.

    스폰 물성·솔버는 스톡 UNITREE_GO2_CFG와 동일 (셀프 충돌 끔, 솔버 4/0),
    액추에이터는 DCMotor(속도-토크 포화 — 스톡 Unitree 표준)에 URDF 한계 사용,
    게인은 스탠스 토크 앵커 규칙 (위 상수 주석). 반환: (cfg, meta).
    """
    meta = json.loads((usd_dir / "meta.json").read_text())
    joints = json.loads((usd_dir / "joints.json").read_text())["joints"]

    # 관절별 게인: 스탠스 요구 앵커, 없으면 질량 비례 폴백
    stance_tau = meta["params"].get("stance_torque", {})
    kp_fallback = max(KP_PER_KG * meta["params"]["total_mass"], KP_MIN)
    kp = {j["name"]: (KP_PER_TAU * stance_tau[j["name"]] if j["name"] in stance_tau
                      else kp_fallback) for j in joints}
    gains = {"stiffness": kp,
             "damping": {n: KD_RATIO * v for n, v in kp.items()}}

    cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_dir / "robot.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False,
                linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4, solver_velocity_iteration_count=0,
            ),
        ),
        # 스폰 = 기립 자세 + 기립고 (+ 약간의 낙하 여유)
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, meta["metrics"]["base_height"] + 0.04),
            joint_pos=dict(meta["standing_pose"]),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators=_dcmotor_actuators(joints, gains, armature=0.0),
    )
    return cfg, meta


def _diag_foot_pairs(contact_links: list[str]) -> tuple:
    """4족 접촉 링크 이름에서 트롯 대각 발 쌍 2개를 유도한다.

    지원 규약: 생성기 leg{row}{l|r}_* (row 최솟값 = 전방 열) / go2 계열 {F|R}{L|R}_*.
    반환: ((전좌, 후우), (전우, 후좌)) — Spot GaitReward synced_feet_pair_names 형식.
    """
    info = {}
    for n in contact_links:
        m = re.match(r"^leg(\d+)([lr])_", n)
        if m:
            info[n] = (int(m.group(1)), m.group(2))
            continue
        m = re.match(r"^([FR])([LR])_", n)
        if m:
            info[n] = (0 if m.group(1) == "F" else 1, m.group(2).lower())
    if len(info) != 4:
        raise SystemExit(f"gait 팔은 4족 이름 규약만 지원 (인식 {len(info)}/4): {contact_links}")
    front = min(r for r, _ in info.values())

    def pick(is_front: bool, side: str) -> str:
        return next(n for n, (r, s) in info.items() if (r == front) == is_front and s == side)

    return ((pick(True, "l"), pick(False, "r")), (pick(True, "r"), pick(False, "l")))


def _joint_power_penalty(env, asset_cfg: SceneEntityCfg):
    """기계 일률 페널티: sum |tau x qdot| [W] — 에너지 최소화 계열의 표준 항.

    에너지 낭비(요동·헛디딤·과토크)가 클수록 커져 자연 걸음을 유도한다
    (Fu et al. "에너지 최소화가 걸음 창발을 이끈다" 계열).
    """
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)


def build_env_cfg(usd_dir: Path, num_envs: int) -> UnitreeGo2RoughEnvCfg:
    """스톡 go2 rough env cfg에 로봇과 로봇 종속 참조만 교체한다.

    로봇 종속 참조 = 베이스 링크(base -> base_link: 높이 스캐너 장착·종료 판정·
    질량 DR·외란 대상)와 발 링크(.*_foot -> meta contact_links: 체공 보상 센서).
    표준 오버라이드 = 발 체공 가중 0.25 + 전진 위주 명령 (go2 대조군과 동일).
    """
    robot_cfg, meta = build_robot_cfg(usd_dir)
    jdata = json.loads((usd_dir / "joints.json").read_text())
    joints = jdata["joints"]
    base = jdata.get("base_link", "base_link")
    cfg = UnitreeGo2RoughEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.robot = robot_cfg
    cfg.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + base
    cfg.terminations.base_contact.params["sensor_cfg"].body_names = base
    cfg.events.add_base_mass.params["asset_cfg"].body_names = base
    cfg.events.base_external_force_torque.params["asset_cfg"].body_names = base
    cfg.rewards.feet_air_time.params["sensor_cfg"].body_names = [f"^{n}$" for n in meta["contact_links"]]
    cfg.rewards.feet_air_time.weight = 0.25
    cfg.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)

    # 토크 페널티 스케일 정규화: 스톡 -2e-4는 go2(한계 23.5Nm) 절대 스케일 튜닝이라
    # 토크 수백 Nm급 로봇에서는 추적 보상을 수십 배 압도한다 (파일럿 실측 — 보상
    # 0 고착). go2 등가 가중 = -2e-4 x (23.5 / 평균 토크 한계)^2 — (tau/한계)^2
    # 정규화와 동치이면서 보상 함수는 스톡 그대로 유지
    mean_effort = sum(float(j["effort"]) for j in joints) / len(joints)
    cfg.rewards.dof_torques_l2.weight = -2.0e-4 * (23.5 / mean_effort) ** 2

    # ---- 보행 품질 실험 팔 (오픈소스 3계열 테스트 — base는 무변경) ----
    if args.arm == "gait":
        # Isaac Lab 내장 Spot 레시피의 걸음새 항 이식 (위상 강제 계열).
        # 가중 스케일 0.3 = go2 추적 가중(1.5) / Spot 추적 가중(5.0) — 레시피 내부
        # 비율 보존. 제외 2항: foot_clearance(발 절대 z 목표 = 평지 전용이라 rough
        # 지형과 불성립), base_orientation(수평 강제 = 경사 정렬과 상충)
        feet = [f"^{n}$" for n in meta["contact_links"]]
        cfg.rewards.gait = RewardTermCfg(
            func=spot_mdp.GaitReward, weight=3.0,
            params={"std": 0.1, "max_err": 0.2, "velocity_threshold": 0.5,
                    "synced_feet_pair_names": _diag_foot_pairs(meta["contact_links"]),
                    "asset_cfg": SceneEntityCfg("robot"),
                    "sensor_cfg": SceneEntityCfg("contact_forces")},
        )
        cfg.rewards.air_time_variance = RewardTermCfg(
            func=spot_mdp.air_time_variance_penalty, weight=-0.3,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=feet)},
        )
        cfg.rewards.foot_slip = RewardTermCfg(
            func=spot_mdp.foot_slip_penalty, weight=-0.15,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=feet),
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=feet),
                    "threshold": 1.0},
        )
        cfg.rewards.base_motion = RewardTermCfg(
            func=spot_mdp.base_motion_penalty, weight=-0.6,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
    elif args.arm == "energy":
        # 에너지 최소화 계열: 가중 앵커 = A1 문헌값 -0.001에 토크 규모 정규화
        # (A1 평균 토크 한계 33.5 Nm 기준 — 토크 페널티와 같은 스케일 불변 원칙)
        cfg.rewards.joint_power = RewardTermCfg(
            func=_joint_power_penalty, weight=-0.001 * (33.5 / mean_effort),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
    return cfg


def train(usd_dir: Path, log_dir: Path):
    """스톡 rsl-rl 학습 루프 (Isaac Lab train.py와 동일 구성) + 최종본만 유지."""
    env_cfg = build_env_cfg(usd_dir, args.num_envs)
    agent_cfg = UnitreeGo2RoughPPORunnerCfg()
    agent_cfg.max_iterations = args.iterations

    log_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 배포용 jit 내보내기 (정규화 포함 여부는 runner 구성 그대로)
    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(log_dir), filename="policy.pt")

    # 무누적 규칙: 중간 체크포인트 삭제, 최종본만 유지
    ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for p in ckpts[:-1]:
        p.unlink()
    print(f"[완료] 정책 저장: {log_dir} (최종 {ckpts[-1].name})", flush=True)
    env.close()


def play(usd_dir: Path, log_dir: Path):
    """고정 명령 시각 확인 (code.md visual check 표준): 전방/횡/계단 비디오 1편.

    명령 고정 = 명령 범위를 한 값으로 + 스폰 yaw 0 + heading 목표 0 (전 로봇
    +x 정렬), 정지 명령 env 0. stairs 씬은 계단 2종만 난이도 1.0으로 남긴다.
    """
    env_cfg = build_env_cfg(usd_dir, num_envs=12)

    # 스톡 Play cfg 관례: 축소 격자 + 커리큘럼 off + 랜덤 레벨 스폰
    tg0 = env_cfg.scene.terrain.terrain_generator
    tg0.num_rows, tg0.num_cols, tg0.curriculum = 4, 4, False
    env_cfg.scene.terrain.max_init_terrain_level = None

    # 카메라 = env 0 로봇 추적 (로봇 크기 비례 거리 — 고정 원점 카메라는
    # 풀 지형에서 로봇이 화면 밖인 것 실측)
    meta = json.loads((usd_dir / "meta.json").read_text())
    s = max(1.0, float(meta["metrics"].get("overall_length", 1.0)))
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (3.0 * s, 3.0 * s, 2.0 * s)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    c = env_cfg.commands.base_velocity
    c.rel_standing_envs = 0.0
    c.ranges.heading = (0.0, 0.0)
    env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    vx, vy = {"fwd": (0.8, 0.0), "lat": (0.0, 0.5), "stairs": (0.8, 0.0)}[args.scene]
    c.ranges.lin_vel_x = (vx, vx)
    c.ranges.lin_vel_y = (vy, vy)
    if args.scene == "stairs":
        tg = env_cfg.scene.terrain.terrain_generator
        for name, sub in tg.sub_terrains.items():
            sub.proportion = 0.5 if "stairs" in name else 0.0
        tg.difficulty_range = (1.0, 1.0)

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    name = usd_dir.name
    # 파일명에 팔 포함 (팔별 병렬 촬영이 같은 이름을 덮어쓰는 사고 방지)
    env = gym.wrappers.RecordVideo(env, video_folder=str(VIDEO_DIR), video_length=args.video_length,
                                   name_prefix=f"{name}__{args.scene}__{args.arm}", disable_logger=True)

    # 정책이 없으면 zero-action 홀드 (PD가 기립 자세 유지하는지 보는 스폰 진단)
    policy_path = log_dir / "policy.pt"
    if policy_path.exists():
        policy = torch.jit.load(str(policy_path)).to(env.unwrapped.device).eval()
    else:
        print("[진단] 정책 없음 -> zero-action 기립 홀드", flush=True)
        n_act = env.unwrapped.action_manager.total_action_dim
        policy = lambda obs: torch.zeros((obs.shape[0], n_act), device=obs.device)  # noqa: E731

    # ---- 보행 품질 지표 (문헌 표준) — 비디오 롤아웃에서 동시 측정 ----
    # 미끄럼 = 접지 중 발 수평 이동 / 몸체 경로, CoT = sum|tau qdot|dt / (m g 경로),
    # 접촉 규칙성 = 체공·접지 시간 분산 (Spot air_time_variance 정의),
    # 대각 비동기율 = 트롯 위상 어긋남, 출렁임 = 수직 속도·roll/pitch 각속도 RMS.
    # 합불 앵커는 go2_proxy 측정값 (상대 기준 — 로봇 크기 무관)
    art = env.unwrapped.scene["robot"]
    sensor = env.unwrapped.scene.sensors["contact_forces"]
    # 접촉 링크 순서를 열 순서로 고정 (find_bodies의 정렬 규약과 무관하게)
    links = meta["contact_links"]
    s_ids = [sensor.find_bodies([f"^{n}$"])[0][0] for n in links]
    b_ids = [art.find_bodies([f"^{n}$"])[0][0] for n in links]
    col = {n: i for i, n in enumerate(links)}
    try:
        p_idx = [(col[a], col[b]) for a, b in _diag_foot_pairs(links)]
    except SystemExit:
        p_idx = None
    dt = env.unwrapped.step_dt
    mass = meta["params"]["total_mass"]
    acc = {"power": 0.0, "slip": 0.0, "path": 0.0, "async": 0.0, "vz2": 0.0, "wxy2": 0.0, "falls": 0}

    obs, _ = env.reset()
    for _ in range(args.video_length):
        with torch.inference_mode():
            obs, _, term, _, _ = env.step(policy(obs["policy"]))
        contact = sensor.data.current_contact_time[:, s_ids] > 0.0
        foot_v = torch.norm(art.data.body_lin_vel_w[:, b_ids, :2], dim=-1)
        acc["slip"] += float((foot_v * contact.float()).mean() * dt)
        acc["power"] += float(torch.abs(art.data.applied_torque * art.data.joint_vel).sum(dim=1).mean() * dt)
        acc["path"] += float(torch.norm(art.data.root_lin_vel_w[:, :2], dim=1).mean() * dt)
        acc["vz2"] += float(torch.square(art.data.root_lin_vel_b[:, 2]).mean())
        acc["wxy2"] += float(torch.square(art.data.root_ang_vel_b[:, :2]).sum(dim=1).mean())
        acc["falls"] += int(term.sum())
        if p_idx:
            # 대각 쌍의 접촉 상태 불일치율 (0 = 완전 동기 트롯)
            xor = sum((contact[:, a] ^ contact[:, b]).float().mean() for a, b in p_idx)
            acc["async"] += float(xor / len(p_idx))
    # 접촉 규칙성: 마지막 스냅샷의 발별 체공·접지 시간 분산 (0.5s 클립 — Spot 정의)
    air_var = float(torch.var(torch.clip(sensor.data.last_air_time[:, s_ids], max=0.5), dim=1).mean()
                    + torch.var(torch.clip(sensor.data.last_contact_time[:, s_ids], max=0.5), dim=1).mean())
    T = args.video_length
    metrics = {
        "falls": acc["falls"],
        "path_m": round(acc["path"], 2),
        "slip_per_m": round(acc["slip"] / max(acc["path"], 1e-6), 4),
        "cot": round(acc["power"] / max(mass * 9.81 * acc["path"], 1e-6), 3),
        "contact_time_var": round(air_var, 4),
        "diag_async": round(acc["async"] / T, 3) if p_idx else None,
        "vz_rms": round((acc["vz2"] / T) ** 0.5, 3),
        "wxy_rms": round((acc["wxy2"] / T) ** 0.5, 3),
    }
    env.close()

    # 지표 저장 (단일 json에 키별 병합 — {로봇}__{씬}__{팔})
    mpath = VIDEO_DIR / "gait_metrics.json"
    allm = json.loads(mpath.read_text()) if mpath.exists() else {}
    allm[f"{name}__{args.scene}__{args.arm}"] = metrics
    mpath.write_text(json.dumps(allm, indent=1, ensure_ascii=False))
    print(f"[지표] {name}__{args.scene}__{args.arm}: {metrics}", flush=True)
    print(f"[완료] 비디오: {VIDEO_DIR}/{name}__{args.scene}*.mp4", flush=True)


def main():
    """진입점: usd 폴더 확인 -> train 또는 play."""
    usd_dir = USD_ROOTS[args.mode] / args.robot
    if not (usd_dir / "robot.usd").exists():
        raise SystemExit(f"USD 없음: {usd_dir} (01_sim 변환 선행 필요)")
    # 실험 팔은 별도 접미사 폴더 (판정 후 승자만 정본으로 승격, 패자 삭제 — 무누적)
    suffix = "" if args.arm == "base" else f"-{args.arm}"
    log_dir = POLICY_ROOT / (args.robot + suffix)
    if args.cmd == "train":
        train(usd_dir, log_dir)
    else:
        play(usd_dir, log_dir)


main()
simulation_app.close()

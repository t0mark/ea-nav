"""멀티 임바디먼트 속도 추종 학습 환경 — rsl-rl VecEnv 구현 (SimEnvironment 재사용).

Isaac 앱 기동 후에만 임포트할 수 있다 (tools/utils/sim.py launch_app 참고).

구성 (legged_gym / Isaac Lab velocity rough 표준 세팅과 정합):
- 지형: TerrainGenerator 격자 (계단·역계단·박스·러프·경사·역경사) 위 스폰 +
  자동 난이도 커리큘럼 — env마다 에피소드 종료 시 "반 칸 이상 걸으면 승급,
  명령 거리의 절반도 못 걸으면 강등"을 지형 임포터가 원점 재배정으로 수행
  (Rudin et al. game-inspired curriculum. 사람 개입·재학습 없음).
  terrain_cfg=None이면 평지 (디버그용 폴백)
- 관측: 고유수용(몸체 속도·중력·관절 상태·직전 액션) + 형태 벡터 +
  높이 스캔 (RayCaster 격자 — 표준 measured heights)
- 보상: 속도 추종 exp + 자세·에너지 페널티 + 발 체공 시간 (ContactSensor)
- 종료: 몸통(루트 링크) 접촉 or 과기울기 or 시간 초과 (time_out 분리)

멀티 임바디먼트: form 1개의 로봇 K종을 spawn_robot_groups로 각 M개 env씩
한 스테이지에 올린다 (전역 N = K x M). PhysX 뷰 제약(동일 DoF)은 그룹별
Articulation·센서로 풀고, 관측·액션은 form 고정 슬롯 규약(low_rl)으로
패딩해 전역 (N, ·) 텐서로 합친다 — 정책 1개가 형태 분포를 커버한다
(GenLoco식, plan 제어기 섹션). 명령 상한은 low_rl.cmd_limits 규칙
(배포와 동일 — 배포 명령이 학습 분포 안에 있도록).
"""
from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from isaaclab.sensors import (ContactSensor, ContactSensorCfg, RayCaster,
                              RayCasterCfg, patterns)

from scripts.sim.controller.legged import low_rl
from scripts.sim.controller.legged.train import rewards
from scripts.sim.utils.environment import SimEnvironment

logger = logging.getLogger(__name__)

# 그룹 프림 이름 규칙 (SimEnvironment.spawn_robot_groups와 동일)
_ENV_NS = "/World/envs"

# 페널티 램프 대상 항 (음수 가중치 성격 — 추적·체공 등 양수 항은 제외)
_PENALTY_TERMS = ("lin_vel_z", "ang_vel_xy", "orientation", "torque",
                  "dof_acc", "action_rate", "undesired", "dof_limits",
                  "joint_dev", "feet_slide")


class LeggedTrainEnv(VecEnv):
    """form 1개의 멀티 임바디먼트 보행 학습 환경 (시뮬·센서·버퍼 상태 보유).

    좌표 규약은 제어기 공통 (core.base): +x 전진, +z 상방, 몸체 좌표 관측.
    rsl-rl 계약: get_observations() / step(actions) -> (obs, rew, dones,
    extras), 관측 그룹은 "policy" 하나 (critic 동일 관측).
    """

    def __init__(self, form: str, robot_dirs: list[tuple[Path, Path]],
                 rl_cfg: dict, sim_cfg: dict, envs_per_robot: int,
                 device: str, seed: int, terrain_cfg: dict | None):
        """로봇 K종 스폰·지형·센서·전역 버퍼를 구성하고 첫 리셋을 수행한다.

        robot_dirs = [(robot.urdf 경로, usd 산출 폴더)] (정적 검사 통과 셋).
        envs_per_robot = 로봇당 env 수 M. seed = 명령·리셋 노이즈 난수 시드.
        terrain_cfg = 지형 설정 (sim.yaml terrain에 rl.yaml 규모 오버레이,
        None = 평지 폴백 — 스캔 관측은 0, 커리큘럼 없음).
        """
        env_cfg = rl_cfg["env"]
        self._rl_cfg = rl_cfg
        self._form = form
        self._physics_dt = float(env_cfg["physics_dt"])
        self._decimation = int(env_cfg["decimation"])
        self._ctrl_dt = self._physics_dt * self._decimation
        self._action_scale = float(env_cfg["action_scale"])
        self._action_clip = float(env_cfg["action_clip"])
        self._rough = terrain_cfg is not None
        # 승급 문턱 = 서브지형 반 칸 (Isaac terrain_levels_vel 규칙과 동일)
        self._curr_up = 0.5 * float(terrain_cfg["size"][0]) if self._rough else 0.0
        self._num_rays = low_rl.scan_rays(rl_cfg["scan"])
        self._scan_clip = float(rl_cfg["scan"]["clip"])

        # rsl-rl VecEnv 계약 속성
        K, M = len(robot_dirs), envs_per_robot
        self.num_envs = K * M
        self.num_actions = low_rl.FORM_SLOTS[form][0] * low_rl.FORM_SLOTS[form][1]
        self.max_episode_length = int(round(float(env_cfg["episode_len_s"])
                                            / self._ctrl_dt))
        self.device = device
        self.cfg = rl_cfg
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long,
                                              device=device)

        # 슬롯 규약·게인 (로봇별) — 학습·배포 공용 규칙 (low_rl)
        self._robot_names = [Path(d).name for _, d in robot_dirs]
        self._slots = [low_rl.build_slots(u, d, rl_cfg["morph"])
                       for u, d in robot_dirs]
        overrides = [low_rl.loco_gain_overrides(s, rl_cfg["gains"])
                     for s in self._slots]

        # 씬 구성: 지형(커리큘럼 원점) 또는 평지 + K 그룹 스폰.
        # 지형 원점은 스트라이드 순열로 배정: Isaac 임포터는 지형 종류 열을
        # env 인덱스 연속 블록으로 나누는데, 그룹(로봇 종)도 연속 구간이라
        # 그대로 쓰면 로봇 종 x 지형 종류가 완전 교락된다 (독립 검증 지적 —
        # full 24종 x 20열이면 로봇 1종 = 지형 1종). 전역 env i = g*M+k <->
        # 임포터 env k*K+g 순열이면 각 로봇이 전 지형 종류를 고르게 본다
        self._env = SimEnvironment(self._physics_dt, device)
        if self._rough:
            self._env.add_terrain(terrain_cfg, num_envs=self.num_envs)
            g_idx = torch.arange(self.num_envs, device=device) // M
            k_idx = torch.arange(self.num_envs, device=device) % M
            self._perm = k_idx * K + g_idx
            spawn_origins = self._env.terrain.env_origins[self._perm]
        else:
            spacing = float(env_cfg["spacing"])
            half = 0.5 * spacing * math.ceil(math.sqrt(self.num_envs)) + spacing
            self._env.add_ground(size=max(20.0, half),
                                 friction=sim_cfg["contact"]["ground_friction"])
            self._perm = None
            spawn_origins = None
        # 발 마찰 명시 (복제 전 원본 바인딩 — 미명시 시 발-지면 유효 마찰
        # 0.5로 걸음이 스케이팅에 기우는 것 실측)
        self._env.spawn_robot_groups(
            [d for _, d in robot_dirs], sim_cfg["drive"], envs_per_robot,
            spacing=float(env_cfg["spacing"]),
            spawn_margin=sim_cfg["sim_run"]["spawn_margin"],
            # 보행 RL 표준 스폰 물리 (rl.yaml env 주석 — 셀프 충돌 끔·솔버 4/0)
            self_collision=bool(env_cfg["self_collision"]),
            gain_overrides_list=overrides, origins=spawn_origins,
            activate_contact_sensors=True,
            friction_links_list=[list(s.contact_links) for s in self._slots],
            friction_links_mu=float(sim_cfg["contact"]["foot_friction"]),
            solver_iters=tuple(env_cfg["solver_iters"]),
            # DCMotor 속도-토크 포화 (스톡 Unitree 표준 — 순응 동역학이
            # 전도·회복 신호의 동역학 축. robot_spawn._dcmotor_actuators)
            actuator_model=str(env_cfg["actuator_model"]))
        # 센서는 reset 전에 만들어야 초기화가 물리 초기화에 묶인다
        # (SceneCamera와 동일 규약)
        self._make_sensors(rl_cfg)
        self._env.reset()

        # 그룹별 상수 텐서 (슬롯 인덱스·마스크·기립 자세·토크 한계·형태 벡터)
        S = self.num_actions
        soft = float(rl_cfg["rewards"]["dof_soft_ratio"])
        self._slot_idx, self._masks, self._defaults = [], [], []
        self._valid, self._valid_idx, self._efforts, self._morphs = [], [], [], []
        self._base_heights, self._feet_body_ids = [], []
        self._soft_lower, self._soft_upper, self._dev_masks = [], [], []
        self._air_thresh = []
        for g, ((urdf, usd_dir), slots) in enumerate(zip(robot_dirs, self._slots)):
            art = self._env.robots[g]
            # 발 링크의 articulation body 인덱스 (미끄럼 페널티의 속도 조회용).
            # 순서를 접촉 센서 body 순서와 일치시켜야 접지 마스크가 정합한다
            # -> 둘 다 find 계열의 순서(스테이지 순서)를 그대로 쓴다
            ids, _ = art.find_bodies([f"^{n}$" for n in slots.contact_links])
            self._feet_body_ids.append(torch.tensor(ids, dtype=torch.long,
                                                    device=device))
            jmap = {n: i for i, n in enumerate(art.joint_names)}
            idx = low_rl.slot_index_tensor(slots, jmap, device)
            mask = torch.tensor(slots.mask, dtype=torch.float32, device=device)
            self._slot_idx.append(idx)
            self._masks.append(mask)
            self._defaults.append(torch.tensor(slots.default, dtype=torch.float32,
                                               device=device))
            self._valid.append(mask > 0)
            self._valid_idx.append(idx[mask > 0])
            self._efforts.append(torch.tensor(slots.effort, dtype=torch.float32,
                                              device=device))
            self._morphs.append(torch.tensor(slots.morph, dtype=torch.float32,
                                             device=device).unsqueeze(0).expand(M, -1))
            # soft 한계 = 가동 범위를 중심 기준 dof_soft_ratio로 좁힌 구간
            lower = torch.tensor(slots.lower, dtype=torch.float32, device=device)
            upper = torch.tensor(slots.upper, dtype=torch.float32, device=device)
            center, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
            self._soft_lower.append(center - soft * half)
            self._soft_upper.append(center + soft * half)
            # 기립 이탈 페널티 대상: humanoid 고관절 yaw/roll (생성기 이름
            # 규약 — 학습 셋은 정본 생성 URDF만 쓰므로 이름 매칭이 안전)
            dev = [1.0 if (n and "hip" in n and ("yaw" in n or "roll" in n))
                   else 0.0 for n in slots.names]
            # 가중치가 켜져 있는데 대상 슬롯이 없으면 조용히 무효가 되므로
            # 경고를 남긴다 (이름 규약 변경 탐지 — 라운드 2 검증 제안)
            if float(rl_cfg["rewards"]["joint_deviation"]) != 0.0 \
                    and sum(dev) == 0:
                logger.warning("그룹 %d: joint_deviation 대상 슬롯 0개 — "
                               "고관절 이름 규약 확인 필요", g)
            self._dev_masks.append(torch.tensor(dev, dtype=torch.float32,
                                                device=device))
            with open(Path(usd_dir) / "meta.json") as f:
                meta = json.load(f)
            self._base_heights.append(float(meta["metrics"]["base_height"]))
            # 체공 문턱은 로봇 크기 비례 (진자 주기 ∝ sqrt(기립고/g)) —
            # 고정 문턱은 소형 로봇의 정상 걸음까지 벌점화 (rl.yaml 주석)
            w = rl_cfg["rewards"]
            thresh = float(w["feet_air_thresh_scale"]) \
                * math.sqrt(float(meta["params"]["stance_height"]) / 9.81)
            self._air_thresh.append(min(max(thresh, float(w["feet_air_thresh_min"])),
                                        float(w["feet_air_thresh_max"])))

        # 로봇별 명령 상한 (N,) — 배포와 같은 규칙 (모듈 docstring)
        v_max = torch.zeros(self.num_envs, device=device)
        for g, (_, usd_dir) in enumerate(robot_dirs):
            with open(Path(usd_dir) / "meta.json") as f:
                params = json.load(f)["params"]
            lim = low_rl.cmd_limits(params, rl_cfg["cmd"])
            v_max[self._env.group_slice(g)] = lim["v_max"]
        self._v_max = v_max
        self._wz_max = float(rl_cfg["cmd"]["wz_max"])
        self._heading_mode = bool(rl_cfg["cmd"].get("heading_command", False))
        self._heading_kp = float(rl_cfg["cmd"].get("heading_kp", 0.5))
        # 명령 속도 커리큘럼 (rl.yaml cmd.curriculum_* 주석): env별 현재
        # 상한 v_lim은 v_start에서 시작, 추적 성적으로 확장. 추적 성적은
        # 에피소드 누적 정규화 추적 보상 / 경과 스텝 (_since_reset 기준 —
        # init_at_random_ep_len의 episode_length_buf 오염과 무관)
        self._v_lim = torch.minimum(
            torch.full_like(v_max, float(rl_cfg["cmd"]["curriculum_v_start"])),
            v_max)
        # 성적 집계는 유효 명령 스텝만 (정지 명령은 기립만으로 추적 1.0이라
        # 무임 확장 — 라운드 4 검증 지적. 채점은 timeout 에피소드 한정)
        self._track_sum = torch.zeros(self.num_envs, device=device)
        self._track_steps = torch.zeros(self.num_envs, dtype=torch.long,
                                        device=device)
        # 에피소드당 보행 거리 집계 (그룹별 — 보행 창발 금본위 지표)
        self._walked_sum = [0.0] * K
        self._walked_n = [0] * K

        # 전역 버퍼 (관측·명령·직전 상태·수익 집계)
        self._obs_buf = torch.zeros(self.num_envs,
                                    low_rl.obs_dim(form, self._num_rays,
                                                   rl_cfg["obs"]),
                                    device=device)
        self._commands = torch.zeros(self.num_envs, 3, device=device)
        self._prev_action = torch.zeros(self.num_envs, S, device=device)
        self._prev_qd = torch.zeros(self.num_envs, S, device=device)
        # 보상용 평균 속도의 기준 위치 (제어 스텝 시작 시점 루트 xy)
        self._prev_root_xy = torch.zeros(self.num_envs, 2, device=device)
        # heading 명령 모드 (스톡 정렬): 목표 방위각 버퍼 + 정지 env 마스크
        # (wz는 매 제어 스텝 P제어로 갱신 — _update_heading_cmd)
        self._heading = torch.zeros(self.num_envs, device=device)
        self._standing = torch.zeros(self.num_envs, dtype=torch.bool,
                                     device=device)
        self._cur_return = torch.zeros(self.num_envs, device=device)
        self._return_hist = deque(maxlen=200)
        self._len_hist = deque(maxlen=200)
        self._term_count = 0
        # 리셋 후 경과 스텝 (종료 유예 판정용 — episode_length_buf는 rsl-rl의
        # init_at_random_ep_len이 랜덤 대입해 유예 판정에 못 쓴다: 검증 지적)
        self._since_reset = torch.zeros(self.num_envs, dtype=torch.long,
                                        device=device)
        # 페널티 램프 카운터 (URMA식 — 초기 탐색 구간의 페널티를 선형 증가)
        self._global_step = 0
        self._rng = torch.Generator(device=device)
        self._rng.manual_seed(seed)
        # 조기 종료 기울기 문턱 (베이스 접촉 판정과 병행 — 접촉 없는 전복도 잡음)
        self._tilt_cos = math.cos(math.radians(float(env_cfg["max_tilt_deg"])))
        self._contact_thresh = float(env_cfg["contact_force_thresh"])
        # 기립고 붕괴 종료 비율 (_group_termination 주석 — 기립 홀드 처짐
        # 0.7x 실측과 붕괴 자세 0.3-0.45x 사이를 가르는 값)
        self._term_height_ratio = float(env_cfg["term_height_ratio"])
        self._only_positive = bool(rl_cfg["rewards"]["only_positive_rewards"])
        # 추적 보상 속도원: false = 순간 몸체 속도 (스톡 정합), true = 제어
        # 주기 평균 속도 (변위/시간 — 진동 위조 방지 변형. 스톡 클론 판정
        # 에서 순간 속도로 복귀: 평균 속도는 낙하·회복 국면에서 액션과의
        # 인과가 흐려져 크레딧 할당이 무뎌진다)
        self._track_avg_vel = bool(rl_cfg["rewards"]["track_avg_vel"])

        # 첫 에피소드 상태: 전 env 리셋 + 명령 샘플 + 관측
        self._reset_envs(torch.arange(self.num_envs, device=device))
        self._compute_obs()
        logger.info("학습 환경 구성: form=%s 로봇 %d종 x env %d = %d envs, "
                    "obs %d (스캔 %d) / act %d, 제어 %.0fHz, 지형 %s",
                    form, K, M, self.num_envs, self._obs_buf.shape[1],
                    self._num_rays, S, 1.0 / self._ctrl_dt,
                    "커리큘럼" if self._rough else "평지")

    def _make_sensors(self, rl_cfg: dict):
        """그룹별 센서 생성: 높이 스캔(RayCaster, 지형 전용) + 발/베이스 접촉.

        접촉 센서는 평지에서도 동작하므로 항상 만든다 (발 체공 보상·베이스
        접촉 종료가 지형/평지에서 동일 규칙). 스캔은 지형 메시가 있어야
        레이캐스트가 성립 — 평지는 관측을 0으로 대체 (low_rl 정규화 규약).
        """
        scan_cfg = rl_cfg["scan"]
        self._scanners: list[RayCaster | None] = []
        self._feet_sensors: list[ContactSensor] = []
        self._base_sensors: list[ContactSensor] = []
        self._mid_sensors: list[ContactSensor] = []
        for g, slots in enumerate(self._slots):
            robot = f"{_ENV_NS}/env_.*/Robot_g{g:03d}"
            if self._rough:
                # yaw 정렬: 격자가 몸체 기울기와 무관하게 수평 유지
                # (표준 measured heights 규약), 시작 고도는 충분히 위에서 하향
                ray_cfg = RayCasterCfg(
                    prim_path=f"{robot}/{slots.root_link}",
                    mesh_prim_paths=["/World/ground"],
                    ray_alignment="yaw",
                    pattern_cfg=patterns.GridPatternCfg(
                        resolution=float(scan_cfg["resolution"]),
                        size=tuple(scan_cfg["size"])),
                    offset=RayCasterCfg.OffsetCfg(
                        pos=(0.0, 0.0, float(scan_cfg["offset_z"]))))
                self._scanners.append(RayCaster(ray_cfg))
            else:
                self._scanners.append(None)
            feet = "|".join(slots.contact_links)
            self._feet_sensors.append(ContactSensor(ContactSensorCfg(
                prim_path=f"{robot}/({feet})", track_air_time=True)))
            # 베이스는 200Hz 갱신(step 본문) x 이력 4틱 = 제어 주기 전체의
            # 접촉을 max 판정 — 순간값 50Hz 샘플은 착지·전도 충격을 놓쳐
            # 종료가 과소 발화 (스톡 velocity 원본의 이력 max 판정 정합)
            self._base_sensors.append(ContactSensor(ContactSensorCfg(
                prim_path=f"{robot}/{slots.root_link}", history_length=4)))
            # 다리 중간 링크(허벅지·정강이 등) 접촉 페널티용 — 무릎 보행 차단
            mid = "|".join(slots.undesired_links)
            self._mid_sensors.append(ContactSensor(ContactSensorCfg(
                prim_path=f"{robot}/({mid})", history_length=3)))

    # ---------- rsl-rl 계약 ----------

    def get_observations(self) -> TensorDict:
        """현재 관측 (그룹 "policy" 하나)."""
        return TensorDict({"policy": self._obs_buf}, batch_size=[self.num_envs],
                          device=self.device)

    def step(self, actions: torch.Tensor):
        """액션 1 제어 스텝 적용 -> (관측, 보상, 종료, extras).

        물리는 decimation회 진행 (위치 목표는 스텝 초에 1회 기록 — PD가
        사이 스텝 동안 같은 목표를 추종). 센서는 제어 스텝당 1회 갱신
        (본문 주석 — 물리 스텝마다는 처리량 붕괴 실측). 보상은 스텝 후
        상태로 계산하고 ctrl_dt를 곱한다 (Isaac 보상 관례).
        """
        actions = torch.clamp(actions.to(self.device),
                              -self._action_clip, self._action_clip)

        # 빈 슬롯 액션 마스킹 — 동역학엔 무영향이지만 마스킹이 없으면
        # (a) prev_action 관측에 노이즈 원본이 유입되고 (b) action_rate
        # 페널티가 빈 슬롯 노이즈에 벌점을 문다 (검증 지적. 배포측
        # controller.py도 동일 마스킹 — 규약 일치)
        for g in range(len(self._env.robots)):
            sl = self._env.group_slice(g)
            actions[sl] *= self._masks[g]

        # 그룹별 관절 목표: 로코모션 슬롯 = 기립 + scale x 액션, 나머지 홀드
        targets = []
        for g, art in enumerate(self._env.robots):
            sl = self._env.group_slice(g)
            slot_target = self._defaults[g] + self._action_scale * actions[sl]
            full = art.data.default_joint_pos.clone()
            full[:, self._valid_idx[g]] = slot_target[:, self._valid[g]]
            targets.append(full)
        # 보상용 평균 속도의 기준 위치 기록 (물리 진행 전 — 스텝 종료 시점의
        # 변위/시간이 이번 제어 주기의 평균 속도가 된다)
        for g, art in enumerate(self._env.robots):
            self._prev_root_xy[self._env.group_slice(g)] = \
                art.data.root_pos_w[:, :2]
        # heading 명령: wz를 목표 방위각 오차의 P제어로 매 스텝 갱신
        # (스톡 velocity_command 규약 — 자기교정 폐루프라 정렬 후 소값)
        if self._heading_mode:
            self._update_heading_cmd()

        # 주기적 베이스 푸시 (legged_gym 원본 기본 규약 — rl.yaml 주석):
        # 전 env 수평 속도에 균등 노이즈를 더해 회복 스텝 경험을 공급
        push_every = max(1, int(round(float(self._rl_cfg["env"]["push_interval_s"])
                                      / self._ctrl_dt)))
        if self._global_step % push_every == 0 and self._global_step > 0:
            pv = float(self._rl_cfg["env"]["push_max_vel"])
            for art in self._env.robots:
                vel = art.data.root_vel_w.clone()
                vel[:, :2] += (torch.rand(vel.shape[0], 2, generator=self._rng,
                                          device=self.device) * 2.0 - 1.0) * pv
                art.write_root_velocity_to_sim(vel)

        for i in range(self._decimation):
            self._env.step_multi(targets if i == 0 else None)
            # 베이스 접촉만 물리 스텝(200Hz)마다 샘플 — 종료 판정의 원천.
            # 50Hz 스냅샷은 착지·전도 충격(1-2 물리 스텝)을 대부분 놓쳐
            # 종료가 과소 발화하는 것 실측 (스톡은 scene.update로 전 센서
            # 200Hz — 우리는 종료에 쓰는 베이스만 좁혀 오버헤드 최소화)
            for sensor in self._base_sensors:
                sensor.update(self._physics_dt)
        # 나머지 센서는 제어 스텝당 1회 갱신 (전 센서 물리 스텝 갱신은
        # 파이썬 오버헤드로 GPU 활용률 붕괴 실측 — 체공 시간 해상도는
        # 제어 주기 20ms로 충분: 문턱 0.5s 대비 2.5%)
        for sensor in (self._feet_sensors + self._mid_sensors):
            sensor.update(self._ctrl_dt)
        for scanner in self._scanners:
            if scanner is not None:
                scanner.update(self._ctrl_dt)
        self.episode_length_buf += 1
        self._since_reset += 1
        self._global_step += 1

        # 종료 판정. 유예(termination_grace_steps)는 리셋 후 경과 카운터
        # 기준 (episode_length_buf는 init_at_random_ep_len이 랜덤 대입해
        # 부적합). 기본 0 = 스톡 정합: 스폰 착지 충격 종료가 "무너진 상태는
        # 즉시 끝"이라는 가치 신호의 원천 — 유예 3스텝이 이를 정확히 억압해
        # 조기 종료가 스톡 대비 소멸하는 것 실측 (code.md)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for g, art in enumerate(self._env.robots):
            terminated[self._env.group_slice(g)] = self._group_termination(g, art)
        grace = int(self._rl_cfg["env"]["termination_grace_steps"])
        if grace > 0:
            terminated &= self._since_reset > grace
        timeout = self.episode_length_buf >= self.max_episode_length
        dones = terminated | timeout

        # 보상 (그룹별 계산 -> 전역 산포). 항별 로그는 GPU 텐서로만 모은다
        # — 스텝마다 float 변환(GPU 동기화) 18회가 처리량을 죽이는 것 실측
        # (rsl-rl 로그는 텐서를 받으면 평균을 알아서 낸다)
        rew = torch.zeros(self.num_envs, device=self.device)
        log_terms: dict[str, torch.Tensor] = {}
        for g, art in enumerate(self._env.robots):
            sl = self._env.group_slice(g)
            rew[sl], terms = self._group_reward(g, art, actions[sl],
                                                terminated[sl])
            # 로그는 항별 전 그룹 평균 (그룹 크기 동일 — 단순 평균 허용)
            for k, v in terms.items():
                log_terms[k] = log_terms.get(k, 0.0) + v / len(self._env.robots)

        # 수익 집계 (완료 에피소드 통계 — ppo의 곡선 기록용)
        self._cur_return += rew
        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            self._return_hist.extend(self._cur_return[done_ids].tolist())
            self._len_hist.extend(self.episode_length_buf[done_ids].tolist())
            self._term_count += int(terminated.sum())
            self._cur_return[done_ids] = 0.0
            self._reset_envs(done_ids, timed_out=timeout[done_ids])

        # 명령 주기 재샘플 (리셋 env는 위에서 이미 새 명령)
        resample_every = max(1, int(round(float(self._rl_cfg["env"]["resample_cmd_s"])
                                          / self._ctrl_dt)))
        tick = (self.episode_length_buf % resample_every == 0) & ~dones
        if tick.any():
            self._sample_commands(tick.nonzero(as_tuple=False).squeeze(-1))

        # 직전 상태 갱신 -> 관측 조립 (리셋 env는 초기화된 버퍼 기준)
        self._prev_action = actions.clone()
        self._prev_action[dones] = 0.0
        self._compute_obs()

        extras = {"time_outs": timeout,
                  "log": {f"/reward/{k}": v for k, v in log_terms.items()}}
        extras["log"]["/curriculum/cmd_v_lim"] = self._v_lim.mean()
        if self._rough:
            extras["log"]["/curriculum/terrain_level"] = \
                self._env.terrain.terrain_levels.float().mean()
        return (self.get_observations(), rew, dones.to(torch.long), extras)

    def reset(self):
        """전 env 강제 리셋 (rsl-rl은 미호출 — 수동 사용 대비 보조 API)."""
        self._reset_envs(torch.arange(self.num_envs, device=self.device))
        self._compute_obs()
        return self.get_observations(), {}

    # ---------- 내부 단계 ----------

    def _penalty_ramp(self) -> float:
        """페널티 가중치 램프 배율 [0,1] (URMA식 커리큘럼).

        초기 탐색 구간에 페널티를 선형 증가시켜, 표준 가중치가 처음부터
        추적 보상을 압도해 클립 0 고착 지대(기울기 소멸)를 만드는 것을
        막는다 (파일럿 붕괴 실측의 구조적 처방 — 가중치 반값 완화를 대체).
        """
        ramp_steps = float(self._rl_cfg["rewards"]["penalty_ramp_iters"]) \
            * float(self._rl_cfg["ppo"]["num_steps_per_env"])
        return min(self._global_step / max(ramp_steps, 1.0), 1.0)

    def _group_reward(self, g: int, art, actions_g: torch.Tensor,
                      terminated_g: torch.Tensor):
        """그룹 g의 보상 (M,)과 항별 평균 로그를 계산한다 (rewards.py 조합).

        음수 가중치 항(페널티)과 종료 감점에는 _penalty_ramp 배율을 곱한다.
        terminated_g = 이번 스텝 조기 종료 (M,) bool (유예 반영 후 값) —
        종료 감점은 하한 0 클립 뒤에 더한다 (rl.yaml termination 주석).
        """
        w = self._rl_cfg["rewards"]
        ramp = self._penalty_ramp()
        sl = self._env.group_slice(g)
        cmd = self._commands[sl]
        vel_b = art.data.root_lin_vel_b
        # 추적 속도원 선택 (__init__ _track_avg_vel 주석): 기본 = 순간 몸체
        # 속도 (스톡). 평균 속도 변형은 변위/시간을 몸체 프레임으로 회전
        if self._track_avg_vel:
            dxy = (art.data.root_pos_w[:, :2] - self._prev_root_xy[sl]) \
                / self._ctrl_dt
            yaw_q = art.data.root_quat_w
            w_, x_, y_, z_ = yaw_q[:, 0], yaw_q[:, 1], yaw_q[:, 2], yaw_q[:, 3]
            yaw = torch.atan2(2.0 * (w_ * z_ + x_ * y_),
                              1.0 - 2.0 * (y_ * y_ + z_ * z_))
            c, s = torch.cos(yaw), torch.sin(yaw)
            vel_track = torch.stack([c * dxy[:, 0] + s * dxy[:, 1],
                                     -s * dxy[:, 0] + c * dxy[:, 1],
                                     torch.zeros_like(yaw)], dim=1)
        else:
            vel_track = vel_b
        ang_b = art.data.root_ang_vel_b
        grav = art.data.projected_gravity_b
        q = low_rl.slot_gather(art.data.joint_pos, self._slot_idx[g],
                               self._masks[g])
        q_err = q - self._defaults[g] * self._masks[g]
        qd = low_rl.slot_gather(art.data.joint_vel, self._slot_idx[g],
                                self._masks[g])
        tau = low_rl.slot_gather(art.data.applied_torque, self._slot_idx[g],
                                 self._masks[g])
        feet = self._feet_sensors[g]
        # 정규화 추적 보상 (0-1)은 명령 커리큘럼의 성적 지표로도 누적한다.
        # 유효 명령 스텝만 집계 — 정지 명령 구간은 기립만으로 1.0이라
        # 성적을 오염시킨다 (라운드 4 검증 지적)
        track_lin = rewards.track_lin_vel_exp(vel_track, cmd,
                                              float(w["track_lin_sigma"]))
        # 채점 하한은 보상 데드밴드보다 높다 — 정지 track이 문턱을 넘는
        # 저명령 무임 대역 폐쇄 (rl.yaml curriculum_score_min_cmd 주석)
        score_min = float(self._rl_cfg["cmd"]["curriculum_score_min_cmd"])
        valid_cmd = torch.norm(cmd[:, :2], dim=1) > score_min
        self._track_sum[sl] += track_lin * valid_cmd
        self._track_steps[sl] += valid_cmd
        terms = {
            "track_lin": float(w["track_lin"]) * track_lin,
            "track_ang": float(w["track_ang"]) * rewards.track_ang_vel_exp(
                ang_b, cmd, float(w["track_ang_sigma"])),
            "lin_vel_z": float(w["lin_vel_z"]) * rewards.lin_vel_z_l2(vel_b),
            "ang_vel_xy": float(w["ang_vel_xy"]) * rewards.ang_vel_xy_l2(ang_b),
            "orientation": float(w["orientation"]) * rewards.flat_orientation_l2(grav),
            "torque": float(w["torque"]) * rewards.torque_ratio_l2(
                tau, self._efforts[g], self._masks[g]),
            "dof_acc": float(w["dof_acc"]) * rewards.dof_acc_l2(
                qd, self._prev_qd[sl], self._ctrl_dt),
            "action_rate": float(w["action_rate"]) * rewards.action_rate_l2(
                actions_g, self._prev_action[sl]),
            # 무릎·정강이 접촉 페널티 — 발 아닌 링크로 걷는 퇴행 차단
            "undesired": float(w["undesired_contact"]) * rewards.undesired_contacts(
                self._mid_sensors[g].data.net_forces_w, self._contact_thresh),
        }
        # 관절 soft 한계 침범 (0 가중치면 생략 — form별 오버라이드)
        if float(w["dof_pos_limits"]) != 0.0:
            terms["dof_limits"] = float(w["dof_pos_limits"]) * rewards.dof_pos_limits(
                q, self._soft_lower[g], self._soft_upper[g], self._masks[g])
        # 고관절 yaw/roll 기립 이탈 (humanoid 오버라이드 — 측방 벌림 억제)
        if float(w["joint_deviation"]) != 0.0:
            terms["joint_dev"] = float(w["joint_deviation"]) \
                * rewards.joint_deviation_l1(q_err, self._dev_masks[g])
        if self._form == "humanoid":
            # 2족: single-stance 보상 (점프·끌기 차단) + 발 미끄럼 페널티.
            # 표준 feet_air_time은 두 발 동시 점프가 치트가 된다 (rewards.py)
            terms["feet_air"] = float(w["feet_air_time_biped"]) \
                * rewards.feet_air_time_biped(
                    feet.data.current_air_time, feet.data.current_contact_time,
                    cmd, float(w["feet_air_threshold_biped"]),
                    float(w["cmd_deadband"]))
            in_contact = torch.norm(feet.data.net_forces_w, dim=-1) \
                > self._contact_thresh
            feet_vel = art.data.body_lin_vel_w[:, self._feet_body_ids[g], :2]
            terms["feet_slide"] = float(w["feet_slide"]) * rewards.feet_slide(
                feet_vel, in_contact)
        else:
            first_contact = feet.compute_first_contact(self._ctrl_dt)
            terms["feet_air"] = float(w["feet_air_time"]) * rewards.feet_air_time(
                feet.data.last_air_time, first_contact, cmd,
                self._air_thresh[g], float(w["cmd_deadband"]))
        self._prev_qd[sl] = qd
        # 페널티 램프: 페널티 성격 항에만 배율 (양수 항 — 추적·체공 — 은 즉시)
        if ramp < 1.0:
            for k in _PENALTY_TERMS:
                if k in terms:
                    terms[k] = terms[k] * ramp
        total = sum(terms.values()) * self._ctrl_dt
        # 총보상 하한 0 클립 (legged_gym only_positive_rewards — 순음수
        # 스텝 보상에서 조기 종료가 이득이 되는 자멸 정책 차단. rl.yaml 주석)
        if self._only_positive:
            total = torch.clamp(total, min=0.0)
        # 종료 감점은 클립 밖 + 램프 미적용 (라운드 2 검증 합의: 전도 비용은
        # 부트스트랩 구간부터 온전해야 기립 기울기가 선다 — H1 표준도 첫
        # 스텝부터 풀 감점. 자멸 유인은 only_positive 클립이 이미 차단)
        term_pen = float(w["termination"]) * terminated_g.float()
        terms["termination"] = term_pen
        return total + term_pen, {k: v.mean() for k, v in terms.items()}

    def _group_termination(self, g: int, art) -> torch.Tensor:
        """그룹 g의 조기 종료 (M,) bool: 베이스 접촉·과기울기·기립고 붕괴.

        베이스 접촉 = 몸통이 지면·장애물에 닿음 (legged_gym 표준 종료) —
        이력 3틱 max 판정 (순간값은 충격성 접촉을 놓쳐 과소 발화. 스톡
        history 규약). 기울기는 접촉 없는 전복(계단 모서리 등)을 보강한다
        (직립에서 gravity_b z = -1, 기울기 t에서 -cos(t)).

        기립고 붕괴 = 베이스 높이(지면 기준)가 기립고 x term_height_ratio
        미만. 무릎·배 깔림 자세는 베이스 접촉도 기울기도 없이 지속 가능해
        (기립 수렴 병리의 은신처 — code.md 리셋 분포 판정) 높이로 자른다.
        지면 고도: 지형 모드 = 스캔 중앙값, 평지 = 0 (원점 격자 z=0).
        """
        hist = self._base_sensors[g].data.net_forces_w_history[:, :, 0]
        base_force = torch.norm(hist, dim=-1).max(dim=1).values
        tilted = art.data.projected_gravity_b[:, 2] > -self._tilt_cos
        if self._rough and self._scanners[g] is not None:
            ground = self._scanners[g].data.ray_hits_w[:, :, 2] \
                .median(dim=1).values
        else:
            ground = 0.0
        collapsed = (art.data.root_pos_w[:, 2] - ground) \
            < self._term_height_ratio * self._base_heights[g]
        return (base_force > self._contact_thresh) | tilted | collapsed

    def _origins_of(self, env_ids: torch.Tensor) -> torch.Tensor:
        """env_ids의 현재 스폰 원점 (n,3) — 지형은 순열 경유 최신값.

        지형 모드는 커리큘럼이 임포터의 env_origins를 제자리 갱신하므로
        항상 임포터에서 순열 인덱스로 읽는다 (스폰 때 만든 복사본은 낡음).
        """
        if self._rough:
            return self._env.terrain.env_origins[self._perm[env_ids]]
        return self._env.origins[env_ids]

    def _reset_envs(self, env_ids: torch.Tensor,
                    timed_out: torch.Tensor | None = None):
        """env 부분 리셋: 커리큘럼 갱신 -> 기립 자세 + 노이즈 + 새 명령.

        timed_out = env_ids별 시간 만료 여부 (None = 채점 없는 강제 리셋).
        커리큘럼(지형 모드): 에피소드가 끝난 env에 대해 걸은 거리로 승급·
        강등을 판정하고 지형 임포터가 원점을 재배정한다 (임포터 인덱스는
        스트라이드 순열 경유 — __init__ 주석).

        리셋 분포는 스톡 velocity 원본 정합 (rl.yaml env 주석): 로코모션
        관절 = 기립각 x 균등 스케일, 루트 xy·yaw 포즈 노이즈 + 6축 속도
        노이즈. 깨끗한 기립 리셋은 전도가 전무해 어드밴티지가 노이즈화
        (양수 surrogate 고착)되는 것 실측 — 역이식 판정의 결함 교정.
        시드 고정 generator로 재현 가능.
        """
        env_cfg = self._rl_cfg["env"]
        cmd_cfg = self._rl_cfg["cmd"]
        js_lo, js_hi = (float(env_cfg["reset_joint_scale"][0]),
                        float(env_cfg["reset_joint_scale"][1]))
        xy_n = float(env_cfg["reset_pos_xy_noise"])
        yn = float(env_cfg["yaw_noise"])

        # 명령 속도 커리큘럼 확장 판정: timeout(완주) 에피소드만, 유효 명령
        # 스텝 평균으로 채점 — legged_gym 원본의 고정 분모 규약과 등가
        # ("질주 후 전도"가 만점이 되는 생존 스텝 평균의 결함 교정 +
        # 정지 명령 무임 확장 차단. 라운드 4 검증 합의)
        if timed_out is not None:
            steps = self._track_steps[env_ids]
            perf = self._track_sum[env_ids] \
                / torch.clamp(steps.float(), min=1.0)
            good = timed_out & (steps > 0) \
                & (perf > float(cmd_cfg["curriculum_track_threshold"]))
            grow = env_ids[good]
            self._v_lim[grow] = torch.minimum(
                self._v_lim[grow] + float(cmd_cfg["curriculum_v_step"]),
                self._v_max[grow])
        self._track_sum[env_ids] = 0.0
        self._track_steps[env_ids] = 0

        for g, art in enumerate(self._env.robots):
            sl = self._env.group_slice(g)
            in_group = (env_ids >= sl.start) & (env_ids < sl.stop)
            if not in_group.any():
                continue
            ids = env_ids[in_group]
            local = ids - sl.start

            # 에피소드당 보행 거리 = 보행 창발의 금본위 지표 (라운드 7 —
            # 지형 레벨·cmd_v_ratio는 위조·잔상 가능, 이 값은 불가.
            # 평지 모드 포함 전 모드 집계)
            walked = torch.norm(art.data.root_pos_w[local, :2]
                                - self._origins_of(ids)[:, :2], dim=1)
            self._walked_sum[g] += float(walked.sum())
            self._walked_n[g] += len(ids)

            # 커리큘럼 판정은 원점 이동 전 위치·직전 명령 기준 (이동 후에는
            # 걸은 거리가 무의미해진다)
            if self._rough:
                required = torch.norm(self._commands[ids, :2], dim=1) \
                    * self.max_episode_length * self._ctrl_dt
                move_up = walked > self._curr_up
                move_down = (walked < 0.5 * required) & ~move_up
                self._env.terrain.update_env_origins(self._perm[ids],
                                                     move_up, move_down)

            # 루트: 기립 스폰 상태 + xy·yaw 포즈 노이즈 (z축 회전 쿼터니언
            # (w,0,0,sin)) + 6축 속도 노이즈 (스톡 reset_root_state_uniform)
            root = art.data.default_root_state[local].clone()
            root[:, :3] += self._origins_of(ids)
            root[:, :2] += (torch.rand(len(local), 2, generator=self._rng,
                                       device=self.device) * 2.0 - 1.0) * xy_n
            yaw = (torch.rand(len(local), generator=self._rng,
                              device=self.device) * 2.0 - 1.0) * yn
            root[:, 3] = torch.cos(0.5 * yaw)
            root[:, 4:6] = 0.0
            root[:, 6] = torch.sin(0.5 * yaw)
            vn = float(env_cfg["reset_vel_noise"])
            root[:, 7:] = (torch.rand(len(local), 6, generator=self._rng,
                                      device=self.device) * 2.0 - 1.0) * vn
            art.write_root_pose_to_sim(root[:, :7], env_ids=local)
            art.write_root_velocity_to_sim(root[:, 7:], env_ids=local)

            # 관절: 기립 자세 x 균등 스케일 (스톡 reset_joints_by_scale —
            # 반쯤 무너진 자세 스폰이 전도·회복 데이터의 원천), soft 한계로
            # 클램프. 비로코모션 관절(팔·머리)은 정확히 홀드
            q = art.data.default_joint_pos[local].clone()
            scale = torch.rand(len(local), len(self._valid_idx[g]),
                               generator=self._rng, device=self.device) \
                * (js_hi - js_lo) + js_lo
            v = self._valid[g]
            q[:, self._valid_idx[g]] = torch.clamp(
                q[:, self._valid_idx[g]] * scale,
                self._soft_lower[g][v], self._soft_upper[g][v])
            art.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=local)
            # 상태 기록 직후 관측 버퍼 동기화 (다음 관측 조립이 새 상태를 읽게)
            art.update(0.0)

            # 접촉 이력(체공 시간) 초기화 — 이전 에피소드가 새어들면 첫
            # 접지에서 가짜 체공 보상이 생긴다
            self._feet_sensors[g].reset(env_ids=local)
            self._base_sensors[g].reset(env_ids=local)
            self._mid_sensors[g].reset(env_ids=local)
            if self._scanners[g] is not None:
                self._scanners[g].reset(env_ids=local)

        self.episode_length_buf[env_ids] = 0
        self._since_reset[env_ids] = 0
        self._prev_action[env_ids] = 0.0
        self._prev_qd[env_ids] = 0.0
        self._sample_commands(env_ids)

    def _sample_commands(self, env_ids: torch.Tensor):
        """env_ids의 속도 명령을 로봇별 상한 안에서 재샘플한다.

        vx = [-back_ratio, 1] x v_max, vy = ±vy_ratio x v_max.
        yaw: heading 모드면 목표 방위각(±π)을 샘플하고 wz는 매 스텝 P제어로
        갱신 (스톡 규약), 아니면 ±wz_max 독립 샘플.
        zero_cmd_prob 확률로 정지 명령 (제자리 기립 능력 학습).
        """
        env_cfg = self._rl_cfg["env"]
        n = len(env_ids)
        u = torch.rand(n, 5, generator=self._rng, device=self.device)
        # 커리큘럼 현재 상한 (능력에 맞는 명령 대역 — 학습 신호 밀도 유지)
        v_max = self._v_lim[env_ids]
        back = float(env_cfg["back_ratio"])
        vy_r = float(env_cfg["vy_ratio"])
        cmd = torch.zeros(n, 3, device=self.device)
        cmd[:, 0] = (u[:, 0] * (1.0 + back) - back) * v_max
        cmd[:, 1] = (u[:, 1] * 2.0 - 1.0) * vy_r * v_max
        if not self._heading_mode:
            cmd[:, 2] = (u[:, 2] * 2.0 - 1.0) * self._wz_max
        standing = u[:, 3] < float(env_cfg["zero_cmd_prob"])
        cmd[standing] = 0.0
        # 미세 수평 명령은 0으로 (legged_gym 원본 규약 — 추적 불가능한
        # 크리프 명령 제거. wz는 유지)
        small = torch.norm(cmd[:, :2], dim=1) \
            < float(self._rl_cfg["cmd"]["min_cmd_norm"])
        cmd[small, :2] = 0.0
        self._commands[env_ids] = cmd
        # heading 목표·정지 마스크 갱신 (정지 env는 wz도 0 유지)
        self._heading[env_ids] = (u[:, 4] * 2.0 - 1.0) * math.pi
        self._standing[env_ids] = standing

    def _update_heading_cmd(self):
        """heading 모드: 전 env의 wz 명령을 방위각 오차 P제어로 갱신한다.

        wz = clip(kp x wrap(목표 - yaw), ±wz_max) — 정렬 후엔 소값이 되는
        자기교정 폐루프 (스톡 velocity_command 규약). 정지 env는 0 유지.
        """
        for g, art in enumerate(self._env.robots):
            sl = self._env.group_slice(g)
            q = art.data.root_quat_w
            yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                              1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            err = self._heading[sl] - yaw
            # 각도 랩 (-π, π]
            err = torch.atan2(torch.sin(err), torch.cos(err))
            wz = torch.clamp(self._heading_kp * err, -self._wz_max, self._wz_max)
            self._commands[sl, 2] = torch.where(self._standing[sl],
                                                torch.zeros_like(wz), wz)

    def _noisy(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """학습 관측 노이즈 (rl.yaml obs_noise — 원시 단위 균등, 스톡 정렬).

        배포는 무노이즈 관측 (스톡 play 규약과 동일) — 학습 전용 부가.
        """
        nc = self._rl_cfg["obs_noise"]
        if not nc["enabled"]:
            return x
        n = (torch.rand(x.shape, generator=self._rng, device=self.device)
             * 2.0 - 1.0) * float(nc[key])
        return x + n

    def _compute_obs(self):
        """전 그룹 관측을 조립해 전역 버퍼에 산포한다 (low_rl 단일 출처)."""
        for g, art in enumerate(self._env.robots):
            sl = self._env.group_slice(g)
            q_err = self._noisy(
                low_rl.slot_gather(art.data.joint_pos, self._slot_idx[g],
                                   self._masks[g])
                - self._defaults[g] * self._masks[g], "joint_pos") \
                * self._masks[g]
            qd = self._noisy(
                low_rl.slot_gather(art.data.joint_vel, self._slot_idx[g],
                                   self._masks[g]), "joint_vel") \
                * self._masks[g]
            if self._scanners[g] is not None:
                scan = low_rl.scan_obs(art.data.root_pos_w[:, 2],
                                       self._scanners[g].data.ray_hits_w[:, :, 2],
                                       self._base_heights[g], self._scan_clip)
            else:
                # 평지 폴백: 정규화 규약상 0 = 기립 높이의 평지 (low_rl)
                scan = torch.zeros(sl.stop - sl.start, self._num_rays,
                                   device=self.device)
            self._obs_buf[sl] = low_rl.assemble_obs(
                self._noisy(art.data.root_lin_vel_b, "lin_vel"),
                self._noisy(art.data.root_ang_vel_b, "ang_vel"),
                self._noisy(art.data.projected_gravity_b, "gravity"),
                self._commands[sl], q_err, qd,
                self._prev_action[sl], self._morphs[g], scan,
                self._rl_cfg["obs"])

    # ---------- 학습 모니터링 ----------

    def per_robot_reached(self) -> dict[str, float]:
        """로봇별 명령 커리큘럼 도달 상한 {이름: 평균 v_lim (m/s)}.

        배포 클램프용 — 전 로봇 평균 도달률은 v_max가 큰 로봇에게 학습
        대역 밖 명령을 만들어 전도시키는 것 실측 (z-점수 진단: 명령 3.5σ
        -> 액션 포화). 번들에 저장되어 로봇별로 클램프한다.
        """
        return {name: round(float(self._v_lim[self._env.group_slice(g)].mean()), 3)
                for g, name in enumerate(self._robot_names)}

    def pop_stats(self) -> dict:
        """직전 구간의 완료 에피소드 통계를 반환하고 카운터를 비운다."""
        stats = {
            "episodes": len(self._return_hist),
            "mean_return": (sum(self._return_hist) / len(self._return_hist))
            if self._return_hist else 0.0,
            "mean_ep_len_s": (sum(self._len_hist) / len(self._len_hist)
                              * self._ctrl_dt) if self._len_hist else 0.0,
            "terminations": self._term_count,
        }
        # 명령 커리큘럼 진행률 (1.0 = 전 env가 로봇별 상한 도달)
        stats["cmd_v_ratio"] = round(
            float((self._v_lim / self._v_max).mean()), 2)
        # 로봇별 에피소드당 평균 보행 거리 (m)
        stats["walked_m"] = {name: round(self._walked_sum[g] / max(self._walked_n[g], 1), 2)
                             for g, name in enumerate(self._robot_names)}
        self._walked_sum = [0.0] * len(self._robot_names)
        self._walked_n = [0] * len(self._robot_names)
        if self._rough:
            stats["terrain_level"] = round(
                float(self._env.terrain.terrain_levels.float().mean()), 2)
        self._return_hist.clear()
        self._len_hist.clear()
        self._term_count = 0
        return stats

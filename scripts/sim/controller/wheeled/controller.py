"""wheeled 제어기 조립: 상위 pure pursuit + 하위(low_ik 또는 low_lqr) 캐스케이드.

- WheeledRobotController  : diff / skid / ackermann / omni (+ wheeled_humanoid
  베이스). 명령 = pure pursuit, 배분 = 역기구학 (바퀴 속도·조향각 목표)
- BalancingRobotController: diff_balancing 전용. 명령 = pure pursuit(unicycle),
  배분 = LQR 균형 토크 (바퀴 드라이브 게인 0 스폰 + effort 목표 전제)

제어 주기: 일반 타입은 물리 스텝의 decimation배 (기본 50Hz)로 compute를
호출하고, balancing은 매 물리 스텝(200Hz) LQR을 돌리되 명령 갱신만
decimation 주기로 늦춘다 (역진자 안정화는 지연에 민감하고, 게인 행렬곱은
스텝당 비용이 무시할 수준이라 가능한 구성).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from ..core.base import (BaseController, ControlObs, JointTargets,
                         RobotCtrlParams)
from ..core.high_pp import PurePursuit
from . import low_ik, low_lqr


def _yaw_rate_limit(A: torch.Tensor, limits: torch.Tensor) -> float:
    """제자리 회전(v=0)에서 바퀴 한계가 허용하는 최대 yaw 속도를 구한다.

    바퀴 속도 = w x A[:,2] 이므로 상한 = min_i (한계_i / |A_i,2|).
    """
    arm = torch.abs(A[:, 2])
    return float(torch.min(limits / torch.clamp(arm, min=1e-6)))


class WheeledRobotController(BaseController):
    """일반 wheeled 제어기 (pure pursuit 명령 + 역기구학 배분)."""

    def __init__(self, params: RobotCtrlParams, joint_names: list[str],
                 default_pose: torch.Tensor, num_envs: int, device: str,
                 cfg: dict, physics_dt: float, *, on_terrain: bool):
        """배분 행렬·명령 경계를 로봇 물성에서 구성하고 pure pursuit을 만든다.

        on_terrain = 지형 롤아웃 여부 (make_controller docstring — 지형에서는
        적응형 yaw 보상을 자동 차단).
        """
        super().__init__(params, joint_names, default_pose, num_envs, device)
        self.decimation = int(cfg["ctrl"]["decimation"])

        # 배분 행렬 (로봇당 1회)과 바퀴 조인트 인덱스
        self._wheel_names, self._A, self._limits = \
            low_ik.build_wheel_matrix(params, device,
                                      skid_yaw_scale=cfg["ctrl"]["skid_yaw_scale"])
        self._wheel_idx = self._index_of(self._wheel_names)

        # 전륜(조향) 구동 바퀴의 (배분 열, 해당 측 조향 조인트): 조향각만큼
        # 굴림 방향이 돌아가므로 속도 목표에 1/cos(조향각) 보정이 필요
        # (배분 행렬은 조향 0 자세 FK 고정이라 이 회전을 모른다)
        self._front_wheels = []
        if params.base_tag == "ackermann":
            for i, wf in enumerate(params.wheels):
                if wf.pos[0] > 0:
                    # 같은 측 조향 조인트 = 조향축 y부호가 바퀴 y부호와 일치
                    # (이름 리터럴 의존 제거)
                    side = max(params.steer_y,
                               key=lambda n: params.steer_y[n] * float(wf.pos[1]))
                    self._front_wheels.append((i, side))

        # 명령 경계: 전진 = 생성기 목표 선속도, yaw는 물성에서 유도.
        # yaw 상한은 절대 캡을 함께 적용 — 바퀴 한계만으로는 (트랙이 좁은
        # 개체에서) 수십 rad/s가 나와 기하 모델과 실물(관성)이 괴리된다
        margin = float(cfg["ctrl"]["limit_margin"])
        v_max = params.max_lin_vel * margin
        w_max = min(_yaw_rate_limit(self._A, self._limits) * margin,
                    float(cfg["ctrl"]["yaw_rate_cap"]))
        # 전도 여유 (상체 무거운 개체): 기하 유도 전도 한계 가속도의 안전율
        # 이내로 가감속·원심(v x w) 가속을 묶는다 — 일률 한계로는 확률적
        # 전도가 남는 것 실측 (개체별 수동 튜닝 없이 meta 기하에서 자동)
        tip = params.tip_accel * float(cfg["ctrl"]["tip_safety"])
        lin_accel = float(cfg["ctrl"]["lin_accel"])
        lat_accel = float(cfg["pp"]["lat_accel"])
        if tip > 0.0:
            lin_accel = min(lin_accel, tip)
            lat_accel = min(lat_accel, tip)
            w_max = min(w_max, tip / max(v_max, 1e-6))
        # 적응 게인 적용 후의 실현 yaw까지 횡가속 상한을 강제하기 위한 저장
        # (compute의 클램프 — 낮은 마찰에서는 초과분이 미끄러져 소산되지만
        # 지형 정합 마찰(1.0)에서는 그대로 횡가속이 되어 전복 실측)
        self._lat_accel = lat_accel

        if params.base_tag == "ackermann":
            # bicycle 명령 = (v, 중심 조향각). 중심각 상한은 조인트 한계가
            # 아니라 "내륜이 조인트 한계에 닿는 중심각"으로 축소한다 —
            # 내륜각은 tan(d_in) = wb*k/(1 - y*k)로 중심각보다 크므로,
            # 조인트 한계를 중심각 상한으로 쓰면 내륜 포화로 실 회전
            # 반경이 모델보다 커져 예측-실물 괴리가 생긴다.
            # k_max = tan(L) / (wb + y_in*tan(L))  (내륜각 = L 역산)
            tan_l = math.tan(params.steer_range)
            y_in = max(abs(y) for y in params.steer_y.values())
            kappa_max = tan_l / (params.wheelbase + y_in * tan_l)
            delta_max = math.atan(params.wheelbase * kappa_max)
            bounds = [[-v_max, v_max], [-delta_max, delta_max]]
            model = "bicycle"
            # 내륜 한계 정합 중심각 기준의 실효 최소 회전 반경
            r_turn = 1.0 / kappa_max
        elif params.holonomic:
            # holonomic 명령 = (vx, vy, w). vy 상한은 vx와 동일 스케일로 두고
            # 초과분은 feasible_scale이 바퀴 한계로 잘라낸다
            bounds = [[-v_max, v_max], [-v_max, v_max], [-w_max, w_max]]
            model = "holonomic"
            r_turn = v_max / w_max
        else:
            bounds = [[-v_max, v_max], [-w_max, w_max]]
            model = "unicycle"
            r_turn = v_max / w_max

        # 진입점의 물성 비례 제한 시간 산정용 (실제 명령 경계 그대로 노출 —
        # meta 기하값·전역 캡 근사보다 정확하다)
        self.nav_limits = {"v_max": v_max, "w_max": w_max, "r_turn": r_turn}

        # 선회 우선 모드의 저속 전진(creep)은 unicycle 전 타입 적용 — 정지
        # 제자리 회전이 접촉 마찰에 잠기는 것은 skid만이 아니라 트랙이 좁은
        # diff에서도 실측됨. 미세 전진이 정지 마찰을 깨고, pivot 가능한
        # 개체에는 반경 수십 cm의 작은 원호가 될 뿐이라 무해
        creep = float(cfg["pp"]["creep_ratio"])
        # 전방 주시 하한은 회전 반경에 비례해 올린다 — 대회전 개체에서
        # 주시 거리 << 회전 반경이면 조향 요구 곡률이 항상 포화해 진동·
        # 재정렬 반복 위험 (pure pursuit 표준 튜닝 관행 L_d ~ R)
        pp_cfg = dict(cfg["pp"])
        pp_cfg["lookahead_min"] = max(
            float(pp_cfg["lookahead_min"]),
            float(pp_cfg["lookahead_turn_ratio"]) * r_turn)
        self._planner = PurePursuit(
            model, torch.tensor(bounds, dtype=torch.float32, device=device),
            pp_cfg, num_envs, device, wheelbase=params.wheelbase,
            min_turn_radius=r_turn if model == "bicycle" else 0.0,
            decel=lin_accel, lat_accel=lat_accel, pivot_creep=creep)

        # 명령 슬루(가감속) 제한: 급가감속 저크가 상체 무거운 개체를
        # 전도시키는 것 실측 -> 제어 주기당 명령 변화량을 제한한다
        # (lin_accel은 위 전도 여유가 이미 반영된 값)
        period = self.decimation * physics_dt
        lin_step = lin_accel * period
        yaw_step = float(cfg["ctrl"]["yaw_accel"]) * period
        if params.base_tag == "ackermann":
            # bicycle 명령 = (v, 조향각): 조향 슬루 = 조향 조인트 속도 한계
            du = [lin_step, params.steer_vel_limit * period]
        elif params.holonomic:
            du = [lin_step, lin_step, yaw_step]
        else:
            du = [lin_step, yaw_step]
        self._du = torch.tensor(du, dtype=torch.float32, device=device)
        self._last_u = torch.zeros(num_envs, len(du), device=device)

        # 적응형 yaw 보상 (diff·skid): 명령 대비 실측 yaw 비율로 차동
        # 배율을 온라인 갱신해 개체별 선회 저항을 흡수한다 — skid의
        # 옆미끄럼만이 아니라 협트랙 diff의 캐스터·관성 저항도 같은
        # "명령-실측 yaw 괴리"로 나타나 광궤도 공전을 만드는 것 실측
        # (표준 skid-steer ICR 적응 기법의 일반화).
        # mecanum은 전도 여유 조건부 — mecanum의 yaw 괴리는 롤러 접촉의
        # 요동(명령 주위 +-2배 진동)이 섞여 있어 배율 증폭이 지터를 키운다.
        # 저상 개체(tip_accel 큼)는 증폭해도 안전하고 보상이 없으면 선회
        # 권한 부족으로 시간 초과, 상체 무거운 개체는 증폭 지터가 전도
        # 경계를 넘는 것 실측 -> 전도 한계로 분기 (둘 다 실측 근거).
        # 지형에서는 장애물 걸림을 저항으로 오인하므로 on_terrain이 자동
        # 차단한다 (config 문서 의존이 아닌 코드 인터록 — 감사 지적)
        mecanum_safe = not params.mecanum_sign \
            or params.tip_accel >= float(cfg["ctrl"]["mecanum_adapt_min_tip"])
        self._yaw_adapt = model == "unicycle" and mecanum_safe \
            and bool(cfg["ctrl"]["slip_adapt_enabled"]) and not on_terrain
        self._slip_gain = torch.ones(num_envs, device=device)
        self._prev_w = torch.zeros(num_envs, device=device)
        self._slip_rate = float(cfg["ctrl"]["slip_adapt_rate"])
        self._slip_min = float(cfg["ctrl"]["slip_gain_min"])
        self._slip_max = float(cfg["ctrl"]["slip_gain_max"])
        self._slip_valid_w = float(cfg["ctrl"]["slip_valid_w"])

        # diff 전용 최소 선회 반경 (m): 2륜+캐스터 협트랙 구성은 안쪽 바퀴가
        # 정지·역회전 근방에 가면 하중이 빠지며 스틱-슬립 교착(구동 바퀴
        # 헛돎 + 차체 동결)이 생기는 것 실측 -> 안쪽 바퀴가 바깥의 절반
        # 이상 속도로 확실히 구르는 반경(트랙 배수 config)까지만 선회 허용
        if params.base_tag == "diff":
            track = 2.0 * max(abs(float(w.pos[1])) for w in params.wheels)
            self._min_turn_r = float(cfg["ctrl"]["diff_turn_radius_tracks"]) * track
        else:
            self._min_turn_r = 0.0

    def reset(self, env_ids: torch.Tensor | None = None):
        """기준선·슬루·적응 보상 상태 초기화 (env_ids = 부분 리셋)."""
        self._planner.reset(env_ids)
        if env_ids is None:
            self._last_u.zero_()
            self._slip_gain.fill_(1.0)
            self._prev_w.zero_()
        else:
            self._last_u[env_ids] = 0.0
            self._slip_gain[env_ids] = 1.0
            self._prev_w[env_ids] = 0.0

    def _update_slip(self, obs: ControlObs):
        """직전 스텝에 실제 적용된 yaw 명령 대비 실측 비율로 배율을 갱신한다.

        기준(_prev_w)은 feasible_scale 이후의 모델 수준 w — 배율 상승이
        바퀴 한계 축소를 부르고 그 감속이 다시 미끄럼으로 오인되는 양의
        되먹임을 차단한다. 정지·미세 명령 구간은 비율이 무의미하므로 건너
        뛴다. 하한 < 1은 정적 배율이 과보상인 개체의 하향 적응 허용.
        """
        valid = self._prev_w.abs() > self._slip_valid_w
        ratio = torch.where(valid,
                            obs.ang_b[:, 2] / torch.where(
                                valid, self._prev_w, torch.ones_like(self._prev_w)),
                            torch.ones_like(self._prev_w))
        # 실측이 명령보다 작으면(비율 < 1) 배율을 올린다 (안전 클램프 포함)
        self._slip_gain = torch.clamp(
            self._slip_gain + self._slip_rate * (1.0 - torch.clamp(ratio, -1.0, 2.0)),
            self._slip_min, self._slip_max)

    def compute(self, obs: ControlObs, goal_xy: torch.Tensor) -> JointTargets:
        """pure pursuit 명령 -> 슬루 제한 -> (조향 배분) -> 바퀴 속도 목표."""
        u = self._planner.plan(obs.pos_xy, obs.yaw, goal_xy)
        u = torch.clamp(u, self._last_u - self._du, self._last_u + self._du)
        self._last_u = u
        pos = self._default_pose.clone()
        steer = {}
        if self._params.base_tag == "ackermann":
            # (v, delta) -> 좌우 조향각 + 등가 yaw 속도 (자전거 모델 곡률)
            v, delta = u[:, 0], u[:, 1]
            omega = v * torch.tan(delta) / self._params.wheelbase
            cmd = torch.stack([v, torch.zeros_like(v), omega], dim=1)
            steer = low_ik.ackermann_steer(delta, self._params)
            for name, angle in steer.items():
                pos[:, self._joint_index[name]] = angle
        elif self._params.holonomic:
            cmd = u
        else:
            w = u[:, 1]
            if self._yaw_adapt:
                # 적응형 yaw 보상: IK에 들어가는 yaw만 배율 (모델 무관).
                # 배율 적용 후에도 실현 횡가속 |v x w|가 전도 한계를 넘지
                # 않게 클램프 (init의 _lat_accel 주석 — 게인 최대 3배가
                # 그대로 실현되면 상체 무거운 개체가 전복하는 것 실측)
                self._update_slip(obs)
                w = w * self._slip_gain
                cap = self._lat_accel / torch.clamp(torch.abs(u[:, 0]), min=0.1)
                w = torch.clamp(w, -cap, cap)
            if self._min_turn_r > 0.0:
                # diff 최소 선회 반경: |w| <= v / R_min (init 주석 참고)
                cap = torch.abs(u[:, 0]) / self._min_turn_r
                w = torch.clamp(w, -cap, cap)
            cmd = torch.stack([u[:, 0], torch.zeros_like(u[:, 0]), w], dim=1)

        # 바퀴 한계 초과분은 (v, vy, w) 통째 축소로 경로 형상을 보존
        cmd = low_ik.feasible_scale(self._A, self._limits, cmd)

        # 모델 수준 명령 (보상 배율 이전) — GT 추종 점수·보상 갱신의 기준
        cmd_model = cmd.clone()
        if self._yaw_adapt:
            cmd_model[:, 2] = cmd[:, 2] / self._slip_gain
            self._prev_w = cmd_model[:, 2].clone()

        speeds = low_ik.wheel_speeds(self._A, cmd)
        # 전륜 구동 바퀴 cos 보정 (init의 _front_wheels 주석 참고). 보정으로
        # 개별 한계를 넘는 몫은 그 바퀴만 클램프 (미세 슬립 허용)
        for col, sname in self._front_wheels:
            scale = 1.0 / torch.clamp(torch.cos(steer[sname]), min=0.5)
            speeds[:, col] = torch.clamp(speeds[:, col] * scale,
                                         -self._limits[col], self._limits[col])
        vel = torch.zeros_like(self._default_pose)
        vel[:, self._wheel_idx] = speeds
        return JointTargets(pos=pos, vel=vel, effort=None, cmd=cmd_model)


class BalancingRobotController(BaseController):
    """balancing 2륜 제어기 (pure pursuit 명령 + LQR 균형 토크)."""

    def __init__(self, params: RobotCtrlParams, joint_names: list[str],
                 default_pose: torch.Tensor, num_envs: int, device: str,
                 cfg: dict, physics_dt: float, urdf_path: Path):
        """URDF에서 역진자 모델·LQR 게인을 유도하고 pure pursuit을 만든다."""
        super().__init__(params, joint_names, default_pose, num_envs, device)
        # 매 물리 스텝 LQR 계산 (모듈 docstring), 명령 갱신만 늦춘다
        self.decimation = 1
        self._plan_every = int(cfg["ctrl"]["decimation"])
        self._calls = 0

        self._lqr = low_lqr.BalancingLQR(urdf_path, params, cfg["lqr"], device)
        self._idx_l = self._joint_index[self._lqr.left_joint]
        self._idx_r = self._joint_index[self._lqr.right_joint]

        # 명령 경계: 균형 여유를 위해 일반 타입보다 보수적으로 잡는다
        A_names, A, limits = low_ik.build_wheel_matrix(params, device)
        ratio = float(cfg["lqr"]["cmd_ratio"])
        v_max = params.max_lin_vel * ratio
        w_max = min(_yaw_rate_limit(A, limits) * ratio,
                    float(cfg["ctrl"]["yaw_rate_cap"]))
        bounds = torch.tensor([[-v_max, v_max], [-w_max, w_max]],
                              dtype=torch.float32, device=device)
        # 감속 가정은 역진자 실측 수준으로 별도 하향 (brake_ratio) — 균형
        # 로봇은 감속 전에 몸을 뒤로 젖혀야 하는 비최소위상 + LQR의 균형
        # 우선 가중치 때문에 유효 감속이 일반 로봇의 절반 이하라, 일반
        # 가정으로는 과속 진입 -> 관통 -> 루프백 재진입이 생긴다 (실측)
        self._planner = PurePursuit(
            "unicycle", bounds, cfg["pp"], num_envs, device,
            decel=float(cfg["ctrl"]["lin_accel"]) * float(cfg["lqr"]["brake_ratio"]),
            lat_accel=float(cfg["pp"]["lat_accel"]) * ratio)
        self._cmd = torch.zeros(num_envs, 2, device=device)
        # 진입점의 물성 비례 제한 시간 산정용 (실제 명령 경계 그대로 노출)
        self.nav_limits = {"v_max": v_max, "w_max": w_max,
                           "r_turn": v_max / max(w_max, 1e-6)}

        # 명령 슬루: 속도 명령 급변이 피치 여기를 키우므로 갱신 주기당
        # 변화량을 제한한다 (일반 타입과 동일 원칙)
        period = self._plan_every * physics_dt
        self._du = torch.tensor(
            [float(cfg["ctrl"]["lin_accel"]) * period,
             float(cfg["ctrl"]["yaw_accel"]) * period],
            dtype=torch.float32, device=device)

    @property
    def balance_model(self) -> low_lqr.BalanceModel:
        """유도된 역진자 모델 (진입점 로그·보고용)."""
        return self._lqr.model

    def reset(self, env_ids: torch.Tensor | None = None):
        """기준선·명령 캐시 초기화 (env_ids = 부분 리셋).

        명령 갱신 카운터는 전 env 공통 시계라 부분 리셋에서는 유지한다.
        """
        self._planner.reset(env_ids)
        if env_ids is None:
            self._cmd.zero_()
            self._calls = 0
        else:
            self._cmd[env_ids] = 0.0

    def compute(self, obs: ControlObs, goal_xy: torch.Tensor) -> JointTargets:
        """LQR 균형 토크를 계산한다 (속도 명령은 주기적으로만 갱신)."""
        if self._calls % self._plan_every == 0:
            new_cmd = self._planner.plan(obs.pos_xy, obs.yaw, goal_xy)
            self._cmd = torch.clamp(new_cmd, self._cmd - self._du,
                                    self._cmd + self._du)
        self._calls += 1

        tau_l, tau_r = self._lqr.wheel_torques(
            pitch=obs.pitch, pitch_rate=obs.ang_b[:, 1], v=obs.vel_b[:, 0],
            yaw_rate=obs.ang_b[:, 2], v_cmd=self._cmd[:, 0], w_cmd=self._cmd[:, 1])
        effort = torch.zeros_like(self._default_pose)
        effort[:, self._idx_l] = tau_l
        effort[:, self._idx_r] = tau_r
        # 몸체 명령 노출 (GT 추종 점수용) — (v, 0, w)
        cmd_model = torch.stack([self._cmd[:, 0],
                                 torch.zeros_like(self._cmd[:, 0]),
                                 self._cmd[:, 1]], dim=1)
        # 바퀴 드라이브 게인은 스폰 시 0 (진입점 gain override) — 토크만 작용
        return JointTargets(pos=self._default_pose.clone(), vel=None,
                            effort=effort, cmd=cmd_model)

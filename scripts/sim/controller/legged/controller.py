"""legged 제어기 조립: 상위 pure pursuit + 하위 RL 정책 (low_rl) 캐스케이드.

wheeled 조립(controller.py)과 동일 구조 — 상위 명령 계층(core.high_pp)이
목표점 -> 몸체 속도 명령을 만들고, 하위 계층이 명령 -> 관절 목표로 배분한다.
legged의 배분은 역기구학 대신 학습된 RL 정책(PolicyBundle)이며, 명령은
정책의 관측으로 들어간다 (속도 추종은 학습된 능력).

제어 주기 = 물리 스텝 x 정책 번들의 decimation (학습 제어 주기와 동일해야
정책이 유효 — 번들 메타로 검증). compute는 진입점이 decimation 경계에서만
호출한다 (02_controller 실행 규약 — 사이 물리 스텝은 직전 목표 유지).

결정성: pure pursuit(난수 없음) + 평균 액션 추론이라 같은 초기 상태·목표면
같은 주행이 나온다 (GT 재현성 — wheeled와 동일 원칙).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from ..core.base import (BaseController, ControlObs, JointTargets,
                         RobotCtrlParams)
from ..core.high_pp import PurePursuit
from . import low_rl

logger = logging.getLogger(__name__)


class LeggedRobotController(BaseController):
    """legged 제어기 (pure pursuit 명령 + RL 정책 배분). quad/hex/humanoid 공용.

    로코모션 관절만 정책이 소유하고, 나머지 position 관절(팔·머리·waist)은
    기립 자세 홀드 (plan 롤아웃 명목 자세 규칙).
    """

    def __init__(self, params: RobotCtrlParams, joint_names: list[str],
                 default_pose: torch.Tensor, num_envs: int, device: str,
                 cfg: dict, physics_dt: float, urdf_path: Path,
                 usd_dir: Path, policy_dir: Path):
        """정책 번들·슬롯 규약을 로드하고 pure pursuit을 구성한다.

        policy_dir = form 정책 폴더 (policy.pt + bundle.json). cfg =
        configs/controller.yaml 전체 (pp·ctrl 상수는 wheeled와 공유).
        """
        super().__init__(params, joint_names, default_pose, num_envs, device)
        self._bundle = low_rl.PolicyBundle.load(policy_dir, device)
        meta = self._bundle.meta

        # 슬롯 규약은 학습과 같은 코드(low_rl.build_slots)로 재구성 — 번들에
        # 저장된 morph 스케일을 써야 형태 벡터가 학습 분포와 정합한다
        self._slots = low_rl.build_slots(urdf_path, usd_dir, meta["morph"])
        if self._slots.form != meta["form"]:
            raise ValueError(f"정책 form 불일치: 번들 {meta['form']} vs "
                             f"로봇 {self._slots.form}")
        self._num_rays = low_rl.scan_rays(meta["scan"])
        if low_rl.obs_dim(self._slots.form, self._num_rays,
                          meta["obs"]) != int(meta["obs_dim"]):
            raise ValueError("관측 차원 불일치 — 번들과 코드의 규약 버전이 다름")
        # 제어 주기는 학습 조건의 일부 — 물리 스텝까지 함께 검증한다
        self.decimation = int(meta["decimation"])
        if abs(float(meta["physics_dt"]) - physics_dt) > 1e-9:
            logger.warning("물리 스텝 불일치: 학습 %.4fs vs 실행 %.4fs — 정책 유효성 저하",
                           float(meta["physics_dt"]), physics_dt)

        # 슬롯 -> articulation 인덱스와 정책 입출력 상수 텐서
        S = self._slots.num_slots
        self._slot_idx = low_rl.slot_index_tensor(self._slots, self._joint_index,
                                                  device)
        self._mask = torch.tensor(self._slots.mask, dtype=torch.float32,
                                  device=device)
        self._slot_default = torch.tensor(self._slots.default,
                                          dtype=torch.float32, device=device)
        self._morph = torch.tensor(self._slots.morph, dtype=torch.float32,
                                   device=device).unsqueeze(0).expand(num_envs, -1)
        self._valid = self._mask > 0
        self._valid_idx = self._slot_idx[self._valid]
        self._action_scale = float(meta["action_scale"])
        self._action_clip = float(meta["action_clip"])
        self._prev_action = torch.zeros(num_envs, S, device=device)

        # 명령 경계: 학습 명령 샘플과 같은 규칙 (번들 저장)에 배포 여유율을
        # 곱한다 — 학습 상한을 상시 명령하면 추적 보상이 희박한 분포 꼬리라
        # 정책이 얼어붙는 것 실측 (rl.yaml cmd.deploy_ratio 주석).
        # 명령 커리큘럼이 상한까지 못 간 번들은 도달률로 추가 제한 —
        # 안 그러면 배포 명령이 다시 학습 분포 밖이 된다 (라운드 4 검증)
        with open(Path(usd_dir) / "meta.json") as f:
            robot_meta = json.load(f)
        lim = low_rl.cmd_limits(robot_meta["params"], meta["cmd"])
        ratio = float(meta["cmd"].get("deploy_ratio", 1.0))
        v_max = ratio * lim["v_max"]
        w_max = ratio * lim["wz_max"]
        # 이 로봇의 커리큘럼 도달 상한으로 추가 클램프 — 전 로봇 평균
        # 도달률은 v_max 큰 로봇에 학습 대역 밖 명령을 만들어 전도시키는
        # 것 실측 (관측 z-점수: 명령 3.5σ -> 액션 포화). 구번들 폴백 =
        # 전역 도달률
        train_info = meta.get("train", {})
        reached_v = train_info.get("reached_v", {}).get(robot_meta["name"])
        if reached_v is not None:
            # 곱 적용 (min이 아님): 도달 상한 자체도 분포 꼬리라 여유율을
            # 곱해 "잘 추종되는 대역"으로 내린다 (라운드 6 — 전역 평균
            # 클램프 0.44 x v_max = z 3.6으로 여전히 분포 밖이었던 실측)
            v_new = ratio * float(reached_v)
            if v_new < v_max:
                logger.warning("배포 상한 %.2f -> %.2f m/s (%.2f x 로봇 도달 %.2f)",
                               v_max, v_new, ratio, float(reached_v))
            v_max = v_new
        else:
            global_ratio = train_info.get("cmd_reached_ratio")
            if global_ratio is not None and float(global_ratio) < ratio:
                logger.warning("명령 커리큘럼 도달률 %.2f < deploy_ratio %.2f — "
                               "배포 상한을 도달률로 제한", float(global_ratio), ratio)
                v_max = float(global_ratio) * lim["v_max"]
                w_max = float(global_ratio) * lim["wz_max"]
        self._cmd_low = torch.tensor([-v_max, 0.0, -w_max], device=device)
        self._cmd_high = torch.tensor([v_max, 0.0, w_max], device=device)
        # 진입점의 물성 비례 제한 시간 산정용 (wheeled와 동일 계약)
        self.nav_limits = {"v_max": v_max, "w_max": w_max,
                           "r_turn": v_max / max(w_max, 1e-6)}

        # 상위 명령: unicycle (전진 + yaw — vy 명령은 배포에서 0 고정).
        # 제자리 선회가 가능하므로 creep 불필요, 후진 재정렬도 없음
        pp_cfg = dict(cfg["pp"])
        bounds = torch.tensor([[-v_max, v_max], [-w_max, w_max]], device=device)
        self._pp = PurePursuit("unicycle", bounds, pp_cfg, num_envs, device,
                               decel=float(cfg["ctrl"]["lin_accel"]),
                               lat_accel=float(pp_cfg["lat_accel"]),
                               pivot_creep=0.0)

    def reset(self, env_ids: torch.Tensor | None = None):
        """추종 기준선과 직전 액션 버퍼를 초기화한다 (env_ids = 부분 리셋)."""
        self._pp.reset(env_ids)
        if env_ids is None:
            self._prev_action.zero_()
        else:
            self._prev_action[env_ids] = 0.0

    def compute(self, obs: ControlObs, goal_xy: torch.Tensor) -> JointTargets:
        """목표점 -> 속도 명령(pp) -> 정책 추론 -> 관절 위치 목표 1스텝."""
        # 상위 명령 (v, w) -> 정책 명령 (vx, 0, wz) — 학습 범위로 클램프
        u = self._pp.plan(obs.pos_xy, obs.yaw, goal_xy)
        cmd = torch.stack([u[:, 0], torch.zeros_like(u[:, 0]), u[:, 1]], dim=1)
        cmd = torch.clamp(cmd, self._cmd_low, self._cmd_high)

        # 관측 조립 -> 평균 액션 (학습과 같은 assemble_obs — 단일 출처 규약).
        # 높이 스캔 미제공(평지 씬) = 0 벡터 — scan_obs 정규화에서 "기립
        # 높이의 평지"와 동일한 값이라 의미가 정확하다 (low_rl docstring)
        q_err = low_rl.slot_gather(obs.joint_pos, self._slot_idx, self._mask) \
            - self._slot_default * self._mask
        qd = low_rl.slot_gather(obs.joint_vel, self._slot_idx, self._mask)
        scan = obs.height_scan if obs.height_scan is not None else \
            torch.zeros(self._num_envs, self._num_rays, device=self._device)
        obs_vec = low_rl.assemble_obs(obs.vel_b, obs.ang_b, obs.gravity_b, cmd,
                                      q_err, qd, self._prev_action, self._morph,
                                      scan, self._bundle.meta["obs"])
        # 클램프 + 빈 슬롯 마스킹 (학습 env와 동일 순서 — prev_action 관측
        # 규약 일치. 마스킹이 없으면 빈 슬롯 값이 관측으로 되먹임된다)
        action = torch.clamp(self._bundle.act(obs_vec),
                             -self._action_clip, self._action_clip) * self._mask
        # 직전 액션 관측은 클램프·마스킹 후 값 — 학습 관측 규약과 동일해야 한다
        self._prev_action = action

        # 로코모션 슬롯 -> 관절 목표, 비로코모션 관절은 기립 자세 홀드
        targets = self._default_pose.clone()
        slot_target = self._slot_default + self._action_scale * action
        targets[:, self._valid_idx] = slot_target[:, self._valid]
        return JointTargets(pos=targets, vel=None, effort=None, cmd=cmd)

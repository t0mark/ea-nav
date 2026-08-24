"""상위 명령 계층: pure pursuit — 목표점 -> 몸체 속도 명령 (결정적).

wheeled·legged 전 form 공통 (plan 제어기 섹션의 "공통 명령 계층").
wheeled는 명령을 역기구학/LQR로 배분하고, legged는 RL 정책의 속도 명령
관측으로 넣는다 — 명령 의미론이 form 간 동일해야 3단계 GT(명령 추종
점수)가 일관된다.

개활지 웨이포인트 주행의 표준 기법. 장애물 회피가 없는 이 연구의 사용
형태(파일럿 평지·3단계 지형 통과 시도)에서는 샘플링 MPC가 불필요하고,
결정적 기하 제어가 재현성(GT 라벨)·궤적 품질·속도 활용 모두에서 낫다.

동작:
- 구간 기준선 = 목표가 바뀐 시점의 위치 -> 목표. 로봇을 기준선에 투영한
  점에서 전방 주시 거리만큼 앞의 주시점(lookahead)을 추종한다 (기준선이
  있어야 횡이탈이 스스로 교정된다 — 목표점 직접 추적은 경로를 자른다)
- pure pursuit 조향 기하: 주시점 방위각 alpha, 명목 주시 거리 L_d에서
  곡률 k = 2 sin(alpha) / L_d. 실거리 L이 아니라 명목 L_d로 나누는 것이
  표준식 — 경로에서 멀리 밀려나도(주시점까지 실거리 증가) 조향 교정력이
  유지된다 (실거리 제곱으로 나누면 이탈할수록 조향이 약해져 목표를
  중심에 둔 광궤도 공전이 생기는 것 실측)
- 속도 프로파일 = min(상한, 정지 프로파일 sqrt(2 a d), 곡률 감속
  sqrt(a_lat/|k|)) — 능력만큼 달리고 목표·코너 앞에서 스스로 감속한다

모델 3종 (명령 공간은 하위 배분 계층과 공유):
- unicycle  (diff/skid/mecanum): u = (v, w), w = v k. 방위 오차가 크면
  선회 우선 — 전 타입 저속 전진(creep) 선회 (정지 제자리 회전이 접촉
  마찰에 잠기는 실측: skid 전반 + 협트랙 diff)
- bicycle   (ackermann): u = (v, delta), delta = atan(wb k). 목표가
  뒤쪽이거나 최소 회전원 안이면 조향 후진 재정렬 (코를 목표 쪽으로
  돌리는 표준 3점 선회 — 방위각 히스테리시스 래치)
- holonomic (omni3/4): u = (vx, vy, w) — 주시점 방향 속도 벡터 +
  진행 방향 정렬 heading P
"""
from __future__ import annotations

import torch


class PurePursuit:
    """pure pursuit 명령 생성기 (로봇 1종·env N개 배치, 결정적).

    상태 = 구간 기준선(목표 변경 시점 위치)·직전 속도(주시 거리 산정용).
    명령 경계는 생성 시점에 로봇 물성에서 받아 고정한다. 좌표는 월드
    평면 (m, rad), 몸체 프레임 +x 전진.
    """

    def __init__(self, model: str, bounds: torch.Tensor, pp_cfg: dict,
                 num_envs: int, device: str, wheelbase: float = 0.0,
                 min_turn_radius: float = 0.0, decel: float = 1.5,
                 lat_accel: float = 2.0, pivot_creep: float = 0.0):
        """model = unicycle|holonomic|bicycle, bounds = (U,2) 채널별 [하한, 상한].

        min_turn_radius = bicycle 후진 재정렬 판정용 (m). decel = 정지
        프로파일 감속 (m/s^2 — 슬루 한계와 정합하도록 호출측이 전달).
        lat_accel = 곡률 감속 횡가속 한계 (m/s^2, 전도 한계 반영값).
        pivot_creep = 선회 우선 모드의 전진 속도 비율 (0 = 제자리 선회 —
        정지 마찰 잠김 실측 때문에 unicycle 전 타입이 > 0을 쓴다).
        """
        self._model = model
        self._bounds = bounds
        self._cfg = pp_cfg
        self._num_envs = num_envs
        self._device = device
        self._wheelbase = wheelbase
        self._min_turn_radius = min_turn_radius
        self._decel = decel
        self._lat_accel = lat_accel
        self._pivot_creep = pivot_creep
        self._seg_start = torch.zeros(num_envs, 2, device=device)
        self._last_goal = torch.full((num_envs, 2), float("nan"), device=device)
        self._last_v = torch.zeros(num_envs, device=device)
        self._reversing = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):
        """구간 기준선·속도·후진 래치를 초기화한다 (env_ids = 부분 리셋).

        기준선은 다음 plan()에서 목표 변경 감지로 다시 잡힌다.
        """
        if env_ids is None:
            self._last_goal.fill_(float("nan"))
            self._last_v.zero_()
            self._reversing.fill_(False)
        else:
            self._last_goal[env_ids] = float("nan")
            self._last_v[env_ids] = 0.0
            self._reversing[env_ids] = False

    def _speed_limit(self, dist_goal: torch.Tensor,
                     kappa: torch.Tensor) -> torch.Tensor:
        """속도 프로파일 (모듈 docstring): 정지·곡률 감속의 최솟값 (N,).

        정지 프로파일은 목표 앞 stop_dist에서 0이 되도록 잡아, 도달 반경
        안 체류(감속 정지)가 성립하게 한다.
        """
        cfg = self._cfg
        v_max = self._bounds[0, 1]
        v_stop = torch.sqrt(2.0 * cfg["decel_margin"] * self._decel
                            * torch.clamp(dist_goal - cfg["stop_dist"], min=0.0))
        v_curv = torch.sqrt(self._lat_accel
                            / torch.clamp(torch.abs(kappa), min=1e-6))
        return torch.minimum(torch.minimum(v_stop, v_curv),
                             torch.full_like(v_stop, float(v_max)))

    def plan(self, pos_xy: torch.Tensor, yaw: torch.Tensor,
             goal_xy: torch.Tensor) -> torch.Tensor:
        """현재 자세 (N,2)·(N,)와 목표점 (N,2)로 이번 스텝 명령 (N,U)을 만든다."""
        cfg = self._cfg

        # 구간 기준선 갱신: 목표가 바뀐 env는 현재 위치가 새 구간의 시작점
        moved = torch.norm(goal_xy - self._last_goal, dim=1) \
            > float(cfg["goal_change_eps"])
        new_seg = moved | torch.isnan(self._last_goal[:, 0])
        self._seg_start = torch.where(new_seg.unsqueeze(1), pos_xy,
                                      self._seg_start)
        self._last_goal = goal_xy.clone()

        # 주시점: 기준선 투영 호길이 + 전방 주시 거리 (목표에서 클램프)
        seg = goal_xy - self._seg_start
        seg_len = torch.clamp(torch.norm(seg, dim=1), min=1e-6)
        seg_dir = seg / seg_len.unsqueeze(1)
        # (텐서 상한 clamp는 torch 버전 의존이라 minimum/maximum으로 통일)
        proj = ((pos_xy - self._seg_start) * seg_dir).sum(dim=1)
        lookahead = torch.minimum(
            torch.maximum(torch.full_like(proj, cfg["lookahead_min"]),
                          self._last_v * cfg["lookahead_time"]),
            seg_len)
        s = torch.minimum(torch.clamp(proj + lookahead, min=0.0), seg_len)
        look = self._seg_start + s.unsqueeze(1) * seg_dir

        # 몸체 프레임 변환 (주시점·목표)과 조향 곡률 k = 2 sin(alpha) / L_d
        # (모듈 docstring — 명목 주시 거리 기준 표준식)
        c, sn = torch.cos(yaw), torch.sin(yaw)
        dl = look - pos_xy
        x_l = c * dl[:, 0] + sn * dl[:, 1]
        y_l = -sn * dl[:, 0] + c * dl[:, 1]
        L = torch.clamp(torch.sqrt(x_l ** 2 + y_l ** 2), min=1e-6)
        kappa = 2.0 * (y_l / L) / lookahead
        dg = goal_xy - pos_xy
        x_g = c * dg[:, 0] + sn * dg[:, 1]
        y_g = -sn * dg[:, 0] + c * dg[:, 1]
        dist_goal = torch.norm(dg, dim=1)

        v = self._speed_limit(dist_goal, kappa)
        # 정지점 안쪽은 완전 정지 (체류 판정·부호 반전 진동 방지)
        parked = dist_goal < cfg["stop_dist"]
        v = torch.where(parked, torch.zeros_like(v), v)

        if self._model == "bicycle":
            u = self._bicycle(v, kappa, x_g, y_g, parked)
        elif self._model == "holonomic":
            u = self._holonomic(v, x_l, y_l, parked)
        else:
            u = self._unicycle(v, kappa, x_g, y_g, parked)
        # 주시 거리 산정용 속도 크기 (holonomic은 vx·vy 합성 — vx만 쓰면
        # 횡·사선 이동에서 주시 거리가 하한으로 붕괴한다)
        if self._model == "holonomic":
            self._last_v = torch.hypot(u[:, 0], u[:, 1])
        else:
            self._last_v = torch.abs(u[:, 0])
        return u

    def _unicycle(self, v, kappa, x_g, y_g, parked):
        """(v, w) 산출: 방위 오차 크면 선회 우선, 아니면 곡률 추종.

        w 경계 내에서 곡률을 보존하기 위해 v를 w_max/|k|로 추가 제한한다
        (바퀴 한계 클램프처럼 경로 형상 왜곡을 막는 원칙).
        """
        cfg = self._cfg
        v_max, w_max = self._bounds[0, 1], self._bounds[1, 1]
        bearing = torch.atan2(y_g, x_g)
        turn_first = (torch.abs(bearing) > cfg["pivot_bearing"]) & ~parked

        v = torch.minimum(v, w_max / torch.clamp(torch.abs(kappa), min=1e-6))
        w = v * kappa
        # 선회 우선: 저속 전진(pivot_creep) 선회 — unicycle 전 타입 공통
        v = torch.where(turn_first, self._pivot_creep * v_max
                        * torch.ones_like(v), v)
        w = torch.where(turn_first,
                        torch.sign(bearing) * cfg["turn_w_ratio"] * w_max, w)
        w = torch.where(parked, torch.zeros_like(w), w)
        return torch.stack([v, w], dim=1)

    def _bicycle(self, v, kappa, x_g, y_g, parked):
        """(v, delta) 산출: 전방이면 pure pursuit, 아니면 조향 후진 재정렬.

        후진 조건 = 목표가 몸체 뒤쪽 또는 최소 회전원 2개(중심 (0, +-R),
        반지름 R) 안 — 전진 원호로는 도달 불가한 기하. 후진하며 코를
        목표 쪽으로 돌려(3점 선회) 목표를 앞·원 밖으로 되돌린다.
        진입/이탈을 방위각 문턱으로 분리한 래치로 판정한다 — 즉시 판정은
        옆 목표 경계에서 전/후진이 매 스텝 뒤집혀 슬루에 걸린 속도가 0
        부근에서 진동하며 정체하는 채터링이 생긴다.
        """
        cfg = self._cfg
        v_max = self._bounds[0, 1]
        delta_max = self._bounds[1, 1]
        R = self._min_turn_radius
        inside_circle = (x_g ** 2 + (torch.abs(y_g) - R) ** 2) < R ** 2
        # 방위각 히스테리시스: 옆 목표(+-90도)는 진입도 이탈도 아니어서
        # 래치가 유지된다 (x 문턱 판정은 옆 목표에서 채터링 실측)
        bearing = torch.atan2(y_g, x_g)
        enter = (torch.abs(bearing) > float(cfg["reverse_enter_bearing"])) \
            | inside_circle
        leave = (torch.abs(bearing) < float(cfg["reverse_exit_bearing"])) \
            & ~inside_circle
        self._reversing = (self._reversing | enter) & ~leave
        reverse = self._reversing & ~parked

        delta = torch.clamp(torch.atan(self._wheelbase * kappa),
                            -delta_max, delta_max)
        # 후진은 코가 목표 쪽으로 돌게 조향 (v < 0에서 yaw율 = -sign(delta)
        # 이므로 delta = -sign(y_g)) — 직선 후진은 전진 전환 슬루 구간에서
        # 음속+조향이 겹쳐 목표를 계속 뒤로 밀어내는 소용돌이 실측
        delta_rev = -torch.sign(y_g) * cfg["turn_w_ratio"] * delta_max
        v = torch.where(reverse, -cfg["reverse_ratio"] * v_max
                        * torch.ones_like(v), v)
        delta = torch.where(reverse, delta_rev, delta)
        delta = torch.where(parked, torch.zeros_like(delta), delta)
        return torch.stack([v, delta], dim=1)

    def _holonomic(self, v, x_l, y_l, parked):
        """(vx, vy, w) 산출: 주시점 방향 속도 벡터 + 진행 방향 정렬 P.

        헤딩 정렬은 GT 관측(전방 카메라)과 자연스러운 주행 자세를 위해
        이동 방향으로 몸체를 돌려 둔다 (홀로노믹이라 이동과 독립).
        """
        cfg = self._cfg
        w_max = self._bounds[2, 1]
        L = torch.clamp(torch.sqrt(x_l ** 2 + y_l ** 2), min=1e-6)
        vx, vy = v * x_l / L, v * y_l / L
        w = torch.clamp(cfg["yaw_kp"] * torch.atan2(y_l, x_l), -w_max, w_max)
        w = torch.where(parked, torch.zeros_like(w), w)
        return torch.stack([vx, vy, w], dim=1)

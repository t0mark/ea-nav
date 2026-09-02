from __future__ import annotations

import torch

class PurePursuit:
    """단일 waypoint 추종기.

    ROS 계열 mobile base controller와 같은 역할 분리를 쓴다. 이 클래스는 로봇 몸체 기준
    속도 명령만 만들고, 바퀴 반지름·트랙·조향 관절 같은 하위 구동계 변환은 각 wheeled
    controller가 맡는다. 경사는 여기서 회피 비용으로 다루지 않는다. 같은 명령을 평지와
    경사에서 모두 내보내고, 성공 여부는 물리·구동계 검증 결과로 판단한다.

    Nav2 Regulated Pure Pursuit와 같은 규약으로 lookahead를 다룬다. 속도에 비례한
    lookahead 거리 L_d = clamp(lookahead_time * v, L_min, L_max)를 잡고, goal까지의
    직선 위에서 그 거리에 있는 carrot을 추종한다. 곡률이 큰(=선회반경이 작은) 명령에는
    속도 regulation을 걸어 미끄러짐과 전복을 줄인다. lookahead 자체는 여전히 해석 가능한
    스칼라 파라미터라, RL은 이 값의 배율만 로봇별로 정한다.
    """

    def __init__(self, model: str, bounds: torch.Tensor, pp_cfg: dict,
                 num_envs: int, device: str, wheelbase: float = 0.0,
                 turn_radius: float = 0.0, decel: float = 1.5,
                 pivot_creep: float = 0.0):

        self._model = model
        self._bounds = bounds
        self._cfg = pp_cfg
        self._num_envs = num_envs
        self._device = device
        self._wheelbase = wheelbase
        # 명목 선회반경. bicycle에서는 물리 최소 선회반경, unicycle 계열에서는
        # v_max/w_max로 정의한 등가 반경이며 곡률 regulation 임계값으로도 쓴다
        self._turn_radius = turn_radius
        self._decel = decel
        self._pivot_creep = pivot_creep
        self._reversing = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):
        """후진 hysteresis 상태를 초기화한다."""

        if env_ids is None:
            self._reversing.fill_(False)
        else:
            self._reversing[env_ids] = False

    def plan(self, pos_xy: torch.Tensor, yaw: torch.Tensor,
             goal_xy: torch.Tensor,
             params: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """현재 위치에서 waypoint까지의 body-frame 명령을 계산한다."""

        params = params or {}
        dx = goal_xy[:, 0] - pos_xy[:, 0]
        dy = goal_xy[:, 1] - pos_xy[:, 1]
        c, s = torch.cos(yaw), torch.sin(yaw)
        x_b = c * dx + s * dy
        y_b = -s * dx + c * dy
        dist = torch.norm(goal_xy - pos_xy, dim=1)
        parked = dist < float(self._cfg["stop_dist"])
        if self._model == "bicycle":
            return self._bicycle(x_b, y_b, dist, parked, params)
        if self._model == "holonomic":
            return self._holonomic(x_b, y_b, dist, parked, params)
        return self._unicycle(x_b, y_b, dist, parked, params)

    def _speed(self, dist: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
        """정지거리식 v^2 = 2as로 waypoint 근처에서만 감속한다."""

        stop = float(self._cfg["stop_dist"])
        margin = float(self._cfg["decel_margin"])
        v_stop = torch.sqrt(2.0 * margin * self._decel
                            * torch.clamp(dist - stop, min=0.0))
        return torch.minimum(v_stop, torch.ones_like(v_stop) * limit)

    def _scale(self, params: dict[str, torch.Tensor], name: str,
               like: torch.Tensor) -> torch.Tensor:
        """RL parameter scale이 있으면 쓰고, 없으면 1.0 tensor를 반환한다."""

        return params.get(name, torch.ones_like(like))

    def _lookahead(self, dist: torch.Tensor, v: torch.Tensor,
                   params: dict[str, torch.Tensor]) -> torch.Tensor:
        """속도 비례 lookahead 거리를 RL 배율까지 반영해 계산한다.

        Nav2 Regulated Pure Pursuit 규약: L_d = clamp(lookahead_time * v, L_min, L_max).
        RL은 그렇게 정해진 L_d에 lookahead_scale 배율을 곱해 로봇별로 늘리거나 줄인다
        (짧으면 민첩하지만 진동하고, 길면 안정적이지만 코너를 크게 돈다). 대역 경계에만
        배율을 걸면 속도 항이 대역 내부에 있는 흔한 구간에서 배율이 아무 효과도 내지 못하므로,
        결과 거리 자체에 곱한다. carrot은 경로 끝인 goal을 넘을 수 없어 남은 거리로 자른다.
        """

        look = torch.clamp(float(self._cfg["lookahead_time"]) * v.abs(),
                           min=float(self._cfg["lookahead_min"]),
                           max=float(self._cfg["lookahead_max"]))
        look = look * self._scale(params, "lookahead_scale", dist)
        return torch.clamp(torch.minimum(look, dist), min=1e-3)

    def _carrot_curvature(self, y_b: torch.Tensor, dist: torch.Tensor,
                          look: torch.Tensor) -> torch.Tensor:
        """carrot을 지나는 원호의 곡률을 계산한다.

        표준 Pure Pursuit 곡률식 kappa = 2 * y_L / L_d^2에서, carrot이 goal 방향 직선 위
        거리 L_d 지점이므로 y_L = y_b * L_d / dist다. 대입하면 kappa = 2 * y_b / (dist * L_d)로
        정리된다. dist 대신 L_d로 나누는 항이 생겨, 멀리 있는 goal에서도 곡률이 0으로
        무너지지 않는다.
        """

        return 2.0 * y_b / (torch.clamp(dist, min=1e-6) * look)

    def _regulated_speed(self, v: torch.Tensor,
                         kappa: torch.Tensor) -> torch.Tensor:
        """선회반경이 기준 반경보다 작으면 그 비율만큼 속도를 줄인다.

        Nav2 Regulated Pure Pursuit의 곡률 regulation과 같은 식이다. 기준 반경을 상수로
        박지 않고 명목 선회반경의 배수(regulated_radius_ratio)로 두는 이유는, 로봇 크기와
        조향 능력에 따라 "급선회"의 절대 반경이 다르기 때문이다. carrot이 가까울수록 곡률이
        커지므로 lookahead 배율이 여기서 속도로 이어진다.
        """

        limit = float(self._cfg["regulated_radius_ratio"]) * self._turn_radius
        if limit <= 1e-6:
            return v
        radius = 1.0 / torch.clamp(kappa.abs(), min=1e-6)
        ratio = torch.clamp(radius / limit,
                            min=float(self._cfg["regulated_min_speed_ratio"]),
                            max=1.0)
        return v * ratio

    def _unicycle(self, x_b, y_b, dist, parked, params):
        """diff/skid 계열 명령 (v, w)을 만든다."""

        v_max, w_max = self._bounds[0, 1], self._bounds[1, 1]
        bearing = torch.atan2(y_b, x_b)
        pivot = float(self._cfg["pivot_bearing"]) * self._scale(
            params, "pivot_bearing_scale", bearing)
        turn_first = (torch.abs(bearing) > pivot) & ~parked
        v = self._speed(dist, v_max)
        look = self._lookahead(dist, v, params)
        v = self._regulated_speed(v, self._carrot_curvature(y_b, dist, look))
        yaw_kp = float(self._cfg["yaw_kp"]) * self._scale(
            params, "yaw_kp_scale", bearing)
        w = torch.clamp(yaw_kp * bearing, -w_max, w_max)
        creep = self._pivot_creep * self._scale(
            params, "creep_ratio_scale", bearing) * v_max
        v = torch.where(turn_first, torch.ones_like(v) * creep, v)
        v = torch.where(parked, torch.zeros_like(v), v)
        w = torch.where(parked, torch.zeros_like(w), w)
        return torch.stack([v, w], dim=1)

    def _bicycle(self, x_b, y_b, dist, parked, params):
        """ackermann 계열 명령 (v, delta)을 만든다."""

        v_max = self._bounds[0, 1]
        delta_max = self._bounds[1, 1]
        bearing = torch.atan2(y_b, x_b)
        inside = (x_b ** 2 + (torch.abs(y_b) - self._turn_radius) ** 2) < self._turn_radius ** 2
        enter = (torch.abs(bearing) > float(self._cfg["reverse_enter_bearing"])) | inside
        leave = (torch.abs(bearing) < float(self._cfg["reverse_exit_bearing"])) & ~inside
        self._reversing = (self._reversing | enter) & ~leave
        reverse = self._reversing & ~parked
        v = self._speed(dist, v_max)
        look = self._lookahead(dist, v, params)
        kappa = self._carrot_curvature(y_b, dist, look)
        delta = torch.clamp(torch.atan(self._wheelbase * kappa), -delta_max, delta_max)
        v = self._regulated_speed(v, kappa)
        reverse_delta = -torch.sign(y_b) * float(self._cfg["turn_w_ratio"]) * delta_max
        reverse_ratio = float(self._cfg["reverse_ratio"]) * self._scale(
            params, "reverse_ratio_scale", bearing)
        v = torch.where(reverse, -reverse_ratio * v_max * torch.ones_like(v), v)
        delta = torch.where(reverse, reverse_delta, delta)
        v = torch.where(parked, torch.zeros_like(v), v)
        delta = torch.where(parked, torch.zeros_like(delta), delta)
        return torch.stack([v, delta], dim=1)

    def _holonomic(self, x_b, y_b, dist, parked, params):
        """omni 계열 명령 (vx, vy, w)을 만든다.

        홀로노믹 베이스는 goal 방향으로 직선 평행이동하므로 추종 경로에 곡률이 없고,
        lookahead carrot을 잡아도 명령이 달라지지 않는다. 그래서 lookahead 계열 파라미터를
        여기서 쓰지 않고, adapter의 omni schema에도 그 이름을 넣지 않는다.
        """

        v_max = self._bounds[0, 1]
        w_max = self._bounds[2, 1]
        L = torch.clamp(torch.sqrt(x_b ** 2 + y_b ** 2), min=1e-6)
        v = self._speed(dist, v_max)
        vx = v * x_b / L
        vy = v * y_b / L
        bearing = torch.atan2(y_b, x_b)
        yaw_kp = float(self._cfg["yaw_kp"]) * self._scale(
            params, "yaw_kp_scale", bearing)
        w = torch.clamp(yaw_kp * bearing, -w_max, w_max)
        vx = torch.where(parked, torch.zeros_like(vx), vx)
        vy = torch.where(parked, torch.zeros_like(vy), vy)
        w = torch.where(parked, torch.zeros_like(w), w)
        return torch.stack([vx, vy, w], dim=1)

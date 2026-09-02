from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ...core.base import ControlObs, RobotCtrlParams
from . import obs as obs_lib
from .bundle import PolicyBundle

# 정책 action 벡터의 고정 배치(layout). controller type마다 해석하는 parameter 개수가
# 다르므로, 벡터 자체는 모든 schema의 합집합(=최대 차원)으로 잡아 padding하고 "이 type이
# 실제로 무엇을 해석하는가"는 ControllerParameterSchema가 정한다. 여기 순서는 저장된 정책
# 번들과의 호환 규약이라 임의로 바꾸면 안 된다.
ACTION_PARAM_NAMES = (
    "drive_speed_scale",
    "lateral_speed_scale",
    "yaw_speed_scale",
    "lin_accel_scale",
    "yaw_accel_scale",
    "yaw_kp_scale",
    "lookahead_scale",
    "pivot_bearing_scale",
    "creep_ratio_scale",
    "reverse_ratio_scale",
    "wheel_yaw_scale",
)
_IDX = {name: i for i, name in enumerate(ACTION_PARAM_NAMES)}

# Pure Pursuit가 tracker 단계에서 직접 읽는 parameter. 나머지는 controller의 속도·가속도
# 제한 단계에서 쓰인다
_TRACKER_PARAMS = ("yaw_kp_scale", "lookahead_scale", "pivot_bearing_scale",
                   "creep_ratio_scale", "reverse_ratio_scale")

@dataclass(frozen=True)
class ControllerParameterSchema:
    """controller type 하나가 해석하는 parameter set 정의.

    RL은 controller action(휠 속도·조향각)을 직접 만들지 않는다. 이 schema에 든 이름은
    모두 기존 controller가 이미 쓰고 있는 해석 가능한 parameter에 곱해지는 배율이고,
    배율 1.0이면 controller 기본 설정 그대로다. schema에 없는 이름은 그 type의 제어
    경로에서 값이 읽히지 않으므로 항상 1.0 no-op으로 고정한다.
    """

    base_tag: str
    names: tuple[str, ...]

    def __post_init__(self):
        """schema 이름이 action layout 안에 있는지, 중복이 없는지 확인한다."""

        unknown = [name for name in self.names if name not in _IDX]
        if unknown:
            raise ValueError(f"{self.base_tag} schema에 없는 parameter 이름: {unknown}")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"{self.base_tag} schema에 중복된 parameter 이름이 있다")

    @property
    def indices(self) -> tuple[int, ...]:
        """schema 이름이 action 벡터에서 차지하는 위치를 반환한다."""

        return tuple(_IDX[name] for name in self.names)

    def contains(self, name: str) -> bool:
        """해당 parameter를 이 controller type이 해석하는지 반환한다."""

        return name in self.names

    def tracker_names(self) -> tuple[str, ...]:
        """이 type에서 Pure Pursuit가 직접 읽는 parameter 이름만 반환한다."""

        return tuple(name for name in self.names if name in _TRACKER_PARAMS)

    def active_mask(self, device: str) -> torch.Tensor:
        """schema에 든 차원만 True인 (1, N) mask를 만든다."""

        values = [name in self.names for name in ACTION_PARAM_NAMES]
        return torch.tensor(values, dtype=torch.bool, device=device).unsqueeze(0)

def _schema(base_tag: str, names: tuple[str, ...]) -> ControllerParameterSchema:
    """이름을 action layout 순서로 정렬해 schema를 만든다."""

    ordered = tuple(name for name in ACTION_PARAM_NAMES if name in set(names))
    return ControllerParameterSchema(base_tag, ordered)

# controller type별 parameter schema. 기준은 하나뿐이다 — "그 type의 제어 경로에서 값이
# 실제로 읽히는가".
#  - diff/skid: bearing P 제어(yaw_kp)로 yaw를 만들고, carrot 곡률로 속도를 regulation하며
#    (lookahead), 제자리 선회 후 전진하는 구간이 있다(pivot_bearing, creep_ratio).
#    skid는 여기에 고정축 다륜이 옆미끄럼으로 도는 실효 선회반경 보정(wheel_yaw)이 붙는다.
#  - ackermann: 조향각이 carrot 곡률에서 바로 나오므로 yaw_kp가 없고, 제자리 선회 대신
#    후진 전환(reverse_ratio)을 쓴다. 횡방향 명령은 0이라 lateral_speed도 없다.
#  - omni: goal 방향으로 직선 평행이동해 경로 곡률이 없으므로 lookahead 계열이 모두 빠지고,
#    body y축 명령이 살아 있어 lateral_speed가 들어간다.
_UNICYCLE_PARAMS = ("drive_speed_scale", "yaw_speed_scale", "lin_accel_scale",
                    "yaw_accel_scale", "yaw_kp_scale", "lookahead_scale",
                    "pivot_bearing_scale", "creep_ratio_scale")
SCHEMAS: dict[str, ControllerParameterSchema] = {
    "diff": _schema("diff", _UNICYCLE_PARAMS),
    "skid": _schema("skid", _UNICYCLE_PARAMS + ("wheel_yaw_scale",)),
    "ackermann": _schema("ackermann", ("drive_speed_scale", "yaw_speed_scale",
                                       "lin_accel_scale", "yaw_accel_scale",
                                       "lookahead_scale", "reverse_ratio_scale")),
    "omni": _schema("omni", ("drive_speed_scale", "lateral_speed_scale",
                             "yaw_speed_scale", "lin_accel_scale",
                             "yaw_accel_scale", "yaw_kp_scale")),
}

def schema_for(base_tag: str) -> ControllerParameterSchema:
    """wheeled base type의 parameter schema를 반환한다."""

    if base_tag not in SCHEMAS:
        raise ValueError(f"지원하지 않는 wheeled base_tag: {base_tag}")
    return SCHEMAS[base_tag]

def schema_names() -> dict[str, list[str]]:
    """controller type별 schema 이름 목록을 설정·번들 기록용으로 반환한다."""

    return {tag: list(schema.names) for tag, schema in SCHEMAS.items()}

def validate_schema_cfg(cfg_schema: dict | None):
    """설정 파일에 적힌 schema가 코드 schema와 같은지 확인한다.

    설정은 문서 겸 검증용이다. 의미는 controller 코드가 정하므로, 설정이 코드와 어긋나면
    조용히 무시하지 않고 여기서 바로 실패시킨다.
    """

    if not cfg_schema:
        return
    code = schema_names()
    if set(cfg_schema) != set(code):
        raise ValueError(f"wheeled_rl.param_schema의 controller 목록이 코드와 다르다: "
                         f"{sorted(cfg_schema)} != {sorted(code)}")
    for tag, names in cfg_schema.items():
        if sorted(names) != sorted(code[tag]):
            raise ValueError(f"wheeled_rl.param_schema[{tag}]가 코드 schema와 다르다: "
                             f"{sorted(names)} != {sorted(code[tag])}")

class WheeledRlAdapter:
    """URDF별 controller parameter 배율을, decision_period_s마다 실시간 상태를 보는 TD3
    정책(Daffan/APPLR·ros_jackal 포팅)으로 정한다.

    두 경로가 이 클래스를 쓴다:
      - 배포/평가 경로(update()): policy_path가 있으면 decision_period_s마다 이 클래스가
        직접 관측을 조립해 actor를 호출하고 배율을 갱신한다. 결정 시점이 아닌 tick은 저장해
        둔 배율을 그대로 쓴다 — Daffan/APPLR의 _take_action이 파라미터를 한 번 설정한 뒤
        time_step초 동안 그대로 두는 것과 같다.
      - 학습 경로(set_action()): WheeledParameterTrainEnv가 TD3 rollout이 낸 action을
        decision마다 외부에서 주입한다. 여기서는 이 클래스가 actor를 호출하지 않는다.

    schema 밖 차원은 매핑 후 1.0으로 덮어써 명시적 no-op으로 만든다. 배율 1.0은 controller
    기본 설정 그대로라는 뜻이라, 정규화 항에서도 penalty가 0이 된다.
    """

    def __init__(self, params: RobotCtrlParams, cfg: dict, num_envs: int,
                 device: str, ctrl_dt: float):

        self._params = params
        self._schema = schema_for(params.base_tag)
        # 배포 경로는 policy_path가 있을 때만 켜진다 (없으면 배율 1.0으로 통과)
        path = cfg.get("policy_path")
        self._bundle = PolicyBundle.load(Path(path), device) if path else None
        # 학습 설정은 정책 번들에 실려 오므로, 번들이 있으면 그 값이 controller 설정을 덮는다
        source = dict(cfg)
        if self._bundle is not None:
            source.update(self._bundle.meta)
        validate_schema_cfg(source.get("param_schema"))
        self._obs_spec = obs_lib.ObsSpec.from_cfg(source.get("obs"))
        if self._bundle is not None:
            self._validate(self._bundle, self._obs_spec, self._schema)
        self._param_min, self._param_max = self._bounds_tensors(source, device)
        self._check_neutral_inside_bounds()
        self._active = self._schema.active_mask(device)
        self._state_cfg = dict(source.get("state", {}))
        morph = obs_lib.morphology_vector(params, device, source.get("morph", {}))
        self._static = self._obs_spec.build_static(params, morph, num_envs)
        self._scales = torch.ones(num_envs, len(ACTION_PARAM_NAMES), device=device)
        self._prev_action = torch.zeros(num_envs, len(ACTION_PARAM_NAMES), device=device)

        # decision_period_s마다만 배포 경로에서 정책을 다시 호출한다. env별로 reset 시점이
        # 다를 수 있어 tick 카운터를 env마다 따로 둔다
        decision_period_s = float(source.get("decision_period_s", 1.0))
        self._decision_ticks = max(1, round(decision_period_s / max(ctrl_dt, 1e-9)))
        self._tick = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def schema(self) -> ControllerParameterSchema:
        """이 로봇 controller type의 parameter schema를 반환한다."""

        return self._schema

    @property
    def active_mask(self) -> torch.Tensor:
        """schema에 든 parameter 차원 mask (1, N)를 반환한다."""

        return self._active

    @property
    def effective_scales(self) -> torch.Tensor:
        """schema 밖 차원이 1.0으로 채워진 현재 env별 배율 (E, N)을 반환한다."""

        return self._scales

    @property
    def param_bounds(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """action 매핑에 쓰는 parameter 하한/상한 (1, N)을 반환한다."""

        return self._param_min, self._param_max

    @property
    def drive_speed_scale(self) -> torch.Tensor:
        """현재 env별 전진 속도 상한 배율을 반환한다."""

        return self._scales[:, _IDX["drive_speed_scale"]]

    @property
    def lateral_speed_scale(self) -> torch.Tensor:
        """현재 env별 횡방향 속도 상한 배율을 반환한다 (omni 외에는 1.0)."""

        return self._scales[:, _IDX["lateral_speed_scale"]]

    @property
    def yaw_speed_scale(self) -> torch.Tensor:
        """현재 env별 yaw 속도 상한 배율을 반환한다."""

        return self._scales[:, _IDX["yaw_speed_scale"]]

    @property
    def lin_accel_scale(self) -> torch.Tensor:
        """현재 env별 선형 가속도 한계 배율을 반환한다."""

        return self._scales[:, _IDX["lin_accel_scale"]]

    @property
    def yaw_accel_scale(self) -> torch.Tensor:
        """현재 env별 yaw 가속도 한계 배율을 반환한다."""

        return self._scales[:, _IDX["yaw_accel_scale"]]

    @property
    def wheel_yaw_scale(self) -> torch.Tensor:
        """휠 yaw 배분 보정 배율을 반환한다 (skid 외에는 1.0)."""

        return self._scales[:, _IDX["wheel_yaw_scale"]]

    def pp_params(self) -> dict[str, torch.Tensor]:
        """Pure Pursuit가 읽는 배율만 schema에 맞춰 골라 반환한다.

        schema 밖 이름은 아예 넘기지 않는다. tracker는 없는 이름을 1.0으로 보므로,
        "이 type이 해석하지 않는 parameter"가 tracker 코드까지 흘러가지 않는다.
        """

        return {name: self._scales[:, _IDX[name]]
                for name in self._schema.tracker_names()}

    def adapt(self, nominal_cmd: torch.Tensor) -> torch.Tensor:
        """현재 유지 중인 controller parameter 배율을 body command에 적용한다."""

        cmd = nominal_cmd.clone()
        cmd[:, 0] = cmd[:, 0] * self._scales[:, _IDX["drive_speed_scale"]]
        cmd[:, 1] = cmd[:, 1] * self._scales[:, _IDX["lateral_speed_scale"]]
        cmd[:, 2] = cmd[:, 2] * self._scales[:, _IDX["yaw_speed_scale"]]
        return cmd

    def reset(self, env_ids: torch.Tensor | None = None):
        """episode 경계에서 직전 action·배율·decision tick 카운터를 되돌린다.

        tick을 0으로 되돌려야 reset 직후 첫 tick에 바로 새 결정을 내린다.
        """

        if env_ids is None:
            self._prev_action.zero_()
            self._scales.fill_(1.0)
            self._tick.zero_()
        else:
            self._prev_action[env_ids] = 0.0
            self._scales[env_ids] = 1.0
            self._tick[env_ids] = 0

    def update(self, obs: ControlObs, goal_xy: torch.Tensor):
        """배포/평가 경로: decision_period_s마다만 actor를 호출해 배율을 갱신한다.

        policy_path가 없으면 아무 것도 하지 않는다. 학습 경로는 이 메서드를 부르지 않고
        set_action()으로 외부 action을 직접 주입한다 (거기서는 env.step() 자체가 이미
        decision 1번 단위라 tick 게이팅이 필요 없다).
        """

        if self._bundle is None:
            self._tick += 1
            return
        due = (self._tick % self._decision_ticks) == 0
        if due.any():
            obs_vec = self._obs_spec.assemble(self._static, obs, goal_xy,
                                              self._prev_action, self._state_cfg)
            action = torch.clamp(self._bundle.act(obs_vec), -1.0, 1.0)
            new_scales = self._scales_from_action(action)
            mask = due.unsqueeze(1)
            self._prev_action = torch.where(mask, action, self._prev_action)
            self._scales = torch.where(mask, new_scales, self._scales)
        self._tick += 1

    def set_action(self, action: torch.Tensor):
        """학습 환경에서 받은 이번 decision의 parameter action을 배율로 저장한다."""

        action = torch.clamp(action.to(self._scales.device), -1.0, 1.0)
        if action.shape != self._scales.shape:
            raise ValueError("wheeled RL parameter action shape이 controller env 수와 맞지 않음")
        self._prev_action = action.clone()
        self._scales.copy_(self._scales_from_action(action))

    def _scales_from_action(self, action: torch.Tensor) -> torch.Tensor:
        """[-1, 1] action을 parameter 범위로 옮기고 schema 밖 차원을 1.0으로 막는다."""

        if self._param_min is None or self._param_max is None:
            raise ValueError("wheeled RL parameter 범위(param_bounds)가 설정에 없다")
        unit = 0.5 * (action + 1.0)
        mapped = self._param_min + unit * (self._param_max - self._param_min)
        return torch.where(self._active, mapped, torch.ones_like(mapped))

    def _check_neutral_inside_bounds(self):
        """배율 1.0(= controller 기본 설정)이 parameter 범위 안에 있는지 확인한다.

        정규화 항이 "기본 설정에서 얼마나 멀어졌는가"를 1.0 기준으로 재기 때문에, 1.0이
        범위 밖이면 어떤 action을 내도 penalty를 피할 수 없어 의미가 뒤틀린다.
        """

        if self._param_min is None or self._param_max is None:
            return
        bad = (self._param_min > 1.0) | (self._param_max < 1.0)
        if bool(bad.any()):
            names = [ACTION_PARAM_NAMES[i]
                     for i in bad.squeeze(0).nonzero().flatten().tolist()]
            raise ValueError(f"wheeled RL parameter 범위가 1.0을 포함하지 않음: {names}")

    @staticmethod
    def _validate(bundle: PolicyBundle, obs_spec: "obs_lib.ObsSpec",
                  schema: ControllerParameterSchema):
        """정책 번들 규약이 현재 관측·schema 코드와 맞는지 확인한다."""

        meta = bundle.meta
        if meta.get("kind") != "wheeled_controller_parameter_policy":
            raise ValueError("wheeled RL 정책 종류가 controller parameter 규약과 맞지 않음")
        if int(meta["obs_dim"]) != obs_spec.dim(len(ACTION_PARAM_NAMES)):
            raise ValueError("wheeled RL 정책 관측 차원이 현재 코드와 맞지 않음")
        if int(meta["num_actions"]) != len(ACTION_PARAM_NAMES):
            raise ValueError("wheeled RL 정책 action 차원이 parameter layout과 맞지 않음")
        if tuple(meta.get("action_param_names", ())) != ACTION_PARAM_NAMES:
            raise ValueError("wheeled RL 정책 action layout 순서가 현재 코드와 맞지 않음")
        trained = meta.get("param_schema", {}).get(schema.base_tag)
        if trained is not None and tuple(trained) != schema.names:
            raise ValueError(f"wheeled RL 정책의 {schema.base_tag} schema가 "
                             f"현재 controller 코드와 맞지 않음")

    @staticmethod
    def _bounds_tensors(cfg: dict,
                        device: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """설정의 name -> [min, max] 표를 action layout 순서의 (1, N) tensor로 만든다."""

        bounds = cfg.get("param_bounds")
        if not bounds:
            return None, None
        missing = [name for name in ACTION_PARAM_NAMES if name not in bounds]
        if missing:
            raise ValueError(f"wheeled_rl.param_bounds에 빠진 parameter: {missing}")
        lo, hi = [], []
        for name in ACTION_PARAM_NAMES:
            low, high = (float(v) for v in bounds[name])
            if low > high:
                raise ValueError(f"wheeled_rl.param_bounds[{name}] 범위가 뒤집혀 있다")
            lo.append(low)
            hi.append(high)
        return (torch.tensor(lo, dtype=torch.float32, device=device).unsqueeze(0),
                torch.tensor(hi, dtype=torch.float32, device=device).unsqueeze(0))

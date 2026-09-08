"""configs/ 아래 yaml 두 종류를 파싱해 값 객체로 만든다.

- RobotProfile: configs/robots/legged/multi-legged/{robot_id}.yaml - 로봇마다 실제로 다른 "물리적
  사실"(usd 경로·바디/관절 이름·액추에이터 수치·기본 자세·reward_weights·reward_type·도메인 랜덤화)만 담는다.
- RLPreset: configs/rl/legged/presets/{preset}.yaml - "학습 설계 선택"(어떤 관측/보상/종료/이벤트/액션
  로직을 쓸지 + PPO 하이퍼파라미터 + 커리큘럼 파라미터). robot yaml이 `rl_preset:`으로 이 파일을 가리킨다.
  4개 로봇이 같은 preset을 가리키면 embodiment 비교가 성립한다(학습 설계 동일).

두 파서 모두 "yaml 한 장 -> frozen dataclass" 로 형태가 같아 한 파일에 둔다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[5]
_ROBOT_CONFIG_ROOT = _REPO_ROOT / "configs" / "robots" / "legged" / "multi-legged"
_RL_PRESET_ROOT = _REPO_ROOT / "configs" / "rl" / "legged" / "presets"


@dataclass
class RLPreset:
    """RL 설계 번들 yaml 한 장(configs/rl/legged/presets/{preset}.yaml)에 대응하는 값 객체.

    observations/rewards/terminations/events/action은 scripts/.../rl/{모듈}.py 레지스트리에 등록된
    이름이고, build_loco_rl_env_cfg()가 그 이름으로 빌더를 골라 조립한다. algorithm/curriculum은
    딕셔너리 그대로 들고 있다가 agent_cfg.build_agent_cfg()·rl_trainer.py의 StageTrainer가 읽는다.
    """

    observations: str
    rewards: str
    terminations: str
    events: str
    action: str
    num_envs: int
    algorithm: dict
    curriculum: dict

    @classmethod
    def load(cls, preset_name: str) -> RLPreset:
        """configs/rl/legged/presets/{preset_name}.yaml을 읽어 RLPreset으로 변환한다."""
        yaml_path = _RL_PRESET_ROOT / f"{preset_name}.yaml"
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        return cls(
            observations=raw["observations"],
            rewards=raw["rewards"],
            terminations=raw["terminations"],
            events=raw["events"],
            action=raw["action"],
            num_envs=raw["num_envs"],
            algorithm=raw["algorithm"],
            curriculum=raw["curriculum"],
        )


@dataclass
class RobotProfile:
    """로봇 yaml 한 장에 대응하는, 학습에 필요한 모든 선언을 담는 값 객체."""

    usd_path: str
    base_body_name: str
    foot_body_names: str
    undesired_contact_body_names: str
    hip_joint_names: str
    # 로봇 USD가 다리 관절 외에 별도 액추에이터가 없는 관절(카메라 페이로드 마운트 등)을 더 갖고
    # 있을 때만 좁힌다 - 물리적 사실이라 로봇마다 다를 수 있는 유일한 관절 선택 필드다.
    controlled_joint_names: str
    default_joint_pos: dict[str, float]
    # default_joint_pos가 "0이 리밋을 벗어나는 관절만 담은 override 목록"인지, "로봇 관절 전체에 대한
    # 기본 자세"인지 - loco_rl_env.py가 나머지 관절에 0.0 와일드카드를 넣을지 이 값으로 정확히 판단한다.
    default_joint_pos_complete: bool
    actuator: dict
    action_scale: float
    domain_randomization: dict
    # "light_quadruped" | "standard_quadruped" - scripts/.../rl/rewards.py의 RewardBuilder 서브클래스
    # 중 어느 걸 쓸지 고르는 물리적 사실(허벅지·정강이 접촉이 정상 보행에서도 잦은 몸집인지).
    reward_type: str
    # 보상 항목별 가중치 - rewards.py는 이 딕셔너리 값만 그대로 쓰고 코드 쪽 기본값을 두지 않는다.
    reward_weights: dict[str, float]
    # 이 로봇이 가리키는 RL 설계 번들 이름(configs/rl/legged/presets/{rl_preset}.yaml).
    rl_preset: str = "baseline"
    # PPO 하이퍼파라미터 중 이 로봇만 preset 기본값과 달라야 하는 키만 담는 override(보통 비어 있음).
    # 예: anymal 계열은 entropy_coef를 낮게 쓴다. build_agent_cfg()가 {**preset.algorithm, **agent}로 병합한다.
    agent: dict = field(default_factory=dict)
    # 스폰 시 베이스 높이(m) - 오픈소스 공식 설정이 쓰는 값이 있는 로봇만 적는다. 비워두면 usd의
    # authored 자세 bbox로 추정하는데(robot_spawn.ground_clearance), 그 추정은 실제 기본 자세가 아니라
    # usd에 저장된 자세 기준이라 웅크린 기본 자세를 쓰는 로봇은 몇 cm 높게 잡혀 리셋마다 낙하한다.
    spawn_height: float | None = None

    @classmethod
    def load(cls, robot_id: str) -> RobotProfile:
        """configs/robots/legged/multi-legged/{robot_id}.yaml을 읽어 RobotProfile로 변환한다."""
        yaml_path = _ROBOT_CONFIG_ROOT / f"{robot_id}.yaml"
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        return cls(
            usd_path=raw["usd_path"],
            base_body_name=raw["base_body_name"],
            foot_body_names=raw["foot_body_names"],
            undesired_contact_body_names=raw["undesired_contact_body_names"],
            hip_joint_names=raw["hip_joint_names"],
            controlled_joint_names=raw.get("controlled_joint_names", ".*"),
            default_joint_pos=raw.get("default_joint_pos") or {},
            default_joint_pos_complete=raw.get("default_joint_pos_complete", False),
            actuator=raw["actuator"],
            action_scale=raw.get("action_scale", 0.25),
            domain_randomization=raw.get("domain_randomization") or {},
            reward_type=raw["reward_type"],
            reward_weights=raw["reward_weights"],
            rl_preset=raw.get("rl_preset", "baseline"),
            agent=raw.get("agent") or {},
            spawn_height=raw.get("spawn_height"),
        )

    def build_joint_pos_cfg(self) -> dict[str, float]:
        """관절 기본 자세 딕셔너리를 만든다 - ArticulationCfg.InitialStateCfg.joint_pos에 그대로 쓴다.

        - default_joint_pos_complete=False(기본값): default_joint_pos는 "0이 리밋을 벗어나는 관절만"
          담은 override 목록이다. 이 경우 나머지 관절은 전부 0.0이어야 하므로, override한 이름을
          제외한 나머지에만 적용되는 부정 전방탐색 정규식을 와일드카드로 쓴다(와일드카드 ".*"와 정확한
          이름을 같은 딕셔너리에 같이 쓰면 "패턴 두 개에 매칭" 에러가 난다).
        - default_joint_pos_complete=True: default_joint_pos가 이미 로봇 관절 전체에 대한 기본
          자세다(Isaac Lab 공식 자세를 그대로 옮긴 경우). 이 경우 위 와일드카드를 추가하면 매칭할
          관절이 하나도 안 남아 "패턴이 아무 것도 매칭 안 함" 에러가 나므로, override 목록을 그대로 쓴다.
        """
        if not self.default_joint_pos:
            return {".*": 0.0}
        if self.default_joint_pos_complete:
            return dict(self.default_joint_pos)
        excluded = "|".join(re.escape(name) for name in self.default_joint_pos)
        return {f"^(?!({excluded})$).*": 0.0, **self.default_joint_pos}

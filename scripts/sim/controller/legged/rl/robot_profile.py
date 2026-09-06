"""로봇 yaml(configs/robots/legged/{category}/{robot_id}.yaml)을 파싱해 RobotProfile로 만든다.

각 필드는 "이 로봇이 무엇인가"(usd_path·base/foot 링크 이름 등, 로봇마다 다른 사실)와 "이 로봇을 어떻게
학습시킬 것인가"(actuator/action/rewards/termination/policy_architecture/algorithm/domain_randomization
축 선택 + agent 하이퍼파라미터, 로봇마다 다른 선택)를 함께 담는다.

quadruped/humanoid라는 로봇 타입으로 후자를 통째로 묶어 상속하는 "프로필 버킷" 방식은 쓰지 않는다 -
go2w(다족)와 tron2a_wf(2족)처럼 실제 오픈소스 세팅이 로봇 타입과 무관하게 같은 축 값을 고르는 경우가
있어, 타입 버킷이 오히려 억지 분류가 되기 때문이다. 각 로봇 yaml은 축 선택을 전부 직접, 완결된
형태로 선언한다(configs/robots/legged/{multi-legged,humanoid}/_template.yaml 참고).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[5]
_ROBOT_CONFIG_ROOT = _REPO_ROOT / "configs" / "robots" / "legged"


@dataclass
class RobotProfile:
    """로봇 yaml 한 장에 대응하는, 학습에 필요한 모든 선언을 담는 값 객체."""

    usd_path: str
    base_body_name: str
    foot_body_names: str
    undesired_contact_body_names: str | None
    default_joint_pos: dict[str, float]
    # default_joint_pos가 "0이 리밋을 벗어나는 관절만 담은 override 목록"(False, usd_export_config.py
    # 자동 생성 기본값)인지, "로봇 관절 전체에 대한 기본 자세"(True, go2/go1처럼 Isaac Lab 공식 자세를
    # 그대로 옮긴 경우)인지 - loco_rl_env.py가 나머지 관절에 0.0 와일드카드를 넣을지 말지 이 값으로
    # 정확히 판단한다(usd를 다시 열어 추론하지 않는다).
    default_joint_pos_complete: bool
    actuator: dict
    action: dict
    rewards: list[dict]
    termination: list  # 항목은 이름 문자열("contact") 또는 파라미터가 있는 딕셔너리({name: orientation, ...})
    policy_architecture: str
    algorithm: str
    domain_randomization: dict
    agent: dict = field(default_factory=dict)

    @classmethod
    def load(cls, category: str, robot_id: str) -> RobotProfile:
        """configs/robots/legged/{category}/{robot_id}.yaml을 읽어 RobotProfile로 변환한다."""
        yaml_path = _ROBOT_CONFIG_ROOT / category / f"{robot_id}.yaml"
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        return cls(
            usd_path=raw["usd_path"],
            base_body_name=raw["base_body_name"],
            foot_body_names=raw["foot_body_names"],
            undesired_contact_body_names=raw.get("undesired_contact_body_names"),
            default_joint_pos=raw.get("default_joint_pos") or {},
            default_joint_pos_complete=raw.get("default_joint_pos_complete", False),
            actuator=raw["actuator"],
            action=raw["action"],
            rewards=raw["rewards"],
            termination=raw.get("termination", ["contact"]),
            policy_architecture=raw.get("policy_architecture", "mlp"),
            algorithm=raw.get("algorithm", "ppo"),
            # 문자열("basic")과 딕셔너리({type: basic, reset_joint_position_range: [1.0, 1.0]}) 둘 다
            # 받는다 - 문자열이면 오버라이드 없는 딕셔너리로 정규화한다
            domain_randomization=cls._normalize_domain_randomization(raw.get("domain_randomization", "basic")),
            agent=raw.get("agent") or {},
        )

    @staticmethod
    def _normalize_domain_randomization(raw_value: str | dict) -> dict:
        """domain_randomization 필드를 항상 {"type": ..., 오버라이드...} 딕셔너리 형태로 통일한다."""
        return raw_value if isinstance(raw_value, dict) else {"type": raw_value}

"""관절 그룹(정규식 패턴)마다 다른 stiffness/damping을 쓰는 액추에이터 축.

H1/G1처럼 hip·knee·ankle·팔 관절의 실제 최대 토크·강성이 서로 크게 달라, 단일 게인 하나로는 발목이
과도하게 뻣뻣해지거나 hip이 물러지는 로봇을 재현한다(공식 리포에서 확인된 패턴).

각 그룹을 실제로 어떤 물리 모델로 구동할지는 그룹별 "model" 키가 결정한다(motor_model.py 참고) -
그룹마다 다른 model을 섞어 쓸 수도 있다(예: 다리는 dc_motor, 팔은 implicit).
"""

from __future__ import annotations

from isaaclab.actuators import ActuatorBaseCfg

from .motor_model import build_actuator_group


def build(actuator_cfg: dict) -> dict[str, ActuatorBaseCfg]:
    """actuator_cfg["groups"](패턴별 설정 리스트)를 그룹 개수만큼의 액추에이터로 만든다."""
    actuators: dict[str, ActuatorBaseCfg] = {}
    for index, group in enumerate(actuator_cfg["groups"]):
        actuators[f"group_{index}"] = build_actuator_group([group["pattern"]], group)
    return actuators

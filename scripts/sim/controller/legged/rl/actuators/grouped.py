"""관절 그룹(정규식 패턴)마다 다른 stiffness/damping을 쓰는 액추에이터 모델.

H1/G1처럼 hip·knee·ankle·팔 관절의 실제 최대 토크·강성이 서로 크게 달라, 단일 게인 하나로는 발목이
과도하게 뻣뻣해지거나 hip이 물러지는 로봇을 재현한다(공식 리포에서 확인된 패턴).
"""

from __future__ import annotations

from isaaclab.actuators import ActuatorBaseCfg, ImplicitActuatorCfg


def build(actuator_cfg: dict) -> dict[str, ActuatorBaseCfg]:
    """actuator_cfg["groups"](패턴별 stiffness/damping 리스트)를 그룹 개수만큼의 ImplicitActuatorCfg로 만든다."""
    actuators: dict[str, ActuatorBaseCfg] = {}
    for index, group in enumerate(actuator_cfg["groups"]):
        actuators[f"group_{index}"] = ImplicitActuatorCfg(
            joint_names_expr=[group["pattern"]], stiffness=group["stiffness"], damping=group["damping"]
        )
    return actuators

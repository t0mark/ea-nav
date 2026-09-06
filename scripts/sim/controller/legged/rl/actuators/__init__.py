"""관절 구동 모델 축 레지스트리 - 로봇 yaml의 actuator.type으로 ArticulationCfg.actuators 딕셔너리를 만든다."""

from __future__ import annotations

from isaaclab.actuators import ActuatorBaseCfg

from . import grouped, simple

_BUILDERS = {"simple": simple.build, "grouped": grouped.build}


def build(actuator_cfg: dict) -> dict[str, ActuatorBaseCfg]:
    """actuator_cfg["type"]에 맞는 빌더를 찾아 ArticulationCfg.actuators에 바로 넣을 딕셔너리를 만든다."""
    actuator_type = actuator_cfg["type"]
    if actuator_type not in _BUILDERS:
        raise KeyError(f"등록되지 않은 actuator 타입: {actuator_type} (사용 가능: {sorted(_BUILDERS)})")
    return _BUILDERS[actuator_type](actuator_cfg)

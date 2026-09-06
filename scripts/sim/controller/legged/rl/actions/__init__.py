"""액션 구성 축 레지스트리 - 관절을 위치제어만 할지, 위치+속도(바퀴-다리 혼합)로 나눌지 로봇 yaml이 고른다."""

from __future__ import annotations

from . import position_only, position_velocity_mixed

_BUILDERS = {"position_only": position_only.build, "position_velocity_mixed": position_velocity_mixed.build}


def build(action_cfg: dict):
    """action_cfg["type"]에 맞는 빌더로 ActionsCfg 인스턴스를 조립한다."""
    action_type = action_cfg["type"]
    if action_type not in _BUILDERS:
        raise KeyError(f"등록되지 않은 action 타입: {action_type} (사용 가능: {sorted(_BUILDERS)})")
    return _BUILDERS[action_type](action_cfg)

"""종료조건 축 레지스트리 - time_out(시간 초과)은 항상 포함되는 기반 조건이라 여기서 다루지 않고
loco_rl_env.py가 직접 추가한다. 이 레지스트리는 그 위에 로봇이 추가로 고르는 조건만 다룬다."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from . import contact, orientation

if TYPE_CHECKING:
    from ..robot_profile import RobotProfile

_REGISTRY = {"contact": contact.base_contact, "orientation": orientation.bad_orientation}


@configclass
class TerminationsCfg:
    """빈 컨테이너 - loco_rl_env.py가 time_out을 먼저 채우고, build()가 나머지를 채운다."""


def build(termination_entries: list, profile: RobotProfile) -> TerminationsCfg:
    """로봇 yaml의 termination 리스트를 레지스트리에서 찾아 TerminationsCfg에 채운다.

    각 항목은 이름 문자열("contact")이거나 파라미터가 필요한 딕셔너리({name: orientation,
    limit_angle: 0.7})일 수 있다.
    """
    cfg = TerminationsCfg()
    for entry in termination_entries:
        params = {} if isinstance(entry, str) else {k: v for k, v in entry.items() if k != "name"}
        name = entry if isinstance(entry, str) else entry["name"]
        if name not in _REGISTRY:
            raise KeyError(f"등록되지 않은 종료조건: {name} (사용 가능: {sorted(_REGISTRY)})")
        setattr(cfg, name, _REGISTRY[name](profile, **params))
    return cfg

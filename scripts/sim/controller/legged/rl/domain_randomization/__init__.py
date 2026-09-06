"""도메인 랜덤화 범위 축 레지스트리 - basic(기존 동작)과 extended(액추에이터 게인·링크별 질량까지) 중
로봇 yaml이 고른다.

로봇 yaml의 domain_randomization 필드는 문자열("basic")이나 딕셔너리({type: basic,
reset_joint_position_range: [1.0, 1.0]}) 둘 다 받는다 - robot_profile.py가 문자열을
{"type": "basic"}로 정규화해서 넘겨준다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import basic, extended

if TYPE_CHECKING:
    from ..robot_profile import RobotProfile

_BUILDERS = {"basic": basic.build, "extended": extended.build}


def build(domain_randomization: dict, profile: RobotProfile):
    """domain_randomization["type"]에 맞는 빌더로 EventCfg를 만들고, 나머지 키는 오버라이드 params로 넘긴다."""
    dr_type = domain_randomization["type"]
    if dr_type not in _BUILDERS:
        raise KeyError(f"등록되지 않은 도메인 랜덤화: {dr_type} (사용 가능: {sorted(_BUILDERS)})")
    params = {key: value for key, value in domain_randomization.items() if key != "type"}
    return _BUILDERS[dr_type](profile, params)

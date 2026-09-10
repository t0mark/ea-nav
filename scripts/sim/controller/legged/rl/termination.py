"""종료조건 세트 - preset의 `terminations:` 키로 고른다.

time_out(시간 초과)은 어느 세트든 항상 포함되는 기반 조건이라 여기서 다루지 않고 loco_rl_env.py가 직접 추가한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .robot_profile import RobotProfile

# 지형 밖으로 나가 허공으로 떨어진 것으로 보는 몸통 높이(m, 월드 기준). 생성 지형의 타일은 z=0 근방에
# 놓이고 단차는 최대 0.11m라, 이 아래로 내려갔다면 지면 위에 있는 상태가 아니다.
_FALLEN_BASE_HEIGHT_M = -1.0


@configclass
class TerminationsCfg:
    """빈 컨테이너 - build()가 base_contact·base_fell을 채우고, loco_rl_env.py가 time_out을 추가한다."""


def _build_base_contact(profile: RobotProfile) -> TerminationsCfg:
    """몸통이 부딪히거나(넘어짐) 지형 밖으로 떨어지면(허공 낙하) 종료되는 조건을 만든다.

    낙하 조건이 필요한 이유는 생성 지형이 유한하기 때문이다 - 로봇이 가장자리를 넘어가면 아무것도
    닿지 않아 base_contact가 걸리지 않고, 에피소드가 끝날 때까지 계속 떨어진다. 그동안 height scan은
    지형 밖 값을, 발 높이 기준 지면은 가장자리 값을 돌려주므로 관측도 보상도 의미를 잃는다.
    """
    cfg = TerminationsCfg()
    cfg.base_contact = DoneTerm(
        func=core_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.base_body_name), "threshold": 1.0},
    )
    cfg.base_fell = DoneTerm(func=core_mdp.root_height_below_minimum, params={"minimum_height": _FALLEN_BASE_HEIGHT_M})
    return cfg


TERMINATION_SETS = {
    "base_contact": _build_base_contact,
}


def build(profile: RobotProfile, set_name: str) -> TerminationsCfg:
    """preset.terminations 이름으로 종료조건 세트를 만든다(time_out은 loco_rl_env.py가 따로 추가)."""
    return TERMINATION_SETS[set_name](profile)

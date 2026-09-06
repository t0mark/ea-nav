"""보상 항목 레지스트리 - 로봇 yaml의 rewards 리스트가 이름으로 이 레지스트리를 찾아 조립한다.

quadruped/humanoid라는 로봇 타입으로 보상 세트를 통째로 고르던 방식(과거 QuadrupedRewardsCfg/
HumanoidRewardsCfg)을 버리고, 개별 보상 항목(track_lin_vel_xy_exp, feet_air_time_biped 등)을 로봇마다
직접 나열해 고르게 한다 - go2w(다족)와 tron2a_wf(2족)처럼 실제 오픈소스 세팅이 로봇 타입과 무관하게
같은 항목을 쓰는 경우를 타입 버킷 없이 자연스럽게 재사용하기 위함이다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from . import energy_penalty, foot_contact_biped, foot_contact_multi, gait_sync, posture, velocity_tracking

if TYPE_CHECKING:
    from ..robot_profile import RobotProfile

# 이름 -> 빌더 함수 레지스트리 - 로봇 yaml의 rewards[].name이 이 딕셔너리 키와 정확히 일치해야 한다
_REGISTRY = {
    "track_lin_vel_xy_exp": velocity_tracking.track_lin_vel_xy_exp,
    "track_ang_vel_z_exp": velocity_tracking.track_ang_vel_z_exp,
    "lin_vel_z_l2": energy_penalty.lin_vel_z_l2,
    "ang_vel_xy_l2": energy_penalty.ang_vel_xy_l2,
    "dof_torques_l2": energy_penalty.dof_torques_l2,
    "dof_acc_l2": energy_penalty.dof_acc_l2,
    "dof_vel_l2": energy_penalty.dof_vel_l2,
    "action_rate_l2": energy_penalty.action_rate_l2,
    "flat_orientation_l2": posture.flat_orientation_l2,
    "dof_pos_limits": posture.dof_pos_limits,
    "base_height_l2": posture.base_height_l2,
    "joint_deviation_l1": posture.joint_deviation_l1,
    "stand_still": posture.stand_still,
    "termination_penalty": posture.termination_penalty,
    "alive": posture.alive,
    "feet_air_time_multi": foot_contact_multi.feet_air_time_multi,
    "undesired_contacts": foot_contact_multi.undesired_contacts,
    "feet_air_time_biped": foot_contact_biped.feet_air_time_biped,
    "feet_slide": foot_contact_biped.feet_slide,
    "no_jumps": foot_contact_biped.no_jumps,
    "feet_contact": foot_contact_biped.feet_contact,
    "feet_swing_height": foot_contact_biped.feet_swing_height,
    "feet_gait": gait_sync.feet_gait,
    "joint_mirror": gait_sync.joint_mirror,
}


@configclass
class RewardsCfg:
    """빈 컨테이너 - 로봇이 고른 보상 항목만 인스턴스 속성으로 채워진다(reward manager가 인스턴스
    속성을 순회하며 RewardTermCfg를 찾으므로, 클래스 정의 시점에 필드를 고정해둘 필요가 없다)."""


def build(reward_entries: list[dict], profile: RobotProfile) -> RewardsCfg:
    """로봇 yaml의 rewards 리스트를 레지스트리에서 찾아 RewardsCfg 인스턴스로 조립한다.

    같은 함수(name)를 관절군만 바꿔 두 번 쓰는 로봇이 있다(예: tron2a_wf의 dof_vel_l2를 바퀴용/
    비바퀴용으로 따로 둠) - name은 레지스트리 조회에만 쓰고, RewardsCfg 인스턴스 속성 이름은
    entry의 key(선택, 로그에 남는 이름)를 우선 쓰거나 없으면 name에 등장 순번을 붙여 구분한다.
    """
    cfg = RewardsCfg()
    seen_names: dict[str, int] = {}
    for entry in reward_entries:
        name = entry["name"]
        if name not in _REGISTRY:
            raise KeyError(f"등록되지 않은 보상 항목: {name} (사용 가능: {sorted(_REGISTRY)})")
        term = _REGISTRY[name](weight=entry["weight"], profile=profile, **entry.get("params", {}))
        occurrence = seen_names.get(name, 0)
        seen_names[name] = occurrence + 1
        attr_name = entry.get("key") or (name if occurrence == 0 else f"{name}_{occurrence}")
        setattr(cfg, attr_name, term)
    return cfg

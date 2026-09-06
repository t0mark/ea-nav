"""전 관절(또는 관절 부분집합)에 동일한 stiffness/damping을 쓰는 가장 단순한 PD 액추에이터 모델.

joint_names(선택, 기본 ".*")로 액추에이터 적용 대상을 제한할 수 있다 - Digit처럼 다리에 닫힌 루프
(로드 구속) 구조가 있는 로봇은 전 관절(".*")에 액추에이터를 걸면 안 된다(관측됨: Isaac Lab 공식
digit/rough_env_cfg.py 주석 - "Digit has closed loops (mechanisms)... explicit joint names produce
a concrete index tensor that works correctly", 전체 관절에 걸면 joint_pos 텐서 길이가 관절 개수와
안 맞아 인덱싱이 깨진다). 이런 로봇은 joint_names에 LEG_JOINT_NAMES+ARM_JOINT_NAMES(로드 관절 제외)를
지정해야 한다.
"""

from __future__ import annotations

from isaaclab.actuators import ActuatorBaseCfg, ImplicitActuatorCfg


def build(actuator_cfg: dict) -> dict[str, ActuatorBaseCfg]:
    """actuator_cfg의 stiffness/damping을 joint_names(기본 전 관절)에 적용한 액추에이터 그룹 1개를 만든다."""
    joint_names = actuator_cfg.get("joint_names", ".*")
    if isinstance(joint_names, str):
        joint_names = [joint_names]
    return {
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=joint_names, stiffness=actuator_cfg["stiffness"], damping=actuator_cfg["damping"]
        )
    }

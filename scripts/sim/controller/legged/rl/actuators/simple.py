"""전 관절(또는 관절 부분집합)을 한 그룹으로 묶어 구동하는 액추에이터 축.

joint_names(선택, 기본 ".*")로 액추에이터 적용 대상을 제한할 수 있다 - Digit처럼 다리에 닫힌 루프
(로드 구속) 구조가 있는 로봇은 전 관절(".*")에 액추에이터를 걸면 안 된다(관측됨: Isaac Lab 공식
digit/rough_env_cfg.py 주석 - "Digit has closed loops (mechanisms)... explicit joint names produce
a concrete index tensor that works correctly", 전체 관절에 걸면 joint_pos 텐서 길이가 관절 개수와
안 맞아 인덱싱이 깨진다). 이런 로봇은 joint_names에 LEG_JOINT_NAMES+ARM_JOINT_NAMES(로드 관절 제외)를
지정해야 한다.

이 그룹을 실제로 어떤 물리 모델(순수 PD인지 토크 포화가 있는 DC 모터인지)로 구동할지는
actuator_cfg["model"]이 결정한다(motor_model.py 참고) - "몇 그룹으로 나누는가"(이 파일의 책임)와
"그룹을 어떤 모델로 구동하는가"는 서로 독립인 축이라 여기서 분기하지 않는다.
"""

from __future__ import annotations

from isaaclab.actuators import ActuatorBaseCfg

from .motor_model import build_actuator_group


def build(actuator_cfg: dict) -> dict[str, ActuatorBaseCfg]:
    """actuator_cfg를 joint_names(기본 전 관절)에 적용한 액추에이터 그룹 1개를 만든다."""
    joint_names = actuator_cfg.get("joint_names", ".*")
    if isinstance(joint_names, str):
        joint_names = [joint_names]
    return {"all_joints": build_actuator_group(joint_names, actuator_cfg)}

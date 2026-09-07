"""actuator.model(기본 implicit)에 따라 그룹 하나를 실제 ActuatorBaseCfg로 만드는 공용 로직.

simple.py(전 관절 한 그룹)와 grouped.py(패턴별 여러 그룹)는 "관절을 몇 그룹으로 나누는가"만 다루고,
"그 그룹을 어떤 물리 모델로 구동하는가"는 이 모듈에 위임한다 - 두 축(그룹핑 방식 vs 액추에이터
물리 모델)이 서로 독립이라, 조합이 늘어나도(예: 그룹별로 다른 모델) type을 새로 늘리지 않고 각
그룹 dict에 model 키만 얹으면 된다.
"""

from __future__ import annotations

from isaaclab.actuators import ActuatorBaseCfg, DCMotorCfg, ImplicitActuatorCfg


def build_actuator_group(joint_names_expr: list[str], group_cfg: dict) -> ActuatorBaseCfg:
    """group_cfg["model"](기본 "implicit")에 맞는 액추에이터 그룹 1개를 만든다.

    model: dc_motor - effort_limit·saturation_effort·velocity_limit로 토크·속도 포화가 있는 실제
    모터를 재현한다(Isaac Lab 공식 Unitree Go2/H1 등이 쓰는 DCMotorCfg). 목표 관절 위치가 극단값으로
    튀어도 모터 자체가 그 이상 토크를 못 내므로, 포화가 없는 순수 PD(ImplicitActuatorCfg)보다 학습
    발산에 안전하다(관측됨: unitree_b2·deeprobotics_lite3가 ImplicitActuatorCfg + 큰 stiffness
    조합에서 가치함수 loss 폭주로 반복 발산).
    """
    model = group_cfg.get("model", "implicit")
    if model == "implicit":
        return ImplicitActuatorCfg(
            joint_names_expr=joint_names_expr, stiffness=group_cfg["stiffness"], damping=group_cfg["damping"]
        )
    if model == "dc_motor":
        return DCMotorCfg(
            joint_names_expr=joint_names_expr,
            stiffness=group_cfg["stiffness"],
            damping=group_cfg["damping"],
            effort_limit=group_cfg["effort_limit"],
            saturation_effort=group_cfg.get("saturation_effort", group_cfg["effort_limit"]),
            velocity_limit=group_cfg["velocity_limit"],
            friction=group_cfg.get("friction", 0.0),
        )
    raise KeyError(f"등록되지 않은 actuator model: {model} (사용 가능: implicit, dc_motor)")

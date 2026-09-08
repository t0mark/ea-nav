"""관절 구동 모델 조립 - 관절군을 하나로 묶어 PD 또는 DC-motor로 구동한다.

model: dc_motor - effort_limit·saturation_effort·velocity_limit로 토크·속도 포화가 있는 실제 모터를
재현한다(Isaac Lab 공식 Unitree Go2/A1이 쓰는 DCMotorCfg). model 기본값(implicit)은 속도에 따른
토크 감소가 없는 순수 PD이고, effort_limit/velocity_limit을 주면 그 값으로 클램프만 건다 - Spot
공식 설정(DelayedPDActuatorCfg)이 쓰는 "게인 + 토크 한계" 조합이 이쪽이다.

joint_names는 profile.controlled_joint_names에서 온다(actuator·action이 같은 관절 집합이어야 하므로
공유) - ANYmal-D의 USD는 다리 12관절 외에 카메라 페이로드 마운트 관절(inspection_payload_mount_to_pan/
pan_to_tilt)이 더 있어, 관절군을 ".*"로 두면 액추에이터가 그 2개까지 잡아 정책이 불필요하게
제어하게 된다. 이건 학습 설계 선택이 아니라 로봇마다 다른 물리적 사실이라 robot_profile이 그대로 담는다.
"""

from __future__ import annotations

from isaaclab.actuators import ActuatorBaseCfg, DCMotorCfg, ImplicitActuatorCfg


def build(actuator_cfg: dict, joint_names: str) -> dict[str, ActuatorBaseCfg]:
    """actuator_cfg["model"](기본 implicit)에 맞는 액추에이터를 joint_names에 적용해 만든다."""
    model = actuator_cfg.get("model", "implicit")
    joint_names_expr = [joint_names]
    if model == "implicit":
        # effort_limit은 PD 출력 토크의 하드 클램프다 - 안 주면 토크가 무제한이라 실제 하드웨어보다
        # 강한 로봇이 되므로, 공식 설정에 토크 한계가 있는 로봇은 반드시 yaml에 적어야 한다.
        actuator: ActuatorBaseCfg = ImplicitActuatorCfg(
            joint_names_expr=joint_names_expr,
            stiffness=actuator_cfg["stiffness"],
            damping=actuator_cfg["damping"],
            effort_limit=actuator_cfg.get("effort_limit"),
            velocity_limit=actuator_cfg.get("velocity_limit"),
        )
    elif model == "dc_motor":
        actuator = DCMotorCfg(
            joint_names_expr=joint_names_expr,
            stiffness=actuator_cfg["stiffness"],
            damping=actuator_cfg["damping"],
            effort_limit=actuator_cfg["effort_limit"],
            saturation_effort=actuator_cfg.get("saturation_effort", actuator_cfg["effort_limit"]),
            velocity_limit=actuator_cfg["velocity_limit"],
            friction=actuator_cfg.get("friction", 0.0),
        )
    else:
        raise KeyError(f"등록되지 않은 actuator model: {model} (사용 가능: implicit, dc_motor)")
    return {"all_joints": actuator}

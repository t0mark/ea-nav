"""USD 파일로부터 로봇을 Articulation으로 스폰하는 모듈 (wheeled/legged 공통 재사용)."""

from __future__ import annotations

import re
from pathlib import Path

import isaaclab.sim as sim_utils
import isaacsim.core.utils.stage as stage_utils
from isaaclab.actuators import ActuatorBaseCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

from scripts.sim.utils.capture import compute_prim_world_bounds, usd_local_bbox

_OUT_OF_RANGE_PATTERN = re.compile(r"'([^']+)':\s*[-\d.eE]+\s*not in \[\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\]")

# legged RL 학습 스폰(scripts/sim/controller/legged/rl/loco_rl_env.py)이 쓰는 물리 속성 그대로 -
# 관절 제어를 검증할 때(actuators를 주는 경우) 이 값도 같이 줘야 학습 때와 같은 물리 조건이 된다.
# max_depenetration_velocity=1.0이 핵심: 기본값(제한 없음)으로 스폰하면 접힌 다리 기본 자세가
# usd 원본 자세(대개 직립) 기준 지면 여유 높이보다 낮게 착지하면서 첫 충돌을 완충 없이 그대로
# 받아 관절이 리밋을 순간적으로 넘는다.
LOCO_RIGID_BODY_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)
LOCO_ARTICULATION_PROPS = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
)
LOCO_SOFT_JOINT_POS_LIMIT_FACTOR = 0.9  # 리밋 끝까지 밀어붙이면 제어가 불안정해지므로 10% 여유를 둔다


def spawn_robot_from_usd(
    usd_path: str,
    prim_path: str,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    joint_pos: dict[str, float] | None = None,
    actuators: dict[str, ActuatorBaseCfg] | None = None,
    activate_contact_sensors: bool = False,
    rigid_props: sim_utils.RigidBodyPropertiesCfg | None = None,
    articulation_props: sim_utils.ArticulationRootPropertiesCfg | None = None,
    soft_joint_pos_limit_factor: float = 1.0,
) -> Articulation:
    """USD 경로의 로봇을 prim_path 위치에 Articulation으로 스폰한다.

    actuators를 안 주면(기본값) 관절 드라이브 타입·PD 게인이 URDF -> USD 변환 시 이미 USD에
    반영돼 있다고 보고, 모든 관절에 USD가 가진 값을 그대로 쓰는 ImplicitActuatorCfg로만 감싼다(40종
    USD가 문제없이 열리는지만 볼 때 - 이 경우 로봇마다 다른 실제 학습용 액추에이터 모델은 필요 없다).
    actuators를 주면 그걸 그대로 쓴다(로봇 프로필의 실제 액추에이터 모델로 관절 제어 자체를 검증할
    때 - scripts/sim/controller/legged/rl/actuator.py의 build() 결과를 그대로 넘기면 된다). 이때는
    rigid_props/articulation_props/soft_joint_pos_limit_factor도 이 모듈의 LOCO_* 상수로 같이 줘야
    학습 스폰과 같은 물리 조건이 된다.
    joint_pos를 안 주면 전 관절 기본 자세 0(USD 자체 기본값)으로 스폰한다.
    activate_contact_sensors는 이 로봇의 몸체에서 ContactSensor로 접촉력을 읽어야 할 때만 켠다.
    """
    robot_cfg = ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=activate_contact_sensors,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=position, joint_pos=joint_pos or {".*": 0.0}),
        soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
        actuators=actuators or {"all_joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
    )
    return Articulation(robot_cfg)


def clear_prim(prim_path: str) -> None:
    """prim 하나를 씬에서 지운다 (재스폰 전 정리용).

    저수준 Usd.Stage.RemovePrim()은 Hydra 렌더 캐시·USD 네이티브 인스턴싱(prototype)을 정리하지
    않아, 이전에 스폰한 지오메트리가 다음 스폰 화면에 잔상으로 남는다. Kit의 공식 삭제 커맨드
    (clear_stage가 내부적으로 쓰는 DeletePrimsCommand)로 지워야 렌더 상태까지 정리된다.
    """
    stage_utils.clear_stage(predicate=lambda path: path == prim_path)


def ground_clearance(usd_path: Path) -> float:
    """usd에 저장된 기본 자세 기준으로, 가장 낮은 지점이 지면(z=0) 위로 오도록 띄울 높이를 구한다.

    base_link 원점이 지면이 아니라 몸통 중심에 있는 다족보행/휴머노이드 로봇을 원점(z=0)에 그대로
    스폰하면 다리 길이만큼 땅에 파묻힌 채로 스폰된다.
    """
    bbox_min_z = usd_local_bbox(usd_path)[0][2]
    return max(0.0, -bbox_min_z) + 0.02  # 살짝 여유를 둬 스폰 직후 지면과 완전히 겹치지 않게 한다


def resolve_spawn_height(declared_height: float | None, usd_path: Path) -> float:
    """로봇 yaml이 선언한 스폰 높이를 쓰고, 없으면 usd bbox로 추정한다.

    ground_clearance()는 usd에 저장된 자세 기준이라 실제 기본 자세가 웅크린 로봇은 몇 cm 높게
    잡힌다 - 리셋마다 그만큼 낙하하고 낙하량이 로봇마다 달라 embodiment 비교에 교란이 되므로,
    공식 설정값이 있는 로봇은 robot yaml의 spawn_height로 고정한다.
    """
    return declared_height if declared_height is not None else ground_clearance(usd_path)


def parse_out_of_range_joints(message: str) -> dict[str, float]:
    """Isaac Lab이 던지는 "관절 기본 위치가 리밋 밖" 에러 메시지에서 관절 이름과 중간값을 뽑아낸다.

    usd를 직접 파싱해 관절 이름을 미리 알아내면, Isaac Lab이 중복 이름에 접미사를 붙이거나 다른
    이름 체계를 쓰는 경우 어긋난다. 대신 Isaac Lab이 실제로 인식한 정확한 이름을 담고 있는 에러
    메시지 자체를 파싱해서 쓴다.
    """
    return {name: (float(lower) + float(upper)) / 2.0 for name, lower, upper in _OUT_OF_RANGE_PATTERN.findall(message)}


def spawn_robot_safely(sim, usd_path: Path, prim_path: str) -> Articulation:
    """usd_path의 로봇을 prim_path에 흔한 실패 유형을 자동으로 피해가며 스폰한다.

    - 기본 자세(전부 0)로 먼저 시도한다. 관절 가동범위가 0을 포함하지 않아 실패하면(사족보행 무릎
      관절 등) Isaac Lab 에러 메시지가 알려주는 정확한 관절 이름만 중간값으로 덮어써 재시도한다 -
      와일드카드(".*")와 정확한 이름을 같이 쓰면 "패턴 두 개에 매칭" 에러가 나므로, 오버라이드한
      이름을 제외한 나머지에만 적용되는 부정 전방탐색 정규식을 와일드카드 대신 쓴다.
    - 높이는 usd 파일의 authored 자세로 1차 추정하고, 실제 스폰 자세가 그것과 달라 여전히 땅에
      파묻혀 있으면 부족분만큼 높여 재스폰한다.
    - 위치는 항상 스폰 설정(init_state.pos)으로만 정한다 - 물리 텐서에 직접 쓰면
      (write_root_pose_to_sim) USD 스테이지 기준 bbox 조회와 어긋나 이후 카메라가 엉뚱한 곳을
      보게 된다.
    """

    def try_spawn(joint_pos: dict[str, float], height: float) -> Articulation:
        robot = spawn_robot_from_usd(str(usd_path), prim_path, position=(0.0, 0.0, height), joint_pos=joint_pos)
        sim.reset()
        return robot

    height = ground_clearance(usd_path)
    joint_pos = {".*": 0.0}
    try:
        robot = try_spawn(joint_pos, height)
    except ValueError as exc:
        overrides = parse_out_of_range_joints(str(exc))
        if not overrides:
            raise
        excluded = "|".join(re.escape(name) for name in overrides)
        joint_pos = {f"^(?!({excluded})$).*": 0.0, **overrides}
        clear_prim(prim_path)
        robot = try_spawn(joint_pos, height)

    bbox_min, _ = compute_prim_world_bounds(prim_path)
    shortfall = -bbox_min[2]
    if shortfall > 0.0:
        clear_prim(prim_path)
        robot = try_spawn(joint_pos, height + shortfall + 0.02)
    return robot

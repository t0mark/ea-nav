"""USD 파일로부터 로봇을 Articulation으로 스폰하는 모듈 (wheeled/legged 공통 재사용)."""

from __future__ import annotations

import re
from pathlib import Path

import isaaclab.sim as sim_utils
import isaacsim.core.utils.stage as stage_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

from scripts.sim.utils.capture import compute_prim_world_bounds, usd_local_bbox

_OUT_OF_RANGE_PATTERN = re.compile(r"'([^']+)':\s*[-\d.eE]+\s*not in \[\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\]")


def spawn_robot_from_usd(
    usd_path: str,
    prim_path: str,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    joint_pos: dict[str, float] | None = None,
) -> Articulation:
    """USD 경로의 로봇을 prim_path 위치에 Articulation으로 스폰한다.

    관절 드라이브 타입·PD 게인은 URDF -> USD 변환 시 이미 USD에 반영돼 있으므로, 여기서는
    모든 관절에 USD가 가진 값을 그대로 쓰는 ImplicitActuatorCfg로만 감싼다. joint_pos를 안 주면
    전 관절 기본 자세 0(USD 자체 기본값)으로 스폰한다.
    """
    robot_cfg = ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(pos=position, joint_pos=joint_pos or {".*": 0.0}),
        actuators={"all_joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
    )
    return Articulation(robot_cfg)


def clear_prim(prim_path: str) -> None:
    """prim 하나를 씬에서 지운다 (재스폰 전 정리용).

    저수준 Usd.Stage.RemovePrim()은 Hydra 렌더 캐시·USD 네이티브 인스턴싱(prototype)을 제대로
    정리하지 않아, 이전에 스폰했던 지오메트리가 다음 스폰 화면에 남아 겹쳐 보이는 문제가 있었다
    (관측됨: USD 네이티브 인스턴싱을 쓰는 로봇 이후 여러 로봇에서 잔상). Kit의 공식 삭제 커맨드
    (clear_stage가 내부적으로 쓰는 DeletePrimsCommand)를 통해 지워야 렌더 상태까지 정리된다.
    """
    stage_utils.clear_stage(predicate=lambda path: path == prim_path)


def ground_clearance(usd_path: Path) -> float:
    """usd에 저장된 기본 자세 기준으로, 가장 낮은 지점이 지면(z=0) 위로 오도록 띄울 높이를 구한다.

    base_link 원점이 지면이 아니라 몸통 중심에 있는 다족보행/휴머노이드 로봇을 원점(z=0)에 그대로
    스폰하면 다리 길이만큼 땅에 파묻힌 채로 스폰된다(관측됨: 사족보행·휴머노이드 거의 전부).
    """
    bbox_min_z = usd_local_bbox(usd_path)[0][2]
    return max(0.0, -bbox_min_z) + 0.02  # 살짝 여유를 둬 스폰 직후 지면과 완전히 겹치지 않게 한다


def parse_out_of_range_joints(message: str) -> dict[str, float]:
    """Isaac Lab이 던지는 "관절 기본 위치가 리밋 밖" 에러 메시지에서 관절 이름과 중간값을 뽑아낸다.

    usd를 직접 파싱해 관절 이름을 미리 알아내려 하면, 이름이 중복돼 Isaac Lab이 접미사를 붙이거나
    애초에 다른 이름 체계를 쓰는 경우 어긋난다(관측됨). 대신 Isaac Lab이 실제로 인식한 정확한
    이름을 담고 있는 에러 메시지 자체를 파싱해서 쓴다.
    """
    return {name: (float(lower) + float(upper)) / 2.0 for name, lower, upper in _OUT_OF_RANGE_PATTERN.findall(message)}


def spawn_robot_safely(sim, usd_path: Path, prim_path: str) -> Articulation:
    """usd_path의 로봇을 prim_path에 흔한 실패 유형을 자동으로 피해가며 스폰한다.

    - 기본 자세(전부 0)로 먼저 시도한다. 관절 가동범위가 0을 포함하지 않아 실패하면(사족보행 무릎
      관절 등에서 관측됨) Isaac Lab 에러 메시지가 알려주는 정확한 관절 이름만 중간값으로 덮어써
      재시도한다 - 와일드카드(".*")와 정확한 이름을 같이 쓰면 "패턴 두 개에 매칭" 에러가 나므로,
      오버라이드한 이름을 제외한 나머지에만 적용되는 부정 전방탐색 정규식을 와일드카드 대신 쓴다.
    - 높이는 usd 파일의 authored 자세로 1차 추정하고, 실제 스폰 자세가 그것과 달라 여전히 땅에
      파묻혀 있으면(관측됨: 다리 관절 기본값이 파일 기본 자세와 크게 다른 로봇) 부족분만큼 높여
      재스폰한다.
    - 위치는 항상 스폰 설정(init_state.pos)으로만 정한다 - 물리 텐서에 직접 쓰면
      (write_root_pose_to_sim) USD 스테이지 기준 bbox 조회와 어긋나 이후 카메라가 엉뚱한 곳을
      보는 문제가 있었다(관측됨).
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

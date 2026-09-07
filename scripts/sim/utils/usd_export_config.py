"""wheeled/legged 로봇 USD로부터 configs/robots/ yaml을 자동으로 뽑아내는 도구.

바퀴 관절 이름, 반지름, 좌우 바퀴 간 거리 같은 값은 제조사 스펙이 아니라 실제로 시뮬레이션되는 USD
자체에서 나와야 정확하다 - 콜리전 형상이 스펙과 다를 수 있고, 관절 이름도 Isaac Lab이 스폰 시 실제로
인식한 이름을 써야 나중에 그 이름으로 관절을 다시 찾을 수 있다.

바퀴인지 아닌지는 이름이 아니라 관절 성질(회전 제한 없음 + 모터 드라이브 있음)과 바디 형상으로
판단한다 - 이름 규칙은 로봇마다 달라 신뢰할 수 없고, 회전 제한이 없다는 것만으로는 캐스터·스위블
같은 비구동 회전체와 구분되지 않는다. 관절-바디 관계는 USD의 body0/body1 relationship을 직접
읽는다 - 배열 인덱스로 추정하면 트리가 여러 갈래로 갈라지는 로봇에서는 대응이 보장되지 않는다.

구동 방향 부호(direction_sign/rotation_sign)와 pure pursuit 속도(nav_linear_velocity/
nav_max_angular_velocity)는 전부 실측한다 - 바퀴 축이 authored된 방향이나 메카넘 롤러 배치에 따라
"양수 명령 = 전진"이 항상 성립하지는 않고, 접지 마찰의 임계값 특성도 로봇마다 반대 방향일 수 있어
기하학적으로 미리 계산할 수 없다.

ackermann은 구동 바퀴 자체는 관절 성질로 골라낼 수 있어도, 조향 관절과 바퀴를 짝짓고 wheelbase·
track_width까지 뽑아내려면 조향 너클 구조를 별도로 추적해야 해서 아직 자동 추출을 지원하지 않는다 -
configs/robots/wheeled/ackermann/_template.yaml을 참고해 직접 작성해야 한다.

legged의 몸통·발은 물리 시뮬레이션이 전혀 필요 없다 - 둘 다 로봇 설계 시점에 이미 구조·형상으로
결정돼 있는 값이기 때문이다.
    - base(몸통): 관절 트리에서 "어떤 관절의 자식으로도 등장하지 않는 바디" = 루트. 플로팅 베이스
      로봇은 base가 월드에 관절로 고정되지 않으므로 이 조건을 만족하는 바디가 정확히 하나뿐이다.
    - foot(발): 관절 트리의 말단(어떤 관절의 부모로도 등장하지 않는 바디) 중, 루트까지 거슬러
      올라가는 동안 actuated(회전/직동) 관절을 2개 이상 거치는 것만 다리·팔 후보로 남긴다 - 라이다·
      카메라 같은 센서 부착물은 보통 fixed 관절 1개로 몸통에 바로 붙어 이 조건에서 걸러진다.
      multi-legged(팔이 없음)는 이 후보 전부가 발이고, humanoid(팔도 같은 조건을 만족)는 usd에
      저장된 기본 자세에서 가장 낮은 2개만 발로 뽑는다(발이 손보다 낮은 건 로봇 설계상 항상 참).
    이 판단은 usd 파일을 스폰하지 않고 그대로 열어서(Usd.Stage.Open) 구조와 authored 자세만 읽으면
    끝나므로, 관절 기본 자세가 안정적으로 서 있는 자세인지 여부와 무관하게 항상 같은 결과가 나온다 -
    이전에 로봇을 실제로 스폰해 중력으로 정지시킨 뒤 접지를 실측하던 방식은, 기본 자세가 서 있는
    자세라는 보장이 없어 로봇마다 다른 이유로 실패해서 폐기했다.

    관절 PD 게인(actuator_stiffness/damping)도 같은 이유로 USD authored 값을 그대로 안 믿는다 -
    URDF->USD 변환기가 정지 자세를 딱딱하게 붙잡아두려고 넣은 값(관측됨: unitree_go2 stiffness 1e7,
    damping 1e5 - Isaac Lab 공식 값 25/0.5의 40만 배)이라 RL 학습에 못 쓴다. 대신 authored 관절
    effort 한계(최대 토크)에서 Isaac Lab 공식 로봇들의 비율을 참고해 역산한다(_compute_actuator_gains).

pxr는 Kit 프로세스가 뜬 뒤에만 임포트할 수 있으므로, 이 모듈의 함수는 AppLauncher 부팅이 끝난
tools/ 진입점에서만 호출해야 한다(capture.py와 동일한 전제) - legged 쪽은 스폰은 안 하지만 pxr
자체가 Isaac Sim 번들 파이썬에서만 임포트되므로 이 제약이 똑같이 적용된다.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import euler_xyz_from_quat

from scripts.sim.env.robot_spawn import clear_prim, spawn_robot_from_usd

# 좌/우 바퀴로 인정하는 중심선 최소 이격 거리(m) - 캐스터·피벗처럼 중앙에 있는 회전체를 제외한다
_WHEEL_MIN_LATERAL_OFFSET = 0.02

# 다리·팔로 인정하는 최소 actuated(회전/직동) 관절 개수 - 센서 부착물은 보통 fixed 관절 1개뿐이다
_MIN_LIMB_ACTUATED_DEPTH = 2

# humanoid에서 발로 뽑는 개수 - 다리 2개(발), 팔 2개(손) 중 발만 남긴다
_HUMANOID_FOOT_COUNT = 2


@dataclass
class _WheelCandidate:
    """구동 바퀴로 추정되는 관절-바디 하나."""

    joint_name: str
    body_name: str
    world_position: tuple[float, float, float]
    radius: float


class _WheelInspector:
    """스폰된 로봇의 prim 트리에서 실제 구동 바퀴 후보를 찾아낸다."""

    _SHAPE_CLOSE_RATIO = 0.95  # 두 축이 이 비율 이상 비슷해야 원판(두 축이 지름) 취급
    _SHAPE_THIN_RATIO = 0.9  # 세 번째 축(두께)이 지름의 이 비율보다 작아야 원판 취급
    _MAX_WHEELS_PER_PARENT = 4  # 부모 하나에 이보다 많은 바퀴 모양 자식이 달려 있으면 허브+롤러로 본다

    def __init__(self, prim_path: str) -> None:
        """바퀴를 찾을 로봇 prim의 루트 경로를 저장한다."""
        self._prim_path = prim_path

    def find_candidates(self, joint_names: list[str]) -> list[_WheelCandidate]:
        """joint_names 중 실제로 구동되는 바퀴 후보를 관절 성질 + 형상 + 형제 수로 걸러 모은다.

        형제 수까지 보는 이유는, 메카넘 휠은 허브 하나에 회전 제한 없는 롤러가 5개 이상 달려 있어
        롤러도 형상만으로는 바퀴처럼 보이기 때문이다. 실제 차량은 한 부모(섀시)에 바퀴가 많아야
        4개 정도이므로, 같은 부모를 공유하는 바퀴 모양 후보가 그보다 많으면 허브에 달린 롤러로 보고
        전부 제외한다 - "부모 자신이 바퀴 모양인가"로 판단하면 섀시가 우연히 둥근 로봇에서 진짜
        바퀴까지 함께 제외되므로, 부모의 형상이 아니라 부모가 거느린 바퀴 모양 자식의 수를 본다.
        """
        raw_candidates: list[tuple[str, str, str | None, float]] = []  # (joint, child, parent, extent 최대값)
        for joint_name in joint_names:
            joint_prim_path = self._find_joint_prim(joint_name)
            if joint_prim_path is None or not self._is_driven_continuous_revolute(joint_prim_path):
                continue
            parent_path, child_path = self._joint_bodies(joint_prim_path)
            if child_path is None:
                continue
            child_extent = self._local_extent(child_path)
            if not self._is_wheel_shaped(child_extent):
                continue
            raw_candidates.append((joint_name, child_path, parent_path, max(child_extent) / 2.0))

        wheels_per_parent: dict[str, int] = {}
        for _, _, parent_path, _ in raw_candidates:
            if parent_path is not None:
                wheels_per_parent[parent_path] = wheels_per_parent.get(parent_path, 0) + 1

        candidates = []
        for joint_name, child_path, parent_path, radius in raw_candidates:
            sibling_count = wheels_per_parent.get(parent_path, 0) if parent_path is not None else 0
            if sibling_count > self._MAX_WHEELS_PER_PARENT:
                continue
            candidates.append(
                _WheelCandidate(
                    joint_name=joint_name,
                    body_name=child_path.rsplit("/", 1)[-1],
                    world_position=self._world_origin(child_path),
                    radius=radius,
                )
            )
        return candidates

    def _find_joint_prim(self, joint_name: str) -> str | None:
        """joint_name과 이름이 같은 prim을 트리에서 찾는다 - 못 찾으면 조용히 후보에서 제외한다."""
        import isaacsim.core.utils.stage as stage_utils
        from pxr import Usd

        stage = stage_utils.get_current_stage()
        root_prim = stage.GetPrimAtPath(self._prim_path)
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == joint_name:
                return str(prim.GetPath())
        return None

    def _joint_bodies(self, joint_prim_path: str) -> tuple[str | None, str | None]:
        """관절의 body0(부모)/body1(자식) prim 경로를 relationship에서 직접 읽는다."""
        from pxr import UsdPhysics
        import isaacsim.core.utils.stage as stage_utils

        stage = stage_utils.get_current_stage()
        prim = stage.GetPrimAtPath(joint_prim_path)
        joint = UsdPhysics.Joint(prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        body0 = str(body0_targets[0]) if body0_targets else None
        body1 = str(body1_targets[0]) if body1_targets else None
        return body0, body1

    def _is_driven_continuous_revolute(self, joint_prim_path: str) -> bool:
        """회전 제한이 없는(또는 ±무한대인) 리볼루트이면서 모터 드라이브가 authored돼 있는지 본다.

        회전 제한이 없다는 것만으로는 캐스터·스위블처럼 자유 회전하되 구동은 안 되는 관절과
        구분되지 않는다. 실제 구동 바퀴는 각속도 제어용 DriveAPI가 authored돼 있고, 캐스터·스위블은
        순수 물리 접촉으로만 굴러가므로 이 API 자체가 없다.

        리밋은 아예 authored 안 하는 경우뿐 아니라, 값 자체를 ±무한대로 authored해서 "무제한"을
        표현하는 경우도 있다 - 둘 다 계속 도는 바퀴로 봐야 하고, 유한한 값이 authored된 경우만
        팔·조향처럼 실제 가동 범위가 있는 관절로 취급한다.
        """
        from pxr import UsdPhysics
        import isaacsim.core.utils.stage as stage_utils

        stage = stage_utils.get_current_stage()
        prim = stage.GetPrimAtPath(joint_prim_path)
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            return False
        if not prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            return False
        revolute = UsdPhysics.RevoluteJoint(prim)
        lower_attr, upper_attr = revolute.GetLowerLimitAttr(), revolute.GetUpperLimitAttr()
        if not lower_attr.HasAuthoredValue() and not upper_attr.HasAuthoredValue():
            return True
        lower = lower_attr.Get() if lower_attr.HasAuthoredValue() else float("-inf")
        upper = upper_attr.Get() if upper_attr.HasAuthoredValue() else float("inf")
        return math.isinf(lower) or math.isinf(upper)

    def _is_wheel_shaped(self, extent: tuple[float, float, float]) -> bool:
        """바운딩 박스가 얇은 원판(두 축이 비슷하고 나머지 한 축은 짧음) 모양인지 본다."""
        a, b, c = sorted(extent)
        if c <= 0.0:
            return False
        return (b / c) > self._SHAPE_CLOSE_RATIO and (a / c) < self._SHAPE_THIN_RATIO

    @staticmethod
    def _local_extent(prim_path: str) -> tuple[float, float, float]:
        """prim의 로컬 프레임 기준 바운딩 박스 한 변 길이(x, y, z)를 구한다."""
        from pxr import Usd, UsdGeom
        import isaacsim.core.utils.stage as stage_utils

        stage = stage_utils.get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"바운딩 박스를 계산할 prim이 존재하지 않습니다: {prim_path}")
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        aligned_range = bbox_cache.ComputeLocalBound(prim).ComputeAlignedRange()
        bbox_min, bbox_max = aligned_range.GetMin(), aligned_range.GetMax()
        return tuple(bbox_max[i] - bbox_min[i] for i in range(3))

    @staticmethod
    def _world_origin(prim_path: str) -> tuple[float, float, float]:
        """prim의 월드 좌표계 원점을 구한다.

        바운딩 박스 중심이 아니라 prim 원점을 쓴다 - URDF/USD 변환 관례상 바퀴 링크의 원점이 곧 차축
        (회전축) 위치이므로, 바퀴 메시가 원점 기준으로 완벽히 대칭이 아니어도 이게 좌우 바퀴 간
        거리를 재는 정확한 기준점이 된다.
        """
        from pxr import Usd, UsdGeom
        import isaacsim.core.utils.stage as stage_utils

        stage = stage_utils.get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = transform.ExtractTranslation()
        return (translation[0], translation[1], translation[2])


class _DriveCalibrator:
    """로봇을 실제로 짧게 굴려서 구동 방향 부호와 pure pursuit 속도 파라미터를 실측한다."""

    _NUM_STEPS = 90  # 과도응답이 끝나고 정상상태에 가까워질 시간 여유
    _LINEAR_WHEEL_SPEED_CANDIDATES = (2.0, 4.0, 6.0, 8.0)  # rad/s
    _ANGULAR_VELOCITY_CANDIDATES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.5)  # rad/s
    _LINEAR_VELOCITY_SAFETY_MARGIN = 0.5

    def __init__(self, sim: sim_utils.SimulationContext, robot: Articulation) -> None:
        """실측에 쓸 시뮬레이션 컨텍스트와 스폰된 로봇을 저장한다."""
        self._sim = sim
        self._robot = robot

    def calibrate_forward(self, joint_ids: list[int]) -> tuple[float, float]:
        """여러 바퀴 속도를 시도해 방향 부호와 순항 속도(m/s, 안전 마진 적용)를 함께 잰다.

        접지 마찰의 임계값 특성은 로봇마다 반대 방향일 수 있다 - 저속에서 잘 나가다 고속에서
        나빠지는 로봇도, 그 반대인 로봇도 있어 "낮거나 높은 속도가 항상 안전하다"는 가정 자체가
        성립하지 않는다. 그래서 여러 속도를 다 시도해 실제로 가장 빨리 달성되는 속도를 채택한다.
        """
        best_speed = 0.0
        direction_sign = 1.0
        for wheel_speed in self._LINEAR_WHEEL_SPEED_CANDIDATES:
            signed_speed = self._drive_and_measure_linear_speed(joint_ids, wheel_speed)
            if abs(signed_speed) > best_speed:
                best_speed = abs(signed_speed)
                direction_sign = 1.0 if signed_speed >= 0.0 else -1.0
        return direction_sign, round(best_speed * self._LINEAR_VELOCITY_SAFETY_MARGIN, 4)

    def calibrate_rotation(self, wheel_speeds_for_test_w) -> tuple[float, float]:
        """여러 후보 각속도 중 실제로 가장 잘 도는 값과 회전 부호를 함께 잰다.

        wheel_speeds_for_test_w(test_w)는 후보 각속도 하나를 (joint_ids, 바퀴별 rad/s 목록)으로
        바꾸는 함수다 - diff/omni의 좌우·4륜 공식이 서로 달라 호출부가 넘겨준다. 크기는 요 변화율이
        가장 큰 후보를 그대로 쓰고, 부호는 후보 하나가 아니라 전체 다수결로 정한다 - 저속 후보는
        실제 요 변화가 거의 0에 가까워 수치 잡음만으로 부호가 뒤집힐 수 있다.
        """
        results = []
        for test_w in self._ANGULAR_VELOCITY_CANDIDATES:
            joint_ids, wheel_speeds = wheel_speeds_for_test_w(test_w)
            results.append((test_w, self._drive_and_measure_delta_yaw(joint_ids, wheel_speeds)))

        sign_votes = sum(1 if delta_yaw >= 0.0 else -1 for _, delta_yaw in results)
        rotation_sign = 1.0 if sign_votes >= 0 else -1.0
        best_test_w = max(results, key=lambda pair: abs(pair[1]))[0]
        return rotation_sign, best_test_w

    def _drive_and_measure_linear_speed(self, joint_ids: list[int], wheel_speed: float) -> float:
        """joint_ids 전부에 wheel_speed를 명령해 굴리고, 시작 헤딩 기준 전진 속도(부호 있음)를 잰다."""
        self._sim.reset()
        start_xy = self._robot.data.root_pos_w[0, :2].clone()
        _, _, start_yaw = euler_xyz_from_quat(self._robot.data.root_quat_w[0:1])
        self._run(joint_ids, [wheel_speed] * len(joint_ids))
        displacement = self._robot.data.root_pos_w[0, :2] - start_xy
        forward = torch.stack([torch.cos(start_yaw[0]), torch.sin(start_yaw[0])])
        return torch.dot(displacement, forward).item() / (self._NUM_STEPS * self._sim.get_physics_dt())

    def _drive_and_measure_delta_yaw(self, joint_ids: list[int], wheel_speeds: list[float]) -> float:
        """joint_ids에 wheel_speeds를 명령해 굴리고, 시작 대비 요(yaw) 변화량을 [-pi, pi]로 잰다."""
        self._sim.reset()
        _, _, start_yaw = euler_xyz_from_quat(self._robot.data.root_quat_w[0:1])
        self._run(joint_ids, wheel_speeds)
        _, _, end_yaw = euler_xyz_from_quat(self._robot.data.root_quat_w[0:1])
        return (end_yaw.item() - start_yaw.item() + math.pi) % (2 * math.pi) - math.pi

    def _run(self, joint_ids: list[int], wheel_speeds: list[float]) -> None:
        """joint_ids에 wheel_speeds를 정해진 스텝 수만큼 계속 명령하며 물리를 진행시킨다."""
        target = torch.tensor([wheel_speeds], device=self._robot.device)
        for _ in range(self._NUM_STEPS):
            self._robot.set_joint_velocity_target(target, joint_ids=joint_ids)
            self._robot.write_data_to_sim()
            self._sim.step()
            self._robot.update(self._sim.get_physics_dt())


def _build_diff_config(candidates: list[_WheelCandidate], usd_filename: str) -> dict:
    """좌/우 바퀴 후보를 y좌표 부호로 나누고, wheel_radius/wheel_base를 평균으로 계산한다.

    +y가 좌측이라는 부호 규약은 URDF/USD 변환에서 널리 쓰이는 관례다. 회전축이 중심선에 가까운
    캐스터·피벗은 좌/우 어느 쪽도 아니므로 _WHEEL_MIN_LATERAL_OFFSET보다 중심선에 가까우면 제외한다.
    """
    left = [c for c in candidates if c.world_position[1] > _WHEEL_MIN_LATERAL_OFFSET]
    right = [c for c in candidates if c.world_position[1] < -_WHEEL_MIN_LATERAL_OFFSET]
    if not left or not right:
        found = [c.body_name for c in candidates]
        raise RuntimeError(f"좌/우 바퀴를 자동으로 분류하지 못했습니다 - 바퀴 후보: {found}")

    all_wheels = left + right
    wheel_radius = sum(c.radius for c in all_wheels) / len(all_wheels)
    lateral_offset = sum(abs(c.world_position[1]) for c in all_wheels) / len(all_wheels)
    return {
        "usd_path": usd_filename,
        "wheel_radius": round(wheel_radius, 5),
        "wheel_base": round(lateral_offset * 2.0, 5),
        "left_wheel_joint_names": [c.joint_name for c in left],
        "right_wheel_joint_names": [c.joint_name for c in right],
    }


def _build_omni_config(candidates: list[_WheelCandidate], usd_filename: str) -> dict:
    """4개 바퀴 후보를 x(앞/뒤)·y(좌/우) 부호로 4분면 분류한다.

    후보가 정확히 4개가 아니면 OmniDriveKinematics(4륜 메카넘 전용)가 애초에 맞지 않는 로봇이라는
    뜻이므로, 억지로 끼워 맞추지 않고 예외를 던진다.
    """
    if len(candidates) != 4:
        found = [c.body_name for c in candidates]
        raise RuntimeError(
            f"omni 자동 추출은 4륜 메카넘만 지원합니다 - 바퀴 후보 {len(candidates)}개 발견({found}). "
            "3륜 등 다른 구성은 config를 직접 작성하세요."
        )

    def pick(is_front: bool, is_left: bool) -> _WheelCandidate:
        matches = [
            c for c in candidates if (c.world_position[0] > 0.0) == is_front and (c.world_position[1] > 0.0) == is_left
        ]
        if len(matches) != 1:
            raise RuntimeError("omni 바퀴 4분면(앞/뒤 x 좌/우) 분류에 실패했습니다 - 위치가 애매합니다.")
        return matches[0]

    front_left, front_right = pick(True, True), pick(True, False)
    rear_left, rear_right = pick(False, True), pick(False, False)
    wheel_radius = sum(c.radius for c in candidates) / 4.0
    half_wheelbase = sum(abs(c.world_position[0]) for c in candidates) / 4.0
    half_track_width = sum(abs(c.world_position[1]) for c in candidates) / 4.0

    return {
        "usd_path": usd_filename,
        "wheel_radius": round(wheel_radius, 5),
        "half_wheelbase": round(half_wheelbase, 5),
        "half_track_width": round(half_track_width, 5),
        "wheel_joint_names": {
            "front_left": front_left.joint_name,
            "front_right": front_right.joint_name,
            "rear_left": rear_left.joint_name,
            "rear_right": rear_right.joint_name,
        },
    }


def _calibrate_diff(calibrator: _DriveCalibrator, robot: Articulation, config: dict) -> None:
    """diff config에 direction_sign/rotation_sign/nav 속도를 실측해 채워 넣는다."""
    left_ids = robot.find_joints(config["left_wheel_joint_names"], preserve_order=True)[0]
    right_ids = robot.find_joints(config["right_wheel_joint_names"], preserve_order=True)[0]
    all_ids = left_ids + right_ids

    config["direction_sign"], config["nav_linear_velocity"] = calibrator.calibrate_forward(all_ids)

    def wheel_speeds(test_w: float) -> tuple[list[int], list[float]]:
        arm = test_w * config["wheel_base"] / 2.0 / config["wheel_radius"]
        return all_ids, [-arm] * len(left_ids) + [arm] * len(right_ids)

    config["rotation_sign"], config["nav_max_angular_velocity"] = calibrator.calibrate_rotation(wheel_speeds)


def _calibrate_omni(calibrator: _DriveCalibrator, robot: Articulation, config: dict) -> None:
    """omni config에 direction_sign/rotation_sign/nav 속도를 실측해 채워 넣는다."""
    joint_names = [config["wheel_joint_names"][key] for key in ("front_left", "front_right", "rear_left", "rear_right")]
    joint_ids = robot.find_joints(joint_names, preserve_order=True)[0]

    config["direction_sign"], config["nav_linear_velocity"] = calibrator.calibrate_forward(joint_ids)

    lateral_sum = config["half_wheelbase"] + config["half_track_width"]

    def wheel_speeds(test_w: float) -> tuple[list[int], list[float]]:
        arm = lateral_sum * test_w / config["wheel_radius"]
        return joint_ids, [-arm, arm, -arm, arm]

    config["rotation_sign"], config["nav_max_angular_velocity"] = calibrator.calibrate_rotation(wheel_speeds)


def export_wheeled_robot_config(
    sim: sim_utils.SimulationContext, usd_path: Path, robot_category: str, output_path: Path
) -> None:
    """usd_path 로봇을 스폰해 config 값을 자동 계산하고, 방향 부호까지 실측해 output_path에 저장한다."""
    if robot_category == "ackermann":
        raise NotImplementedError(
            "ackermann은 조향/구동 관절을 형상·위치만으로 구분할 수 없어 자동 추출을 지원하지 않습니다 - "
            "configs/robots/wheeled/ackermann/_template.yaml을 참고해 직접 작성하세요."
        )
    if robot_category not in ("diff", "omni"):
        raise ValueError(f"알 수 없는 wheeled 카테고리: {robot_category}")

    prim_path = "/World/Robot"
    robot = spawn_robot_from_usd(str(usd_path), prim_path)
    sim.reset()
    robot.update(sim.get_physics_dt())

    candidates = _WheelInspector(prim_path).find_candidates(robot.joint_names)
    calibrator = _DriveCalibrator(sim, robot)

    if robot_category == "diff":
        config = _build_diff_config(candidates, usd_path.name)
        _calibrate_diff(calibrator, robot, config)
    else:
        config = _build_omni_config(candidates, usd_path.name)
        _calibrate_omni(calibrator, robot, config)

    clear_prim(prim_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"[usd_export_config] {output_path} 자동 생성 완료: {config}")


def _build_kinematic_tree(
    stage, root_prim_path: str
) -> tuple[dict[str, str], dict[str, bool], dict[str, str], set[str]]:
    """관절의 body0(부모)/body1(자식) relationship과 그 관절이 fixed인지를 모아 트리를 구성한다.

    스폰된 라이브 스테이지든, Usd.Stage.Open()으로 직접 연 파일이든 똑같이 동작한다 - USD
    관절 스키마(body0/body1 relationship)는 물리 시뮬레이션 여부와 무관한 저장된 데이터이기
    때문이다. 반환값: {자식: 부모}, {자식: 그 관절이 fixed인지}, {자식: 그 관절의 이름}, 전체 바디
    경로 집합.
    """
    from pxr import Usd, UsdPhysics

    root_prim = stage.GetPrimAtPath(root_prim_path)
    parent_of: dict[str, str] = {}
    is_fixed_joint: dict[str, bool] = {}
    joint_name_of: dict[str, str] = {}
    all_bodies: set[str] = set()
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if not body0_targets or not body1_targets:
            continue
        parent_path, child_path = str(body0_targets[0]), str(body1_targets[0])
        parent_of[child_path] = parent_path
        is_fixed_joint[child_path] = prim.IsA(UsdPhysics.FixedJoint)
        joint_name_of[child_path] = prim.GetName()
        all_bodies.add(parent_path)
        all_bodies.add(child_path)
    return parent_of, is_fixed_joint, joint_name_of, all_bodies


def _leg_chain_joint_names(
    foot_leaves: list[str], parent_of: dict[str, str], joint_name_of: dict[str, str]
) -> set[str]:
    """각 발 링크에서 루트까지 거슬러 올라가며 지나는 관절 이름을 모은다 (다리 체인 전용).

    로봇 전체 관절이 아니라 다리 체인 관절만 걸러야 하는 이유: humanoid는 손가락처럼 다리와 무관한
    관절이 많아서(관측됨: fourier_gr1 - 다리 관절 12개 vs 손가락 등 나머지 44개, 손가락 토크가 훨씬
    작아 관절 전체 중앙값을 쓰면 다리에 필요한 토크보다 훨씬 작은 값이 나옴), actuator 게인은 실제로
    체중을 지탱·보행하는 다리 관절 기준으로만 잡아야 한다.
    """
    names: set[str] = set()
    for foot_path in foot_leaves:
        current = foot_path
        while current in parent_of:
            names.add(joint_name_of[current])
            current = parent_of[current]
    return names


def _find_root_body(parent_of: dict[str, str], all_bodies: set[str]) -> str:
    """어떤 관절의 자식으로도 등장하지 않는 바디 = 트리의 루트(base/torso).

    다리·팔 달린 로봇은 전부 플로팅 베이스라 base가 월드에 관절로 고정되지 않으므로, "누구의
    자식도 아닌 바디"가 정확히 하나만 남는다.
    """
    children = set(parent_of.keys())
    roots = all_bodies - children
    if len(roots) != 1:
        raise RuntimeError(f"루트(base) 바디를 하나로 특정하지 못했습니다 - 후보: {roots}")
    return next(iter(roots))


def _find_leaf_bodies(parent_of: dict[str, str], all_bodies: set[str]) -> set[str]:
    """어떤 관절의 부모로도 등장하지 않는 바디 = 트리의 말단(발·손·머리·센서 부착물 등 후보)."""
    parents = set(parent_of.values())
    return all_bodies - parents


def _actuated_depth(body_path: str, parent_of: dict[str, str], is_fixed_joint: dict[str, bool]) -> int:
    """루트까지 거슬러 올라가며 지나는 actuated(non-fixed) 관절 수를 센다.

    센서 부착물(라이다·카메라·IMU 등)은 보통 fixed 관절 1개로 몸통에 바로 붙어 이 값이 0이고,
    다리·팔은 hip/thigh/knee 등 여러 actuated 관절을 거쳐야 말단에 닿아 값이 크다 - 이 차이로
    "말단 링크가 다리·팔의 끝인지, 그냥 부착물인지"를 물리 시뮬레이션 없이 구조만으로 구분한다.
    """
    depth = 0
    current = body_path
    while current in parent_of:
        if not is_fixed_joint.get(current, False):
            depth += 1
        current = parent_of[current]
    return depth


def _authored_height(stage, body_path: str) -> float:
    """usd에 저장된 기본(authored) 자세 기준, 이 바디의 bbox 중심 z 높이를 구한다.

    스폰하지 않은 파일 자체를 그대로 쓰므로 관절 기본값 그대로가 기준이 된다 - humanoid에서 발(낮음)과
    손(높음)을 구분하는 유일한 근거로 쓴다.
    """
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(body_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return (aligned_range.GetMin()[2] + aligned_range.GetMax()[2]) / 2.0


def _compute_default_joint_overrides(stage, root_prim_path: str) -> dict[str, float]:
    """0.0이 가동범위를 벗어나는 관절만 골라, 범위 중앙값을 기본 자세로 제안한다.

    ArticulationCfg는 관절 기본 자세가 리밋 안에 있어야만 스폰을 허용하는데, 무릎처럼 애초에 0도를
    포함하지 않게 설계된 관절이 있다(관측됨: unitree_go2의 calf_joint 리밋이 [-2.723, -0.838]로 전부
    음수). 중앙값은 "물리적으로 유효하다"만 보장하는 값이고, 실제로 자연스럽게 서는 자세는 학습
    (reset_robot_joints 이벤트의 랜덤 스케일 + PPO)이 그 위에서 찾아가므로 이걸로 충분하다.

    UsdPhysics.RevoluteJoint의 lower/upper limit은 USD Physics 스키마 규약상 도(degree) 단위로
    authored되어 있는데, ArticulationCfg.init_state.joint_pos(및 이 값을 검증하는 PhysX 쪽 리밋)는
    라디안을 기대한다 - 변환 없이 그대로 쓰면 값이 57배(180/pi)가량 부풀려져 리밋을 한참 벗어난다
    (관측됨: unitree_b2/go2w calf_joint, deeprobotics_lite3 knee_joint).
    """
    from pxr import Usd, UsdPhysics

    root_prim = stage.GetPrimAtPath(root_prim_path)
    overrides: dict[str, float] = {}
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        lower_attr, upper_attr = joint.GetLowerLimitAttr(), joint.GetUpperLimitAttr()
        if not lower_attr.HasAuthoredValue() or not upper_attr.HasAuthoredValue():
            continue
        lower, upper = math.radians(lower_attr.Get()), math.radians(upper_attr.Get())
        if lower <= 0.0 <= upper:
            continue
        overrides[prim.GetName()] = round((lower + upper) / 2.0, 5)
    return overrides


def _compute_actuator_gains(stage, root_prim_path: str, leg_joint_names: set[str]) -> tuple[float, float]:
    """다리 체인 관절의 authored 최대 토크(effort limit) 중앙값으로 PD 게인을 추정한다.

    USD에 authored된 stiffness/damping 값 자체는 신뢰하지 않는다 - URDF->USD 변환기가 "정지 자세를
    딱딱하게 붙잡아두기 위한" 임의의 큰 값(관측됨: unitree_go2 stiffness 1e7, damping 1e5)을 넣는
    경우가 흔해서, RL 학습에 그대로 쓰면 관절이 사실상 위치 고정에 가깝게 거동해 정책이 자연스러운
    토크를 못 낸다. 대신 Isaac Lab 공식 로봇 설정(A1, Go2 등: effort_limit 23.5~45 -> stiffness=25,
    damping=0.5 / H1: effort_limit 100~300 -> stiffness 20~200, damping 4~10)을 대조해보면, 로봇
    스케일이 달라도 stiffness가 대략 그 로봇 관절의 최대 토크(effort limit)와 같은 자릿수이고
    damping은 stiffness의 약 2%인 경향이 있다 - 이 비율을 일반 공식으로 채택한다.

    leg_joint_names로 다리 체인 관절만 걸러서 본다 - 로봇 전체 관절로 중앙값을 내면 humanoid의 손가락
    관절(수가 많고 토크가 훨씬 작음)에 밀려 다리에 필요한 값보다 훨씬 작게 나온다(관측됨: fourier_gr1
    - 다리 관절 12개는 최대 133인데 손가락 등 나머지 44개가 대부분 1~10이라, 전체로 계산하면
    stiffness가 4 근처까지 떨어짐 - 다리 12개만 걸러야 133 근처의 제대로 된 값이 나온다).
    """
    from pxr import Usd, UsdPhysics

    root_prim = stage.GetPrimAtPath(root_prim_path)
    effort_limits: list[float] = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdPhysics.RevoluteJoint) or prim.GetName() not in leg_joint_names:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive is None:
            continue
        max_force_attr = drive.GetMaxForceAttr()
        if max_force_attr.HasAuthoredValue() and max_force_attr.Get() > 0.0:
            effort_limits.append(max_force_attr.Get())

    if not effort_limits:
        return 25.0, 0.5  # authored effort 한계가 전혀 없으면 Isaac Lab 사족보행 기본값으로 대체

    effort_limits.sort()
    median_effort = effort_limits[len(effort_limits) // 2]
    stiffness = round(median_effort, 3)
    damping = round(stiffness * 0.02, 4)
    return stiffness, damping


# 자동 생성 시 채워 넣는 기본 보상 목록 - scripts/sim/controller/legged/rl/rewards/ 레지스트리 이름 기준.
# 로봇마다 실제 오픈소스 세팅과 대조해 축(actuator/action/rewards 등)을 다시 채우기 전까지 쓰는
# "일단 학습이 도는" 최소 기본값이다 - configs/robots/legged/_profiles/ 같은 런타임 공유 버킷이
# 아니라, 이 생성 함수 코드 안에서만 쓰는 부트스트랩 상수라는 점이 다르다.
_DEFAULT_MULTI_LEGGED_REWARDS = [
    {"name": "track_lin_vel_xy_exp", "weight": 1.0},
    {"name": "track_ang_vel_z_exp", "weight": 0.5},
    {"name": "lin_vel_z_l2", "weight": -2.0},
    {"name": "ang_vel_xy_l2", "weight": -0.05},
    {"name": "dof_torques_l2", "weight": -1.0e-5},
    {"name": "dof_acc_l2", "weight": -2.5e-7},
    {"name": "action_rate_l2", "weight": -0.01},
    {"name": "flat_orientation_l2", "weight": -1.0},
    {"name": "dof_pos_limits", "weight": -1.0},
    {"name": "feet_air_time_multi", "weight": 0.125, "params": {"threshold": 0.5}},
    {"name": "undesired_contacts", "weight": -1.0, "params": {"threshold": 1.0}},
]
_DEFAULT_HUMANOID_REWARDS = [
    {"name": "track_lin_vel_xy_exp", "weight": 1.0},
    {"name": "track_ang_vel_z_exp", "weight": 0.5},
    {"name": "ang_vel_xy_l2", "weight": -0.05},
    {"name": "dof_torques_l2", "weight": -1.0e-5},
    {"name": "dof_acc_l2", "weight": -2.5e-7},
    {"name": "action_rate_l2", "weight": -0.01},
    {"name": "flat_orientation_l2", "weight": -1.0},
    {"name": "dof_pos_limits", "weight": -1.0},
    {"name": "termination_penalty", "weight": -200.0},
    {"name": "feet_air_time_biped", "weight": 0.25, "params": {"threshold": 0.4}},
    {"name": "feet_slide", "weight": -0.25},
]


def export_legged_robot_config(usd_path: Path, robot_category: str, output_path: Path) -> None:
    """usd 파일을 열어(스폰·물리 없이) 관절 구조와 authored 자세만으로 config를 만들어 저장한다.

    발의 부모 링크(허벅지·정강이 등)는 multi-legged에서만 undesired_contact_body_names로 함께 담는다 -
    그 부위가 지면에 닿으면 쓰러진 것으로 볼 수 있는 multi-legged와 달리, humanoid는 발이 지면에 닿는
    것 자체가 정상 보행이라 이 개념이 안 맞고(feet_air_time_biped 보상이 대신 호핑 억제로 자세를
    잡는다). actuator/action/rewards 등 학습 방식 축은 여기서는 "일단 도는" 기본값만 채우고, 실제
    오픈소스 리포와 대조한 정확한 값은 사람이 로봇 yaml을 직접 다시 채워야 한다(usd 구조만으로는
    PD 게인 그룹핑·보상 커스터마이징까지 알아낼 수 없기 때문).
    """
    if robot_category not in ("multi-legged", "humanoid"):
        raise ValueError(f"알 수 없는 legged 카테고리: {robot_category}")

    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    root_prim_path = str(stage.GetDefaultPrim().GetPath())

    parent_of, is_fixed_joint, joint_name_of, all_bodies = _build_kinematic_tree(stage, root_prim_path)
    root_body_path = _find_root_body(parent_of, all_bodies)
    leaf_body_paths = _find_leaf_bodies(parent_of, all_bodies)

    # actuated 관절을 충분히 거치는 말단만 다리·팔 후보로 남기고, 센서 부착물은 걸러낸다
    limb_leaves = [
        path for path in leaf_body_paths if _actuated_depth(path, parent_of, is_fixed_joint) >= _MIN_LIMB_ACTUATED_DEPTH
    ]
    if not limb_leaves:
        raise RuntimeError(
            f"actuated 관절을 {_MIN_LIMB_ACTUATED_DEPTH}개 이상 거치는 말단 링크(다리 후보)를 찾지 못했습니다."
        )

    if robot_category == "multi-legged":
        # 팔이 없는 다족보행이므로 다리 말단 후보 전부가 발이다
        foot_leaves = limb_leaves
    else:
        # 휴머노이드는 팔(손)도 같은 조건을 만족하므로, authored 자세에서 가장 낮은 것들만 발로 뽑는다
        foot_leaves = sorted(limb_leaves, key=lambda path: _authored_height(stage, path))[:_HUMANOID_FOOT_COUNT]

    foot_short_names = sorted(path.rsplit("/", 1)[-1] for path in foot_leaves)
    leg_joint_names = _leg_chain_joint_names(foot_leaves, parent_of, joint_name_of)
    stiffness, damping = _compute_actuator_gains(stage, root_prim_path, leg_joint_names)

    config = {
        "usd_path": usd_path.name,
        "base_body_name": root_body_path.rsplit("/", 1)[-1],
        "foot_body_names": "|".join(re.escape(name) for name in foot_short_names),
        "actuator": {"type": "simple", "stiffness": stiffness, "damping": damping},
        "action": {"type": "position_only", "scale": 0.5},
        "rewards": _DEFAULT_MULTI_LEGGED_REWARDS if robot_category == "multi-legged" else _DEFAULT_HUMANOID_REWARDS,
        "termination": ["contact"],
        "policy_architecture": "mlp",
        "algorithm": "ppo",
        "domain_randomization": "basic",
        "agent": {
            "num_envs": 4096,
            "max_iterations": 1500 if robot_category == "multi-legged" else 3000,
            "learning_rate": 1.0e-3,
            "entropy_coef": 0.01,
            "hidden_dims": [512, 256, 128],
        },
    }
    if robot_category == "multi-legged":
        parent_short_names = {parent_of[path].rsplit("/", 1)[-1] for path in foot_leaves}
        config["undesired_contact_body_names"] = "|".join(re.escape(name) for name in sorted(parent_short_names))

    default_joint_pos = _compute_default_joint_overrides(stage, root_prim_path)
    if default_joint_pos:
        config["default_joint_pos"] = default_joint_pos

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"[usd_export_config] {output_path} 자동 생성 완료: {config}")

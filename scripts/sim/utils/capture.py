"""Isaac Sim 씬을 카메라로 렌더링해 이미지 파일로 저장하는 유틸리티.

시각 검증이 필요한 모든 파일럿 테스트(01_sim_test 뿐 아니라 이후 controller/GT 단계)가
공통으로 재사용하도록 scripts/sim/main.py 의 개별 작업 함수와 분리해 둔다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
import isaacsim.core.utils.stage as stage_utils
from isaaclab.sensors.camera import Camera, CameraCfg


def spawn_capture_camera(prim_path: str, width: int = 1280, height: int = 720) -> Camera:
    """스크린샷 캡처용 카메라 센서를 스폰한다.

    omni.kit.viewport.utility의 뷰포트 캡처는 완료를 await로 기다려야 하는 비동기 API라
    정지 스크립트에서 완료 시점을 보장하기 어렵다. 대신 렌더 결과를 동기적으로 텐서로
    읽을 수 있는 Camera 센서를 스폰해서 쓴다.
    """
    camera_cfg = CameraCfg(
        prim_path=prim_path,
        update_period=0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    return Camera(cfg=camera_cfg)


def capture_scene_to_file(
    sim: sim_utils.SimulationContext,
    camera: Camera,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    output_path: str,
    max_warm_up_steps: int = 100,
) -> None:
    """카메라를 지정한 시점으로 옮기고, 렌더가 준비되면 결과를 이미지 파일로 저장한다."""
    # world-frame eye/target을 배치 텐서로 변환해 카메라 포즈를 지정
    eye_tensor = torch.tensor([list(eye)], device=sim.device)
    target_tensor = torch.tensor([list(target)], device=sim.device)
    camera.set_world_poses_from_view(eye_tensor, target_tensor)

    # 초기 몇 스텝은 렌더 프로덕트가 인스턴싱 중이라 camera.data.output이 비어있을 수 있어 대기
    for step in range(max_warm_up_steps):
        sim.step()
        camera.update(dt=sim.get_physics_dt())
        if "rgb" in camera.data.output:
            print(f"[capture] 카메라 렌더 완료 (step {step})")
            break
        if step % 10 == 0:
            print(f"[capture] warm-up step {step}/{max_warm_up_steps}")
    else:
        raise RuntimeError("카메라 렌더 결과를 받지 못했습니다 (warm-up 스텝 초과)")

    # 렌더된 RGB 텐서를 이미지 파일로 저장
    rgb_image = camera.data.output["rgb"][0].cpu().numpy().astype(np.uint8)
    Image.fromarray(rgb_image).save(output_path)
    print(f"[capture] 스크린샷 저장 완료: {output_path}")


def set_camera_view(
    sim: sim_utils.SimulationContext,
    camera: Camera,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> None:
    """카메라 포즈만 지정한다(고정된 시점에서 여러 프레임을 녹화할 때 한 번만 호출)."""
    eye_tensor = torch.tensor([list(eye)], device=sim.device)
    target_tensor = torch.tensor([list(target)], device=sim.device)
    camera.set_world_poses_from_view(eye_tensor, target_tensor)


def capture_camera_frame(camera: Camera) -> np.ndarray | None:
    """현재 카메라 렌더 결과 한 프레임을 RGB 배열로 읽어온다. 아직 렌더가 준비 안 됐으면 None."""
    if "rgb" not in camera.data.output:
        return None
    return camera.data.output["rgb"][0].cpu().numpy().astype(np.uint8)


def record_frames_to_video(frames: list[np.ndarray], output_path: str, fps: int = 30) -> None:
    """RGB 프레임 시퀀스를 mp4(H.264)로 저장한다. VS Code 탐색기에서 바로 재생 가능한 포맷."""
    import imageio

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, codec="libx264", format="FFMPEG", pixelformat="yuv420p") as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"[capture] 영상 저장 완료 ({len(frames)}프레임): {output_path}")


def compute_prim_world_bounds(prim_path: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """스테이지에 있는 prim의 월드 좌표계 바운딩 박스를 (min, max)로 계산한다.

    로봇처럼 임포트 시점에 크기를 알 수 없는 대상을 화면 전체에 들어오게 카메라를 잡을 때 사용한다.
    """
    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"바운딩 박스를 계산할 prim이 존재하지 않습니다: {prim_path}")

    # default(=변위 반영)와 render 퍼포즈를 모두 포함해야 스폰 직후의 실제 지오메트리를 놓치지 않는다
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return tuple(aligned_range.GetMin()), tuple(aligned_range.GetMax())


def usd_local_bbox(usd_path: str | Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """usd 파일 자체(라이브 스테이지 아님)를 열어, default prim 자신의 원점 기준 바운딩 박스를 구한다.

    "usd에 저장된 기본(authored) 자세" 기준이라, 조인트 기본값 적용 후 실제 스폰 자세와는 다를
    수 있다 - 접지 높이 추정치·카메라 프레이밍 크기처럼 대략적인 값이면 충분한 곳에 쓴다.
    """
    stage = Usd.Stage.Open(str(usd_path))
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    aligned_range = bbox_cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    return tuple(aligned_range.GetMin()), tuple(aligned_range.GetMax())


def compute_diagonal_view_pose(
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    distance_scale: float = 1.6,
    elevation_ratio: float = 0.35,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """바운딩 박스 중심이 프레임 정중앙에 오는 대각선 위 시점의 (eye, target)을 계산한다.

    거리는 축 하나의 최대 길이가 아니라 대각선 길이 기준으로 잡아야, 옆으로 길거나 위아래로
    긴 대상도 잘리지 않고 전체가 프레임 안에 들어온다. elevation_ratio가 낮을수록(기본 0.35)
    눈높이에 가까워져, 바퀴·다리처럼 아래쪽에 낮게 붙은 부위까지 시야에 들어온다.
    """
    center = tuple((bbox_min[i] + bbox_max[i]) / 2.0 for i in range(3))
    dx, dy, dz = (bbox_max[i] - bbox_min[i] for i in range(3))
    diagonal = (dx**2 + dy**2 + dz**2) ** 0.5
    distance = max(diagonal * distance_scale, 0.5)
    eye = (center[0] + distance, center[1] - distance, center[2] + distance * elevation_ratio)
    return eye, center

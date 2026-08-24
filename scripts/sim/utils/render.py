"""헤드리스 오프스크린 렌더 (Camera 센서 기반): 정지 캡처·주행 비디오·씬 장식.

Isaac 앱이 기동된 뒤에만 임포트할 수 있고, 앱 기동 시 enable_cameras=True가
필요하다 (tools/utils/sim.py launch_app 참고).

뷰포트 캡처(capture_viewport_to_file)는 창 없는 헤드리스에서 파일을 쓰지 못하는
것을 실측으로 확인 -> Camera 센서의 render product 경로를 쓴다 (H200에서 RGB·뎁스
정상 캡처 실증). 3단계 뎁스 렌더링도 같은 경로를 쓰게 된다.

비디오 인코딩: imageio_ffmpeg 번들 ffmpeg로 H.264(yuv420p) mp4를 만든다 —
VSCode 내장 미디어 미리보기가 재생하는 코덱 (시스템 ffmpeg 불필요).

씬 장식(spawn_goal_markers·spawn_breadcrumb·add_ground_grid)은 순수 시각 프림
(물리 속성 없음)이라 시뮬 재생 중에 만들어도 물리에 영향이 없다 — 파일럿
시각 확인 전용 (본 실행·GT 롤아웃에서는 쓰지 않는다).

로봇 색상은 여기서 다루지 않는다 — 스테이지 프림 직접 수정(displayColor)이
렌더에 반영되지 않는 것을 실측해서, 스폰 시점 visual_material로 입힌다
(environment.spawn_robot color 인자).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext

logger = logging.getLogger(__name__)


class SceneCamera:
    """씬 1개에 붙는 캡처용 카메라 (스테이지 재생성 시 함께 재생성해야 함).

    좌표계: eye/target은 월드 좌표 (m). 이미지 좌표는 좌상단 원점 픽셀.
    """

    _PRIM_PATH = "/World/check_camera"

    def __init__(self, render_cfg: dict):
        """카메라 센서 프림을 만든다 (sim.reset() 전에 생성해야 초기화가 묶인다).

        render_cfg: configs/sim.yaml render 섹션 (해상도·워밍업 프레임 수).
        """
        self._cfg = render_cfg
        self._camera = Camera(CameraCfg(
            prim_path=self._PRIM_PATH,
            width=render_cfg["width"],
            height=render_cfg["height"],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.05, 200.0)),
        ))

    def capture_array(self, eye: tuple, target: tuple,
                      warmup: int | None = None) -> np.ndarray | None:
        """지정 시점에서 현재 씬을 렌더해 RGB 배열 (H,W,3) uint8을 반환한다.

        렌더 파이프라인 워밍업을 위해 렌더 프레임만 여러 번 돌린다 (물리는
        진행하지 않음). warmup = 렌더 프레임 수 (None = config 기본값 —
        비디오 연속 캡처는 파이프라인이 데워져 있어 작은 값으로 충분).
        반환: 프레임이 비면 None.
        """
        sim = SimulationContext.instance()
        device = self._camera.device
        self._camera.set_world_poses_from_view(
            eyes=torch.tensor([eye], dtype=torch.float32, device=device),
            targets=torch.tensor([target], dtype=torch.float32, device=device))
        for _ in range(self._cfg["warmup_frames"] if warmup is None else warmup):
            sim.render()
        self._camera.update(dt=0.0)

        rgb = self._camera.data.output["rgb"][0].cpu().numpy()
        if rgb.size == 0 or rgb.max() == rgb.min():
            return None
        return rgb[..., :3].astype(np.uint8)

    def capture_rgb(self, path: Path, eye: tuple, target: tuple) -> bool:
        """지정 시점에서 현재 씬을 렌더해 PNG로 저장한다 (01_sim 정지 렌더용).

        반환: 저장 성공 여부 (프레임이 비면 False).
        """
        rgb = self.capture_array(eye, target)
        if rgb is None:
            logger.warning("렌더 프레임이 비어 있음: %s", path)
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(path)
        return True


class VideoRecorder:
    """주행 비디오 레코더 (프레임 누적 상태 보유 — 시나리오 1회당 1개 생성).

    add_frame으로 시뮬 시각별 프레임을 모으고 save로 mp4(H.264)를 만든다.
    재생 속도 = 실시간 (인코딩 fps = 1 / 캡처 간격).
    """

    def __init__(self, camera: SceneCamera, video_cfg: dict):
        """video_cfg = configs/controller.yaml scenario.video 섹션."""
        self._camera = camera
        self._cfg = video_cfg
        self._frames: list[np.ndarray] = []

    def add_frame(self, eye: tuple, target: tuple):
        """현재 씬을 1프레임 캡처해 누적한다 (첫 프레임만 전체 워밍업).

        연속 캡처는 파이프라인이 데워져 있어 소수 렌더 프레임으로 충분
        (config warmup_frames — 전체 워밍업을 매번 돌리면 비디오가 벽시계를
        지배한다).
        """
        warmup = None if not self._frames else int(self._cfg["warmup_frames"])
        rgb = self._camera.capture_array(eye, target, warmup=warmup)
        if rgb is not None:
            self._frames.append(rgb)

    def save(self, path: Path) -> bool:
        """누적 프레임을 H.264 mp4로 인코딩해 저장하고 버퍼를 비운다.

        yuv420p + libx264는 VSCode 내장 미리보기 재생 코덱 (모듈 docstring).
        홀수 해상도는 yuv420p가 거부하므로 짝수로 자른다. 반환: 성공 여부.
        """
        import imageio_ffmpeg

        if not self._frames:
            logger.warning("비디오 프레임 없음: %s", path)
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = self._frames[0].shape[:2]
        w2, h2 = w - w % 2, h - h % 2
        fps = 1.0 / float(self._cfg["capture_interval_s"])
        writer = imageio_ffmpeg.write_frames(
            str(path), (w2, h2), fps=fps, codec="libx264",
            output_params=["-pix_fmt", "yuv420p"])
        writer.send(None)
        for frame in self._frames:
            writer.send(np.ascontiguousarray(frame[:h2, :w2]))
        writer.close()
        n = len(self._frames)
        self._frames.clear()
        logger.info("비디오 저장: %s (%d프레임, %.1ffps)", path.name, n, fps)
        return True


# ---------- 씬 장식 (파일럿 시각 확인 전용 — 순수 시각 프림) ----------

def _spawn_visual(path: str, cfg, pos: tuple, orientation: tuple | None = None,
                  no_shadow: bool = False):
    """시각 전용 프림 1개를 스폰한다 (물리 속성 없음 — 재생 중 생성 무해).

    no_shadow: RTX 그림자 차단 — 반투명 비콘이 본체는 안 보이고 그림자
    덩어리만 남기는 것 실측 (표식은 그림자가 정보가 아니라 소음).
    """
    prim = cfg.func(path, cfg, translation=pos, orientation=orientation)
    if no_shadow:
        from pxr import Sdf

        prim.CreateAttribute("primvars:doNotCastShadows",
                             Sdf.ValueTypeNames.Bool).Set(True)


def _surface_z(x: float, y: float, fallback: float) -> float:
    """지형 표면 높이를 PhysX 레이캐스트로 샘플한다 (실패 시 fallback).

    경사·계단 씬은 목적지 표면 높이가 스폰 원점과 달라 마커가 파묻히거나
    떠 보인다 — 물리 초기화(env.reset) 이후에만 유효하다.
    """
    try:
        import carb
        from omni.physx import get_physx_scene_query_interface

        probe = 10.0
        hit = get_physx_scene_query_interface().raycast_closest(
            carb.Float3(x, y, fallback + probe), carb.Float3(0.0, 0.0, -1.0),
            2.0 * probe)
        if hit.get("hit"):
            return float(hit["position"][2])
    except Exception:
        logger.warning("표면 높이 레이캐스트 실패 (%.1f, %.1f) — 원점 높이 사용", x, y)
    return fallback


def spawn_goal_markers(goals_xy: list, base_z: float, colors: list,
                       radius: float):
    """웨이포인트마다 바닥 디스크 + 공중 발광 구 마커를 만든다 (색 순환).

    디스크 반지름 = 도달 판정 반경이라 "이 영역에 들어가면 도달"이 화면에서
    그대로 읽히고, 납작한 디스크 + 떠 있는 구라 장애물로 오독되지 않는다
    (기둥형은 장애물로 보이고, 반투명은 RTX 실시간에서 렌더되지 않는 것
    실측). 표면 높이는 레이캐스트 샘플 — env.reset() 이후에 호출할 것.
    goals_xy = 월드 [(x, y)], base_z = 레이캐스트 실패 시 대체 높이 (m).
    """
    for i, (x, y) in enumerate(goals_xy):
        color = tuple(colors[i % len(colors)])
        glow = tuple(min(1.0, 1.3 * c) for c in color)
        z = _surface_z(x, y, base_z)
        disc = sim_utils.CylinderCfg(
            radius=radius, height=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, emissive_color=glow))
        _spawn_visual(f"/World/markers/goal_{i}/disc", disc,
                      (x, y, z + 0.03), no_shadow=True)
        orb = sim_utils.SphereCfg(
            radius=0.09,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, emissive_color=glow))
        _spawn_visual(f"/World/markers/goal_{i}/orb", orb,
                      (x, y, z + 0.85), no_shadow=True)


def spawn_path_segment(index: int, p_prev: tuple, p_cur: tuple, width: float,
                       color: tuple):
    """직전 -> 현재 위치를 잇는 궤적 선분(얇은 박스) 1개를 남긴다.

    시나리오 실행 중 주기 호출 — 점이 아니라 연속선으로 경로가 남는다.
    선분 방향 정렬 = x축을 세그먼트 방향으로 회전 (yaw 후 pitch, roll 0).
    길이가 폭보다 짧으면(정지 구간) 생략한다 (퇴화 방지·클러터 억제).
    """
    dx = p_cur[0] - p_prev[0]
    dy = p_cur[1] - p_prev[1]
    dz = p_cur[2] - p_prev[2]
    length = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if length < width:
        return
    # q = Rz(yaw) Ry(pitch): x축 -> 방향 벡터. 오일러 -> 쿼터니언 (w,x,y,z),
    # roll 0이라 w = cy cp, x = -sy sp, y = cy sp, z = sy cp
    yaw = np.arctan2(dy, dx)
    pitch = -np.arcsin(np.clip(dz / length, -1.0, 1.0))
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    quat = (float(cy * cp), float(-sy * sp), float(cy * sp), float(sy * cp))
    seg = sim_utils.CuboidCfg(
        size=(length, width, 0.4 * width),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)))
    mid = (0.5 * (p_prev[0] + p_cur[0]), 0.5 * (p_prev[1] + p_cur[1]),
           0.5 * (p_prev[2] + p_cur[2]))
    _spawn_visual(f"/World/markers/path_{index}", seg, mid, orientation=quat,
                  no_shadow=True)


def add_ground_grid(half_size: float, spacing: float = 1.0,
                    color: tuple = (0.45, 0.45, 0.5)):
    """평지 씬에 격자선(가는 박스)을 깐다 — 단색 지면의 거리·이동 기준선.

    half_size = 격자 반폭 (m). 지형 씬은 높이 색상이 있어 불필요.
    """
    n = int(half_size / spacing)
    line = sim_utils.CuboidCfg(
        size=(2.0 * half_size, 0.02, 0.005),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
    for i in range(-n, n + 1):
        _spawn_visual(f"/World/markers/grid_x{i + n}", line,
                      (0.0, i * spacing, 0.005))
    line_y = sim_utils.CuboidCfg(
        size=(0.02, 2.0 * half_size, 0.005),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
    for i in range(-n, n + 1):
        _spawn_visual(f"/World/markers/grid_y{i + n}", line_y,
                      (i * spacing, 0.0, 0.005))

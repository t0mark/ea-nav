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

    _PRIM_PATH = "/World/check_camera"

    def __init__(self, render_cfg: dict):

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

        rgb = self.capture_array(eye, target)
        if rgb is None:
            logger.warning("렌더 프레임이 비어 있음: %s", path)
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(path)
        return True

class VideoRecorder:

    def __init__(self, camera: SceneCamera, video_cfg: dict):

        self._camera = camera
        self._cfg = video_cfg
        self._frames: list[np.ndarray] = []

    def add_frame(self, eye: tuple, target: tuple):

        warmup = None if not self._frames else int(self._cfg["warmup_frames"])
        rgb = self._camera.capture_array(eye, target, warmup=warmup)
        if rgb is not None:
            self._frames.append(rgb)

    def save(self, path: Path) -> bool:

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

def _spawn_visual(path: str, cfg, pos: tuple, orientation: tuple | None = None,
                  no_shadow: bool = False):

    prim = cfg.func(path, cfg, translation=pos, orientation=orientation)
    if no_shadow:
        from pxr import Sdf

        prim.CreateAttribute("primvars:doNotCastShadows",
                             Sdf.ValueTypeNames.Bool).Set(True)

def _surface_z(x: float, y: float, fallback: float) -> float:

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

    dx = p_cur[0] - p_prev[0]
    dy = p_cur[1] - p_prev[1]
    dz = p_cur[2] - p_prev[2]
    length = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if length < width:
        return

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

def structure_band_center(terrain_size: tuple[float, float],
                          band: tuple[float, float], gap: float)        -> tuple[float, float]:

    return (0.5 * terrain_size[0] + gap + 0.5 * band[0], 0.0)

def topview_camera(terrain_size: tuple[float, float], band: tuple[float, float],
                   band_center: tuple[float, float], height_scale: float)        -> tuple[tuple[float, float, float], tuple[float, float, float]]:

    x_min, x_max = -0.5 * terrain_size[0], 0.5 * terrain_size[0]
    y_min, y_max = -0.5 * terrain_size[1], 0.5 * terrain_size[1]
    if band[0] > 0.0:
        x_min = min(x_min, band_center[0] - 0.5 * band[0])
        x_max = max(x_max, band_center[0] + 0.5 * band[0])
        y_min = min(y_min, band_center[1] - 0.5 * band[1])
        y_max = max(y_max, band_center[1] + 0.5 * band[1])
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    height = height_scale * max(x_max - x_min, y_max - y_min)

    return (cx, cy - 0.02 * height, height), (cx, cy, 0.0)

def chase_camera_context(camera: SceneCamera, video_cfg: dict, render_cfg: dict,
                         video_path: Path, size: float, color: tuple | None,
                         base_height: float) -> dict:

    return {
        "recorder": VideoRecorder(camera, video_cfg),
        "video_path": video_path,
        "dist": max(render_cfg["cam_dist_min"], render_cfg["cam_dist_scale"] * size),
        "line_width": max(0.025, 0.03 * size),
        "line_color": tuple(0.35 * c for c in (color or (0.6, 0.6, 0.6))),
        "line_drop": max(0.0, base_height - 0.04),
    }

def add_ground_grid(half_size: float, spacing: float = 1.0,
                    color: tuple = (0.45, 0.45, 0.5)):

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

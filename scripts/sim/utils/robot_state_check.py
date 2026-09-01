from __future__ import annotations

import json
import logging
from pathlib import Path

from scripts.sim.utils import robot_spawn
from scripts.sim.utils.environment import SimEnvironment
from scripts.sim.utils.render import SceneCamera
from scripts.sim.utils.robot_behavior import StandingCriteria

logger = logging.getLogger(__name__)

def standing_hold(usd_dir: Path, cfg: dict, device: str, steps: int,
                  color: tuple | None = None,
                  render_path: Path | None = None) -> dict:

    run_cfg, contact_cfg, render_cfg = cfg["sim_run"], cfg["contact"], cfg["render"]
    with open(usd_dir / "meta.json") as f:
        meta = json.load(f)

    env = SimEnvironment(run_cfg["physics_dt"], device)
    env.add_ground(run_cfg["ground_size"], friction=contact_cfg["ground_friction"])
    robot_spawn.spawn_with_standard_friction(
        env, usd_dir, meta, cfg["drive"], contact_cfg, num_envs=1,
        spawn_margin=run_cfg["spawn_margin"], color=color,
        self_collision=run_cfg["self_collision"])

    camera = SceneCamera(render_cfg) if render_path is not None else None
    env.reset()
    for _ in range(steps):
        env.hold_step()

    data = env.robot.data
    pos = data.root_pos_w[0].tolist()
    quat = data.root_quat_w[0].tolist()
    result = StandingCriteria(run_cfg["success"], meta).evaluate(pos, quat)
    result["rendered"] = False

    if camera is not None:

        metrics = meta["metrics"]
        size = max(metrics["overall_length"], metrics["overall_width"],
                   metrics["overall_height"])
        dist = max(render_cfg["cam_dist_min"], render_cfg["cam_dist_scale"] * size)
        target = (pos[0], pos[1], 0.5 * metrics["overall_height"])
        eye = (target[0] + dist, target[1] + dist, target[2] + 0.6 * dist)
        result["rendered"] = camera.capture_rgb(render_path, eye, target)
    return result

def standing_batch(robot_dirs: list[Path], robots_root: Path, usd_root: Path,
                   cfg: dict, device: str, convert_report: dict,
                   validate_report: dict, render_dir: Path | None = None) -> dict:

    run_cfg, render_cfg = cfg["sim_run"], cfg["render"]
    steps = int(round(run_cfg["run_time"] / run_cfg["physics_dt"]))
    result = {}
    for robot_dir in robot_dirs:
        rel = str(robot_dir.relative_to(robots_root))
        if not convert_report[rel]["ok"] or not validate_report.get(rel, {}).get("pass"):
            continue

        form = rel.split("/")[0]
        color = render_cfg["form_colors"].get(form, render_cfg["form_colors"]["default"])            if render_dir is not None else None

        render_path = render_dir / f"{rel.split('/')[-1]}.png"            if render_dir is not None else None
        try:
            r = standing_hold(usd_root / rel, cfg, device, steps, color, render_path)
            logger.info("시뮬 %s %s (높이 %.3f/%.3fm, 기울기 %.1f/%.1fdeg%s%s)", rel,
                        r["status"].upper(), r["final_height"], r["height_min"],
                        r["tilt_deg"], r["tilt_max_deg"],
                        ", 원인 " + ",".join(r["fail_reasons"]) if r["fail_reasons"] else "",
                        ", 렌더 저장" if r["rendered"] else "")
        except Exception as e:
            r = {"ok": False, "status": "fail", "fail_reasons": ["exception"],
                 "error": str(e), "rendered": False}
            logger.error("시뮬 예외 %s: %s", rel, e)
        result[rel] = r
    return result

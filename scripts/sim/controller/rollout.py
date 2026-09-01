from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from isaaclab.sensors import RayCaster

from scripts.sim.controller.core import scan_terrain
from scripts.sim.controller.core.base import (LEGGED_TAGS, ControlObs,
                                              extract_ctrl_params,
                                              make_controller)
from scripts.sim.controller.core.scan_terrain import TerrainScan
from scripts.sim.controller.legged import low_rl
from scripts.sim.utils import render as render_utils
from scripts.sim.utils import robot_spawn
from scripts.sim.utils.environment import SimEnvironment, attach_contact_sensor
from scripts.sim.utils.render import SceneCamera
from tools.utils.common import save_figure

def timeout(ctrl_cfg: dict, controller, waypoints: list) -> float:

    scn = ctrl_cfg["scenario"]
    nav = controller.nav_limits
    v = float(scn["timeout_v_ratio"]) * nav["v_max"]

    length, prev = 0.0, (0.0, 0.0)
    for wp in waypoints:
        length += math.hypot(wp[0] - prev[0], wp[1] - prev[1])
        prev = tuple(wp)
    need = length / v + len(waypoints) * 2.0 * math.pi * nav["r_turn"] / v
    return min(max(scn["timeout_margin"] * need, scn["timeout_min"]),
               scn["timeout_max"])

def build_scene(env: SimEnvironment, scene: str, sim_cfg: dict, scn: dict,
                decorate: bool) -> tuple:

    if scene == "flat":
        env.add_ground(sim_cfg["sim_run"]["ground_size"],
                       friction=sim_cfg["contact"]["ground_friction"])
        if decorate:
            render_utils.add_ground_grid(half_size=8.0)
        return (0.0, 0.0, 0.0)

    et = scn["eval_terrain"]
    tcfg = dict(sim_cfg["terrain"])
    tcfg.update(num_rows=1, num_cols=1, curriculum=False,
                size=list(et["size"]))
    for key in ("stairs_proportion", "stairs_inv_proportion",
                "boxes_proportion", "rough_proportion",
                "slope_proportion", "slope_inv_proportion"):
        tcfg[key] = 0.0
    if scene == "slope":
        tcfg["slope_inv_proportion"] = 1.0
        tcfg["slope_range"] = [et["slope_angle"], et["slope_angle"]]
    elif scene == "stairs":
        tcfg["stairs_inv_proportion"] = 1.0
        tcfg["step_height"] = [et["stair_height"], et["stair_height"]]
    else:
        raise ValueError(f"알 수 없는 씬: {scene}")

    env.add_terrain(tcfg, num_envs=1, color_scheme="none", use_cache=True)
    return tuple(env.terrain.env_origins[0].tolist())

def prepare_legged_spawn(urdf_path: Path, usd_dir: Path, policy_dir: Path):

    with open(policy_dir / "bundle.json") as f:
        meta = json.load(f)
    slots = low_rl.build_slots(urdf_path, usd_dir, meta["morph"])
    return low_rl.loco_gain_overrides(slots, meta["gains"]), slots, meta

class DebuggingTrace:

    def __init__(self, base_prim: str, wheel_names: list[str] | None,
                 wheel_link_names: list[str] | None,
                 wheel_effort_limit: dict, history_length: int):

        self._base_sensor = attach_contact_sensor(base_prim, history_length)
        self._wheel_names = list(wheel_names or [])
        parent = base_prim.rsplit("/", 1)[0]
        self._wheel_sensors = [
            attach_contact_sensor(f"{parent}/{link}", history_length)
            for link in (wheel_link_names or [])]
        self._wheel_effort_limit = wheel_effort_limit
        self._rows: list[dict] = []

    def update(self, physics_dt: float):

        self._base_sensor.update(physics_dt)
        for s in self._wheel_sensors:
            s.update(physics_dt)

    @staticmethod
    def _contact_force(sensor) -> float:

        hist = sensor.data.net_forces_w_history[:, :, 0]
        return float(torch.norm(hist, dim=-1).max())

    def record(self, t: float, wp: int, origin, robot, wheel_idx,
              targets, goal_dist: float):

        pos3 = robot.data.root_pos_w[0]
        row = {
            "t": round(t, 4), "wp": int(wp),
            "pos_xy": [round(float(pos3[0] - origin[0]), 4),
                       round(float(pos3[1] - origin[1]), 4)],
            "root_z": round(float(pos3[2] - origin[2]), 4),
            "goal_dist_m": round(goal_dist, 4),
            "vel_b": [round(float(v), 4) for v in robot.data.root_lin_vel_b[0].tolist()],
            "ang_b": [round(float(v), 4) for v in robot.data.root_ang_vel_b[0].tolist()],
            "base_contact_force": round(self._contact_force(self._base_sensor), 3),
        }
        if targets.cmd is not None:
            row["cmd"] = [round(float(v), 4) for v in targets.cmd[0].tolist()]
        if wheel_idx is not None and self._wheel_names:
            qd = robot.data.joint_vel[0, wheel_idx]
            tau = robot.data.applied_torque[0, wheel_idx]
            row["wheel_names"] = list(self._wheel_names)
            row["wheel_vel"] = [round(float(v), 4) for v in qd.tolist()]
            row["wheel_tau"] = [round(float(v), 4) for v in tau.tolist()]
            if targets.vel is not None:
                row["wheel_vel_target"] = [round(float(v), 4)
                                           for v in targets.vel[0, wheel_idx].tolist()]
            limits = [float(self._wheel_effort_limit.get(n, 0.0))
                     for n in self._wheel_names]
            if any(limits):
                sat = [abs(float(v)) / max(l, 1e-6)
                      for v, l in zip(tau.tolist(), limits)]
                row["wheel_tau_ratio_max"] = round(max(sat), 4)
            row["wheel_contact_force"] = [round(self._contact_force(s), 3)
                                          for s in self._wheel_sensors]
        self._rows.append(row)

    def save(self, path: Path, result: dict, static_meta: dict):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"static": static_meta, "result": result,
                       "samples": self._rows}, f, indent=1, ensure_ascii=False)

def simulate_scenario(env: SimEnvironment, controller, ctrl_cfg: dict,
                      physics_dt: float, base_sensor, visual: dict | None = None,
                      scan_ctx: dict | None = None,
                      debugging_trace: DebuggingTrace | None = None)        -> tuple[dict, list]:

    scn = ctrl_cfg["scenario"]
    rel_wps = scn["waypoints"]
    timeout_s = timeout(ctrl_cfg, controller, rel_wps)
    device = env.origins.device
    origin = env.origins[0]
    origin_xy = origin[:2]
    goals = [origin_xy + torch.tensor(wp, dtype=torch.float32, device=device)
             for wp in rel_wps]

    tilt_cos = math.cos(math.radians(float(scn["fall_tilt_deg"])))
    contact_thresh = float(scn["base_contact_thresh"])

    if visual is not None:
        render_utils.spawn_goal_markers([g.tolist() for g in goals],
                                        base_z=float(origin[2]),
                                        colors=scn["marker_colors"],
                                        radius=float(scn["reach_radius"]))

    legged_ctrl = controller.params.base_tag in LEGGED_TAGS
    if not legged_ctrl:
        for _ in range(int(round(scn["settle_time"] / physics_dt))):
            env.hold_step()
    controller.reset()

    record_every = max(1, int(round(scn["traj_interval"] / physics_dt)))
    total_steps = int(round(timeout_s / physics_dt))
    video_every = max(1, int(round(scn["video"]["capture_interval_s"] / physics_dt)))
    path_every = max(1, int(round(scn["video"]["path_interval_s"] / physics_dt)))

    wheel_idx = getattr(controller, "_wheel_idx", None)
    wheel_names = getattr(controller, "_wheel_names", None)

    traj, reach_times = [], [None] * len(goals)
    wp_i, fell = 0, False
    path_prev, n_segs = None, 0
    step = 0
    for step in range(total_steps):
        goal = goals[wp_i]
        if step % controller.decimation == 0:
            obs = ControlObs.from_articulation(env.robot)

            if scan_ctx is not None:
                scan_ctx["scanner"].update(physics_dt * controller.decimation)
                if legged_ctrl:
                    obs.height_scan = low_rl.scan_obs(
                        env.robot.data.root_pos_w[:, 2],
                        scan_ctx["scanner"].data.ray_hits_w[:, :, 2],
                        scan_ctx["base_height"], scan_ctx["clip"])
                else:
                    obs.terrain_scan = TerrainScan.from_ray_hits(
                        scan_ctx["scanner"].data.ray_hits_w, scan_ctx["nx"],
                        scan_ctx["ny"], scan_ctx["resolution"])
            targets = controller.compute(obs, goal.unsqueeze(0))
            env.step(targets.pos, targets.vel, targets.effort)
        else:
            env.step()
        base_sensor.update(physics_dt)
        if debugging_trace is not None:
            debugging_trace.update(physics_dt)

        pos3 = env.robot.data.root_pos_w[0]
        pos = pos3[:2]
        if step % record_every == 0:
            rel = (pos - origin_xy).tolist()
            traj.append([round(rel[0], 4), round(rel[1], 4)])

        if visual is not None:
            if step % video_every == 0:
                q = env.robot.data.root_quat_w[0]
                yaw = math.atan2(2.0 * float(q[0] * q[3] + q[1] * q[2]),
                                 1.0 - 2.0 * float(q[2] * q[2] + q[3] * q[3]))
                p = pos3.tolist()
                d = visual["dist"]
                dx, dy = math.cos(yaw), math.sin(yaw)
                eye = (p[0] - d * dx, p[1] - d * dy, p[2] + 0.55 * d)
                tgt = (p[0] + 0.4 * d * dx, p[1] + 0.4 * d * dy, p[2])

                prev = visual.get("_cam")
                if prev is not None:
                    a = 0.25
                    eye = tuple(a * n + (1 - a) * o for n, o in zip(eye, prev[0]))
                    tgt = tuple(a * n + (1 - a) * o for n, o in zip(tgt, prev[1]))
                visual["_cam"] = (eye, tgt)
                visual["recorder"].add_frame(eye, tgt)
            if step % path_every == 0:

                p3 = pos3.tolist()
                p = (p3[0], p3[1], p3[2] - visual["line_drop"])
                if path_prev is not None:
                    render_utils.spawn_path_segment(n_segs, path_prev, p,
                                                    visual["line_width"],
                                                    visual["line_color"])
                    n_segs += 1
                path_prev = p
        dist_to_goal = float(torch.norm(pos - goals[wp_i]))
        if debugging_trace is not None and step % controller.decimation == 0:
            debugging_trace.record(t=step * physics_dt, wp=wp_i, origin=origin,
                                   robot=env.robot, wheel_idx=wheel_idx,
                                   targets=targets, goal_dist=dist_to_goal)

        base_force = float(torch.norm(
            base_sensor.data.net_forces_w_history[:, :, 0], dim=-1).max())
        tilted = bool(obs.gravity_b[0, 2] > -tilt_cos)
        if base_force > contact_thresh or tilted:
            fell = True
            break

        if dist_to_goal < scn["reach_radius"]:
            reach_times[wp_i] = round((step + 1) * physics_dt, 2)
            wp_i += 1
            if wp_i == len(goals):
                break

    final_dist = float(torch.norm(env.robot.data.root_pos_w[0, :2] - goals[-1]))
    result = {
        "ok": wp_i == len(goals) and not fell,
        "control_tag": controller.params.control_tag,
        "reached": wp_i, "n_waypoints": len(goals),
        "reach_times_s": reach_times,
        "final_dist_m": round(final_dist, 3),
        "fell": fell,
        "sim_time_s": round((step + 1) * physics_dt, 2),
        "timeout_s": round(timeout_s, 1),
    }
    if visual is not None:
        result["video"] = visual["recorder"].save(visual["video_path"])
    return result, traj

def save_plot(path: Path, traj: list, waypoints: list, reach_radius: float,
             title: str):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    xs, ys = [p[0] for p in traj], [p[1] for p in traj]
    ax.plot(xs, ys, "-", color="tab:blue", lw=1.5, label="궤적")
    ax.plot(0, 0, "o", color="black", ms=8, label="스폰")
    for i, (wx, wy) in enumerate(waypoints):
        ax.add_patch(plt.Circle((wx, wy), reach_radius, color="tab:green",
                                alpha=0.25))
        ax.plot(wx, wy, "*", color="tab:green", ms=14)
        ax.annotate(f"wp{i}", (wx, wy), textcoords="offset points",
                    xytext=(6, 6))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("x [m] (스폰 기준)")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    save_figure(fig, path)

def evaluate_scene(usd_dir: Path, urdf_path: Path, rel: str, form: str, scene: str,
                   sim_cfg: dict, ctrl_cfg: dict, device: str, policy_dirs: dict,
                   pilot: bool, out_root: Path, debugging: bool = False) -> dict:

    run_cfg = sim_cfg["sim_run"]
    render_cfg = sim_cfg["render"]
    scn = ctrl_cfg["scenario"]
    legged = form in LEGGED_TAGS

    with open(usd_dir / "meta.json") as f:
        meta = json.load(f)
    with open(usd_dir / "joints.json") as f:
        base_link = json.load(f)["base_link"]

    wheel_params = extract_ctrl_params(urdf_path, usd_dir)
    wheel_names = [w.joint for w in wheel_params.wheels]
    wheel_link_names = [w.link for w in wheel_params.wheels]
    wheel_effort_limit = wheel_params.wheel_effort_limit

    overrides, slots, bundle_meta = None, None, None
    if legged:
        overrides, slots, bundle_meta = prepare_legged_spawn(
            urdf_path, usd_dir, policy_dirs[form])

    env = SimEnvironment(run_cfg["physics_dt"], device)
    origin = build_scene(env, scene, sim_cfg, scn, decorate=pilot)
    color = render_cfg["form_colors"].get(form) if pilot else None
    robot_spawn.spawn_with_standard_friction(
        env, usd_dir, meta, sim_cfg["drive"], sim_cfg["contact"], num_envs=1,
        spawn_margin=run_cfg["spawn_margin"], color=color,

        self_collision=(False if legged else run_cfg["self_collision"]),
        gain_overrides=overrides, origin=origin,

        friction_links=(list(slots.contact_links) if slots else None),
        solver_iters=((4, 0) if legged else None),

        actuator_model=(bundle_meta.get("actuator_model", "implicit")
                        if bundle_meta else "implicit"),

        activate_contact_sensors=True)

    camera = SceneCamera(render_cfg) if (pilot and not debugging) else None
    base_prim = f"/World/envs/env_0/Robot/{base_link}"
    base_sensor = attach_contact_sensor(base_prim, int(ctrl_cfg["ctrl"]["decimation"]))
    debugging_trace = DebuggingTrace(
        base_prim=base_prim, wheel_names=wheel_names,
        wheel_link_names=wheel_link_names, wheel_effort_limit=wheel_effort_limit,
        history_length=int(ctrl_cfg["ctrl"]["decimation"])) if debugging else None
    scan_ctx = None
    if scene != "flat":
        if legged:
            # legged RL 정책의 proprioceptive height scan — 로봇 정면 기준(yaw 정렬)
            sc = bundle_meta["scan"]
            prim = f"/World/envs/env_0/Robot/{slots.root_link}"
            scanner = RayCaster(scan_terrain.scanner_cfg(prim, sc, alignment="yaw"))
            scan_ctx = {"scanner": scanner,
                        "base_height": float(meta["metrics"]["base_height"]),
                        "clip": float(sc["clip"])}
        else:
            # wheeled MPPI 로컬 플래너용 지형 격자 — world 축정렬(로봇 위치만 따라 이동)
            sc = ctrl_cfg["mppi"]["scan"]
            prim = f"/World/envs/env_0/Robot/{base_link}"
            scanner = RayCaster(scan_terrain.scanner_cfg(prim, sc, alignment="world"))
            nx, ny = scan_terrain.grid_shape(sc)
            scan_ctx = {"scanner": scanner, "nx": nx, "ny": ny,
                        "resolution": float(sc["resolution"])}
    env.reset()

    controller = make_controller(
        urdf_path, usd_dir, ctrl_cfg,
        joint_names=env.robot.joint_names,
        default_pose=env.robot.data.default_joint_pos.clone(),
        num_envs=1, device=device,
        physics_dt=run_cfg["physics_dt"],
        policy_dir=policy_dirs.get(form))

    stem = f"{rel.split('/')[-1]}__{scene}"

    visual = None
    out_dir = Path(out_root) / ("legged" if legged else "wheeled")
    if pilot and not debugging:
        metrics = meta["metrics"]
        size = max(metrics["overall_length"], metrics["overall_width"],
                   metrics["overall_height"])
        visual = render_utils.chase_camera_context(
            camera, scn["video"], render_cfg,
            out_dir / "videos" / f"{stem}.mp4", size, color,
            float(metrics["base_height"]))

    result, traj = simulate_scenario(env, controller, ctrl_cfg,
                                     run_cfg["physics_dt"], base_sensor,
                                     visual=visual, scan_ctx=scan_ctx,
                                     debugging_trace=debugging_trace)
    if pilot:

        save_plot(out_dir / "plots" / f"{stem}.png",
                  traj, scn["waypoints"], scn["reach_radius"],
                  f"{rel} [{scene}] ({result['control_tag']}) "
                  f"{result['reached']}/{result['n_waypoints']} 도달")
    if debugging_trace is not None:

        static_meta = {
            "control_tag": meta["control_tag"], "form": form,
            "total_mass": meta["metrics"].get("total_mass"),
            "clearance": meta["metrics"].get("clearance"),
            "wheel_radius": meta["params"].get("wheel_radius"),
            "track_width": meta["params"].get("track_width"),
            "wheelbase": meta["params"].get("wheelbase"),
            "tip_accel": controller.params.tip_accel,
        }
        debugging_trace.save(out_dir / "debugging_trace" / f"{stem}.json",
                             result, static_meta)
    return result

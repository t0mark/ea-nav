from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrains
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaacsim.core.cloner import Cloner, GridCloner
from omni.physx.scripts import physicsUtils
from pxr import Gf, Usd, UsdGeom, UsdShade

from scripts.sim.utils.robot_spawn import make_articulation_cfg

logger = logging.getLogger(__name__)

class SimEnvironment:

    _GROUND_PATH = "/World/ground"
    _ENV_NS = "/World/envs"

    def __init__(self, physics_dt: float, device: str):

        self._teardown_previous()
        self._sim = SimulationContext(SimulationCfg(dt=physics_dt, device=device))
        self._dt = physics_dt

        self._groups: list[Articulation] = []
        self._slices: list[slice] = []
        self._origins: torch.Tensor | None = None
        self._terrain: terrains.TerrainImporter | None = None

    @property
    def sim(self) -> SimulationContext:

        return self._sim

    @property
    def robot(self) -> Articulation:

        if len(self._groups) != 1:
            raise RuntimeError("robot은 spawn_robot() 단일 스폰 후에만 유효하다"
                               " (다중 그룹은 robots 사용)")
        return self._groups[0]

    @property
    def robots(self) -> list[Articulation]:

        if not self._groups:
            raise RuntimeError("스폰 이전에는 robots에 접근할 수 없다")
        return self._groups

    def group_slice(self, g: int) -> slice:

        return self._slices[g]

    @property
    def origins(self) -> torch.Tensor:

        if self._origins is None:
            raise RuntimeError("스폰 이전에는 origins에 접근할 수 없다")
        return self._origins

    @staticmethod
    def _teardown_previous():

        prev = SimulationContext.instance()
        if prev is not None:
            prev._disable_app_control_on_stop_handle = True
            prev.stop()
            prev.clear_all_callbacks()
            prev.clear_instance()
        stage_utils.create_new_stage()

    @staticmethod
    def _add_light():

        light = sim_utils.DistantLightCfg(intensity=2500.0)
        light.func("/World/light", light,
                   orientation=(0.9397, 0.0, 0.342, 0.0))

        dome = sim_utils.DomeLightCfg(intensity=150.0)
        dome.func("/World/dome_light", dome)

    def add_ground(self, size: float = 20.0, friction: float = 1.0):

        stage = stage_utils.get_current_stage()
        physicsUtils.add_ground_plane(stage, self._GROUND_PATH,
                                      "Z", size, Gf.Vec3f(0.0), Gf.Vec3f(0.2))
        mat_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=friction, dynamic_friction=friction,
            friction_combine_mode="multiply", restitution_combine_mode="multiply")
        mat_cfg.func("/World/Materials/ground", mat_cfg)
        material = UsdShade.Material(stage.GetPrimAtPath("/World/Materials/ground"))
        plane = stage.GetPrimAtPath(f"{self._GROUND_PATH}/CollisionPlane")

        if not plane.IsValid():
            logger.warning("지면 충돌 프림 미발견 — 마찰 재질 미적용 (기본 마찰로 동작)")
        else:
            UsdShade.MaterialBindingAPI.Apply(plane).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics")
            logger.info("평지 마찰 재질 바인딩: mu %.2f (%s)", friction,
                        plane.GetPath())
        self._add_light()

    @property
    def terrain(self) -> terrains.TerrainImporter:

        if self._terrain is None:
            raise RuntimeError("add_terrain() 이전에는 terrain에 접근할 수 없다")
        return self._terrain

    def add_terrain(self, terrain_cfg: dict, num_envs: int = 1,
                    color_scheme: str = "height",
                    use_cache: bool = False) -> tuple[float, float]:

        sub_terrains = {
            "pyramid_stairs": terrains.MeshPyramidStairsTerrainCfg(
                proportion=terrain_cfg["stairs_proportion"],
                step_height_range=tuple(terrain_cfg["step_height"]),
                step_width=terrain_cfg["step_width"],
                platform_width=terrain_cfg["stairs_platform_width"],
                border_width=terrain_cfg["stairs_border"], holes=False),
            "pyramid_stairs_inv": terrains.MeshInvertedPyramidStairsTerrainCfg(
                proportion=terrain_cfg["stairs_inv_proportion"],
                step_height_range=tuple(terrain_cfg["step_height"]),
                step_width=terrain_cfg["step_width"],
                platform_width=terrain_cfg["stairs_platform_width"],
                border_width=terrain_cfg["stairs_border"], holes=False),
            "boxes": terrains.MeshRandomGridTerrainCfg(
                proportion=terrain_cfg["boxes_proportion"],
                grid_width=terrain_cfg["grid_width"],
                grid_height_range=tuple(terrain_cfg["grid_height"]),
                platform_width=terrain_cfg["platform_width"]),
            "random_rough": terrains.HfRandomUniformTerrainCfg(
                proportion=terrain_cfg["rough_proportion"],
                noise_range=tuple(terrain_cfg["rough_noise"]), noise_step=0.02,
                border_width=terrain_cfg["hf_border"]),
            "hf_pyramid_slope": terrains.HfPyramidSlopedTerrainCfg(
                proportion=terrain_cfg["slope_proportion"],
                slope_range=tuple(terrain_cfg["slope_range"]),
                platform_width=terrain_cfg["platform_width"],
                border_width=terrain_cfg["hf_border"]),
            "hf_pyramid_slope_inv": terrains.HfInvertedPyramidSlopedTerrainCfg(
                proportion=terrain_cfg["slope_inv_proportion"],
                slope_range=tuple(terrain_cfg["slope_range"]),
                platform_width=terrain_cfg["platform_width"],
                border_width=terrain_cfg["hf_border"]),
        }
        generator_cfg = terrains.TerrainGeneratorCfg(
            size=tuple(terrain_cfg["size"]),
            border_width=terrain_cfg["border_width"],
            num_rows=terrain_cfg["num_rows"],
            num_cols=terrain_cfg["num_cols"],
            curriculum=terrain_cfg["curriculum"],
            sub_terrains=sub_terrains,
            color_scheme=color_scheme,
            use_cache=use_cache,
        )

        importer_cfg = terrains.TerrainImporterCfg(
            prim_path=self._GROUND_PATH,
            terrain_type="generator",
            terrain_generator=generator_cfg,
            visual_material=None,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=terrain_cfg["static_friction"],
                dynamic_friction=terrain_cfg["dynamic_friction"]),
            max_init_terrain_level=terrain_cfg["max_init_terrain_level"],
            num_envs=num_envs,
            collision_group=-1,
        )
        self._terrain = terrains.TerrainImporter(importer_cfg)
        if color_scheme == "none":
            self._paint_terrain_by_slope()
        self._add_light()

        size_x = terrain_cfg["num_rows"] * terrain_cfg["size"][0] + 2 * terrain_cfg["border_width"]
        size_y = terrain_cfg["num_cols"] * terrain_cfg["size"][1] + 2 * terrain_cfg["border_width"]
        logger.info("지형 생성: %d x %d 칸, 전체 %.1f x %.1f m",
                    terrain_cfg["num_rows"], terrain_cfg["num_cols"], size_x, size_y)
        return size_x, size_y

    def add_structures(self, structure_cfg: dict,
                       center: tuple[float, float]) -> list[dict]:

        from scripts.sim.utils.structures import StructureBuilder

        if not structure_cfg.get("enabled", False):
            logger.info("구조물 배치 생략 (structures.enabled false)")
            return []
        return StructureBuilder(structure_cfg).build(center)

    def _paint_terrain_by_slope(self):

        from pxr import Sdf

        flat_c = (0.30, 0.30, 0.32)
        steep_c = (0.15, 0.15, 0.17)
        quad_c = {
            "+x": (0.40, 0.30, 0.22), "-x": (0.22, 0.30, 0.40),
            "+y": (0.25, 0.37, 0.25), "-y": (0.37, 0.26, 0.37),
        }
        stage = stage_utils.get_current_stage()
        ground = stage.GetPrimAtPath(self._GROUND_PATH)
        for prim in Usd.PrimRange(ground, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get())
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            if pts.size == 0 or counts.size == 0 or not (counts == 3).all():

                continue
            idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)

            v0, v1, v2 = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
            n = np.cross(v1 - v0, v2 - v0)
            n /= np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)

            n[n[:, 2] < 0.0] *= -1.0

            centers = (v0 + v1 + v2) / 3.0
            bin_size = 0.5
            bx = np.floor(centers[:, 0] / bin_size).astype(np.int64)
            by = np.floor(centers[:, 1] / bin_size).astype(np.int64)
            bx -= bx.min()
            by -= by.min()
            bin_id = bx * (by.max() + 1) + by
            n_bins = int(bin_id.max()) + 1
            avg = np.zeros((n_bins, 3))
            np.add.at(avg, bin_id, n)
            avg /= np.clip(np.linalg.norm(avg, axis=1, keepdims=True), 1e-9, None)

            bin_colors = np.tile(np.array(flat_c), (n_bins, 1))
            bnz = avg[:, 2]
            sloped = bnz <= 0.997
            east = np.abs(avg[:, 0]) >= np.abs(avg[:, 1])
            bin_colors[sloped & east & (avg[:, 0] >= 0)] = quad_c["+x"]
            bin_colors[sloped & east & (avg[:, 0] < 0)] = quad_c["-x"]
            bin_colors[sloped & ~east & (avg[:, 1] >= 0)] = quad_c["+y"]
            bin_colors[sloped & ~east & (avg[:, 1] < 0)] = quad_c["-y"]
            colors = bin_colors[bin_id]
            colors[n[:, 2] < 0.35] = steep_c
            primvar = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.uniform)
            primvar.Set([Gf.Vec3f(*c) for c in colors])

    def spawn_robot(self, usd_dir: Path, drive_cfg: dict, num_envs: int = 1,
                    spacing: float = 3.0, spawn_margin: float = 0.01,
                    color: tuple | None = None,
                    self_collision: bool = True,
                    gain_overrides: dict | None = None,
                    contact_cfg: dict | None = None,
                    origin: tuple | None = None,
                    friction_links: list[str] | None = None,
                    friction_links_mu: float = 1.0,
                    solver_iters: tuple[int, int] | None = None,
                    actuator_model: str = "implicit",
                    activate_contact_sensors: bool = False) -> Articulation:

        usd_dir = Path(usd_dir)

        prim_utils.define_prim(f"{self._ENV_NS}/env_0")
        material = None

        art_props = dict(enabled_self_collisions=self_collision,
                         sleep_threshold=0.0, stabilization_threshold=0.0)
        if solver_iters is not None:
            art_props.update(solver_position_iteration_count=solver_iters[0],
                             solver_velocity_iteration_count=solver_iters[1])
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=str(usd_dir / "robot.usd"),
            visual_material=material,
            activate_contact_sensors=activate_contact_sensors,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(**art_props))
        spawn_cfg.func(f"{self._ENV_NS}/env_0/Robot", spawn_cfg)

        if contact_cfg is not None:
            self._bind_passive_friction(f"{self._ENV_NS}/env_0/Robot", contact_cfg)

        if friction_links:
            self.bind_link_friction(f"{self._ENV_NS}/env_0/Robot",
                                    friction_links, friction_links_mu,
                                    "/World/Materials/contact_links")

        if color is not None:
            self._bind_link_palette(f"{self._ENV_NS}/env_0/Robot", color)
        cloner = GridCloner(spacing=spacing)
        cloner.define_base_env(self._ENV_NS)
        env_paths = cloner.generate_paths(f"{self._ENV_NS}/env", num_envs)
        positions = cloner.clone(f"{self._ENV_NS}/env_0", env_paths,
                                 replicate_physics=True, base_env_path=self._ENV_NS)

        cloner.filter_collisions(self._sim.cfg.physics_prim_path, "/World/collisions",
                                 env_paths, [self._GROUND_PATH])
        self._origins = torch.tensor(positions, dtype=torch.float32, device=self._sim.device)
        if origin is not None:
            self._origins += torch.tensor(origin, dtype=torch.float32,
                                          device=self._sim.device)

        robot_cfg, meta = make_articulation_cfg(usd_dir, drive_cfg,
                                                f"{self._ENV_NS}/env_.*/Robot", spawn_margin,
                                                gain_overrides=gain_overrides,
                                                actuator_model=actuator_model)
        self._groups = [Articulation(robot_cfg)]
        self._slices = [slice(0, num_envs)]
        logger.info("스폰 %s: env %d개, 스폰 높이 %.3fm, 셀프충돌 %s",
                    meta["name"], num_envs, robot_cfg.init_state.pos[2],
                    self._applied_self_collision())
        return self._groups[0]

    def spawn_robot_groups(self, usd_dirs: list[Path], drive_cfg: dict,
                           envs_per_robot: int, spacing: float = 4.0,
                           spawn_margin: float = 0.01,
                           self_collision: bool = True,
                           gain_overrides_list: list[dict | None] | None = None,
                           origins: torch.Tensor | None = None,
                           activate_contact_sensors: bool = False,
                           friction_links_list: list[list[str] | None] | None = None,
                           friction_links_mu: float = 1.0,
                           solver_iters: tuple[int, int] | None = None,
                           actuator_model: str = "implicit")            -> list[Articulation]:

        K, M = len(usd_dirs), envs_per_robot
        total = K * M
        overrides = gain_overrides_list or [None] * K

        if origins is None:

            cols = max(1, math.ceil(math.sqrt(total)))
            rows = math.ceil(total / cols)
            positions = np.zeros((total, 3))
            for i in range(total):
                positions[i, 0] = (i % cols - 0.5 * (cols - 1)) * spacing
                positions[i, 1] = (i // cols - 0.5 * (rows - 1)) * spacing
            self._origins = torch.tensor(positions, dtype=torch.float32,
                                         device=self._sim.device)
        else:

            if origins.shape[0] != total:
                raise ValueError(f"origins 수 {origins.shape[0]} != env 수 {total}")
            positions = origins.detach().cpu().numpy()
            self._origins = origins

        cloner = Cloner()
        self._groups, self._slices = [], []
        all_paths = []
        for g, usd_dir in enumerate(usd_dirs):
            usd_dir = Path(usd_dir)
            base = g * M
            src = f"{self._ENV_NS}/env_{base}"
            prim_utils.define_prim(src)

            art_props = dict(enabled_self_collisions=self_collision,
                             sleep_threshold=0.0, stabilization_threshold=0.0)
            if solver_iters is not None:
                art_props.update(solver_position_iteration_count=solver_iters[0],
                                 solver_velocity_iteration_count=solver_iters[1])
            spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=str(usd_dir / "robot.usd"),
                activate_contact_sensors=activate_contact_sensors,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(**art_props))
            spawn_cfg.func(f"{src}/Robot_g{g:03d}", spawn_cfg)

            if friction_links_list is not None and friction_links_list[g]:
                self.bind_link_friction(f"{src}/Robot_g{g:03d}",
                                        friction_links_list[g],
                                        friction_links_mu,
                                        f"/World/Materials/links_g{g:03d}")
            paths = [f"{self._ENV_NS}/env_{base + k}" for k in range(M)]
            cloner.clone(src, paths, positions=positions[base:base + M],
                         replicate_physics=False)
            all_paths += paths

            robot_cfg, meta = make_articulation_cfg(
                usd_dir, drive_cfg, f"{self._ENV_NS}/env_.*/Robot_g{g:03d}",
                spawn_margin, gain_overrides=overrides[g],
                actuator_model=actuator_model)
            self._groups.append(Articulation(robot_cfg))
            self._slices.append(slice(base, base + M))
            logger.info("그룹 %d/%d 스폰 %s: env %d개, 스폰 높이 %.3fm",
                        g + 1, K, meta["name"], M, robot_cfg.init_state.pos[2])

        cloner.filter_collisions(self._sim.cfg.physics_prim_path,
                                 "/World/collisions", all_paths,
                                 [self._GROUND_PATH])
        return self._groups

    _DRIVE_LINK_TOKENS = ("wheel", "caster", "ball", "roller", "knuckle",
                          "steer", "leg", "hip", "thigh", "shin", "calf",
                          "knee", "foot", "ankle", "coxa", "femur", "tibia")

    def _bind_link_palette(self, robot_path: str, color: tuple):

        palette = {
            "base": tuple(color),
            "light": tuple(min(1.0, 0.5 + 0.5 * c) for c in color),
            "accent": tuple(min(1.0, 1.15 - c) for c in color),
            "black1": (0.06, 0.06, 0.07),
            "black2": (0.20, 0.20, 0.22),
        }
        stage = stage_utils.get_current_stage()
        materials = {}
        for name, rgb in palette.items():
            cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=rgb)
            cfg.func(f"/World/Materials/robot_{name}", cfg)
            materials[name] = UsdShade.Material(
                stage.GetPrimAtPath(f"/World/Materials/robot_{name}"))
        root = stage.GetPrimAtPath(robot_path)
        body_order = ["base", "light", "accent"]
        links = sorted((p for p in root.GetChildren() if p.GetName() != "Looks"),
                       key=lambda p: p.GetName())
        n_drive, n_body = 0, 0
        for prim in links:
            name = prim.GetName().lower()
            if any(tok in name for tok in self._DRIVE_LINK_TOKENS):
                kind = "black1" if n_drive % 2 == 0 else "black2"
                n_drive += 1
            else:
                kind = body_order[n_body % len(body_order)]
                n_body += 1
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                materials[kind],
                bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    def bind_link_friction(self, robot_path: str, link_names: list[str],
                           friction: float, material_path: str):

        cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=friction, dynamic_friction=friction,
            friction_combine_mode="multiply", restitution_combine_mode="multiply")
        cfg.func(material_path, cfg)
        stage = stage_utils.get_current_stage()
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        root = stage.GetPrimAtPath(robot_path)
        names = set(link_names)
        bound = 0
        for prim in Usd.PrimRange(root):
            if prim.GetName() in names:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics")
                bound += 1
        logger.info("링크 마찰 바인딩: %d개 (mu %.2f, %s)", bound, friction,
                    robot_path)

    def _bind_passive_friction(self, robot_path: str, contact_cfg: dict):

        mats = {
            "ball": ("/World/Materials/passive_ball",
                     contact_cfg["ball_friction"]),
            "roller": ("/World/Materials/passive_roller",
                       contact_cfg["roller_friction"]),
        }
        for path, friction in mats.values():
            cfg = sim_utils.RigidBodyMaterialCfg(
                static_friction=friction, dynamic_friction=friction,
                friction_combine_mode="multiply", restitution_combine_mode="multiply")
            cfg.func(path, cfg)

        stage = stage_utils.get_current_stage()
        root = stage.GetPrimAtPath(robot_path)
        bound = {"ball": 0, "roller": 0}
        for prim in Usd.PrimRange(root):
            name = prim.GetName()
            if name.startswith("ball_"):
                kind = "ball"
            elif "_roller_" in name:
                kind = "roller"
            else:
                continue
            material = UsdShade.Material(stage.GetPrimAtPath(mats[kind][0]))
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(material,
                         bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                         materialPurpose="physics")
            bound[kind] += 1
        if bound["ball"] or bound["roller"]:
            logger.info("수동 접촉 마찰 바인딩: 볼 캐스터 %d개(mu %.2f), 롤러 %d개(mu %.2f)",
                        bound["ball"], mats["ball"][1], bound["roller"], mats["roller"][1])

    def _applied_self_collision(self):

        root = stage_utils.get_current_stage().GetPrimAtPath(f"{self._ENV_NS}/env_0/Robot")
        for prim in Usd.PrimRange(root):
            attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
            if attr and attr.HasAuthoredValue():
                return attr.Get()
        return None

    def reset(self):

        self._sim.reset()
        for art, sl in zip(self._groups, self._slices):
            root_state = art.data.default_root_state.clone()
            root_state[:, :3] += self.origins[sl]
            art.write_root_pose_to_sim(root_state[:, :7])
            art.write_root_velocity_to_sim(root_state[:, 7:])
            art.write_joint_state_to_sim(art.data.default_joint_pos.clone(),
                                         art.data.default_joint_vel.clone())
            art.reset()

    def step(self, joint_pos_target: torch.Tensor | None = None,
             joint_vel_target: torch.Tensor | None = None,
             joint_effort_target: torch.Tensor | None = None,
             render: bool = False):

        art = self.robot
        if joint_pos_target is not None:
            art.set_joint_position_target(joint_pos_target)
        if joint_vel_target is not None:
            art.set_joint_velocity_target(joint_vel_target)
        if joint_effort_target is not None:
            art.set_joint_effort_target(joint_effort_target)
        art.write_data_to_sim()
        self._sim.step(render)
        art.update(self._dt)

    def step_multi(self, joint_pos_targets: list[torch.Tensor] | None = None,
                   render: bool = False):

        for g, art in enumerate(self._groups):
            if joint_pos_targets is not None:
                art.set_joint_position_target(joint_pos_targets[g])
            art.write_data_to_sim()
        self._sim.step(render)
        for art in self._groups:
            art.update(self._dt)

    def hold_step(self, render: bool = False):

        art = self.robot
        self.step(joint_pos_target=art.data.default_joint_pos,
                  joint_vel_target=torch.zeros_like(art.data.default_joint_vel),
                  render=render)

def attach_contact_sensor(prim_path: str, history_length: int):

    return ContactSensor(ContactSensorCfg(prim_path=prim_path,
                                          history_length=history_length))

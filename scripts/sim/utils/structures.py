from __future__ import annotations

import logging

import isaaclab.sim as sim_utils

logger = logging.getLogger(__name__)

class StructureBuilder:

    def __init__(self, cfg: dict, prim_root: str = "/World/structures"):

        self._cfg = cfg
        self._root = prim_root
        self._kinds = list(cfg["kinds"])
        self._levels = int(cfg["num_levels"])
        self._cell = tuple(float(v) for v in cfg["cell_size"])

        self._builders = {
            "narrow_wall": self._build_narrow_wall,
            "low_ceiling": self._build_low_ceiling,
            "desk": self._build_desk,
        }

    @property
    def extent(self) -> tuple[float, float]:

        return (len(self._kinds) * self._cell[0], self._levels * self._cell[1])

    def build(self, center: tuple[float, float]) -> list[dict]:

        pad = float(self._cfg.get("ground_pad_thickness", 0.0))
        if pad > 0:
            self._spawn_box(f"{self._root}/ground_pad",
                            (self.extent[0], self.extent[1], pad),
                            (center[0], center[1], -0.5 * pad))

        items = []
        for ci, kind in enumerate(self._kinds):
            for ri in range(self._levels):
                t = ri / max(self._levels - 1, 1)
                cx = center[0] + (ci - 0.5 * (len(self._kinds) - 1)) * self._cell[0]
                cy = center[1] + (ri - 0.5 * (self._levels - 1)) * self._cell[1]
                path = f"{self._root}/{kind}_L{ri}"
                clearance, params = self._builders[kind](path, (cx, cy), t)
                items.append({"kind": kind, "level": ri, "difficulty": t,
                              "center": [cx, cy], "clearance": clearance,
                              "params": params})
        logger.info("구조물 배치: %d종 x %d단계 = %d개, 밴드 %.1f x %.1f m (중심 %.1f, %.1f)",
                    len(self._kinds), self._levels, len(items),
                    self.extent[0], self.extent[1], center[0], center[1])
        return items

    @staticmethod
    def _lerp(rng, t: float) -> float:

        return float(rng[0]) + (float(rng[1]) - float(rng[0])) * t

    def _spawn_box(self, path: str, size: tuple, pos: tuple):

        mu = float(self._cfg.get("friction", 1.0))
        cfg = sim_utils.CuboidCfg(
            size=tuple(float(v) for v in size),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=mu, dynamic_friction=mu,
                friction_combine_mode="multiply", restitution_combine_mode="multiply"),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(self._cfg["color"])),
        )
        cfg.func(path, cfg, translation=tuple(float(v) for v in pos))

    def _build_narrow_wall(self, path: str, center: tuple, t: float) -> tuple[float, dict]:

        p = self._cfg["narrow_wall"]
        gap = self._lerp(p["gap"], t)
        thick, height, length = float(p["thickness"]), float(p["height"]), float(p["length"])
        off = 0.5 * (gap + thick)
        for side, sign in (("l", 1.0), ("r", -1.0)):
            self._spawn_box(f"{path}/wall_{side}", (length, thick, height),
                            (center[0], center[1] + sign * off, 0.5 * height))
        return gap, {"gap": gap, "thickness": thick, "height": height, "length": length}

    def _build_low_ceiling(self, path: str, center: tuple, t: float) -> tuple[float, dict]:

        p = self._cfg["low_ceiling"]
        clearance = self._lerp(p["clearance"], t)
        thick = float(p["slab_thickness"])
        span, length, sup = float(p["span"]), float(p["length"]), float(p["support_thickness"])
        off = 0.5 * (span + sup)
        for side, sign in (("l", 1.0), ("r", -1.0)):
            self._spawn_box(f"{path}/support_{side}", (length, sup, clearance),
                            (center[0], center[1] + sign * off, 0.5 * clearance))
        self._spawn_box(f"{path}/slab", (length, span + 2 * sup, thick),
                        (center[0], center[1], clearance + 0.5 * thick))
        return clearance, {"clearance": clearance, "slab_thickness": thick,
                           "span": span, "length": length}

    def _build_desk(self, path: str, center: tuple, t: float) -> tuple[float, dict]:

        p = self._cfg["desk"]
        h = self._lerp(p["top_height"], t)
        sx, sy = (float(v) for v in p["top_size"])
        thick, leg = float(p["top_thickness"]), float(p["leg_size"])
        for i, (ax, ay) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            self._spawn_box(f"{path}/leg_{i}", (leg, leg, h),
                            (center[0] + ax * (0.5 * sx - 0.5 * leg),
                             center[1] + ay * (0.5 * sy - 0.5 * leg), 0.5 * h))
        self._spawn_box(f"{path}/top", (sx, sy, thick),
                        (center[0], center[1], h + 0.5 * thick))
        return h, {"top_height": h, "top_size": [sx, sy], "leg_size": leg}

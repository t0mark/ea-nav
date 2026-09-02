from __future__ import annotations

from isaaclab.sensors import RayCasterCfg, patterns

def scanner_cfg(prim_path: str, scan_cfg: dict, alignment: str = "world") -> RayCasterCfg:
    """지형 스캔용 그리드 raycaster 설정을 만든다.

    alignment="world"는 로봇 위치를 따라 격자 중심만 이동하고 로봇 yaw에는 회전하지 않는다.
    legged의 proprioceptive height scan은 반대로 alignment="yaw"를 써서 로봇 정면 기준으로 회전한다.
    """
    return RayCasterCfg(
        prim_path=prim_path,
        mesh_prim_paths=["/World/ground"],
        ray_alignment=alignment,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=float(scan_cfg["resolution"]), size=tuple(scan_cfg["size"])),
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, float(scan_cfg["offset_z"]))))

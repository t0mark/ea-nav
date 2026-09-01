from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns

def scanner_cfg(prim_path: str, scan_cfg: dict, alignment: str = "world") -> RayCasterCfg:
    """지형 스캔용 그리드 raycaster 설정을 만든다.

    alignment="world"는 로봇 위치를 따라 격자 중심만 이동하고 로봇 yaw에는 회전하지 않는다
    (MPPI 롤아웃이 world (x,y) 좌표를 그대로 조회하므로 격자가 축정렬이어야 보간이 단순해짐).
    legged의 proprioceptive height scan은 반대로 alignment="yaw"를 써서 로봇 정면 기준으로 회전한다.
    """
    return RayCasterCfg(
        prim_path=prim_path,
        mesh_prim_paths=["/World/ground"],
        ray_alignment=alignment,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=float(scan_cfg["resolution"]), size=tuple(scan_cfg["size"])),
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, float(scan_cfg["offset_z"]))))

def grid_shape(scan_cfg: dict) -> tuple[int, int]:
    """scanner_cfg와 짝을 이루는 격자 크기를 계산한다 (nx: size[0] 축, ny: size[1] 축).

    isaaclab GridPatternCfg는 torch.meshgrid(x, y, indexing="xy")로 광선을 만들어
    ray_hits_w가 (ny, nx) 행-우선(y가 느리게, x가 빠르게) 순서로 펼쳐진다 — 재구성 시 이 순서를 지켜야 한다.
    """
    nx = int(round(float(scan_cfg["size"][0]) / float(scan_cfg["resolution"]))) + 1
    ny = int(round(float(scan_cfg["size"][1]) / float(scan_cfg["resolution"]))) + 1
    return nx, ny

@dataclass
class TerrainScan:
    """한 컨트롤 스텝의 로컬 높이 격자 스냅샷 — MPPI 비용·(추후) governor 지형 조회가 공유하는 입력."""

    height: torch.Tensor   # (num_envs, ny, nx) — 셀별 지면 높이 (world z)
    slope: torch.Tensor    # (num_envs, ny, nx) — 셀별 경사 크기 (rad, atan(|grad z|))
    x0: torch.Tensor       # (num_envs,) — 격자 첫 열의 world x
    y0: torch.Tensor       # (num_envs,) — 격자 첫 행의 world y
    resolution: float

    @classmethod
    def from_ray_hits(cls, ray_hits_w: torch.Tensor, nx: int, ny: int,
                      resolution: float) -> "TerrainScan":
        """raycaster의 평평한 히트 목록을 (ny, nx) 격자로 재구성한다.

        광선이 수직 하방(0,0,-1)이라 히트 x,y는 광선 시작점과 동일 — 히트 좌표를 그대로
        격자 좌표로 쓸 수 있다 (지형 경사와 무관하게 격자가 흐트러지지 않음).
        """
        num_envs = ray_hits_w.shape[0]
        hits = ray_hits_w.reshape(num_envs, ny, nx, 3)
        height = hits[..., 2]

        # 경사 크기: 내부 셀은 중심차분, 가장자리는 인접값 복제로 근사 (그리드 크기 유지)
        dzdx = torch.zeros_like(height)
        dzdy = torch.zeros_like(height)
        dzdx[:, :, 1:-1] = (height[:, :, 2:] - height[:, :, :-2]) / (2.0 * resolution)
        dzdx[:, :, 0] = dzdx[:, :, 1]
        dzdx[:, :, -1] = dzdx[:, :, -2]
        dzdy[:, 1:-1, :] = (height[:, 2:, :] - height[:, :-2, :]) / (2.0 * resolution)
        dzdy[:, 0, :] = dzdy[:, 1, :]
        dzdy[:, -1, :] = dzdy[:, -2, :]
        slope = torch.atan(torch.sqrt(dzdx ** 2 + dzdy ** 2))

        return cls(height=height, slope=slope, x0=hits[:, 0, 0, 0],
                   y0=hits[:, 0, 0, 1], resolution=resolution)

    def sample_slope(self, query_xy: torch.Tensor,
                     env_idx: torch.Tensor) -> torch.Tensor:
        """query_xy (M, 2) world 좌표를, 각 행이 속한 env_idx (M,) 격자에서 쌍선형 보간한다.

        평평화(flatten)된 형태라 env당 쿼리 개수가 달라도(예: MPPI의 env x 샘플 롤아웃)
        env_idx만 맞추면 그대로 쓸 수 있다. 격자 밖으로 나가면(스캔 범위 밖 = 알 수 없는 지형)
        가장자리 셀 값으로 클램프한다 — 알 수 없음을 0비용으로 취급하지 않고 보수적으로 그대로 민다.
        """
        ny, nx = self.slope.shape[1], self.slope.shape[2]

        fx = (query_xy[:, 0] - self.x0[env_idx]) / self.resolution
        fy = (query_xy[:, 1] - self.y0[env_idx]) / self.resolution
        fx = torch.clamp(fx, 0.0, nx - 1.0 - 1e-4)
        fy = torch.clamp(fy, 0.0, ny - 1.0 - 1e-4)

        ix0, iy0 = fx.floor().long(), fy.floor().long()
        ix1, iy1 = ix0 + 1, iy0 + 1
        wx, wy = (fx - ix0.float()), (fy - iy0.float())

        def gather(iy, ix):
            return self.slope[env_idx, iy, ix]

        s00, s01 = gather(iy0, ix0), gather(iy0, ix1)
        s10, s11 = gather(iy1, ix0), gather(iy1, ix1)
        top = s00 * (1.0 - wx) + s01 * wx
        bot = s10 * (1.0 - wx) + s11 * wx
        return top * (1.0 - wy) + bot * wy

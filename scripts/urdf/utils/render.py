"""파일럿 검증용 3D 렌더링 (matplotlib, 헤드리스)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .loader import PosedModel


def render_set(robots_root: str | Path, render_dir: str | Path):
    """생성 셋({root}/{form}/{robot}/)의 모든 로봇을 기립 자세 PNG로 저장한다.

    기립 자세는 각 로봇의 meta.json에 저장된 값을 사용한다.
    """
    log = logging.getLogger("urdf.render")
    for meta_path in sorted(Path(robots_root).glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        png = render_robot(meta_path.parent / "robot.urdf", meta["standing_pose"],
                           Path(render_dir) / f"{meta_path.parent.name}.png")
        log.info("렌더 저장: %s", png)


def render_robot(urdf_path: str | Path, pose: dict, out_png: str | Path, title: str = "") -> Path:
    """URDF를 기립 자세(pose)로 놓고 링크별 색을 입힌 PNG로 저장, 경로 반환.

    사람이 눈으로 확인하는 용도의 근사 렌더다. 3D 투영 특성상 좌우 대칭
    부품이 어긋나 보일 수 있다 (좌표 검증은 validate 쪽에서 수행).
    """
    model = PosedModel(urdf_path, pose)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")

    # 링크별 컬렉션을 나누면 그리기 순서 때문에 가림이 깨지므로
    # 전체 삼각형을 한 컬렉션에 넣어 면 단위 깊이 정렬을 쓴다
    all_v, tris, colors = [], [], []
    for i, link in enumerate(model.links):
        color = cmap(i % 20)
        for mesh in link.meshes:
            tris.append(mesh.triangles)
            colors += [color] * len(mesh.triangles)
            all_v.append(mesh.vertices)
    tri = Poly3DCollection(np.vstack(tris), alpha=0.9)
    tri.set_facecolor(colors)
    tri.set_edgecolor("none")
    ax.add_collection3d(tri)
    vs = np.vstack(all_v)
    lo, hi = vs.min(axis=0), vs.max(axis=0)

    # 지면 표시: 전체 최저점 = 접촉 링크 바닥 높이
    z0 = lo[2]
    gx, gy = np.meshgrid([lo[0] - 0.1, hi[0] + 0.1], [lo[1] - 0.1, hi[1] + 0.1])
    ax.plot_surface(gx, gy, np.full_like(gx, z0), alpha=0.15, color="gray")

    # 등축 스케일: 최대 변 기준 정육면체 뷰박스로 왜곡 방지
    center = (lo + hi) / 2
    span = float((hi - lo).max()) * 0.6 + 0.05
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(z0, z0 + 2 * span)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title or Path(urdf_path).parent.name)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png

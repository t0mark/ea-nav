"""계획 경로(웨이포인트)와 실제 주행 경로를 위에서 본 2D 이미지로 저장하는 유틸리티.

Isaac Sim에 의존하지 않는 순수 후처리 시각화라 Kit 부팅 없이도 호출할 수 있다.
"""

from __future__ import annotations

from pathlib import Path


def save_trajectory_plot(
    planned_waypoints: list[tuple[float, float]],
    actual_positions: list[tuple[float, float]],
    output_path: str | Path,
) -> None:
    """계획 웨이포인트(점+점선)와 실제 주행 궤적(실선)을 같은 축 위에 그려 저장한다."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))

    planned_x, planned_y = zip(*planned_waypoints)
    ax.plot(planned_x, planned_y, "o--", color="tab:red", label="planned waypoints")

    actual_x, actual_y = zip(*actual_positions)
    ax.plot(actual_x, actual_y, "-", color="tab:blue", linewidth=1.5, label="actual trajectory")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[trajectory_plot] 궤적 이미지 저장 완료: {output_path}")

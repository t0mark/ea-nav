from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path("/data/EA-Trav")

def load_config(name: str) -> dict:

    with open(WORKSPACE_ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)

def init_logging():

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def check_dir(stage: str) -> Path:

    return WORKSPACE_ROOT / "check" / stage

PILOT_ROBOTS = check_dir("00_urdf") / "robots"
PILOT_USD = check_dir("01_sim") / "usd"
FULL_ROBOTS = DATA_ROOT / "urdf/synthesis"
FULL_USD = DATA_ROOT / "sim/usd"

def mode_roots(mode: str) -> tuple[Path, Path]:

    return (PILOT_ROBOTS, PILOT_USD) if mode == "pilot" else (FULL_ROBOTS, FULL_USD)

def resolve_robot_dirs(root: Path, robot_arg: str | None,
                       form_filter: tuple[str, ...] | None = None) -> list[Path]:

    from scripts.sim.utils.robot_spawn import iter_robot_dirs

    dirs = iter_robot_dirs(root)
    if form_filter is not None:
        dirs = [d for d in dirs if d.parent.name in form_filter]
    if robot_arg:
        dirs = [d for d in dirs if str(d.relative_to(root)) == robot_arg]
    return dirs

def save_report_json(report: dict, path: Path, single_test: bool) -> Path | None:

    if single_test:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    return path

def save_figure(fig, path: Path) -> None:

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

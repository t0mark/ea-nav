from __future__ import annotations

import argparse
import logging
import sys

from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)

def cli_device(argv: list[str] | None = None) -> str | None:

    argv = sys.argv[1:] if argv is None else argv
    for i, token in enumerate(argv):
        if token.startswith("--device="):
            return token.split("=", 1)[1]
        if token == "--device" and i + 1 < len(argv):
            return argv[i + 1]
    return None

def launch_app(parser: argparse.ArgumentParser, enable_cameras=False,
               device_default: str | None = None):

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.headless = True
    if enable_cameras == "pilot":

        args.enable_cameras = getattr(args, "mode", None) == "pilot"            and not getattr(args, "debug", False)
    else:
        args.enable_cameras = bool(enable_cameras)

    cli = cli_device()
    if cli is None and device_default is not None:
        args.device = device_default
    logger.info("장치: %s (%s)", args.device,
                "CLI --device" if cli is not None else
                "설정 기본값" if device_default is not None else "런처 기본값")

    launcher = AppLauncher(args)
    return args, launcher.app

def close_app(app):

    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    if sim is not None:
        sim._disable_app_control_on_stop_handle = True
    app.close()

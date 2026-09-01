from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.utils.common import DATA_ROOT, check_dir, init_logging, load_config

from scripts.urdf.pipeline import GenerationPipeline
from scripts.urdf.real_robots import RealPoolReport, RealRobotCollector
from scripts.urdf.utils.render import render_set
from scripts.urdf.utils.validate import validate_set

_PILOT_ROOT = check_dir("00_urdf")
_FULL_ROBOTS = DATA_ROOT / "urdf/synthesis"
_REAL_POOL = DATA_ROOT / "urdf/real_robots"

def _robots_root(mode: str) -> Path:

    return _PILOT_ROOT / "robots" if mode == "pilot" else _FULL_ROBOTS

def cmd_generate(args):

    out = _robots_root(args.mode)
    pipeline = GenerationPipeline(load_config("urdf"), out)
    summary = pipeline.run(GenerationPipeline.all_forms(), args.count, args.seed,
                           workers=args.workers)

    with open(out / "generation_summary.json", "w") as f:
        json.dump({"mode": args.mode, "seed": args.seed, "count": args.count,
                   "summary": summary}, f, indent=1)

    if args.mode == "pilot":
        render_set(out, _PILOT_ROOT / "renders")

def cmd_validate(args):

    validate_set(_robots_root(args.mode), load_config("urdf")["validation"])

def cmd_collect_real(args):

    RealRobotCollector(_REAL_POOL).collect()

def cmd_pool_report(args):

    RealPoolReport(_REAL_POOL, _robots_root(args.mode)).run()

def main():

    parser = argparse.ArgumentParser(description="URDF 랜덤 생성·검증·실로봇 풀")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="랜덤 URDF 생성 + 정적 검사 (+파일럿 렌더)")
    gen.add_argument("--mode", choices=["pilot", "full"], required=True,
                     help="pilot: check/ 소규모 + 렌더, full: /data 본 생성")
    gen.add_argument("--count", type=int, required=True, help="form당 생성 개수")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--workers", type=int, default=1)
    gen.set_defaults(func=cmd_generate)

    val = sub.add_parser("validate", help="저장된 생성 셋 재검증")
    val.add_argument("--mode", choices=["pilot", "full"], required=True)
    val.set_defaults(func=cmd_validate)

    col = sub.add_parser("collect-real", help="실로봇 URDF 풀 수집")
    col.set_defaults(func=cmd_collect_real)

    rep = sub.add_parser("pool-report", help="실로봇 풀 파서 검증·표기 통계·커버리지")
    rep.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                     help="커버리지 비교 대상 생성 셋")
    rep.set_defaults(func=cmd_pool_report)

    args = parser.parse_args()
    init_logging()
    args.func(args)

if __name__ == "__main__":
    main()

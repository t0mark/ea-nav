"""URDF 단계 진입점.

서브커맨드·파라미터:
- generate : 랜덤 URDF 생성 + 정적 검사 (+파일럿 렌더 자동)
    --mode pilot|full : pilot = check/00_urdf/ 소규모 + 렌더, full = /data 본 생성 (렌더 없음)
    --count N         : form당 생성 개수 (필수. 규모는 학습 단계에서 확정)
    --seed S          : 전역 시드 (기본 0). 로봇별 시드 = (시드, form, 인덱스, 시도)라 규모 확장 시 재현됨
    --workers W       : 멀티프로세스 워커 수 (기본 1)
- validate : 저장된 생성 셋 재검증
    --mode pilot|full : 대상 셋
- collect-real : 실로봇 URDF 풀 수집 (파라미터 없음)
- pool-report : 실로봇 풀 파서 검증·표기 통계·커버리지
    --mode pilot|full : 커버리지 비교 대상 생성 셋 (기본 pilot)

경로 규약 (전부 파일 기준 상대 계산이라 실행 위치 cwd와 무관):
파일럿 = check/00_urdf/{robots,renders}, 본 생성 = /data/EA-Trav/urdf/synthesis,
실로봇 풀 = /data/EA-Trav/urdf/real_robots, 설정 = configs/urdf.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 저장소 루트를 import 경로에 추가 (공통 유틸을 쓰기 위한 부트스트랩)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.utils.common import DATA_ROOT, check_dir, init_logging, load_config

from scripts.urdf.pipeline import GenerationPipeline
from scripts.urdf.real_robots import RealPoolReport, RealRobotCollector
from scripts.urdf.utils.render import render_set
from scripts.urdf.utils.validate import validate_set

# 이 단계의 고정 경로 (모듈 docstring의 경로 규약)
_PILOT_ROOT = check_dir("00_urdf")
_FULL_ROBOTS = DATA_ROOT / "urdf/synthesis"
_REAL_POOL = DATA_ROOT / "urdf/real_robots"


def _robots_root(mode: str) -> Path:
    """mode에 해당하는 생성 셋 루트 (pilot = 작업 공간, full = 데이터 볼륨)."""
    return _PILOT_ROOT / "robots" if mode == "pilot" else _FULL_ROBOTS


def cmd_generate(args):
    """전체 form 생성 + 정적 검사. 파일럿 모드면 렌더까지 자동 수행한다."""
    out = _robots_root(args.mode)
    pipeline = GenerationPipeline(load_config("urdf"), out)
    summary = pipeline.run(GenerationPipeline.all_forms(), args.count, args.seed,
                           workers=args.workers)

    # 요약에 실행 인자를 함께 남겨 재현 정보를 보존한다
    with open(out / "generation_summary.json", "w") as f:
        json.dump({"mode": args.mode, "seed": args.seed, "count": args.count,
                   "summary": summary}, f, indent=1)

    # 파일럿은 사람 눈 확인이 목적이라 렌더 필수, 본 생성은 렌더 없음
    if args.mode == "pilot":
        render_set(out, _PILOT_ROOT / "renders")


def cmd_validate(args):
    """저장된 생성 셋을 디스크에서 다시 읽어 정적 검사를 재수행한다."""
    validate_set(_robots_root(args.mode), load_config("urdf")["validation"])


def cmd_collect_real(args):
    """실로봇 URDF 풀 수집 (풀 위치 고정)."""
    RealRobotCollector(_REAL_POOL).collect()


def cmd_pool_report(args):
    """실로봇 풀 파서 검증·표기 통계 + mode 생성 셋과의 커버리지 비교."""
    RealPoolReport(_REAL_POOL, _robots_root(args.mode)).run()


def main():
    """서브커맨드 파싱 후 해당 함수 실행 (argparse set_defaults 패턴)."""
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

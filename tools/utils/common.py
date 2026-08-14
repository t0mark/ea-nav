"""진입점 공통 유틸 (tools/*.py 단계 스크립트들이 공유).

모든 경로는 이 파일 위치 기준 상대(저장소 루트)로 계산하므로
실행 위치(cwd)와 무관하게 동작한다. 대용량 산출물 루트는 컨테이너
마운트 규약(/data)을 따른다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

# 저장소 루트 = tools/utils/common.py 기준 두 단계 위
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

# 대용량 산출물 루트 (컨테이너 마운트: ~/jairlab/data -> /data)
DATA_ROOT = Path("/data/EA-Trav")


def load_config(name: str) -> dict:
    """configs/{name}.yaml 을 읽어 dict로 반환한다."""
    with open(WORKSPACE_ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def init_logging():
    """진입점 공통 로깅 설정 (실행 중 진행 로그 출력용)."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def check_dir(stage: str) -> Path:
    """단계별 파일럿 산출물 디렉토리 (check/{stage})."""
    return WORKSPACE_ROOT / "check" / stage

"""커리큘럼 학습의 산출물 경로와 결과 파일을 담당한다.

예전에는 이 모듈이 "단계 하나 = 프로세스 하나"로 자식 프로세스를 띄우는 지휘자였다. 지금은
그렇게 하지 않는다 - 난이도 승급은 지형을 다시 만드는 일이 아니라 이미 만들어 둔 지형 안에서
env를 다른 행으로 옮기는 일이고(TerrainImporter.terrain_levels), 그건 한 프로세스 안에서 텐서
연산으로 끝난다. 그래서 커리큘럼 전체가 프로세스 하나로 돌아가고, 이 모듈에는 경로 규칙과
결과 기록만 남는다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from scripts.sim.env.curriculum.stage_env import DIRECTIONAL_AXES

if TYPE_CHECKING:
    from .rl_trainer import CurriculumOutcome

_REPO_ROOT = Path(__file__).resolve().parents[5]
POLICY_ROOT = _REPO_ROOT / "data" / "sim" / "policies" / "legged"
CURRICULUM_RESULT_FILENAME = "curriculum_result.yaml"


def experiment_name(robot_id: str) -> str:
    """tensorboard 실험 이름 - 로봇 하나의 학습 로그가 이 이름 아래 모인다."""
    return f"multi-legged_{robot_id}"


def curriculum_log_dir(robot_id: str) -> Path:
    """학습 로그·체크포인트 디렉터리 - 커리큘럼 전체가 한 프로세스라 로봇당 하나다."""
    return POLICY_ROOT / "_train_logs" / experiment_name(robot_id)


def policy_output_dir(robot_id: str) -> Path:
    """학습 산출물(policy.pt, curriculum_result.yaml) 디렉터리."""
    return POLICY_ROOT / "multi-legged" / robot_id


def _guaranteed_limit(axis_summaries: dict, key: str) -> float | None:
    """방향을 가리지 않고 보장되는 한계 - 방향 축들이 가진 그 지표의 최솟값.

    오르막만 잘하는 로봇의 오르막 수치를 대표값으로 쓰면 실제 주파 능력을 과장하게 된다. 어느
    방향으로 가더라도 넘는 값은 축별 값 중 가장 낮은 것이다.

    파쿠르 축은 방향이 아니라 장애물 종류라 이 최솟값에 넣지 않는다 - 넣으면 "어느 방향으로든"이
    "고립된 블록까지 포함해"로 바뀌어 지표의 뜻이 달라진다. 그 축의 한계는 difficulty 항목에 남는다.
    """
    values = [
        axis_summaries[axis][key]
        for axis in DIRECTIONAL_AXES
        if axis in axis_summaries and key in axis_summaries[axis]
    ]
    return min(values) if values else None


def cleared_step_height_m(robot_id: str) -> float | None:
    """이 로봇이 어느 방향으로든 넘은 최대 단차 높이(m) - 결과 파일이 없으면 None.

    구동 확인(tools/04_controller_test.py)이 오르막 계단 지형의 난이도를 이 값에서 역산한다.
    """
    result_path = policy_output_dir(robot_id) / CURRICULUM_RESULT_FILENAME
    if not result_path.exists():
        return None
    with open(result_path) as f:
        result = yaml.safe_load(f) or {}
    height = result.get("max_step_height_m")
    return float(height) if height else None


def write_curriculum_result(
    robot_id: str, preset_name: str, outcome: CurriculumOutcome, terrain_summary: dict
) -> Path:
    """축별 최종 클리어 난이도와 종료 사유를 curriculum_result.yaml로 남긴다.

    난이도는 축 이름을 키로 하는 dict 하나로 적는다 - 축이 오르막·내리막으로 갈리므로 축마다
    레벨과 물리 단위(계단 높이 m / 경사비·각도)가 따로 있어야 임베디먼트별 한계를 방향별로 읽을 수
    있다. 그 위에 방향을 가리지 않는 대표값을 최상위에 함께 적는다.
    """
    axis_summaries = terrain_summary["axes"]
    result = {
        "robot_id": robot_id,
        "preset": preset_name,
        "stop_reason": outcome.status,
        "iterations": outcome.iterations,
        "max_step_height_m": _guaranteed_limit(axis_summaries, "max_step_height_m"),
        "max_slope_ratio": _guaranteed_limit(axis_summaries, "max_slope_ratio"),
        "max_slope_deg": _guaranteed_limit(axis_summaries, "max_slope_deg"),
        "difficulty": axis_summaries,
        "noise_range_m": terrain_summary["noise_range_m"],
        "axis_scores": outcome.axis_scores,
        "checkpoint": outcome.checkpoint,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    output_dir = policy_output_dir(robot_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / CURRICULUM_RESULT_FILENAME
    with open(output_path, "w") as f:
        yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)
    print(f"[curriculum] 결과 저장: {output_path}", flush=True)
    return output_path

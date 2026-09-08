"""커리큘럼 전체를 지휘하는 부모 - 단계마다 진입점을 자식 프로세스로 다시 띄운다.

Isaac Lab은 한 프로세스에서 ManagerBasedRLEnv를 두 번 만들 수 없다. env.close()는 파이썬 객체만
정리하고 USD stage의 prim(/World/envs/env_*/Robot, /World/ground, /World/skyLight)은 그대로 남기기
때문에(InteractiveScene에는 prim 삭제 경로가 없고, stage 재생성은 SimulationCfg.create_stage_in_memory가
True일 때만 일어나는데 기본값이 False다), 두 번째 env를 만들면 스포너가 "A prim already exists at
path" ValueError를 던진다. 그래서 "단계 하나 = 프로세스 하나"로 자르고, 이 모듈이 단계마다 진입점을
새 프로세스로 실행한다.

부모는 시뮬레이터를 띄우지 않는다 - 이 모듈은 isaaclab을 import하지 않고 자식의 종료 코드와
stage_outcome.yaml만 읽는다. 자식이 발산·세그폴트 등으로 죽어도 부모는 종료 코드로 알아채고
지금까지의 이력을 curriculum_result.yaml로 남긴 뒤 정상 종료한다.

자식이 남기는 stage_outcome.yaml의 계약(scripts/.../rl/rl_trainer.py가 기록):
    status        "converged" | "plateaued" | "max_iterations" | "diverged"
    iterations    그 단계에서 진행한 이터레이션 수(단계 시작 시점 대비 상대값)
    tracking_lin  정규화(가중치 제거) 평균 선속도 추종 점수
    tracking_ang  정규화 평균 각속도 추종 점수
    best_score    두 추종 점수의 min이 그 단계에서 도달한 최고값
    checkpoint    이 단계 마지막 체크포인트 경로
    summary       stage_terrain_summary()가 만든 지형 난이도 dict
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from .robot_profile import RobotProfile

_REPO_ROOT = Path(__file__).resolve().parents[5]
POLICY_ROOT = _REPO_ROOT / "data" / "sim" / "policies" / "legged"
STAGE_OUTCOME_FILENAME = "stage_outcome.yaml"
# 자식이 "수렴 실패"로 정상 종료했을 때의 상태 -> 커리큘럼 전체의 종료 사유
_STOP_REASON_BY_STATUS = {
    "plateaued": "plateau",
    "max_iterations": "max_iterations",
    "diverged": "divergence",
}


def experiment_name(robot_id: str) -> str:
    """tensorboard 실험 이름 - 로봇 하나의 모든 단계 로그가 이 이름 아래 모인다."""
    return f"multi-legged_{robot_id}"


def stage_log_dir(robot_id: str, stage: int) -> Path:
    """단계 하나의 학습 로그·체크포인트 디렉터리."""
    return POLICY_ROOT / "_train_logs" / experiment_name(robot_id) / f"stage{stage}"


def policy_output_dir(robot_id: str) -> Path:
    """학습 산출물(policy.pt, curriculum_result.yaml) 디렉터리."""
    return POLICY_ROOT / "multi-legged" / robot_id


@dataclass
class StageRecord:
    """메타데이터에 남기는 단계별 이력 한 줄."""

    stage: int
    summary: dict
    outcome: dict | None


class CurriculumDriver:
    """stage 0부터 단계마다 진입점을 재실행하며 커리큘럼을 진행한다.

    수렴한 단계의 체크포인트를 다음 단계 자식에게 --resume-from으로 넘겨 가중치를 이어가고,
    수렴하지 못한 단계(정체·단계상한·발산)나 자식 프로세스 실패가 나오면 커리큘럼을 끝낸다.
    전역 이터레이션 종료 로직은 없다 - 학습이 어디까지 가는지는 로봇이 지형을 못 깰 때 결정된다.
    """

    def __init__(
        self,
        entry_point: Path,
        robot_id: str,
        preset_name: str | None = None,
        num_envs: int | None = None,
        device: str = "cuda:0",
        start_stage: int = 0,
        resume_checkpoint: str | None = None,
    ) -> None:
        self._entry_point = entry_point
        self._robot_id = robot_id
        # 메타데이터에 실제로 쓴 preset 이름을 남기려면 여기서 robot yaml의 기본값을 확정해 둬야 한다
        self._preset_name = preset_name or RobotProfile.load(robot_id).rl_preset
        self._num_envs = num_envs
        self._device = device
        self._start_stage = start_stage
        self._resume_checkpoint = resume_checkpoint

    def run(self) -> None:
        """단계를 하나씩 자식 프로세스로 돌리며 수렴할 때까지 난이도를 올린다."""
        print(
            f"[curriculum] {self._robot_id} | preset={self._preset_name} | device={self._device} "
            f"| start_stage={self._start_stage}",
            flush=True,
        )
        print(
            f"[curriculum] 진행 상황: tensorboard --logdir "
            f"{POLICY_ROOT / '_train_logs' / experiment_name(self._robot_id)}",
            flush=True,
        )
        stage = self._start_stage
        checkpoint = self._resume_checkpoint
        records: list[StageRecord] = []
        stop_reason = "user_interrupt"
        try:
            while True:
                # 이력 항목을 먼저 만들어 둔다 - 중간에 중단돼도 그 단계가 기록에서 통째로 빠지지 않는다
                record = StageRecord(stage=stage, summary={"stage": stage}, outcome=None)
                records.append(record)
                return_code, outcome = self._launch_stage(stage, checkpoint)
                # 결과 파일이 없다는 건 자식이 학습을 마치지 못했다는 뜻이다(발산 전 크래시·OOM 등)
                if outcome is None:
                    print(
                        f"[curriculum] ! stage {stage} 자식 프로세스가 결과를 남기지 못했다 "
                        f"(exit={return_code}) - 커리큘럼을 종료한다",
                        flush=True,
                    )
                    stop_reason = "stage_process_failed"
                    break
                record.summary = outcome.get("summary") or {"stage": stage}
                record.outcome = outcome
                status = outcome["status"]
                print(
                    f"[curriculum] === stage {stage} 종료 === {status} ({outcome['iterations']} iters)",
                    flush=True,
                )
                if status != "converged":
                    stop_reason = _STOP_REASON_BY_STATUS.get(status, status)
                    break
                # 수렴한 단계의 가중치를 이어받아 다음 난이도로 올린다
                checkpoint = outcome["checkpoint"]
                stage += 1
        except KeyboardInterrupt:
            print("\n[curriculum] 사용자 중단 - 지금까지 이력을 저장하고 종료한다", flush=True)
            stop_reason = "user_interrupt"
        self._write_metadata(records, stop_reason)
        self._report(records, stop_reason)

    def _launch_stage(self, stage: int, checkpoint: str | None) -> tuple[int, dict | None]:
        """단계 하나를 자식 프로세스로 실행하고 (종료 코드, 결과 dict)를 돌려준다."""
        outcome_path = stage_log_dir(self._robot_id, stage) / STAGE_OUTCOME_FILENAME
        # 이전 실행이 남긴 결과 파일을 먼저 지운다 - 자식이 죽었을 때 옛 결과를 성공으로 오독하면 안 된다
        outcome_path.unlink(missing_ok=True)
        # sys.executable로 지금 이 프로세스와 같은 파이썬(isaaclab.sh가 띄운 isaac-sim 파이썬)을 쓴다
        command = [
            sys.executable,
            str(self._entry_point),
            "--robot-id",
            self._robot_id,
            "--stage",
            str(stage),
            "--preset",
            self._preset_name,
            "--device",
            self._device,
        ]
        if self._num_envs is not None:
            command += ["--num-envs", str(self._num_envs)]
        if checkpoint is not None:
            command += ["--resume-from", checkpoint]
        print(f"\n[curriculum] === stage {stage} 시작 === 자식 프로세스 실행", flush=True)
        # 표준 입출력을 물려줘 자식의 학습 로그가 부모 로그에 그대로 이어진다
        completed = subprocess.run(command)
        if not outcome_path.exists():
            return completed.returncode, None
        with open(outcome_path) as f:
            return completed.returncode, yaml.safe_load(f)

    def _write_metadata(self, records: list[StageRecord], stop_reason: str) -> None:
        """단계별 이력 + 최종 클리어 난이도를 curriculum_result.yaml로 남긴다."""
        cleared = [r for r in records if r.outcome is not None and r.outcome["status"] == "converged"]
        last_cleared = cleared[-1] if cleared else None
        # 마지막 단계가 수렴하지 못했다면 그 단계가 "못 깬 난이도"다
        failed = None
        if records and (records[-1].outcome is None or records[-1].outcome["status"] != "converged"):
            failed = records[-1]

        stages_log = []
        for record in records:
            row = dict(record.summary)
            if record.outcome is not None:
                row.update(
                    status=record.outcome["status"],
                    iterations=record.outcome["iterations"],
                    tracking_lin=record.outcome["tracking_lin"],
                    tracking_ang=record.outcome["tracking_ang"],
                    checkpoint=record.outcome["checkpoint"],
                )
            else:
                # 결과 파일이 없는 단계 - 사용자가 끊었는지, 자식이 죽었는지를 구분해 남긴다
                row.update(status="interrupted" if stop_reason == "user_interrupt" else "process_failed")
            stages_log.append(row)

        result = {
            "robot_id": self._robot_id,
            "preset": self._preset_name,
            "stop_reason": stop_reason,
            "final_stage_cleared": last_cleared.stage if last_cleared else None,
            "failed_stage": failed.stage if failed else None,
            "policy_from_stage": last_cleared.stage if last_cleared else None,
            "max_step_height_m": last_cleared.summary.get("max_step_height_m") if last_cleared else None,
            "max_slope_ratio": last_cleared.summary.get("max_slope_ratio") if last_cleared else None,
            "max_slope_deg": last_cleared.summary.get("max_slope_deg") if last_cleared else None,
            "stages": stages_log,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        output_dir = policy_output_dir(self._robot_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "curriculum_result.yaml", "w") as f:
            yaml.safe_dump(result, f, allow_unicode=True, sort_keys=False)
        print(f"[curriculum] 메타데이터 저장: {output_dir / 'curriculum_result.yaml'}", flush=True)

    def _report(self, records: list[StageRecord], stop_reason: str) -> None:
        """최종 클리어 난이도를 콘솔에 요약한다."""
        cleared = [r for r in records if r.outcome is not None and r.outcome["status"] == "converged"]
        if not cleared:
            print(f"[curriculum] 완료 - 수렴한 단계 없음 (stop_reason={stop_reason}). policy.pt 미생성.", flush=True)
            return
        last = cleared[-1].summary
        print(
            f"[curriculum] 완료 - 최종 클리어 stage {last['stage']} "
            f"(max_step_height={last.get('max_step_height_m')}m, max_slope={last.get('max_slope_ratio')}) "
            f"| stop_reason={stop_reason}",
            flush=True,
        )
        print("[curriculum] 학습된 정책 구동 확인은 tools/04_controller_test.py를 별도로 실행하세요.", flush=True)

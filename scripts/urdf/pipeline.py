"""생성 -> 검증 -> 저장 파이프라인.

로봇 하나의 시드는 (전역 시드, form 인덱스, 로봇 인덱스, 시도 횟수)로 결정되므로
규모를 나중에 늘려도 앞 인덱스의 로봇은 그대로 재현된다.
표기 랜덤화는 계획상 인코더 입력측 증강(4단계)이라 여기서는 수행하지 않는다.
"""
from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .core.base import write_urdf
from .platform.ackermann import AckermannGenerator
from .platform.diff import DiffGenerator
from .platform.humanoid import HumanoidGenerator
from .platform.multileg import MultilegGenerator
from .platform.omni import OmniGenerator
from .platform.skid import SkidGenerator
from .platform.wheeled_humanoid import WheeledHumanoidGenerator
from .utils.validate import validate_static


class GenerationPipeline:
    """form 목록 x 개수만큼 로봇을 생성·검증·저장하는 오케스트레이터.

    산출물: {out}/{form}/{form}_{idx:04d}/robot.urdf + meta.json.
    정적 검사(공통 + 특수 + 추가 자세) 실패 시 같은 인덱스에서 시도 시드만
    바꿔 재샘플한다.
    """

    _GEN_CLASSES = (DiffGenerator, SkidGenerator, AckermannGenerator, OmniGenerator,
                    MultilegGenerator, HumanoidGenerator, WheeledHumanoidGenerator)

    def __init__(self, cfg: dict, out_root: str | Path):
        """cfg = configs/urdf.yaml 전체 dict, out_root = 산출물 루트 디렉토리."""
        self._cfg = cfg
        self._out = Path(out_root)
        self._gen_of = {f: cls for cls in self._GEN_CLASSES for f in cls.FORMS}

        # form 인덱스는 시드 재현성을 위해 정의 순서로 고정
        self._form_index = {f: i for i, f in enumerate(self.all_forms())}
        self._log = logging.getLogger("urdf.pipeline")

    @classmethod
    def all_forms(cls) -> list[str]:
        """지원하는 전체 form 태그 목록 (생성기 정의 순서 = 시드 인덱스 순서)."""
        return [f for g in cls._GEN_CLASSES for f in g.FORMS]

    def run(self, forms: list[str], count: int, seed: int,
            start: int = 0, workers: int = 1) -> dict:
        """forms 각각에 대해 [start, start+count) 인덱스의 로봇을 생성한다.

        workers > 1이면 로봇 단위로 멀티프로세스 분배 (로봇별 시드가 독립이라
        분배 순서와 무관하게 결과 동일). 반환 = form별 성공·실패·평균 시도 요약.
        """
        jobs = [(form, i) for form in forms for i in range(start, start + count)]
        self._log.info("생성 시작: forms=%s count=%d seed=%d workers=%d", forms, count, seed, workers)
        if workers <= 1:
            results = [self._generate_one(seed, form, i) for form, i in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(self._generate_one,
                                        *zip(*[(seed, f, i) for f, i in jobs])))
        return self._summarize(results)

    def _generate_one(self, seed: int, form: str, index: int) -> dict:
        """로봇 하나를 생성한다: 샘플 -> 저장 -> 검사, 실패 시 재시도.

        시도마다 시드가 (seed, form, index, attempt)로 갈리므로 재시도는
        완전히 새로운 샘플이다. 전부 실패하면 디렉토리를 지우고 실패 기록만 반환.
        """
        gen = self._gen_of[form](self._cfg)
        name = f"{form}_{index:04d}"
        robot_dir = self._out / form / name
        max_attempts = self._cfg["generation"]["max_attempts"]
        last_fail = []
        for attempt in range(max_attempts):
            # 시도별 독립 시드로 샘플 -> URDF 저장
            rng = np.random.default_rng([seed, self._form_index[form], index, attempt])
            spec = gen.sample(form, rng)
            spec.name = name
            urdf_path = write_urdf(spec, robot_dir / "robot.urdf")

            # 정적 검사 (공통 + 특수 + 추가 자세): 예외도 실패로 취급하고 재시도
            try:
                report = validate_static(urdf_path, spec.standing_pose,
                                         spec.contact_links, self._cfg["validation"],
                                         check_poses=spec.check_poses, special=spec.special)
            except Exception as e:
                self._log.warning("%s 시도 %d 검증 오류: %s", name, attempt, e)
                last_fail = [f"error: {e}"]
                continue

            # 통과하면 meta 기록 후 종료, 실패하면 실패 항목을 남기고 재시도
            if report["passed"]:
                self._write_meta(robot_dir, spec, report, seed, attempt)
                self._log.info("%s 통과 (시도 %d, 질량 %.1fkg, 지상고 %.3fm)",
                               name, attempt + 1, report["metrics"]["total_mass"],
                               report["metrics"]["clearance"])
                return {"form": form, "index": index, "ok": True, "attempts": attempt + 1}
            last_fail = [k for k, v in report["checks"].items() if not v]
            self._log.info("%s 시도 %d 실패: %s", name, attempt + 1, last_fail)

        # 전부 실패: 미완성 산출물을 남기지 않는다
        shutil.rmtree(robot_dir, ignore_errors=True)
        self._log.error("%s 포기 (%d회 시도, 마지막 실패: %s)", name, max_attempts, last_fail)
        return {"form": form, "index": index, "ok": False,
                "attempts": max_attempts, "fail": last_fail}

    def _write_meta(self, robot_dir: Path, spec, report: dict, seed: int, attempt: int):
        """meta.json 기록: 태그·시드·속성 딕셔너리·기립 자세·검사 수치.

        params = 보조 손실 라벨용 속성 딕셔너리 (plan 산출물),
        metrics.base_height = 시뮬 스폰 높이.
        """
        meta = {
            "name": spec.name, "family": spec.family, "form": spec.form,
            "control_tag": spec.control_tag,
            "seed": seed, "attempt": attempt,
            "params": spec.params,
            "standing_pose": spec.standing_pose,
            "contact_links": spec.contact_links,
            "special_checks": spec.special,
            "metrics": report["metrics"],
        }
        with open(robot_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)

    def _summarize(self, results: list[dict]) -> dict:
        """로봇별 결과를 form별 {ok, fail, avg_attempts}로 집계하고 로그로 남긴다."""
        summary: dict = {}
        for r in results:
            s = summary.setdefault(r["form"], {"ok": 0, "fail": 0, "attempts": []})
            s["ok" if r["ok"] else "fail"] += 1
            s["attempts"].append(r["attempts"])
        for form, s in summary.items():
            s["avg_attempts"] = float(np.mean(s.pop("attempts")))
            self._log.info("[%s] 성공 %d / 실패 %d (평균 시도 %.1f)",
                           form, s["ok"], s["fail"], s["avg_attempts"])
        return summary

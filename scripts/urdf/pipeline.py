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

    _GEN_CLASSES = (DiffGenerator, SkidGenerator, AckermannGenerator, OmniGenerator,
                    MultilegGenerator, HumanoidGenerator, WheeledHumanoidGenerator)

    def __init__(self, cfg: dict, out_root: str | Path):

        self._cfg = cfg
        self._out = Path(out_root)
        self._gen_of = {f: cls for cls in self._GEN_CLASSES for f in cls.FORMS}

        self._form_index = {f: i for i, f in enumerate(self.all_forms())}
        self._log = logging.getLogger("urdf.pipeline")

    @classmethod
    def all_forms(cls) -> list[str]:

        return [f for g in cls._GEN_CLASSES for f in g.FORMS]

    def run(self, forms: list[str], count: int, seed: int,
            start: int = 0, workers: int = 1) -> dict:

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

        gen = self._gen_of[form](self._cfg)
        name = f"{form}_{index:04d}"
        robot_dir = self._out / form / name
        max_attempts = self._cfg["generation"]["max_attempts"]
        last_fail = []
        for attempt in range(max_attempts):

            rng = np.random.default_rng([seed, self._form_index[form], index, attempt])
            spec = gen.sample(form, rng)
            spec.name = name
            urdf_path = write_urdf(spec, robot_dir / "robot.urdf")

            try:
                report = validate_static(urdf_path, spec.standing_pose,
                                         spec.contact_links, self._cfg["validation"],
                                         check_poses=spec.check_poses, special=spec.special)
            except Exception as e:
                self._log.warning("%s 시도 %d 검증 오류: %s", name, attempt, e)
                last_fail = [f"error: {e}"]
                continue

            if report["passed"]:
                self._write_meta(robot_dir, spec, report, seed, attempt)
                self._log.info("%s 통과 (시도 %d, 질량 %.1fkg, 지상고 %.3fm)",
                               name, attempt + 1, report["metrics"]["total_mass"],
                               report["metrics"]["clearance"])
                return {"form": form, "index": index, "ok": True, "attempts": attempt + 1}
            last_fail = [k for k, v in report["checks"].items() if not v]
            self._log.info("%s 시도 %d 실패: %s", name, attempt + 1, last_fail)

        shutil.rmtree(robot_dir, ignore_errors=True)
        self._log.error("%s 포기 (%d회 시도, 마지막 실패: %s)", name, max_attempts, last_fail)
        return {"form": form, "index": index, "ok": False,
                "attempts": max_attempts, "fail": last_fail}

    def _write_meta(self, robot_dir: Path, spec, report: dict, seed: int, attempt: int):

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

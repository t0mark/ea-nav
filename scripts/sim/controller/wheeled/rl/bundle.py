from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

class PolicyBundle:
    """wheeled RL controller parameter 정책(TD3 actor)의 배포용 번들이다."""

    def __init__(self, module: torch.nn.Module, meta: dict, device: str):
        self._module = module
        self._meta = meta
        self._device = device

    @property
    def meta(self) -> dict:
        """정책 학습과 배포에 사용한 메타데이터를 반환한다."""

        return self._meta

    @classmethod
    def load(cls, policy_dir: Path, device: str) -> "PolicyBundle":
        """TorchScript actor와 bundle metadata를 디스크에서 읽는다."""

        policy_dir = Path(policy_dir)
        module = torch.jit.load(str(policy_dir / "policy.pt"), map_location=device)
        module.eval()
        with open(policy_dir / "bundle.json") as f:
            meta = json.load(f)
        return cls(module, meta, device)

    @staticmethod
    def export(policy_dir: Path, actor: torch.nn.Module, obs_dim: int, meta: dict):
        """배포용 TorchScript actor와 metadata를 저장한다.

        critic·target network는 학습 전용이라 배포 번들에는 actor만 남긴다. 배포·평가 경로는
        policy.pt와 bundle.json 두 파일이 모두 있어야 정책을 인식하므로, 저장 직후 존재를
        확인한다.

        nn.Module.to()는 제자리에서 옮기고 self를 반환하므로, 원본을 그대로 넘기면 학습에
        쓰이는 실제 actor가 CPU로 끌려가 버린다 (device mismatch로 이어짐). 반드시 복사본에만
        적용한다.
        """

        policy_dir = Path(policy_dir)
        policy_dir.mkdir(parents=True, exist_ok=True)
        module = copy.deepcopy(actor).to("cpu").eval()
        example = torch.zeros(1, obs_dim)
        with torch.inference_mode():
            traced = torch.jit.trace(module, example)
        torch.jit.save(traced, str(policy_dir / "policy.pt"))
        with open(policy_dir / "bundle.json", "w") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)

        missing = [name for name in ("policy.pt", "bundle.json")
                   if not (policy_dir / name).exists()]
        if missing:
            raise RuntimeError(f"정책 번들 저장 실패 — 누락 파일: {missing} ({policy_dir})")
        logger.info("정책 번들 저장: %s (policy.pt, bundle.json)", policy_dir)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """관측 batch에 대한 actor action([-1, 1])을 inference mode로 계산한다."""

        with torch.inference_mode():
            return self._module(obs)

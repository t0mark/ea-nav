"""Wheeled/legged 제어기가 공통으로 구현하는 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class RobotController(ABC):
    """"명령을 받아 관절 목표값을 낸다"는 책임 하나로 wheeled(기구학)/legged(RL 정책)를 묶는 추상 클래스.

    두 계열은 목표값을 만드는 방식(닫힌 형태의 기구학 계산 vs 학습된 정책 추론)이 완전히 다르므로,
    공유 로직을 억지로 상위에 두지 않고 이 인터페이스 하나만 강제한다.
    """

    @abstractmethod
    def reset(self) -> None:
        """내부 상태(적분값, 이전 관측·행동 등)를 초기화한다."""

    @abstractmethod
    def compute_joint_targets(
        self, command: torch.Tensor, observation: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """명령(및 필요 시 관측값)을 받아 구동 방식별 관절 목표값을 계산한다.

        반환값은 {"position": 텐서, "velocity": 텐서} 중 실제로 쓰는 키만 담는다(예: 조향 관절은
        position, 구동 바퀴는 velocity). 각 텐서 안의 순서는 컨트롤러 생성 시 전달된 관절 순서와
        일치해야 한다.
        """

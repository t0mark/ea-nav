"""학습 알고리즘 축 레지스트리 - 지금은 PPO만 구현돼 있다.

DR02 공식 세팅은 AMP(모션 판별기 기반 모방학습)를 쓰지만, 참조 모션 데이터 자체가 워크스페이스에 없어
이번 구현 범위에서는 뺐다 - 데이터가 준비되면 amp.py를 추가하고 여기 레지스트리에 등록하면 된다.
"""

from __future__ import annotations

from . import ppo

_BUILDERS = {"ppo": ppo.build}


def build(algorithm: str, agent_cfg: dict):
    """algorithm 이름에 맞는 빌더로 알고리즘 설정을 만든다."""
    if algorithm not in _BUILDERS:
        raise KeyError(f"등록되지 않은 학습 알고리즘: {algorithm} (사용 가능: {sorted(_BUILDERS)})")
    return _BUILDERS[algorithm](agent_cfg)

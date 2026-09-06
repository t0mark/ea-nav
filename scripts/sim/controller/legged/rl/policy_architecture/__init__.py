"""정책 네트워크 구조 축 레지스트리 - mlp(기존 방식)와 recurrent(Unitree 공식 g1/h1/h1_2의 LSTM) 중
로봇 yaml이 고른다."""

from __future__ import annotations

from . import mlp, recurrent

_BUILDERS = {"mlp": mlp.build, "recurrent": recurrent.build}


def build(policy_architecture: str, agent_cfg: dict):
    """policy_architecture 이름에 맞는 빌더로 RslRlPpoActorCriticCfg를 만든다."""
    if policy_architecture not in _BUILDERS:
        raise KeyError(f"등록되지 않은 정책 구조: {policy_architecture} (사용 가능: {sorted(_BUILDERS)})")
    return _BUILDERS[policy_architecture](agent_cfg)

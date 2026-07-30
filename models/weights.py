"""사전학습 가중치·에셋 로드 유틸.

파인튜닝 ckpt 피클에는 habitat Config 객체가 박혀 있어, 참조 불가 클래스를
더미로 대체하는 StubUnpickler로 가중치만 취득한다.

가중치 경로(/data/ETPNav_weights):
  wp_pred/check_cwp_bestdist_hfov90        waypoint predictor (depth 전용)
  ddppo-models/gibson-2plus-resnet50.pth   depth 인코더 백본
  pretrained/ETP/mlm.sap_r2r/...           vln_bert 초기화 (bert.*)
  logs/checkpoints/release_r2r/...         파인튜닝 전체 정책 (net.module.*)
"""

import os
import pickle
import sys
import types

import torch

# ---- 경로 상수 ----

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "assets")
BERT_CONFIG_JSON = os.path.join(_ASSETS, "bert_config.json")
BERT_VOCAB = os.path.join(_ASSETS, "bert_vocab.txt")
WEIGHTS_ROOT = "/data/ETPNav_weights"
WP_PRED = os.path.join(WEIGHTS_ROOT, "wp_pred/check_cwp_bestdist_hfov90")
DDPPO = os.path.join(WEIGHTS_ROOT, "ddppo-models/gibson-2plus-resnet50.pth")
PRETRAIN = os.path.join(
    WEIGHTS_ROOT, "pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt")
FINETUNED = os.path.join(
    WEIGHTS_ROOT, "logs/checkpoints/release_r2r/ckpt.iter12000.pth")


# ---- ckpt 열기 (habitat 의존 피클 대응) ----

class StubUnpickler(pickle.Unpickler):
    """임포트 불가 클래스를 더미 타입으로 대체해 언피클을 통과시킨다."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (dict,), {"__setstate__": lambda s, st: None})


class _PM:
    Unpickler = StubUnpickler

    @staticmethod
    def load(f, **kw):
        return StubUnpickler(f).load()


def load_ckpt(path):
    return torch.load(path, map_location="cpu", pickle_module=_PM,
                      weights_only=False)


# ---- 가중치별 state_dict 정리 ----

def wp_pred_state():
    """waypoint predictor 사전학습 state_dict."""
    ck = load_ckpt(WP_PRED)
    return ck["predictor"]["state_dict"] if "predictor" in ck else ck


def pretrain_state():
    """pretrain state_dict — module. 프리픽스 제거, sap_head 키를 bert.*로
    복제(원본 초기화 코드의 매핑 규칙과 동일)."""
    ck = load_ckpt(PRETRAIN)
    sd = ck.get("state_dict", ck)
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        if "sap_head" in k and not k.startswith("bert."):
            out["bert." + k] = v
        out[k] = v
    return out


def finetuned_state():
    """파인튜닝 state_dict — net.module. 프리픽스 제거."""
    ck = load_ckpt(FINETUNED)
    sd = ck.get("state_dict", ck)
    return {k.replace("net.module.", "", 1): v for k, v in sd.items()}


# ---- 모듈 로드 ----

def load_partial(module, state, prefix="", verbose=True):
    """이름·shape 일치분만 로드. (일치, shape 불일치, 미로드) 수 반환."""
    own = module.state_dict()
    picked, shape_bad = {}, 0
    for k, v in state.items():
        if prefix:
            if not k.startswith(prefix):
                continue
            k = k[len(prefix):]
        if k in own:
            if own[k].shape == v.shape:
                picked[k] = v
            else:
                shape_bad += 1
    module.load_state_dict(picked, strict=False)
    if verbose:
        print(f"  부분 로드: 일치 {len(picked)} / shape 불일치 {shape_bad}"
              f" / 모듈 파라미터 {len(own)}")
    return len(picked), shape_bad, len(own) - len(picked)


# zero-init 확장 파라미터 — 구 ckpt에 없어도 로드 허용(결손 = 동작 동등)
_ZERO_INIT_KEYS = ("frame_embed.", "node_flag_embed.", "node_type_embed.")


def load_compat(module, state):
    """학습 ckpt 정밀 로드.

    load_partial(이름 매칭 관용)과 달리 _ZERO_INIT_KEYS 외의 결손·잉여
    키는 오류로 취급해 ckpt-모델 불일치를 침묵 통과시키지 않는다.
    """
    inc = module.load_state_dict(state, strict=False)
    bad = [k for k in inc.missing_keys
           if not any(k.startswith(p) or ("." + p) in k
                      for p in _ZERO_INIT_KEYS)]
    if bad or inc.unexpected_keys:
        raise RuntimeError(f"ckpt 불일치: 결손 {bad}, "
                           f"잉여 {list(inc.unexpected_keys)}")

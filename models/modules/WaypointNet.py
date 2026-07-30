"""전방 단일 뷰 waypoint 제안기 (ForwardWaypointNet).

관측을 인코딩해 **픽셀 히트맵**을 낸다 — 히트맵의 다중 피크가 후보 지점이고,
각 후보의 3D 위치는 그 픽셀의 depth를 리프트해 얻는다(거리 예측 없음).
단일 뷰에서 방위는 이미지 열의 단조 함수, 거리는 depth에 이미 있으므로
극좌표 빈은 투영 기하의 중복 인코딩이라 폐기했다.

출력:
  heat (B, 1, S, S) 로짓 — 후보 적합도. GT = 로봇별 도달가능 자유공간
       래스터라 다중 양성이며 BCE로 학습(단일 타깃 CE 아님). 손실은 이
       한 항뿐이다
  feat (B, 768, S, S) 후보 feature 맵 — 채택된 셀의 벡터가 그 고스트의
       토큰 feature가 된다(뷰 전체 임베딩 하나를 공유하면 고스트끼리
       구분되지 않음)

도착 heading은 제안기가 예측하지 않는다 — 제안기 입력은 depth뿐이라
학습 가능한 타깃이 기하 대리물밖에 없고, 그건 복도에서만 맞는다.
heading 예측은 지시·기억·선택 결과를 아는 선택기(GraphPlanning) 몫.

입력 토큰:
  depth (B, 128, 4, 4) → 4×4 = 16 패치 토큰(128→768)
  히스토리 스택 (B, K, 128, 4, 4)도 허용 — 프레임당 16토큰 + 프레임 임베딩,
  인덱스 0 = 현재 관측. 선택으로 CLIP RGB 1토큰(512→768) 추가.
  디코더는 현재 프레임 토큰만 공간 복원에 쓴다(히스토리는 문맥 제공).

디코더: 현재 프레임 토큰 (B,768,4,4)을 업샘플하며 원본 depth 스템(고주파
기하)과 합쳐 후보 feature 맵 (B,768,S,S)을 만들고, 여기서 두 헤드가 갈린다.
후보 feature가 768인 이유는 통과성 헤드 g가 노드·스냅샷과 같은 공간에서
동작해야 하기 때문(사전학습 g를 그대로 합류).

z 주입(inject):
  'g'    heat += g(후보 feature, z) 픽셀별 가산 바이어스 (기본)
  'film' 디코더 입력 feature에 FiLM 변조
  'both' 둘 다 / 'none' 주입 없음 (ablation)

사용:
  net = ForwardWaypointNet(z_dim=128, g_head=g)
  net.load_etpnav()                       # TRM 층 가중치 부분 재활용
  out = net(depth_feats, z, depth_m=depth)     # heat 로짓 + feature 맵
  cand = net.propose(out)                      # 임계 통과 셀(점수 내림차순)
  # 월드 0.5m 억제와 상위 K 컷은 리프트 단계(EANav.lift)에서 — 억제 반경이
  # 픽셀이 아니라 월드 거리로 규정돼 있어 depth가 필요하다
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers.modeling_bert import (BertAttention, BertEncoder,
                                                BertIntermediate, BertLayer,
                                                BertOutput, BertSelfAttention,
                                                BertSelfOutput)

import weights as lp
from .TraversabilityHead import FiLM
from .TraversabilityHead import TraversabilityHead

# 히트맵 한 변 크기(정사각) — depth 256² 기준 8px/셀, HFOV 90°에서 2.8°/열
HEATMAP_HW = 32
# 히스토리 최대 프레임 수
N_HIST_MAX = 4
# 후보 추출 규약: 시그모이드 임계 → 월드 억제 반경(m) → 상위 K
PEAK_THRESH = 0.5
NMS_RADIUS_M = 0.5
TOP_K = 5
# propose가 리프트 단계로 넘기는 임계 통과 셀 상한
MAX_CAND = 64


class ForwardWaypointNet(nn.Module):
    def __init__(self, z_dim=128, hidden=768, use_rgb=False,
                 heatmap_hw=HEATMAP_HW, inject="g", g_head=None,
                 stem_ch=64, dec_ch=256):
        """inject: 'g'(기본) | 'film' | 'both' | 'none'."""
        super().__init__()
        from pytorch_transformers import BertConfig

        assert inject in ("g", "film", "both", "none")
        self.inject = inject
        self.hidden = hidden
        self.heatmap_hw = heatmap_hw
        self.stem_ch = stem_ch
        self.use_rgb = use_rgb
        # 토큰화 사영: depth 패치 128→768, (선택) CLIP RGB 512→768
        self.patch_proj = nn.Linear(128, hidden)
        if use_rgb:
            self.rgb_proj = nn.Linear(512, hidden)

        # TRM 본체. waypoint_TRM 및 내부 서브모듈 이름은 사전학습 state_dict와
        # 일치해야 load_etpnav()의 이름 매칭 부분 로드가 성립한다.
        cfg = BertConfig()
        cfg.model_type = "visual"
        cfg.finetuning_task = "waypoint_predictor"
        cfg.hidden_dropout_prob = 0.3
        cfg.hidden_size = hidden
        cfg.num_attention_heads = 12
        cfg.num_hidden_layers = 2
        self.waypoint_TRM = WaypointBert(config=cfg)

        # 히스토리 프레임 임베딩. zero-init이라 학습 전에는 프레임 순서 신호가
        # 없다(과거 프레임은 순서 없는 추가 토큰으로만 섞임). 히스토리 자체는
        # 어텐션 경로로 학습 전에도 출력에 영향을 준다.
        self.frame_embed = nn.Embedding(N_HIST_MAX, hidden)
        nn.init.zeros_(self.frame_embed.weight)

        # depth 스템: 원본 depth를 히트맵 해상도로 직접 내려 고주파 기하를
        # 보존한다(TRM 토큰은 4×4라 자유공간 경계를 낼 해상도가 없음).
        self.depth_stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, stem_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, stem_ch), nn.ReLU(inplace=True),
            nn.Conv2d(stem_ch, stem_ch, 3, padding=1),
            nn.GroupNorm(8, stem_ch), nn.ReLU(inplace=True))

        # 업샘플 디코더: 토큰 격자 4×4 → heatmap_hw. 배율은 2의 거듭제곱
        n_up = int(round(math.log2(heatmap_hw / 4)))
        assert 4 * 2 ** n_up == heatmap_hw, "heatmap_hw는 4의 2배수 배여야 함"
        ups, ch = [], hidden
        for _ in range(n_up):
            ups += [nn.ConvTranspose2d(ch, dec_ch, 4, stride=2, padding=1),
                    nn.GroupNorm(8, dec_ch), nn.ReLU(inplace=True)]
            ch = dec_ch
        self.decoder = nn.Sequential(*ups)
        # 후보 feature 맵 (B, hidden, S, S) — g·헤드가 공유하는 표현
        self.fuse = nn.Sequential(
            nn.Conv2d(dec_ch + stem_ch, dec_ch, 3, padding=1),
            nn.GroupNorm(8, dec_ch), nn.ReLU(inplace=True),
            nn.Conv2d(dec_ch, hidden, 1))
        self.heat_head = nn.Conv2d(hidden, 1, 1)

        if inject in ("g", "both"):
            self.g = g_head or TraversabilityHead(hidden, z_dim)
        if inject in ("film", "both"):
            self.film = FiLM(hidden, z_dim)

    # ---- 인코딩 ----

    def _tokens(self, depth_feats, rgb_feat=None):
        """depth (B,128,4,4) 또는 (B,K,128,4,4) → 토큰 (B, K·16[+1], 768).

        프레임 인덱스 0 = 현재 관측, 1.. = 과거(최신순). K ≤ N_HIST_MAX.
        """
        if depth_feats.dim() == 4:
            depth_feats = depth_feats.unsqueeze(1)
        B, K, _, _, _ = depth_feats.shape
        assert K <= N_HIST_MAX, f"히스토리 {K} > N_HIST_MAX {N_HIST_MAX}"
        t = depth_feats.flatten(3).transpose(2, 3)
        t = self.patch_proj(t)
        t = t + self.frame_embed.weight[:K].view(1, K, 1, -1)
        t = t.reshape(B, -1, t.shape[-1])
        if self.use_rgb and rgb_feat is not None:
            t = torch.cat([t, self.rgb_proj(rgb_feat).unsqueeze(1)], dim=1)
        return t

    def forward(self, depth_feats, z, rgb_feat=None, depth_m=None):
        """관측 → 후보 히트맵.

        depth_feats: (B,128,4,4) 또는 히스토리 (B,K,128,4,4), 인덱스 0=현재.
        z: (B, z_dim). depth_m: 원본 depth (B,H,W) m 단위 — 스템 입력.
        미전달 시 스템은 0으로 대체(토큰 경로만으로 추론, 해상도 열화).
        반환: {"heat": (B,1,S,S) 로짓, "feat": (B,768,S,S) 후보 feature}.
        """
        tok = self._tokens(depth_feats, rgb_feat)
        B, L, _ = tok.shape
        # 가산 어텐션 마스크: 균일 0 = 마스킹 없음 (패치 토큰 전역 어텐션)
        mask = torch.zeros(B, 1, L, L, device=tok.device, dtype=tok.dtype)
        enc = self.waypoint_TRM(tok, attention_mask=mask)
        # 공간 복원은 현재 프레임 16토큰만 사용 (히스토리는 문맥으로 섞임)
        cur = enc[:, :16].transpose(1, 2).reshape(B, self.hidden, 4, 4)
        if self.inject in ("film", "both"):
            cur = self.film(cur.permute(0, 2, 3, 1), z).permute(0, 3, 1, 2)

        S = self.heatmap_hw
        dec = self.decoder(cur)
        if depth_m is None:
            stem = torch.zeros(B, self.stem_ch, S, S,
                               device=dec.device, dtype=dec.dtype)
        else:
            if depth_m.dim() == 3:
                depth_m = depth_m.unsqueeze(1)
            stem = self.depth_stem(depth_m)
            stem = F.interpolate(stem, size=(S, S), mode="bilinear",
                                 align_corners=False)
        feat = self.fuse(torch.cat([dec, stem], dim=1))

        heat = self.heat_head(feat)
        if self.inject in ("g", "both"):
            # 픽셀별 통과성 바이어스 — 노드·스냅샷과 같은 g 인스턴스
            gs = self.g(feat.permute(0, 2, 3, 1), z)
            heat = heat + gs.unsqueeze(1)
        return {"heat": heat, "feat": feat}

    # ---- 후보 추출 ----

    @staticmethod
    def propose(out, thresh=PEAK_THRESH, max_cand=MAX_CAND):
        """히트맵 → 임계 통과 셀을 점수 내림차순으로.

        월드 0.5m 억제와 상위 K 컷은 depth가 필요하므로 리프트 단계가
        맡는다(EANav.lift). 임계를 넘는 셀이 없으면 valid가 전부 False —
        고스트 0개가 정상 결과이고 대체 후보를 억지로 만들지 않는다
        (막다른 곳에서는 노드+STOP만으로 백트래킹이 유도돼야 함).

        반환 dict (B, max_cand 패딩, 패딩은 valid=False):
          uv    (B,N,2) long — 히트맵 좌표 (열 u, 행 v)
          score (B,N) 시그모이드 확률
          feat  (B,N,768) 해당 셀의 후보 feature
          valid (B,N) bool
        """
        heat, fmap = out["heat"], out["feat"]
        B, C, S, _ = fmap.shape
        prob = torch.sigmoid(heat).flatten(1)
        n = min(max_cand, prob.shape[1])
        score, idx = prob.topk(n, dim=1)
        valid = score >= thresh
        v, u = idx // S, idx % S
        feat = fmap.flatten(2).transpose(1, 2).gather(
            1, idx.unsqueeze(-1).expand(B, n, C))
        return {"uv": torch.stack([u, v], dim=-1), "score": score,
                "feat": feat, "valid": valid}

    def load_etpnav(self, verbose=True):
        """사전학습 TRM transformer 층만 부분 재활용 (스템·디코더·헤드는 신규)."""
        state = lp.wp_pred_state()
        return lp.load_partial(self, state, verbose=verbose)


# ---- ETPNav waypoint predictor TRM 본체 (이식) ----
# 사전학습 가중치 이름 매칭을 위해 클래스·서브모듈 이름을 보존한다.
# 계보: Oscar → Recurrent VLN-BERT (MIT).

class CaptionBertSelfAttention(BertSelfAttention):
    """BertSelfAttention + history_state(K/V 연장) 지원."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores \
            / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return (context_layer.view(*new_shape), attention_scores)


class CaptionBertAttention(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = CaptionBertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.config = config

    def forward(self, input_tensor, attention_mask, head_mask=None,
                history_state=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask,
                                 history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        return (attention_output,) + self_outputs[1:]


class CaptionBertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = CaptionBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                                           head_mask, history_state)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return (layer_output,) + attention_outputs[1:]


class CaptionBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [CaptionBertLayer(config)
             for _ in range(config.num_hidden_layers)])
        self.config = config

    def forward(self, hidden_states, attention_mask, head_mask=None,
                encoder_history_states=None):
        for i, layer_module in enumerate(self.layer):
            history_state = None if encoder_history_states is None \
                else encoder_history_states[i]
            layer_outputs = layer_module(hidden_states, attention_mask,
                                         head_mask[i], history_state)
            hidden_states = layer_outputs[0]
            if i == self.config.num_hidden_layers - 1:
                slang_attention_score = layer_outputs[1]
        return (hidden_states, slang_attention_score)


class BertImgModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = CaptionBertEncoder(config)

    def forward(self, input_x, attention_mask=None):
        # (0/1 마스크) → 가산 마스크: 1→0, 0→-10^4
        ext = attention_mask.to(dtype=next(self.parameters()).dtype)
        ext = (1.0 - ext) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers
        out = self.encoder(input_x, ext, head_mask=head_mask)
        return (out[0],) + out[1:]


class WaypointBert(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.bert = BertImgModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_x, attention_mask=None):
        outputs = self.bert(input_x, attention_mask=attention_mask)
        return self.dropout(outputs[0])

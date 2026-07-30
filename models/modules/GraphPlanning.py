"""지시×그래프 cross-modal 내비게이션 — 전역 선택기(ZNavBert) + 그래프 맵(GraphMap).

ZNavBert: backbones.CrossModalTransformer(GlocalTextPathNavCMT)를 본문 수정
없이 감싸고 z 조건화를 얹는다.
  forward_txt         지시 토큰 → 텍스트 임베딩 (언어 인코더)
  forward_panorama    관측 특징 → 노드 임베딩 (기억 그래프·스냅샷 공용)
  forward_navigation  그래프 노드 × [지시 + embodiment 토큰] attention
                      → 노드 선택 로짓 (STOP 포함 전역 선택)
  rank_snapshots      기억 스냅샷 feature → 통과성 점수 (g 직접)

GraphMap: 온라인 토폴로지 맵 — 방문 노드·고스트의 생성·병합·소멸과
forward_navigation 입력(gmap 텐서) 조립까지 담당. ZNavBert의 짝 생산자.

z 주입(inject):
  'token' z 사영을 지시 시퀀스 끝에 embodiment 토큰으로 삽입 (기본)
  'film'  전역 인코딩 출력에 FiLM 변조 (ablation)
  'none'  주입 없음 (ablation)

초기화 = bert_config + pretrain(bert.*) 또는 finetuned(vln_bert.*) 부분
로드. transformers 4.44 필요.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn

import weights as lp
from .TraversabilityHead import FiLM
from .TraversabilityHead import TraversabilityHead


# ---- 백본 구성 ----

def build_config(fix_lang=True):
    """ETPNav r2r 학습 설정과 동일한 백본 config를 구성한다."""
    from transformers import PretrainedConfig

    cfg = PretrainedConfig.from_json_file(lp.BERT_CONFIG_JSON)
    cfg.max_action_steps = 100
    # 관측 feature 규격: CLIP 512 + depth 128 + 각도 피처 4
    cfg.image_feat_size = 512
    cfg.use_depth_embedding = True
    cfg.depth_feat_size = 128
    cfg.angle_feat_size = 4
    # 층 구성: 언어 9 / 파노라마 2 / cross-modal 4
    cfg.num_l_layers = 9
    cfg.num_pano_layers = 2
    cfg.num_x_layers = 4
    cfg.graph_sprels = True
    cfg.glocal_fuse = "global"
    cfg.fix_lang_embedding = fix_lang
    cfg.fix_pano_embedding = False
    cfg.update_lang_bert = not fix_lang
    cfg.output_attentions = True
    cfg.pred_head_dropout_prob = 0.1
    cfg.use_lang2visn_attn = False
    return cfg


def _base_class():
    from backbones.CrossModalTransformer import GlocalTextPathNavCMT
    return GlocalTextPathNavCMT


class ZNavBert(nn.Module):
    """GlocalTextPathNavCMT 래퍼 + embodiment 토큰 + g 스냅샷 랭킹."""

    def __init__(self, z_dim=128, fix_lang=True, pretrained=True,
                 inject="token", g_head=None):
        """inject: 'token'(기본) | 'film' | 'none'."""
        super().__init__()
        assert inject in ("token", "film", "none")
        self.inject = inject
        cls = _base_class()
        cfg = build_config(fix_lang)
        # transformers 4.x에는 from_pretrained(state_dict=...) 경로가 없어
        # 직접 구성 후 이름·shape 일치분만 부분 로드한다.
        self.bert = cls(cfg)
        if pretrained:
            # pretrain ckpt 키는 bert.* 프리픽스 — 스트립 후 매칭
            lp.load_partial(self.bert, lp.pretrain_state(), prefix="bert.")
        H = cfg.hidden_size
        if inject == "token":
            # embodiment 토큰. 값 = zero-init 사영(초기 기여 0),
            # 가시성 = 가산 어텐션 바이어스 버퍼(초기 -10^4 = 완전 차단).
            # set_token_warmup(frac)이 바이어스를 0으로 올려 토큰을 연다 —
            # 사전학습 동작을 정확히 보존한 채 점진 개방하는 구조.
            self.z_token = nn.Linear(z_dim, H)
            nn.init.zeros_(self.z_token.weight)
            nn.init.zeros_(self.z_token.bias)
            self.register_buffer("z_token_bias", torch.tensor(-10000.0))
        elif inject == "film":
            self.film = FiLM(H, z_dim)
        # 전역 선택 후보의 노드 피처 임베딩 (zero-init: 미전달·미학습 시 무영향)
        #   flag: 0=STOP, 1=방문, 2=고스트(미방문 후보)
        #   type: 0=일반, 1=계단, 2=엘리베이터
        #   floor_diff: 후보 층 − 현재 층 (±FLOOR_DIFF_MAX로 클램프)
        # 층 차이는 pos_fts에 필드를 늘리는 대신 별도 임베딩으로 더한다 —
        # 백본의 pos 사영 입력 폭이 바뀌면 사전학습 가중치를 못 쓴다.
        self.node_flag_embed = nn.Embedding(3, H)
        self.node_type_embed = nn.Embedding(3, H)
        self.floor_diff_embed = nn.Embedding(2 * FLOOR_DIFF_MAX + 1, H)
        nn.init.zeros_(self.node_flag_embed.weight)
        nn.init.zeros_(self.node_type_embed.weight)
        nn.init.zeros_(self.floor_diff_embed.weight)
        # 도착 heading 예측 — 후보 토큰 → (sin, cos). 홉당 1개(선택된 후보의
        # 값을 쓴다). 지시·기억·선택 결과를 아는 선택기만 "도착해서 어디를
        # 볼 것인가"에 답할 정보를 갖는다.
        self.heading_head = nn.Linear(H, 2)
        # 지시 없는 프레임용 학습 가능 토큰 — 표본을 버리지 않고 무조건부
        # 주행으로 학습한다
        self.null_instr = nn.Parameter(torch.zeros(1, 1, H))
        nn.init.normal_(self.null_instr, std=0.02)
        # 통과성 헤드 — 스냅샷 랭킹·불가능 판정용 (WaypointNet과 인스턴스 공유)
        self.g = g_head or TraversabilityHead(H, z_dim)

    # ---- 언어 / 파노라마 (백본 그대로) ----

    def forward_txt(self, txt_ids, txt_masks):
        return self.bert.forward_txt(txt_ids, txt_masks)

    def forward_panorama(self, *args, **kwargs):
        return self.bert.forward_panorama(*args, **kwargs)

    # ---- 지시 ----

    def null_instruction(self, batch_size, device):
        """지시 없는 프레임의 대체 입력 → (txt_embeds (B,1,H), masks (B,1)).

        forward_txt 출력 자리에 그대로 넣는다.
        """
        emb = self.null_instr.expand(batch_size, 1, -1).to(device)
        return emb, torch.ones(batch_size, 1, dtype=torch.long,
                               device=device)

    # ---- embodiment 토큰 ----

    def set_token_warmup(self, frac):
        """토큰 가시성 설정. frac 0 = 완전 차단, 1 = 완전 개방."""
        self.z_token_bias.fill_(-10000.0 * (1.0 - float(frac)))

    def _with_z_token(self, z, txt_embeds, txt_masks):
        """지시 시퀀스 끝에 embodiment 토큰 1개를 덧붙인다.

        백본의 마스크 규약이 (1−m)·(−10⁴)을 어텐션에 가산하므로,
        m = 1 + bias/10⁴로 두면 이 토큰의 어텐션 가산이 정확히
        z_token_bias가 된다.
        """
        tok = self.z_token(z).unsqueeze(1)
        txt_embeds = torch.cat([txt_embeds, tok], dim=1)
        m = (1.0 + self.z_token_bias / 10000.0).clamp(0.0, 1.0)
        m = m.expand(txt_masks.shape[0], 1)
        return txt_embeds, torch.cat([txt_masks.float(), m], dim=1)

    # ---- 전역 선택 ----

    def forward_navigation(self, z, txt_embeds, txt_masks,
                           gmap_vpids, gmap_step_ids, gmap_img_fts,
                           gmap_pos_fts, gmap_masks, gmap_visited_masks,
                           gmap_pair_dists,
                           node_flags=None, node_types=None, floor_diff=None,
                           mask_visited=True, g_bias=False, g_tau=None):
        """그래프 노드 선택 로짓 — 후보 = 방문 노드 + 고스트 + STOP(인덱스 0).

        백본을 무수정으로 두기 위해 원본 forward_navigation 절차를 여기서
        재현한다 — 백본이 바뀌면 이 함수도 대조·갱신할 것.

        node_flags/node_types/floor_diff: (B, N) long. 주어지면 zero-init
        임베딩을 노드 임베딩에 가산한다. mask_visited=False면 방문 노드도
        후보로 남긴다(백트래킹 선택 허용). 고스트의 feature(gmap_img_fts·
        pos_fts) 구성은 호출부(GraphMap.gmap_inputs) 책임.

        g_bias=True면 통과성 헤드가 **고스트 후보에만** 바이어스를 얹고
        (미방문 지점의 진입 가부는 g가 판단, 이미 지나온 방문 노드는 불개입),
        g_tau를 주면 그 미만은 −inf로 차단한다. 고스트 판별은 node_flags==2,
        미전달 시 "미방문 ∧ 인덱스>0". STOP(인덱스 0)은 어떤 경로로도
        마스킹되지 않아 항상 선택 가능하다 — 도착 판단의 유일한 출구.
        반환: {"gmap_embeds", "global_logits", "heading"} — 마스크 밖
        로짓은 -inf, heading은 후보별 (B,N,2) 단위 sin·cos이며 선택된
        후보의 값만 쓴다(손실도 정답 후보 위치에서만 건다).
        """
        if self.inject == "token":
            txt_embeds, txt_masks = self._with_z_token(z, txt_embeds,
                                                       txt_masks)
        g = self.bert.global_encoder
        gmap_embeds = (gmap_img_fts
                       + g.gmap_step_embeddings(gmap_step_ids)
                       + g.gmap_pos_embeddings(gmap_pos_fts))
        if node_flags is not None:
            gmap_embeds = gmap_embeds + self.node_flag_embed(node_flags)
        if node_types is not None:
            gmap_embeds = gmap_embeds + self.node_type_embed(node_types)
        if floor_diff is not None:
            fd = floor_diff.clamp(-FLOOR_DIFF_MAX, FLOOR_DIFF_MAX)
            gmap_embeds = gmap_embeds + self.floor_diff_embed(
                fd + FLOOR_DIFF_MAX)
        sprels = (g.sprel_linear(gmap_pair_dists.unsqueeze(3))
                  .squeeze(3).unsqueeze(1)
                  if g.sprel_linear is not None else None)
        gmap_embeds = g.encoder(txt_embeds, txt_masks, gmap_embeds,
                                gmap_masks, graph_sprels=sprels)
        if self.inject == "film":
            gmap_embeds = self.film(gmap_embeds, z)
        logits = self.bert.global_sap_head(gmap_embeds).squeeze(2)
        if g_bias or g_tau is not None:
            if node_flags is not None:
                ghost = node_flags == 2
            else:
                ghost = gmap_visited_masks.logical_not()
                ghost[:, 0] = False
            logits = self.g.bias_logits(logits, gmap_embeds, z,
                                        tau=g_tau, mask=ghost)
        if mask_visited:
            logits = logits.masked_fill(gmap_visited_masks, -float("inf"))
        logits = logits.masked_fill(gmap_masks.logical_not(), -float("inf"))
        heading = self.heading_head(gmap_embeds)
        heading = heading / heading.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return {"gmap_embeds": gmap_embeds, "global_logits": logits,
                "heading": heading}

    # ---- 기억 스냅샷 랭킹 ----

    def rank_snapshots(self, z, snap_img_fts, snap_masks=None):
        """스냅샷 임베딩 (B, K, H) → 통과성 점수 (B, K). 마스크 밖 -inf."""
        scores = self.g(snap_img_fts, z)
        if snap_masks is not None:
            scores = scores.masked_fill(snap_masks.logical_not(),
                                        -float("inf"))
        return scores

    def load_finetuned(self, verbose=True):
        """파인튜닝 가중치(vln_bert.*)를 백본에 부분 로드한다."""
        state = lp.finetuned_state()
        return lp.load_partial(self.bert, state, prefix="vln_bert.",
                               verbose=verbose)


# ---- 온라인 토폴로지 그래프 맵 (전역 선택의 후보 생산측) ----

# pos_fts·pair_dists 거리/스텝 정규화 상수 (ETPNav 동일)
MAX_DIST = 30.0
MAX_STEP = 10.0
# 노드 병합 반경(m), 고스트 병합 반경(m) = 제안기 NMS 억제 반경
NODE_RADIUS = 1.0
GHOST_RADIUS = 0.5
# 전역 후보 상한 — 다층 기억이 커져도 attention 비용을 상수로 묶는다
MAX_CAND = 128
# 층 차이 임베딩 범위 (±FLOOR_DIFF_MAX로 클램프)
FLOOR_DIFF_MAX = 4


class GraphMap:
    """온라인 토폴로지 맵 — ETPNav GraphMap의 전방 단일 뷰·SE(2) 개조판.

    좌표 = 월드 (x, y, z) 3D, 층은 floor 인덱스로 따로 기록한다(y는 층
    높이 오프셋이라 층 간 위치 병합을 막는 역할). heading = rad, 전방
    +z·우 +x (EANav.pixel_to_pose·robot_to_world와 동일 규약).

    반경 2종: node_radius = 같은 방문 노드로 볼 거리(고스트가 이 안에
    들어오면 노드와 중복이라 버린다), ghost_radius = 고스트끼리 병합할
    거리(제안기 NMS 억제 반경과 같은 값).

    홉마다의 사용 (배치 없음, 에피소드당 인스턴스 1개):
      vp = gm.update(pos, node_embed, cand_pos, cand_embeds)  # 관측 반영
      inp = gm.gmap_inputs(vp, pos, heading, device)          # 텐서 조립
      out = policy('navigation', z=z, **txt, **inp)           # 전역 선택
      sel = inp["gmap_vpids"][out["global_logits"][0].argmax()]
      # sel = None(STOP) | 방문 vp | 고스트 gvp. 고스트를 행선으로
      # 확정하면 호출부가 gm.delete_ghost(sel)로 명시 소멸시킨 뒤 주행
      # → 도착 관측을 update()로 반영한다(도착 위치가 곧 새 방문 노드).
      # 자동 소멸 없음 — 실주행 도착 오차에서 오소멸/누락을 막는 계약.

    고스트 = 미방문 후보 노드. **에피소드 내내 누적**되며(프런티어 재방문이
    가능해야 한다) 소멸은 delete_ghost() 명시 호출뿐이다 — 현재 뷰의 후보만
    남기지 않는다. feature = 그 고스트를 관측한 뷰
    node_embed의 평균(합·횟수 누적), 위치 = 관측 위치들의 평균
    (ghost_aug > 0이면 학습용 위치 잡음). front = 고스트를 본 방문 노드들
    — 그래프 거리는 최근접 front를 경유해 계산한다.
    """

    def __init__(self, node_radius=NODE_RADIUS, ghost_radius=GHOST_RADIUS,
                 ghost_aug=0.0, max_cand=MAX_CAND):
        import networkx as nx
        self._nx = nx
        self.graph = nx.Graph()
        self.node_pos = {}
        self.node_embed = {}
        self.node_step = {}
        self.node_type = {}
        self.node_floor = {}
        self.ghost_cnt = 0
        self.ghost_pos = {}
        self.ghost_mean_pos = {}
        self.ghost_aug_pos = {}
        self.ghost_embed = {}
        self.ghost_front = {}
        self.ghost_floor = {}
        self.node_radius = node_radius
        self.ghost_radius = ghost_radius
        self.max_cand = max_cand
        self.ghost_aug = ghost_aug
        self._step = 0
        self._last_vp = None
        self._idx = {}
        self._D = None
        self._H = None

    # -- 내부 유틸 --

    def _localize(self, qpos, kpos_dict, radius):
        """qpos에서 radius 안의 최근접 키 반환 (없으면 None)."""
        best, best_vp = radius, None
        for vp, pos in kpos_dict.items():
            d = float(np.linalg.norm(qpos - pos))
            if d < best:
                best, best_vp = d, vp
        return best_vp

    def _front_dist(self, gvp):
        """고스트의 최근접 front 방문 노드와 그 직선 거리."""
        best, best_vp = float("inf"), None
        for vp in self.ghost_front[gvp]:
            d = float(np.linalg.norm(self.node_pos[vp]
                                     - self.ghost_aug_pos[gvp]))
            if d < best:
                best, best_vp = d, vp
        return best, best_vp

    def _refresh_paths(self):
        """전체 최단거리·경유 홉수 재계산.

        networkx all_pairs_dijkstra는 노드가 100개를 넘으면 홉마다 수십 ms가
        들어 학습이 배로 느려진다(실측 65ms/홉). 같은 결과를 scipy 희소
        행렬 다익스트라로 한 번에 얻는다.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        vps = list(self.graph.nodes)
        self._idx = {vp: i for i, vp in enumerate(vps)}
        n = len(vps)
        if n == 0:
            self._D = self._H = None
            return
        r, c, w = [], [], []
        for a, b, dat in self.graph.edges(data=True):
            i, j = self._idx[a], self._idx[b]
            r += [i, j]; c += [j, i]
            ww = float(dat.get("weight", 1.0))
            w += [ww, ww]
        adj = csr_matrix((w, (r, c)), shape=(n, n)) if r else             csr_matrix((n, n))
        self._D, pred = dijkstra(adj, directed=False,
                                 return_predecessors=True)
        # 경유 홉수 = 선행자 사슬 길이 (거리와 같은 호출에서 얻는다)
        H = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            order = np.argsort(self._D[i])
            for j in order:
                if j == i or not np.isfinite(self._D[i, j]):
                    continue
                pj = pred[i, j]
                H[i, j] = 1 if pj == i else H[i, pj] + 1
        self._H = H

    def sp_dist(self, a, b):
        """a→b 그래프 최단거리 (연결 없으면 MAX_DIST)."""
        if self._D is None or a not in self._idx or b not in self._idx:
            return MAX_DIST
        d = self._D[self._idx[a], self._idx[b]]
        return float(d) if np.isfinite(d) else MAX_DIST

    def sp_steps(self, a, b):
        """a→b 경유 노드 수 (원본 len(path) 규약: 자기 자신은 0)."""
        if self._H is None or a not in self._idx or b not in self._idx:
            return 0
        return int(self._H[self._idx[a], self._idx[b]])

    # -- 갱신 --

    def update(self, cur_pos, cur_embed, cand_pos, cand_embeds,
               node_type=0, floor=0, prev_vp=None):
        """홉 도착 시 호출 — 관측 1회를 그래프에 반영하고 cur_vp를 반환.

        cur_pos: 월드 (x, y, z). cur_embed: 현재 뷰 node_embed (768,).
        cand_pos/cand_embeds: 리프트된 고스트 후보의 월드 위치와 **그 셀의
        디코더 feature**(EANav.lift 반환) — 뷰 전체 임베딩을 공유하면
        고스트끼리 구분되지 않는다.
        node_type: 0=일반, 1=계단, 2=엘리베이터 (현재 노드에 기록).
        floor: 현재 층 인덱스 — 후보 집합 층 필터·층 차이 피처의 근거.
        prev_vp: 직전 노드 강제 지정 (기본 = 마지막 update의 노드).

        처리: 현 위치가 기존 방문 노드와 겹치면(node_radius) 재방문,
        아니면 새 방문 노드. 직전 노드와 엣지 연결. 각 후보는 기존 방문
        노드 근방이면 엣지만 잇고 고스트는 만들지 않으며(중복), 기존
        고스트와 ghost_radius 안이면 병합(feature 합·위치 평균·front
        추가), 아니면 새 고스트 생성. 고스트 소멸은 여기서 하지 않는다 —
        행선으로 확정된 고스트를 호출부가 delete_ghost(id)로 명시
        삭제한다(주행 도착 오차와 무관하게 소비를 보장하는 원본 계약).
        """
        pos = np.asarray(cur_pos, dtype=float)
        prev = self._last_vp if prev_vp is None else prev_vp

        cur_vp = self._localize(pos, self.node_pos, self.node_radius)
        if cur_vp is None:
            cur_vp = str(len(self.node_pos))
        self.graph.add_node(cur_vp)
        if prev is not None and prev != cur_vp:
            d = float(np.linalg.norm(self.node_pos[prev] - pos))
            self.graph.add_edge(prev, cur_vp, weight=d)
        self.node_pos[cur_vp] = pos
        self.node_embed[cur_vp] = cur_embed
        self.node_step[cur_vp] = min(self._step, 99)
        self.node_type[cur_vp] = int(node_type)
        self.node_floor[cur_vp] = int(floor)
        self._step += 1

        for cpos, cemb in zip(cand_pos, cand_embeds):
            cpos = np.asarray(cpos, dtype=float)
            nvp = self._localize(cpos, self.node_pos, self.node_radius)
            if nvp is not None:
                # 이미 방문 노드가 후보에 있으므로 고스트는 만들지 않고
                # 연결 정보만 남긴다
                d = float(np.linalg.norm(pos - self.node_pos[nvp]))
                self.graph.add_edge(cur_vp, nvp, weight=d)
                continue
            gvp = self._localize(cpos, self.ghost_mean_pos,
                                 self.ghost_radius)
            if gvp is None:
                gvp = f"g{self.ghost_cnt}"
                self.ghost_cnt += 1
                self.ghost_pos[gvp] = [cpos]
                self.ghost_mean_pos[gvp] = cpos
                self.ghost_embed[gvp] = [cemb, 1]
                self.ghost_front[gvp] = [cur_vp]
                self.ghost_floor[gvp] = int(floor)
            else:
                self.ghost_pos[gvp].append(cpos)
                self.ghost_mean_pos[gvp] = np.mean(self.ghost_pos[gvp],
                                                   axis=0)
                self.ghost_embed[gvp][0] = self.ghost_embed[gvp][0] + cemb
                self.ghost_embed[gvp][1] += 1
                self.ghost_front[gvp].append(cur_vp)

        # 학습용 고스트 위치 잡음 (ghost_aug=0이면 평균 위치 그대로)
        self.ghost_aug_pos = {g: p.copy()
                              for g, p in self.ghost_mean_pos.items()}
        if self.ghost_aug > 0:
            for gvp in self.ghost_aug_pos:
                noise = np.clip(
                    np.random.normal(0.0, self.ghost_aug, 3),
                    -self.ghost_aug, self.ghost_aug)
                noise[1] = 0.0
                self.ghost_aug_pos[gvp] = self.ghost_aug_pos[gvp] + noise

        self._refresh_paths()
        self._last_vp = cur_vp
        return cur_vp

    def delete_ghost(self, gvp):
        """고스트 소멸 (행선 확정·차단 판정 시 호출)."""
        self.ghost_pos.pop(gvp)
        self.ghost_mean_pos.pop(gvp)
        self.ghost_aug_pos.pop(gvp, None)
        self.ghost_embed.pop(gvp)
        self.ghost_front.pop(gvp)
        self.ghost_floor.pop(gvp, None)

    def connect(self, vp_a, vp_b, dist):
        """방문 노드 간 명시적 엣지 추가 (층 전환 이벤트 등)."""
        self.graph.add_edge(vp_a, vp_b, weight=float(dist))
        self._refresh_paths()

    def ghost_embed_mean(self, gvp):
        return self.ghost_embed[gvp][0] / self.ghost_embed[gvp][1]

    def preload(self, nodes, edges=()):
        """사전 구축 기억을 방문 노드로 일괄 등록 (경로 캐시는 1회만 갱신).

        배포에서는 표준 기억 그래프가 이미 있는 상태로 주행을 시작하므로,
        학습 재생도 같은 조건에서 출발해야 후보 집합 크기가 맞는다(빈
        그래프로 학습하면 후보 5개짜리로 배운 모델이 추론에서 128개를
        받는다). 노드마다 update를 부르면 그때마다 전체 최단경로를 다시
        계산하므로 여기서는 등록만 하고 마지막에 한 번만 갱신한다.

        nodes: [(pos(x,y,z), embed(768,), floor, type)] — 등록 순서가 곧 id
        edges: [(i, j, 길이)] — nodes 인덱스 기준
        반환: 등록된 vp 리스트(입력 순서와 대응)
        """
        vps = []
        for pos, emb, fl, ntype in nodes:
            vp = str(len(self.node_pos))
            self.graph.add_node(vp)
            self.node_pos[vp] = np.asarray(pos, dtype=float)
            self.node_embed[vp] = emb
            self.node_step[vp] = 0
            self.node_type[vp] = int(ntype)
            self.node_floor[vp] = int(fl)
            vps.append(vp)
        for i, j, L in edges:
            if 0 <= i < len(vps) and 0 <= j < len(vps):
                self.graph.add_edge(vps[i], vps[j], weight=float(L))
        self._refresh_paths()
        return vps

    def persistent_nodes(self):
        """영속 기억(토폴로지 맵)에 기록할 노드만 반환.

        고스트는 아직 가보지 않은 자유공간 지점이라 잠재 목적지 레지스트리인
        영속 기억에 들어가면 안 된다 — 기억 기록은 반드시 이 함수를 거쳐
        방문 노드만 내보낸다. 고스트는 세션 그래프(이 인스턴스) 안에서만
        후보로 존재한다.
        반환: {vp: {"pos", "embed", "type", "step"}} (방문 노드 전용)
        """
        return {vp: {"pos": self.node_pos[vp],
                     "embed": self.node_embed[vp],
                     "type": self.node_type[vp],
                     "floor": self.node_floor[vp],
                     "step": self.node_step[vp]}
                for vp in self.node_pos}

    # -- forward_navigation 입력 조립 --

    def gmap_inputs(self, cur_vp, cur_pos, cur_heading, device="cpu",
                    cur_floor=None):
        """전역 선택 입력 텐서 일괄 조립 (B=1).

        후보 집합 = 현재 층 방문 노드 전부 + 다른 층은 전환 노드(계단·
        엘베)만 + 누적 고스트 + STOP(0번 슬롯). 총 MAX_CAND를 넘으면
        노드·고스트를 함께 놓고 현재 위치 근접순으로 자른다 — 다층 기억이 커져도 attention 비용이
        상수로 묶인다. cur_floor 미지정 시 현재 노드의 층을 쓴다.

        반환 dict — forward_navigation 인자 그대로:
          gmap_vpids          [None(STOP)] + 방문 vp + 고스트 gvp (선택
                              인덱스 → id 디코드용, 텐서 아님)
          gmap_step_ids       (1,N) 방문 스텝 (STOP·고스트 = 0)
          gmap_img_fts        (1,N,768) STOP=영벡터, 방문=node_embed,
                              고스트=셀 feature 평균
          gmap_pos_fts        (1,N,7) [sin/cos 상대 heading, sin/cos 고도,
                              직선거리, 그래프 최단거리, 경유 스텝] 정규화
          gmap_masks          (1,N) 전부 True
          gmap_visited_masks  (1,N) 방문 노드만 True
          gmap_pair_dists     (1,N,N) 그래프 최단거리 (고스트는 최근접
                              front 경유) / MAX_DIST
          node_flags          (1,N) 0=STOP, 1=방문, 2=고스트
          node_types          (1,N) 0=일반, 1=계단, 2=엘베 (고스트 = 0)
          floor_diff          (1,N) 후보 층 − 현재 층 (STOP = 0)
        """
        cur_pos = np.asarray(cur_pos, dtype=float)
        if cur_floor is None:
            cur_floor = self.node_floor.get(cur_vp, 0)
        # 층 필터: 현재 층 전부, 다른 층은 전환 노드(계단 1·엘베 2)만
        nodes = [v for v in self.node_pos
                 if self.node_floor.get(v, 0) == cur_floor
                 or self.node_type.get(v, 0) in (1, 2)]
        ghosts = list(self.ghost_mean_pos.keys())
        over = len(nodes) + len(ghosts) + 1 - self.max_cand
        if over > 0:
            # 상한 초과분은 **노드·고스트를 함께 놓고** 현재 위치에서 먼
            # 것부터 제외한다. 고스트는 에피소드 내내 누적되므로 노드만
            # 잘라내면 누적 고스트가 기억 노드를 후보에서 밀어낸다.
            cand = ([(v, self.node_pos[v], False) for v in nodes]
                    + [(g, self.ghost_mean_pos[g], True) for g in ghosts])
            far = sorted(cand,
                         key=lambda t: -float(np.linalg.norm(
                             t[1] - cur_pos)))
            drop = {t[0] for t in far[:over] if t[0] != cur_vp}
            nodes = [v for v in nodes if v not in drop]
            ghosts = [g for g in ghosts if g not in drop]
        vpids = [None] + nodes + ghosts
        N = len(vpids)

        step_ids = [0] + [self.node_step[v] for v in nodes] + [0] * len(ghosts)
        visited = [0] + [1] * len(nodes) + [0] * len(ghosts)
        flags = [0] + [1] * len(nodes) + [2] * len(ghosts)
        types = [0] + [self.node_type[v] for v in nodes] + [0] * len(ghosts)
        fdiff = ([0]
                 + [self.node_floor.get(v, 0) - cur_floor for v in nodes]
                 + [self.ghost_floor.get(g, cur_floor) - cur_floor
                    for g in ghosts])

        embeds = ([self.node_embed[v] for v in nodes]
                  + [self.ghost_embed_mean(g) for g in ghosts])
        img_fts = torch.stack([torch.zeros_like(embeds[0])] + embeds, dim=0)

        pos_fts = np.zeros((N, 7), dtype=np.float32)
        for i, vp in enumerate(vpids):
            if vp is None:
                continue
            if vp in self.node_pos:
                vpos = self.node_pos[vp]
                sp_dist = self.sp_dist(cur_vp, vp)
                sp_step = self.sp_steps(cur_vp, vp)
            else:
                vpos = self.ghost_aug_pos[vp]
                fdist, fvp = self._front_dist(vp)
                sp_dist = self.sp_dist(cur_vp, fvp) + fdist
                sp_step = self.sp_steps(cur_vp, fvp) + 1
            dx, dy, dz = vpos - cur_pos
            line = max(math.hypot(dx, dz), 1e-8)
            line3 = max(float(np.linalg.norm(vpos - cur_pos)), 1e-8)
            rel_h = math.atan2(dx, dz) - cur_heading
            rel_e = math.asin(max(-1.0, min(1.0, dy / line3)))
            pos_fts[i] = [math.sin(rel_h), math.cos(rel_h),
                          math.sin(rel_e), math.cos(rel_e),
                          line3 / MAX_DIST, sp_dist / MAX_DIST,
                          sp_step / MAX_STEP]

        pair = np.zeros((N, N), dtype=np.float32)
        for j in range(1, N):
            for k in range(j + 1, N):
                v1, v2 = vpids[j], vpids[k]
                if v1 in self.node_pos and v2 in self.node_pos:
                    d = self.sp_dist(v1, v2)
                elif v1 in self.node_pos:
                    fd, fv = self._front_dist(v2)
                    d = self.sp_dist(v1, fv) + fd
                elif v2 in self.node_pos:
                    fd, fv = self._front_dist(v1)
                    d = self.sp_dist(v2, fv) + fd
                else:
                    fd1, fv1 = self._front_dist(v1)
                    fd2, fv2 = self._front_dist(v2)
                    d = fd1 + self.sp_dist(fv1, fv2) + fd2
                pair[j, k] = pair[k, j] = d / MAX_DIST

        dev = torch.device(device)
        long = torch.long
        return {
            "gmap_vpids": vpids,
            "gmap_step_ids": torch.tensor([step_ids], dtype=long,
                                          device=dev),
            "gmap_img_fts": img_fts.unsqueeze(0).to(dev),
            "gmap_pos_fts": torch.from_numpy(pos_fts).unsqueeze(0).to(dev),
            "gmap_masks": torch.ones(1, N, dtype=torch.bool, device=dev),
            "gmap_visited_masks": torch.tensor([visited],
                                               dtype=torch.bool, device=dev),
            "gmap_pair_dists": torch.from_numpy(pair).unsqueeze(0).to(dev),
            "node_flags": torch.tensor([flags], dtype=long, device=dev),
            "node_types": torch.tensor([types], dtype=long, device=dev),
            "floor_diff": torch.tensor([fdiff], dtype=long, device=dev),
        }

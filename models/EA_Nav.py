"""EA-Nav 최상위 모델 — mode 디스패치.

modules/의 부품을 한 모델로 엮는다. 학습 대상은 URDFEncoder,
TraversabilityHead(g), WaypointNet, GraphPlanning의 신규 파라미터이고,
나머지(depth 인코더·언어·CLIP·단안 depth)는 프로즌/재활용.

홉 루프 (전역 선택은 홉마다 1회, 저수준 이동 중엔 미호출):
  z   = policy('embodiment', urdf_paths=[...])          # 에피소드당 1회
  txt = policy('language', txt_ids=..., txt_masks=...)   # 지시당 1회
  policy.reset_episode(); gm = GraphMap()                # 에피소드 초기화
  반복:
    obs   = policy('observe', rgb=..., depth=...)        # 히스토리 버퍼 갱신
    out   = policy('propose', obs=obs, z=z)              # 픽셀 히트맵
    prop  = policy.wp.propose(out)                       # 임계 통과 셀
    cand  = policy.lift(prop, obs, pose, floor_y)        # 리프트+월드 NMS
    vp    = gm.update(pos, obs['node_embed'][0],         # 고스트 갱신
                      cand['world'][0], cand['embed'][0], floor=fl)
    inp   = gm.gmap_inputs(vp, pos, heading, device)
    nav   = policy('navigation', z=z, txt_embeds=txt, txt_masks=...,
                   mask_visited=False, g_bias=True, **inp)
    i     = nav['global_logits'][0].argmax()
    sel   = inp['gmap_vpids'][i]                # None = STOP(도착)
    hd    = nav['heading'][0, i]                # 그 홉의 도착 heading
    # 고스트면 gm.delete_ghost(sel) 후 주행. 후보가 0개여도 대체 후보를
    # 만들지 않는다 — 노드+STOP만 남아 백트래킹이 자연 유도된다.
  지시가 없는 프레임은 policy.nav.null_instruction(B, dev)로 대체한다.

후보 표현 = 픽셀 히트맵. 방위는 이미지 열, 거리는 depth에서 읽으므로
극좌표 빈은 쓰지 않는다. 회전 행동은 없고 방향 전환은 이동의 도착
heading으로만 표현된다.

상태 집합 = {주행, 대기중, 도착, 불가능(미탐색), 불가능(embodiment)}.
  도착 = 전역 STOP 선택(+목표형은 레지스트리 규칙 검증), 대기중 = 엘베
  FSM 구간, 불가능(embodiment) = 기억 그래프 도달성 검사(판정 전용 —
  실행 플래닝에 사용 금지), 불가능(미탐색) = 기억에 목적지 없음.
  상태 판정 규칙 자체는 무학습이며 모델 밖에서 수행한다.

관측→노드 임베딩은 forward_panorama를 K=1 뷰로 통과시킨다 — 기억 스냅샷
임베딩도 같은 경로라 rank_memory와 navigation의 feature 공간이 일치한다.
goal grounding용 CLIP 텍스트/이미지는 VisionEncoder.CLIPEncoder가 제공.
"""

import math

import torch
import torch.nn as nn

from modules.VisionEncoder import DepthEncoder
from modules.GraphPlanning import ZNavBert
from modules.TraversabilityHead import TraversabilityHead
from modules.URDFEncoder import URDFEncoder, collate_graphs, urdf_to_graph
from modules.WaypointNet import (ForwardWaypointNet, N_HIST_MAX,
                                 NMS_RADIUS_M, TOP_K)

# 렌더 규약 (데이터 생성부와 동일 — 변경 시 양쪽 동시 수정):
# 정사각 256², HFOV=VFOV 90° → fx=fy=128, 주점 (128,128), pitch 0,
# depth = planar z-depth(m). 픽셀 중심 = 인덱스 + 0.5.
DEPTH_HW = 256
HFOV_DEG = 90.0
# 히트맵 GT와 공유하는 판정 상수 (common.heat_gt와 동일값 유지 의무):
# 바닥 높이 y=0 기준 허용 오차, 후보 최대 거리(m)
HEAT_GROUND_TOL = 0.12
HEAT_MAX_M = 5.0


class EANav(nn.Module):
    def __init__(self, z_dim=128, use_rgb_wp=False, waypoint_inject="g",
                 nav_inject="token", clip_device="cuda", hist_k=N_HIST_MAX,
                 depth_hw=DEPTH_HW, hfov_deg=HFOV_DEG):
        super().__init__()
        assert 1 <= hist_k <= N_HIST_MAX
        # 통과성 헤드 g — 픽셀 후보·고스트 선택·기억 랭킹이 같은 인스턴스 공유
        self.g = TraversabilityHead(768, z_dim)
        self.urdf_enc = URDFEncoder(z_dim=z_dim)
        # depth 인코더는 동결
        self.depth_enc = DepthEncoder(depth_hw=depth_hw)
        self.nav = ZNavBert(z_dim=z_dim, inject=nav_inject, g_head=self.g)
        self.wp = ForwardWaypointNet(z_dim=z_dim, use_rgb=use_rgb_wp,
                                     inject=waypoint_inject, g_head=self.g)
        self.hist_k = hist_k
        self.depth_hw = depth_hw
        self.hfov_deg = hfov_deg
        # 히스토리 버퍼 (최신 우선) — 에피소드 경계에서 reset_episode 필수
        self._hist = []
        # CLIP·단안 depth는 무학습·대용량이라 지연 생성
        self._clip = None
        self._mono = None
        self._clip_device = clip_device

    # ---- 프로즌 보조 모델 (지연 생성) ----

    @property
    def clip(self):
        if self._clip is None:
            from modules.VisionEncoder import CLIPEncoder
            self._clip = CLIPEncoder(self._clip_device)
        return self._clip

    @property
    def mono_depth(self):
        """depth 센서 미부착 로봇용 단안 추정기 (프로즌)."""
        if self._mono is None:
            from modules.VisionEncoder import MonoDepthEstimator
            self._mono = MonoDepthEstimator(device=self._clip_device)
        return self._mono

    # ---- 투영 규약 (모델-데이터 공용) ----

    @property
    def _fx(self):
        return (self.depth_hw / 2.0) / math.tan(math.radians(self.hfov_deg / 2))

    def cell_pixel(self, u, v, heatmap_hw=None):
        """히트맵 셀 → 표본으로 쓸 렌더 픽셀 인덱스 (u, v).

        GT(common.heat_gt)와 같은 규약 = 셀 중심 픽셀 하나(중앙값 풀링
        아님). 라벨과 추론이 다른 픽셀을 보면 학습이 어긋난다.
        """
        k = self.depth_hw // (heatmap_hw or self.depth_hw)
        return u * k + k // 2, v * k + k // 2

    def pixel_to_pose(self, u, v, depth_val, heatmap_hw=None):
        """히트맵 셀 + depth → 로봇 프레임 (dx 우측, dz 전방, dy 위쪽).

        depth_val은 클립 이전 원본 planar z-depth(m)이고 셀 중심 픽셀에서
        표본한다. pitch 0이므로 회전 항은 없다.
        """
        pu, pv = self.cell_pixel(u, v, heatmap_hw)
        c = self.depth_hw / 2.0
        fx = self._fx
        return ((pu + 0.5 - c) * depth_val / fx,     # dx = 우측
                depth_val,                            # dz = 전방(planar)
                -(pv + 0.5 - c) * depth_val / fx)     # dy = 위쪽

    @staticmethod
    def robot_to_world(dx, dz, cx, cz, yaw_deg):
        """로봇 프레임 변위 → 월드 (x, z). 데이터 생성부 규약의 역변환."""
        ry = math.radians(yaw_deg)
        return (cx + dz * math.sin(ry) + dx * math.cos(ry),
                cz + dz * math.cos(ry) - dx * math.sin(ry))

    # ---- 관측 인코딩 ----

    def reset_episode(self):
        """이미지 히스토리 버퍼 비움 — 에피소드/씬 전환마다 호출."""
        self._hist = []

    def observe(self, rgb_uint8, depth_m=None, cam_h=None,
                loc_heading_rad=None, push_history=True):
        """rgb (B,H,W,3 uint8) [+ depth (B,256,256) m] → 관측 특징.

        depth_m 미전달 = depth 센서 미부착 분기 → 프로즌 단안 추정기로
        metric depth를 만든다(cam_h 필수, 지면 정합에 사용).
        반환: {depth_m 원본(리프트·스템용), depth_feats (B,128,4,4),
               depth_hist (B,K,128,4,4) 최신 우선, rgb_feat (B,512),
               node_embed (B,768)}.
        push_history=False면 버퍼를 갱신하지 않는다(오프라인 배치 평가용).
        """
        if depth_m is None:
            assert cam_h is not None, "단안 depth 분기에는 cam_h가 필요"
            depth_m = self.mono_depth(rgb_uint8, cam_h,
                                      hfov_deg=self.hfov_deg,
                                      out_hw=self.depth_hw)
        B = depth_m.shape[0]
        depth_feats = self.depth_enc(depth_m)
        rgb_feat = self.clip(rgb_uint8)

        if push_history:
            # 배치 크기가 달라지면 이전 이력과 스택이 불가능하다 — 이력은
            # 에피소드 재생(B 고정)용이므로 그때는 버퍼를 버린다
            if self._hist and self._hist[0].shape[0] != B:
                self._hist = []
            self._hist = [depth_feats] + self._hist[:self.hist_k - 1]
            hist = list(self._hist)
        else:
            hist = [depth_feats]
        # 에피소드 초반처럼 이력이 모자라면 가장 오래된 프레임을 복제해 항상
        # K장으로 맞춘다 — 학습 배치(고정 K)와 추론이 같은 입력 형태가 된다
        hist += [hist[-1]] * (self.hist_k - len(hist))
        depth_hist = torch.stack(hist, dim=1)

        dep_pooled = depth_feats.flatten(2).mean(dim=2)
        dev = depth_feats.device
        if loc_heading_rad is None:
            loc_heading_rad = torch.zeros(B, device=dev)
        # 각도 피처 규약(4차원): [sin h, cos h, sin e, cos e], 고도 e = 0
        loc = torch.stack([torch.sin(loc_heading_rad),
                           torch.cos(loc_heading_rad),
                           torch.zeros_like(loc_heading_rad),
                           torch.ones_like(loc_heading_rad)],
                          dim=-1).unsqueeze(1)
        nav_types = torch.ones(B, 1, dtype=torch.long, device=dev)
        view_lens = torch.ones(B, dtype=torch.long, device=dev)
        pano, _ = self.nav.bert.forward_panorama(
            rgb_feat.unsqueeze(1), dep_pooled.unsqueeze(1), loc,
            nav_types, view_lens)
        return {"depth_m": depth_m, "depth_feats": depth_feats,
                "depth_hist": depth_hist, "rgb_feat": rgb_feat,
                "node_embed": pano.squeeze(1)}

    # ---- 후보 제안 → 월드 리프트 ----

    def lift(self, cand, obs, pose, floor_y=0.0, top_k=TOP_K,
             nms_radius=NMS_RADIUS_M, ground_tol=HEAT_GROUND_TOL,
             max_dist=HEAT_MAX_M, floor_slack=0.0):
        """제안 셀 → 월드 고스트 후보 (NMS 규약의 뒷단).

        cand: WaypointNet.propose 반환(점수 내림차순). pose: (floor, x, z,
        yaw_deg, cam_y) — 데이터 pose 필드와 같은 순서. floor_y: 층 높이
        오프셋(GraphMap이 층 분리에 쓰는 y 좌표).

        절차 = 점수 순회하며 ① 셀 중심 depth 표본 ② 리프트 ③ 바닥 검사
        (|높이| ≤ ground_tol, y=0 기준 — GT와 동일) ④ 거리 ≤ max_dist
        ⑤ 이미 채택한 후보와 nms_radius 이내면 버림, top_k개 차면 종료.
        임계를 넘는 셀이 없으면 후보 0개를 그대로 반환한다.

        반환: {world (B,K,3) [x, floor_y, z], dist (B,K) planar 거리,
               valid (B,K), embed (B,K,768) 셀 feature}. 좌표류는 CPU
        (그래프 맵이 numpy 기반), embed는 관측 텐서 장치를 유지한다.
        """
        _, cx, cz, yaw_deg, cam_y = pose
        uv = cand["uv"].cpu()
        ok = cand["valid"].cpu()
        B, N = uv.shape[:2]
        depth = obs["depth_m"]
        hm = self.wp.heatmap_hw
        world = torch.zeros(B, top_k, 3, dtype=torch.float32)
        dist = torch.zeros(B, top_k, dtype=torch.float32)
        valid = torch.zeros(B, top_k, dtype=torch.bool)
        pick = torch.zeros(B, top_k, dtype=torch.long)
        for b in range(B):
            taken = 0
            for n in range(N):
                if taken >= top_k:
                    break
                if not bool(ok[b, n]):
                    continue
                u, v = int(uv[b, n, 0]), int(uv[b, n, 1])
                pu, pv = self.cell_pixel(u, v, hm)
                d = float(depth[b, pv, pu])
                if d <= 0.2:
                    continue
                dx, dz, dy = self.pixel_to_pose(u, v, d, hm)
                # 바닥 판정 허용오차는 **셀이 실제로 덮는 높이**보다 작으면
                # 안 된다. GT는 셀 안에 바닥이 하나라도 투영되면 양성인데,
                # 여기서는 셀 중심 픽셀 하나만 표본하므로 그 차이만큼 여유가
                # 필요하다. 셀 반높이 = (k/2+0.5)·d/fx 이고, 카메라가 낮을수록
                # 바닥이 지평선으로 압축돼 이 항이 지배한다(Go2 cam_h 0.363m
                # 실측: 고정 0.12 m만 쓰면 제안 64개 중 유효 0~1개).
                # floor_slack — 렌더 바닥이 로봇 발 높이와 어긋나는 만큼의
                # 여유. habitat navmesh는 실제 바닥 메시보다 위에 뜨고 그
                # 정도가 씬마다 다르다(실측 0.026~0.17 m). MANSION은 바닥이
                # y=0으로 정확해 기본값 0을 쓴다.
                k_cell = self.depth_hw // (hm or self.depth_hw)
                tol = max(ground_tol, floor_slack,
                          (k_cell / 2.0 + 0.5) * d / self._fx)
                if abs(cam_y + dy) > tol:
                    continue
                wx, wz = self.robot_to_world(dx, dz, cx, cz, yaw_deg)
                r = math.hypot(wx - cx, wz - cz)
                if r > max_dist:
                    continue
                if any(math.hypot(wx - float(world[b, j, 0]),
                                  wz - float(world[b, j, 2])) < nms_radius
                       for j in range(taken)):
                    continue
                world[b, taken] = torch.tensor([wx, floor_y, wz])
                dist[b, taken] = r
                valid[b, taken] = True
                pick[b, taken] = n
                taken += 1
        # 고스트 토큰 feature = 채택된 셀의 디코더 feature (뷰 전체 임베딩을
        # 공유하면 같은 뷰의 고스트끼리 구분되지 않음)
        embed = cand["feat"].gather(
            1, pick.to(cand["feat"].device).unsqueeze(-1)
            .expand(B, top_k, cand["feat"].shape[-1]))
        return {"world": world, "dist": dist, "valid": valid, "embed": embed}

    # ---- mode 디스패치 ----

    def forward(self, mode, **kw):
        if mode == "embodiment":
            if "graph" in kw:
                batch = kw["graph"]
            else:
                batch = collate_graphs(
                    [urdf_to_graph(p) for p in kw["urdf_paths"]])
                dev = next(self.urdf_enc.parameters()).device
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
            z, aux = self.urdf_enc(batch)
            return {"z": z, "aux": aux}
        if mode == "language":
            return self.nav.forward_txt(kw["txt_ids"], kw["txt_masks"])
        if mode == "observe":
            return self.observe(kw["rgb"], kw.get("depth"), kw.get("cam_h"),
                                kw.get("loc_heading_rad"),
                                kw.get("push_history", True))
        if mode == "propose":
            # 히스토리 스택을 기본 입력으로 쓰고, 스템에는 원본 depth를 준다
            obs = kw["obs"]
            depth_feats = kw.get("depth_hist", obs.get("depth_hist",
                                                       obs["depth_feats"]))
            return self.wp(depth_feats, kw["z"],
                           rgb_feat=(obs.get("rgb_feat")
                                     if self.wp.use_rgb else None),
                           depth_m=obs.get("depth_m"))
        if mode == "navigation":
            return self.nav.forward_navigation(
                kw["z"], kw["txt_embeds"], kw["txt_masks"],
                kw.get("gmap_vpids"), kw["gmap_step_ids"],
                kw["gmap_img_fts"], kw["gmap_pos_fts"], kw["gmap_masks"],
                kw["gmap_visited_masks"], kw["gmap_pair_dists"],
                node_flags=kw.get("node_flags"),
                node_types=kw.get("node_types"),
                floor_diff=kw.get("floor_diff"),
                mask_visited=kw.get("mask_visited", True),
                g_bias=kw.get("g_bias", False), g_tau=kw.get("g_tau"))
        if mode == "rank_memory":
            return self.nav.rank_snapshots(kw["z"], kw["snap_embeds"],
                                           kw.get("snap_masks"))
        raise ValueError(f"unknown mode: {mode}")

    # ---- 초기화 로드 ----

    def load_pretrained(self, finetuned=True, verbose=True):
        """사전학습 가중치 재활용: waypoint TRM + vln_bert(파인튜닝 우선)."""
        self.wp.load_etpnav(verbose=verbose)
        if finetuned:
            self.nav.load_finetuned(verbose=verbose)

    def set_token_warmup(self, frac):
        if self.nav.inject == "token":
            self.nav.set_token_warmup(frac)

    def freeze_stage1(self):
        """1단계(통과성)에서 학습한 URDF GNN·g를 이후 전 단계 동결.

        z 분포가 흔들리면 g가 깨지고, g는 픽셀·노드·스냅샷·고스트 네
        소비처가 공유하므로 손상이 연쇄된다.
        """
        for m in (self.urdf_enc, self.g):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def tokenizer():
    """지시 토크나이저 (bert-base-uncased vocab, assets 동봉)."""
    import os

    from transformers import BertTokenizer

    import weights as lp
    return BertTokenizer(vocab_file=lp.BERT_VOCAB)

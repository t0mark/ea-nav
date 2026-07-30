"""관측 인코더 — 전방 단일 뷰 RGB-D.

DepthEncoder: ResNet50 depth 백본(backbones.DepthResNet) + gibson-2plus ckpt.
  m 단위 depth를 받아 내부에서 [0,1] 정규화(10m 클립)한다 — 호출부에서
  미리 정규화하면 이중 정규화가 되므로 금지. 단일 프레임 (B,H,W)과
  히스토리 스택 (B,K,H,W)을 모두 받아 (B,128,4,4)/(B,K,128,4,4)를 낸다
  (히스토리는 제안기 전용 입력). 동결 기본.

MonoDepthEstimator: RGB → metric depth (m). depth 센서 미부착 로봇의 분기로,
  프로즌 단안 추정기의 상대 역깊이를 지면 평면 모델에 최소제곱 정합해
  미터 단위로 되돌린다. 센서 보유 로봇은 이 경로를 타지 않는다.

CLIPEncoder: RGB uint8 → CLIP ViT-B/32 512d(프로즌) + encode_text.
  이미지·텍스트가 같은 CLIP 공간이므로 goal grounding 매칭에 사용.
  clip 패키지는 지연 임포트.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.DepthResNet import DepthResNetEncoder
import weights as lp

# depth 정규화 클립 거리(m) — 관측 사양의 가시거리와 동일
DEPTH_CLIP_M = 10.0
# 단안 추정기 기본 가중치 (프로즌, 지연 로드)
MONO_DEPTH_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEncoder(nn.Module):
    def __init__(self, depth_hw=256, checkpoint=lp.DDPPO, trainable=False):
        super().__init__()
        self.enc = DepthResNetEncoder(depth_hw=depth_hw)
        if checkpoint:
            ck = torch.load(checkpoint, map_location="cpu",
                            weights_only=False)
            # ckpt 키 = actor_critic.net.visual_encoder.* — 프리픽스를
            # 벗겨 백본 키로 변환해 로드
            sd = {}
            for k, v in ck["state_dict"].items():
                parts = k.split(".")[2:]
                if parts and parts[0] == "visual_encoder":
                    sd[".".join(parts[1:])] = v
            missing, unexpected = self.enc.load_state_dict(sd, strict=False)
            assert not missing, f"depth 백본 가중치 누락: {missing[:5]}"
        for p in self.enc.parameters():
            p.requires_grad_(trainable)

    def forward(self, depth_m):
        """depth_m m 단위 → 특징맵.

        (B,H,W) 또는 (B,H,W,1) → (B,128,4,4)
        (B,K,H,W) 히스토리 스택 → (B,K,128,4,4), 인덱스 0 = 현재 프레임
        """
        hist = depth_m.dim() == 4 and depth_m.shape[-1] != 1
        if hist:
            B, K = depth_m.shape[:2]
            depth_m = depth_m.reshape(B * K, *depth_m.shape[2:])
        if depth_m.dim() == 3:
            depth_m = depth_m.unsqueeze(-1)
        d = (depth_m / DEPTH_CLIP_M).clamp(0.0, 1.0)
        out = self.enc(d)
        return out.reshape(B, K, *out.shape[1:]) if hist else out


class MonoDepthEstimator(nn.Module):
    """RGB (B,H,W,3 uint8) → metric depth (B,S,S) m. 프로즌·지연 로드.

    추정기는 스케일 미정의 상대 역깊이를 내므로, 지면 평면 모델로 정합한다:
    피치 0·카메라 높이 h에서 지면 픽셀의 역깊이는 행 v에 선형(1/z =
    (v−cy)/(h·fy))이므로, 하단 영역에서 예측 역깊이와 이 모델을 최소제곱
    적합해 스케일·시프트를 구하고 미터로 환산한다.
    """

    def __init__(self, model_id=MONO_DEPTH_ID, device="cuda"):
        super().__init__()
        self.model_id = model_id
        self.device = device
        self._model = None
        self._proc = None

    def _load(self):
        if self._model is None:
            from transformers import AutoImageProcessor
            from transformers import AutoModelForDepthEstimation
            self._proc = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(
                self.model_id).to(self.device).eval()
            for p in self._model.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, rgb_uint8, cam_h, hfov_deg=90.0, out_hw=256,
                ground_frac=0.35):
        """cam_h: 카메라 높이(m) 스칼라 또는 (B,). ground_frac = 정합에
        사용할 이미지 하단 비율. 반환 (B, out_hw, out_hw) m 단위."""
        self._load()
        x = rgb_uint8.to(self.device).permute(0, 3, 1, 2).float() / 255.0
        inp = self._proc(images=x, return_tensors="pt", do_rescale=False)
        pred = self._model(**{k: v.to(self.device)
                              for k, v in inp.items()}).predicted_depth
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        inv = F.interpolate(pred, size=(out_hw, out_hw), mode="bilinear",
                            align_corners=False)[:, 0]

        B, S = inv.shape[0], out_hw
        fy = (S / 2.0) / torch.tan(torch.tensor(
            hfov_deg * torch.pi / 360.0, device=inv.device))
        cy = (S - 1) / 2.0
        cam_h = torch.as_tensor(cam_h, dtype=inv.dtype,
                                device=inv.device).reshape(-1)
        if cam_h.numel() == 1:
            cam_h = cam_h.expand(B)
        v0 = int(S * (1.0 - ground_frac))
        rows = torch.arange(v0, S, device=inv.device, dtype=inv.dtype)
        # 지면 모델 역깊이 (B, n_rows) — 행마다 하나
        model_inv = (rows - cy).view(1, -1) / (cam_h.view(-1, 1) * fy)
        # 예측 역깊이는 행 중앙값으로 대표(장애물·질감 이상치 억제)
        pred_inv = inv[:, v0:, :].median(dim=2).values
        pm, mm = pred_inv.mean(1, keepdim=True), model_inv.mean(1, keepdim=True)
        pc, mc = pred_inv - pm, model_inv - mm
        alpha = (pc * mc).sum(1, keepdim=True) / \
            (pc * pc).sum(1, keepdim=True).clamp(min=1e-8)
        beta = mm - alpha * pm
        metric_inv = (alpha.view(-1, 1, 1) * inv + beta.view(-1, 1, 1))
        depth = 1.0 / metric_inv.clamp(min=1.0 / DEPTH_CLIP_M)
        return depth.clamp(0.0, DEPTH_CLIP_M)


class CLIPEncoder(nn.Module):
    """RGB (B, H, W, 3 uint8) → (B, 512). 프로즌 CLIP ViT-B/32."""

    def __init__(self, device="cuda", model_name="ViT-B/32"):
        super().__init__()
        import clip

        self.model, _ = clip.load(model_name, device=device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        # CLIP 전처리 정규화 상수 — PIL 경유 없이 텐서를 직접 정규화
        self.register_buffer("mean", torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=device).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, rgb_uint8):
        x = rgb_uint8.to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.model.encode_image(x.to(self.device)).float()

    @torch.no_grad()
    def encode_text(self, texts):
        """texts [str] → (N, 512). forward의 이미지 임베딩과 같은 공간."""
        import clip

        tok = clip.tokenize(texts, truncate=True).to(self.device)
        return self.model.encode_text(tok).float()

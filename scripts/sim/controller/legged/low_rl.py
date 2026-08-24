"""legged 저수준 계층: 정책 입출력 규약(슬롯·형태 벡터·관측 조립) + 정책 산출물 수명.

wheeled의 low_ik(속도 배분)·low_lqr(균형 토크)와 대칭인 저수준 계층으로,
속도 명령 -> 관절 위치 목표 변환을 RL 정책이 맡는다. Isaac 비의존
(torch·numpy·표준 라이브러리만) — 학습(train/)과 배포(controller)가 이
파일 하나를 공유해 관측·액션 규약 불일치를 구조적으로 차단한다.

정책 입출력 규약 (form별 고정 차원 — 로봇마다 관절 수가 달라도 동일 정책):
- 조인트 슬롯: 다리 i의 j번째 관절 -> 슬롯 i x spl + j (spl = 다리당 슬롯 수).
  다리 순서는 장착 위치 (전방 우선, 좌측 우선) — 이름 규약이 아니라 기하
  기준이라 실로봇 URDF에도 동일하게 적용된다. 빈 슬롯은 마스크 0 패딩
- 로코모션 관절 판별: base_link -> 접촉 링크(발) 경로 위의 가동 조인트.
  경로 밖 관절(팔·머리·waist)은 정책 밖 — 기립 자세 PD 홀드 (plan 규칙)
- 관측 = [몸체 선속도(3), 각속도(3), 중력 방향(3), 속도 명령(3),
  슬롯 관절각 오차(S), 슬롯 관절 속도(S), 직전 액션(S), 형태 벡터(M),
  높이 스캔(R)]
- 액션 = 슬롯별 기립 자세 대비 관절각 오프셋 (action_scale 배율, 마스크 적용)

높이 스캔 (표준 rough-terrain 세팅 — legged_gym/Isaac velocity rough의
measured heights): 몸체 아래 고정 격자(RayCaster GridPattern)의 지형 높이.
정규화 = clip(몸체 z - 기립 스폰 높이 - 지점 z, ±clip) -> 기립 높이로 평지에
서 있으면 전 지점 0. 이 규약 덕에 평지 씬은 센서 없이 0 벡터가 정확한
관측이 된다 (배포 eval·플랫 학습 겸용).

형태 벡터 (GenLoco식 morphology conditioning — URDF 파라미터를 관측에 포함):
- 전역(7): [log10 질량, 전장, 전폭, 전고, 기립 스폰 높이, 다리 수/6, sprawl]
- 다리별(4 x L): [존재, 장착 위치 x, y, z] (base_link 기준, m)
- 슬롯별(10 x S): [마스크, 회전축 x, y, z (로컬), 부모 관절과의 거리,
  가동 하한, 상한, log1p 토크 한계, 속도 한계, 기립 각도]
전 항목을 configs/rl.yaml morph 스케일로 정규화한다 (고정 스케일 —
로봇 간 비교 가능성이 conditioning의 전제).

정책 산출물 (PolicyBundle): 학습이 export한 {form}/policy.pt(TorchScript,
관측 정규화 포함) + bundle.json(차원·스케일·명령 규칙·게인 규칙)을 배포가
load한다 — 저장 포맷 변경은 이 파일만 고치면 된다.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..core.base import UrdfModel, parse_urdf, zero_pose_frame

# form별 (최대 다리 수, 다리당 슬롯 수). 생성기 구조 축의 상한과 정합:
# multileg = 고관절 2 + 무릎 (+발목) 최대 4관절, humanoid = 고관절 3 +
# 무릎 + 발목 2 = 6관절
FORM_SLOTS = {"quad": (4, 4), "hex": (6, 4), "humanoid": (2, 6)}

# 형태 벡터의 항목별 폭 (모듈 docstring 구성)
_GLOBAL_FEATS = 7
_LEG_FEATS = 4
_SLOT_FEATS = 10

# URDF에서 자유도를 만드는 조인트 타입 (robot_spawn._MOVABLE_TYPES와 동일 규칙)
_MOVABLE_TYPES = ("revolute", "continuous", "prismatic")


@dataclass(frozen=True)
class RobotSlots:
    """로봇 1대의 슬롯 배정·물성 (build_slots 산출, 학습·배포 공용).

    슬롯 배열은 전부 길이 S = 최대 다리 수 x 다리당 슬롯 수. 각도 rad,
    길이 m, 토크 Nm, 속도 rad/s (URDF 규약). names는 빈 슬롯이 None.
    """

    form: str
    n_legs: int
    total_mass: float  # 총질량 (kg) — 질량 비례 게인 규칙의 입력
    root_link: str  # URDF 루트 링크 이름 (센서 부착 프림·베이스 접촉 판정용)
    contact_links: tuple  # 발 링크 이름 (접촉 센서 대상)
    # 다리 중간 링크 (고관절-발 사이, 발 제외) — 접촉 페널티 대상 (무릎·
    # 정강이로 걷는 퇴행 보행 억제, legged_gym undesired contacts)
    undesired_links: tuple
    names: tuple  # (S,) 조인트 이름 또는 None
    mask: np.ndarray  # (S,) 1/0
    default: np.ndarray  # (S,) 기립 자세 각도
    lower: np.ndarray  # (S,)
    upper: np.ndarray  # (S,)
    effort: np.ndarray  # (S,)
    vel_limit: np.ndarray  # (S,)
    axis: np.ndarray  # (S,3) 조인트 로컬 회전축
    offset: np.ndarray  # (S,) 부모 관절(또는 base)과의 거리
    mounts: np.ndarray  # (L,3) 다리 장착 위치 (base_link 기준)
    morph: np.ndarray  # (M,) 정규화된 형태 벡터

    @property
    def num_slots(self) -> int:
        """슬롯 수 S (= 정책 액션 차원)."""
        return len(self.mask)


def _leg_chains(model: UrdfModel, contact_links: list[str]) -> list[list]:
    """접촉 링크별로 base_link까지의 가동 조인트 체인(base->발 순)을 뽑는다.

    반환: [[UrdfJoint, ...], ...] — 다리 1개 = 체인 1개. 경로 위 가동
    조인트만 로코모션 관절이다 (경로 밖 팔·waist는 자연히 제외).
    """
    chains = []
    for foot in contact_links:
        chain, link = [], foot
        # 트리 상행: 발 -> base_link (parent_joint가 없는 링크 = 루트)
        while link in model.parent_joint:
            j = model.joints[model.parent_joint[link]]
            if j.jtype in _MOVABLE_TYPES:
                chain.append(j)
            link = j.parent
        chain.reverse()
        chains.append(chain)
    return chains


def build_slots(urdf_path: Path, usd_dir: Path, morph_cfg: dict) -> RobotSlots:
    """로봇 산출물(robot.urdf + meta·joints)에서 슬롯 배정·형태 벡터를 만든다.

    morph_cfg = configs/rl.yaml morph 섹션 (정규화 스케일). 다리 순서는
    장착 위치 정렬 (전방 우선, 좌측 우선 — 모듈 docstring), 체인 안 순서는
    base -> 발. form 상한(다리 수·체인 길이) 위반은 ValueError.
    """
    usd_dir = Path(usd_dir)
    with open(usd_dir / "meta.json") as f:
        meta = json.load(f)
    with open(usd_dir / "joints.json") as f:
        by_name = {j["name"]: j for j in json.load(f)["joints"]}
    model = parse_urdf(urdf_path)

    form = meta["control_tag"]
    if form not in FORM_SLOTS:
        raise ValueError(f"legged form이 아님: {form}")
    max_legs, spl = FORM_SLOTS[form]
    S = max_legs * spl

    chains = _leg_chains(model, meta["contact_links"])
    if len(chains) > max_legs:
        raise ValueError(f"{form}: 다리 수 {len(chains)} > 상한 {max_legs}")

    # 다리 장착 위치 = 체인 첫 관절의 자식 링크 프레임 원점 (조인트 변위 0
    # 자세 FK — 관절 원점과 동일). 전방(-x 내림차순) 우선, 좌측(y 양수) 우선
    mounts_by_chain = []
    for chain in chains:
        if len(chain) < 2 or len(chain) > spl:
            raise ValueError(f"{form}: 다리 관절 수 {len(chain)}이 범위 [2,{spl}] 밖")
        _, p = zero_pose_frame(model, chain[0].child)
        mounts_by_chain.append(p)
    order = sorted(range(len(chains)),
                   key=lambda i: (-round(mounts_by_chain[i][0], 3),
                                  -round(mounts_by_chain[i][1], 3)))

    # 슬롯 배열 채우기 (빈 슬롯 = 0, names는 None)
    names = [None] * S
    mask = np.zeros(S)
    default = np.zeros(S)
    lower = np.zeros(S)
    upper = np.zeros(S)
    effort = np.zeros(S)
    vel_limit = np.zeros(S)
    axis = np.zeros((S, 3))
    offset = np.zeros(S)
    mounts = np.zeros((max_legs, 3))
    standing = meta["standing_pose"]
    for li, ci in enumerate(order):
        mounts[li] = mounts_by_chain[ci]
        for si, j in enumerate(chains[ci]):
            s = li * spl + si
            info = by_name[j.name]
            names[s] = j.name
            mask[s] = 1.0
            default[s] = float(standing.get(j.name, 0.0))
            lower[s] = info["lower"]
            upper[s] = info["upper"]
            effort[s] = info["effort"]
            vel_limit[s] = info["velocity"]
            axis[s] = j.axis / max(np.linalg.norm(j.axis), 1e-9)
            # 부모 관절과의 거리 = 조인트 원점 오프셋 노름 (세그먼트 길이 대리)
            offset[s] = float(np.linalg.norm(j.xyz))

    # 루트 링크 = 어떤 조인트의 자식도 아닌 링크 (URDF 트리 루트 유일)
    children = {j.child for j in model.joints.values()}
    root_link = next(n for n in model.links if n not in children)
    # 다리 중간 링크 = 체인 조인트의 자식 중 발(마지막)이 아닌 것 전부
    contact_set = set(meta["contact_links"])
    undesired = sorted({j.child for chain in chains for j in chain}
                       - contact_set)

    morph = _morph_vector(meta, mask, default, lower, upper, effort, vel_limit,
                          axis, offset, mounts, len(chains), max_legs, morph_cfg)
    return RobotSlots(form=form, n_legs=len(chains),
                      total_mass=float(meta["metrics"]["total_mass"]),
                      root_link=root_link,
                      contact_links=tuple(meta["contact_links"]),
                      undesired_links=tuple(undesired),
                      names=tuple(names),
                      mask=mask, default=default, lower=lower, upper=upper,
                      effort=effort, vel_limit=vel_limit, axis=axis,
                      offset=offset, mounts=mounts, morph=morph)


def _morph_vector(meta: dict, mask, default, lower, upper, effort, vel_limit,
                  axis, offset, mounts, n_legs: int, max_legs: int,
                  morph_cfg: dict) -> np.ndarray:
    """형태 벡터를 조립·정규화한다 (모듈 docstring 구성·순서).

    스케일은 형태 간 비교 가능성을 위해 config 고정값 (로봇별 정규화 금지).
    """
    metrics = meta["metrics"]
    len_s = float(morph_cfg["len_scale"])
    ang_s = float(morph_cfg["ang_scale"])
    vel_s = float(morph_cfg["vel_scale"])
    # 전역: 질량은 로그 (1-265kg 분포), sprawl은 multileg 장착 축 (humanoid = 0)
    global_part = np.array([
        math.log10(max(float(metrics["total_mass"]), 1e-3)) / float(morph_cfg["mass_log_scale"]),
        float(metrics["overall_length"]) / len_s,
        float(metrics["overall_width"]) / len_s,
        float(metrics["overall_height"]) / len_s,
        float(metrics["base_height"]) / len_s,
        n_legs / 6.0,
        1.0 if meta["params"].get("mount") == "sprawl" else 0.0,
    ])
    leg_part = np.zeros((max_legs, _LEG_FEATS))
    leg_part[:n_legs, 0] = 1.0
    leg_part[:, 1:] = mounts / len_s
    slot_part = np.stack([
        mask,
        axis[:, 0], axis[:, 1], axis[:, 2],
        offset / len_s,
        lower / ang_s,
        upper / ang_s,
        np.log1p(effort) / float(morph_cfg["effort_log_scale"]),
        vel_limit / vel_s,
        default / ang_s,
    ], axis=1)
    return np.concatenate([global_part, leg_part.ravel(), slot_part.ravel()])


def morph_dim(form: str) -> int:
    """form의 형태 벡터 차원 M."""
    max_legs, spl = FORM_SLOTS[form]
    return _GLOBAL_FEATS + _LEG_FEATS * max_legs + _SLOT_FEATS * max_legs * spl


def scan_rays(scan_cfg: dict) -> int:
    """높이 스캔 격자의 레이 수 R (Isaac grid_pattern 규칙: 축별 size/res + 1).

    size가 resolution으로 나누어떨어져야 한다 (config 규약 — 어긋나면
    Isaac 내부 arange 개수와 여기 계산이 달라져 차원 불일치).
    """
    nx = int(round(float(scan_cfg["size"][0]) / float(scan_cfg["resolution"]))) + 1
    ny = int(round(float(scan_cfg["size"][1]) / float(scan_cfg["resolution"]))) + 1
    return nx * ny


def scan_obs(root_z: torch.Tensor, hits_z: torch.Tensor, base_height: float,
             clip: float) -> torch.Tensor:
    """RayCaster 지점 높이 (N,R) -> 정규화 스캔 관측 (모듈 docstring 규약).

    미스 레이(hit z = ±inf)는 클립으로 흡수된다.
    """
    return torch.clamp(root_z.unsqueeze(1) - base_height - hits_z, -clip, clip)


def obs_dim(form: str, num_rays: int, obs_cfg: dict | None = None) -> int:
    """form의 정책 관측 차원 (모듈 docstring 관측 구성 + 스캔 R).

    obs_cfg.include_morph/include_scan = False면 해당 블록 제외 (Run E
    진단 — 상수 패딩 370차원이 학습 동역학에 미치는 영향 분리. 기본 포함).
    """
    S = FORM_SLOTS[form][0] * FORM_SLOTS[form][1]
    d = 12 + 3 * S
    if obs_cfg is None or obs_cfg.get("include_morph", True):
        d += morph_dim(form)
    if obs_cfg is None or obs_cfg.get("include_scan", True):
        d += num_rays
    return d


def slot_index_tensor(slots: RobotSlots, joint_index: dict[str, int],
                      device: str) -> torch.Tensor:
    """슬롯 -> articulation 조인트 인덱스 (S,) long 텐서 (빈 슬롯 = 0).

    빈 슬롯은 mask 0으로 소거되므로 안전한 더미 인덱스 0을 쓴다 (gather용).
    joint_index = {조인트 이름: articulation 인덱스}.
    """
    idx = [joint_index[n] if n is not None else 0 for n in slots.names]
    return torch.tensor(idx, dtype=torch.long, device=device)


def slot_gather(full: torch.Tensor, slot_idx: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """전 DoF 텐서 (N,D)에서 슬롯 값 (N,S)을 뽑는다 (빈 슬롯 = 0)."""
    return full[:, slot_idx] * mask


def assemble_obs(vel_b: torch.Tensor, ang_b: torch.Tensor,
                 gravity_b: torch.Tensor, cmd: torch.Tensor,
                 q_err: torch.Tensor, qd: torch.Tensor,
                 prev_action: torch.Tensor, morph: torch.Tensor,
                 height_scan: torch.Tensor,
                 obs_scales: dict) -> torch.Tensor:
    """정책 관측 1스텝 조립 (N, obs_dim) — 학습·배포 공용 (단일 출처).

    q_err = 슬롯 관절각 - 기립 자세 (N,S), qd = 슬롯 관절 속도 (N,S),
    morph = (N,M) (로봇별 상수 행의 expand), height_scan = 정규화 스캔
    (N,R — scan_obs 규약, 평지는 0). 스케일은 configs/rl.yaml obs.
    """
    lin = float(obs_scales["lin_vel"])
    ang = float(obs_scales["ang_vel"])
    cmd_scale = torch.tensor([lin, lin, ang], device=cmd.device)
    parts = [
        vel_b * lin,
        ang_b * ang,
        gravity_b,
        cmd * cmd_scale,
        q_err * float(obs_scales["joint_pos"]),
        qd * float(obs_scales["joint_vel"]),
        prev_action,
    ]
    # include 플래그 = obs_dim과 동일 규약 (Run E 진단 — 기본 포함)
    if obs_scales.get("include_morph", True):
        parts.append(morph)
    if obs_scales.get("include_scan", True):
        parts.append(height_scan * float(obs_scales["height_scan"]))
    return torch.cat(parts, dim=1)


def cmd_limits(meta_params: dict, cmd_cfg: dict) -> dict[str, float]:
    """로봇 1대의 속도 명령 경계를 기립 높이 기반 규칙으로 만든다.

    v_max = froude x sqrt(g x 기립 높이) (Froude 수 기반 보행 속도 스케일 —
    다리 길이에 비례하는 자연 보행 속도)를 [v_min, v_cap]으로 클램프.
    학습 명령 샘플 범위와 배포 명령 클램프가 같은 규칙을 써야 배포 명령이
    학습 분포 안에 있다 (규칙은 bundle.json에 저장돼 배포 시 재현).
    """
    v = float(cmd_cfg["froude"]) * math.sqrt(9.81 * float(meta_params["stance_height"]))
    v_max = min(max(v, float(cmd_cfg["v_min"])), float(cmd_cfg["v_cap"]))
    return {"v_max": v_max, "wz_max": float(cmd_cfg["wz_max"])}


def loco_gain_overrides(slots: RobotSlots, gain_cfg: dict) -> dict[str, tuple]:
    """로코모션 관절의 스폰 게인 덮어쓰기 {이름: (강성, 감쇠)}를 만든다.

    질량 비례 게인 규칙: kp = kp_per_kg x 총질량을 [kp_min, kp_max]로
    클램프, kd = kp x kd_ratio. 근거 (독립 검증에서 문헌 교차 확인):
    - GenLoco kp = 100 x (질량/12.458kg) ≈ 8 x 질량, ManyQuadrupeds의
      질량 계층표(2-5kg->20, 12kg->100, 30-90kg->430, 200kg->1400)도 동일 직선
    - 이전 규칙(kp = 토크 한계 비례)은 Go2·ANYmal 표본 2개의 우연 일치를
      규칙화한 것 — 절차 생성 로봇의 관대한 토크 한계(고관절 수백 Nm)에서
      과강성이 되어 탐색 노이즈가 격렬한 요동·전도를 만드는 것 실측
    humanoid 발목 하향: ankle_kp_per_kg가 있으면 다리 내 5번째 이후 슬롯
    (발목 pitch/roll — 체인 순서 고관절3·무릎·발목2)에 별도 적용 (Isaac H1
    관례: 다리 kp 150-200 대비 발목 20). 학습·배포가 같은 규칙을 써야
    정책이 유효하다 (규칙은 bundle.json에 저장).
    """
    spl = FORM_SLOTS[slots.form][1]
    kp_body = float(gain_cfg["kp_per_kg"]) * slots.total_mass
    ankle_per_kg = gain_cfg.get("ankle_kp_per_kg")
    kp_min, kp_max = float(gain_cfg["kp_min"]), float(gain_cfg["kp_max"])
    ratio = float(gain_cfg["kd_ratio"])
    # 발목 감쇠 비율은 별도 (H1 kp20/kd4.0 — kp 비례로는 접지 저감쇠 떨림)
    ankle_ratio = float(gain_cfg.get("ankle_kd_ratio", ratio))
    out = {}
    for s, (name, m) in enumerate(zip(slots.names, slots.mask)):
        if m <= 0:
            continue
        # 다리 내 위치(si)로 발목 판별 — 이름 규약 비의존 (체인 순서 규약)
        ankle = ankle_per_kg is not None and s % spl >= 4
        kp = float(ankle_per_kg) * slots.total_mass if ankle else kp_body
        kp = min(max(kp, kp_min), kp_max)
        out[name] = (kp, kp * (ankle_ratio if ankle else ratio))
    return out


class _DeployPolicy(torch.nn.Module):
    """배포용 래퍼: 관측 정규화 + actor MLP (평균 액션 = 결정적 추론)."""

    def __init__(self, normalizer: torch.nn.Module, actor: torch.nn.Module):
        """학습 완료된 정규화 모듈과 actor를 그대로 감싼다 (재학습 없음)."""
        super().__init__()
        self.normalizer = normalizer
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """관측 (N, obs_dim) -> 평균 액션 (N, S)."""
        return self.actor(self.normalizer(obs))


class PolicyBundle:
    """form 정책 산출물의 수명 관리 (export = 학습측, load/act = 배포측).

    산출 폴더 구성: policy.pt (TorchScript — 정규화 포함, 학습 코드 없이
    로드 가능) + bundle.json (차원·스케일·명령/게인 규칙·학습 정보).
    """

    def __init__(self, module, meta: dict, device: str):
        """load()가 호출한다 — 직접 생성 대신 load를 쓸 것."""
        self._module = module
        self._meta = meta
        self._device = device

    @property
    def meta(self) -> dict:
        """bundle.json 내용 (form·차원·스케일·규칙)."""
        return self._meta

    @classmethod
    def load(cls, form_dir: Path, device: str) -> "PolicyBundle":
        """{form_dir}/policy.pt + bundle.json을 로드한다."""
        form_dir = Path(form_dir)
        module = torch.jit.load(str(form_dir / "policy.pt"), map_location=device)
        module.eval()
        with open(form_dir / "bundle.json") as f:
            meta = json.load(f)
        return cls(module, meta, device)

    @staticmethod
    def export(form_dir: Path, normalizer: torch.nn.Module,
               actor: torch.nn.Module, meta: dict):
        """학습 완료 정책을 배포 포맷으로 저장한다 (ppo.train_form이 호출).

        TorchScript trace로 굳혀 rsl-rl 미설치 환경에서도 로드 가능하게
        한다. 정규화 통계가 모듈 버퍼로 함께 저장된다.
        """
        form_dir = Path(form_dir)
        form_dir.mkdir(parents=True, exist_ok=True)
        wrapper = _DeployPolicy(normalizer, actor).to("cpu").eval()
        example = torch.zeros(1, int(meta["obs_dim"]))
        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, example)
        torch.jit.save(traced, str(form_dir / "policy.pt"))
        with open(form_dir / "bundle.json", "w") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """관측 (N, obs_dim) -> 평균 액션 (N, S) (결정적, 무경사)."""
        with torch.inference_mode():
            return self._module(obs)

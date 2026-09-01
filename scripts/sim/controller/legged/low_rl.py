from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..core.base import UrdfModel, parse_urdf, zero_pose_frame

FORM_SLOTS = {"quad": (4, 4), "hex": (6, 4), "humanoid": (2, 6)}

_GLOBAL_FEATS = 7
_LEG_FEATS = 4
_SLOT_FEATS = 10

_MOVABLE_TYPES = ("revolute", "continuous", "prismatic")

@dataclass(frozen=True)
class RobotSlots:

    form: str
    n_legs: int
    total_mass: float
    root_link: str
    contact_links: tuple

    undesired_links: tuple
    names: tuple
    mask: np.ndarray
    default: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    effort: np.ndarray
    vel_limit: np.ndarray
    axis: np.ndarray
    offset: np.ndarray
    mounts: np.ndarray
    morph: np.ndarray

    @property
    def num_slots(self) -> int:

        return len(self.mask)

def _leg_chains(model: UrdfModel, contact_links: list[str]) -> list[list]:

    chains = []
    for foot in contact_links:
        chain, link = [], foot

        while link in model.parent_joint:
            j = model.joints[model.parent_joint[link]]
            if j.jtype in _MOVABLE_TYPES:
                chain.append(j)
            link = j.parent
        chain.reverse()
        chains.append(chain)
    return chains

def build_slots(urdf_path: Path, usd_dir: Path, morph_cfg: dict) -> RobotSlots:

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

    mounts_by_chain = []
    for chain in chains:
        if len(chain) < 2 or len(chain) > spl:
            raise ValueError(f"{form}: 다리 관절 수 {len(chain)}이 범위 [2,{spl}] 밖")
        _, p = zero_pose_frame(model, chain[0].child)
        mounts_by_chain.append(p)
    order = sorted(range(len(chains)),
                   key=lambda i: (-round(mounts_by_chain[i][0], 3),
                                  -round(mounts_by_chain[i][1], 3)))

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

            offset[s] = float(np.linalg.norm(j.xyz))

    children = {j.child for j in model.joints.values()}
    root_link = next(n for n in model.links if n not in children)

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

    metrics = meta["metrics"]
    len_s = float(morph_cfg["len_scale"])
    ang_s = float(morph_cfg["ang_scale"])
    vel_s = float(morph_cfg["vel_scale"])

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

    max_legs, spl = FORM_SLOTS[form]
    return _GLOBAL_FEATS + _LEG_FEATS * max_legs + _SLOT_FEATS * max_legs * spl

def scan_rays(scan_cfg: dict) -> int:

    nx = int(round(float(scan_cfg["size"][0]) / float(scan_cfg["resolution"]))) + 1
    ny = int(round(float(scan_cfg["size"][1]) / float(scan_cfg["resolution"]))) + 1
    return nx * ny

def scan_obs(root_z: torch.Tensor, hits_z: torch.Tensor, base_height: float,
             clip: float) -> torch.Tensor:

    return torch.clamp(root_z.unsqueeze(1) - base_height - hits_z, -clip, clip)

def obs_dim(form: str, num_rays: int, obs_cfg: dict | None = None) -> int:

    S = FORM_SLOTS[form][0] * FORM_SLOTS[form][1]
    d = 12 + 3 * S
    if obs_cfg is None or obs_cfg.get("include_morph", True):
        d += morph_dim(form)
    if obs_cfg is None or obs_cfg.get("include_scan", True):
        d += num_rays
    return d

def slot_index_tensor(slots: RobotSlots, joint_index: dict[str, int],
                      device: str) -> torch.Tensor:

    idx = [joint_index[n] if n is not None else 0 for n in slots.names]
    return torch.tensor(idx, dtype=torch.long, device=device)

def slot_gather(full: torch.Tensor, slot_idx: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:

    return full[:, slot_idx] * mask

def assemble_obs(vel_b: torch.Tensor, ang_b: torch.Tensor,
                 gravity_b: torch.Tensor, cmd: torch.Tensor,
                 q_err: torch.Tensor, qd: torch.Tensor,
                 prev_action: torch.Tensor, morph: torch.Tensor,
                 height_scan: torch.Tensor,
                 obs_scales: dict) -> torch.Tensor:

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

    if obs_scales.get("include_morph", True):
        parts.append(morph)
    if obs_scales.get("include_scan", True):
        parts.append(height_scan * float(obs_scales["height_scan"]))
    return torch.cat(parts, dim=1)

def cmd_limits(meta_params: dict, cmd_cfg: dict) -> dict[str, float]:

    v = float(cmd_cfg["froude"]) * math.sqrt(9.81 * float(meta_params["stance_height"]))
    v_max = min(max(v, float(cmd_cfg["v_min"])), float(cmd_cfg["v_cap"]))
    return {"v_max": v_max, "wz_max": float(cmd_cfg["wz_max"])}

def loco_gain_overrides(slots: RobotSlots, gain_cfg: dict) -> dict[str, tuple]:

    spl = FORM_SLOTS[slots.form][1]
    kp_body = float(gain_cfg["kp_per_kg"]) * slots.total_mass
    ankle_per_kg = gain_cfg.get("ankle_kp_per_kg")
    kp_min, kp_max = float(gain_cfg["kp_min"]), float(gain_cfg["kp_max"])
    ratio = float(gain_cfg["kd_ratio"])

    ankle_ratio = float(gain_cfg.get("ankle_kd_ratio", ratio))
    out = {}
    for s, (name, m) in enumerate(zip(slots.names, slots.mask)):
        if m <= 0:
            continue

        ankle = ankle_per_kg is not None and s % spl >= 4
        kp = float(ankle_per_kg) * slots.total_mass if ankle else kp_body
        kp = min(max(kp, kp_min), kp_max)
        out[name] = (kp, kp * (ankle_ratio if ankle else ratio))
    return out

class _DeployPolicy(torch.nn.Module):

    def __init__(self, normalizer: torch.nn.Module, actor: torch.nn.Module):

        super().__init__()
        self.normalizer = normalizer
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        return self.actor(self.normalizer(obs))

class PolicyBundle:

    def __init__(self, module, meta: dict, device: str):

        self._module = module
        self._meta = meta
        self._device = device

    @property
    def meta(self) -> dict:

        return self._meta

    @classmethod
    def load(cls, form_dir: Path, device: str) -> "PolicyBundle":

        form_dir = Path(form_dir)
        module = torch.jit.load(str(form_dir / "policy.pt"), map_location=device)
        module.eval()
        with open(form_dir / "bundle.json") as f:
            meta = json.load(f)
        return cls(module, meta, device)

    @staticmethod
    def export(form_dir: Path, normalizer: torch.nn.Module,
               actor: torch.nn.Module, meta: dict):

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

        with torch.inference_mode():
            return self._module(obs)

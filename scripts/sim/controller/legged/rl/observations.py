"""정책 관측 세트 - preset의 `observations:` 키로 고른다.

레지스트리(OBSERVATION_SETS)에 이름을 등록해두고 build(profile, name)이 그 이름의 ObservationsCfg를
만든다. 관측 항목의 순서가 그대로 관측 벡터 순서가 되므로(concatenate_terms=True), 세트를 새로 추가할
때는 학습·배포가 같은 순서를 보도록 클래스 하나로 고정해 둔다.

접촉 관측이 필요한 이유: height scan은 "앞에 뭐가 있는지"만 알려주고 "지금 무엇으로 딛고 있는지"는
알려주지 않는다. 발 접촉 상태가 관측에 없으면 정책은 발로 딛는 것과 정강이로 딛는 것을 구분할
근거가 없다(실측: 일부 로봇이 발이 아니라 중간 관절로 보행). Isaac Lab 코어 mdp에는 접촉 관측
함수가 없어 이 파일에 직접 둔다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .robot_profile import RobotProfile

# 접촉 판정 임계(N) - 발이 지면에 "닿았는지"만 보므로 스침도 접촉으로 센다. 하중을 지지하는지까지
# 따지는 보상(rewards.py의 undesired_contacts)과 달리 여기서는 낮은 값이 맞다.
_CONTACT_THRESHOLD_N = 1.0


def _foot_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """발마다 접촉 여부를 0/1로 - 관측 벡터에 다리 수만큼 차원이 붙는다.

    net_forces_w_history의 시간축 최대를 쓰는 이유는 physics 스텝보다 관측 주기가 길어(decimation=4)
    그 사이의 짧은 접촉이 샘플링에서 통째로 빠질 수 있기 때문이다 - Isaac Lab 공식
    mdp.undesired_contacts와 같은 처리다.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    peak_force = torch.max(torch.norm(forces, dim=-1), dim=1)[0]
    return (peak_force > _CONTACT_THRESHOLD_N).float()


@configclass
class ProprioHeightScanObservationsCfg:
    """고유 감각(proprioception) + height scan(로컬 지형 굴곡)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """정책에 들어가는 관측 항목 그룹 (순서가 그대로 관측 벡터 순서가 된다)."""

        base_lin_vel = ObsTerm(func=core_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=core_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=core_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=core_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=core_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=core_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=core_mdp.last_action)
        height_scan = ObsTerm(
            func=core_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            """관측에 노이즈를 섞고, 항목들을 하나의 벡터로 이어 붙인다."""
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ProprioHeightScanContactObservationsCfg(ProprioHeightScanObservationsCfg):
    """위 세트 + 발 접촉 상태 - foot_contact는 build()가 로봇의 발 바디로 채운다.

    접촉은 실제 센서에서도 이진값으로 읽히는 판정이라 노이즈를 섞지 않는다.
    """

    @configclass
    class PolicyCfg(ProprioHeightScanObservationsCfg.PolicyCfg):
        """기존 8항목 뒤에 발 접촉이 붙는다 - 앞쪽 순서는 그대로라 관측 설계가 이어진다."""

        foot_contact = ObsTerm(func=_foot_contact, params={"sensor_cfg": None})

    policy: PolicyCfg = PolicyCfg()


def _build_proprio_heightscan(profile: RobotProfile) -> ProprioHeightScanObservationsCfg:
    """고유 감각 + height scan만 쓰는 세트 - 로봇별로 달라지는 항목이 없다."""
    return ProprioHeightScanObservationsCfg()


def _build_proprio_heightscan_contact(profile: RobotProfile) -> ProprioHeightScanContactObservationsCfg:
    """위 세트에 발 접촉을 더한다 - 접촉을 읽을 바디는 로봇마다 다르므로 여기서 채운다."""
    cfg = ProprioHeightScanContactObservationsCfg()
    cfg.policy.foot_contact.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=profile.foot_body_names
    )
    return cfg


OBSERVATION_SETS = {
    "proprio_heightscan": _build_proprio_heightscan,
    "proprio_heightscan_contact": _build_proprio_heightscan_contact,
}


def build(profile: RobotProfile, set_name: str):
    """preset.observations 이름으로 관측 세트 cfg를 만든다."""
    return OBSERVATION_SETS[set_name](profile)

"""정책 관측 세트 - preset의 `observations:` 키로 고른다.

레지스트리(OBSERVATION_SETS)에 이름을 등록해두고 build(name)이 그 이름의 ObservationsCfg를 만든다.
관측 항목의 순서가 그대로 관측 벡터 순서가 되므로(concatenate_terms=True), 세트를 새로 추가할 때는
학습·배포가 같은 순서를 보도록 클래스 하나로 고정해 둔다.
"""

from __future__ import annotations

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


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


OBSERVATION_SETS = {
    "proprio_heightscan": ProprioHeightScanObservationsCfg,
}


def build(set_name: str):
    """preset.observations 이름으로 관측 세트 cfg를 만든다."""
    return OBSERVATION_SETS[set_name]()

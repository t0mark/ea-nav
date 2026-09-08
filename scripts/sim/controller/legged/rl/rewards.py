"""보상 함수 - Rudin et al.(CoRL 2021/2022, arXiv:2109.11978) 표준 9종 + Extreme Parkour(ICRA 2024,
chengxuxin/extreme-parkour) 계단 전용 2종.

로봇마다 물리적으로 다른 지점(허벅지·정강이 접촉을 벌점 줄지 여부)은 로봇 "타입"별
RewardBuilder 서브클래스가 결정한다 - 로봇은 Isaac Lab 공식 rough_env_cfg.py의 설정을 기준으로
두 부류로 나뉜다:

- LightQuadrupedRewards: 가볍고 토크가 작은 4족(Unitree Go2/A1 등). 정상 보행 중에도 허벅지·
  정강이가 순간적으로 스치는 접촉이 잦은 몸집이라, undesired_contacts를 켜두면 "접촉을 피하려
  안 움직이는" local optimum에 빠지기 쉽다. Isaac Lab 공식 go2/a1 rough_env_cfg.py도
  `self.rewards.undesired_contacts = None`으로 이 항목을 끈다.
- StandardQuadrupedRewards: 무겁고 토크가 큰 4족(ANYmal-D, Spot 등). 몸집이 커서 허벅지 스침
  없이도 안정적으로 학습되므로, Isaac Lab 공식 anymal_d/rough_env_cfg.py처럼 부모 클래스
  (velocity_env_cfg.py) 기본값을 그대로 두고 이 항목을 포함한다.

"타입"(서브클래스)이 정하는 건 어떤 항목을 켤지(구조)뿐이고, 각 항목의 가중치(숫자)는 전부
robot yaml의 reward_weights에서 그대로 읽는다 - 코드 쪽에는 숫자 기본값을 두지 않는다(yaml에
빠지면 KeyError로 바로 드러난다).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .robot_profile import RobotProfile

# ---- 3점 이상 동시 지지가 가능한 보행(4족)의 발 공중 체류 보상.
# isaaclab.envs.mdp(코어 패키지)에는 2족 전용 feet_air_time_positive_biped만 있고, 4족용은 예제
# 패키지(isaaclab_tasks)에만 있어 런타임 의존성으로 끌어오는 대신 이 프로젝트 안에 그대로 옮겨온다
# (isaaclab_tasks의 locomotion/velocity/mdp/rewards.py 원본 로직 그대로). ----


def _feet_air_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """발이 임계 시간 이상 공중에 머문 스텝만큼 보상 - 발을 끌지 않고 제대로 들어 걷도록 유도.

    정지 명령(속도 명령이 거의 0)일 때는 보상을 0으로 죽여, 가만히 서 있어야 하는 상황에서
    불필요하게 발을 드는 행동을 학습하지 않게 한다.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


# ---- Extreme Parkour(ICRA 2024, chengxuxin/extreme-parkour) 원본 로직 포팅 ----


def _feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """발의 수평 방향 접촉힘이 수직 방향의 4배를 넘으면 1 - 계단 옆면(챌면)에 부딪힌 걸 잡아낸다.

    _reward_feet_stumble 원본 그대로: 수평/수직 힘의 비율만으로 판정해, 발을 헛디뎌 옆면을
    걷어차는 상황과 정상적으로 위에서 내리딛는 상황을 구분한다.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    horizontal_force = torch.norm(net_forces[..., :2], dim=-1)
    vertical_force = torch.abs(net_forces[..., 2])
    return torch.any(horizontal_force > 4.0 * vertical_force, dim=1).float()


def _hip_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """힙 관절이 기본 자세에서 벗어난 정도의 제곱합 - _reward_hip_pos 원본 그대로.

    다리를 옆으로 벌리지 않고도 지형을 넘도록 유도해, 부자연스러운 스프롤(splay) 자세를 억제한다.
    """
    asset = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(deviation), dim=1)


@configclass
class RewardsCfg:
    """빈 컨테이너 - RewardBuilder.build()가 필요한 속성을 채운다(타입에 따라 개수가 다를 수 있다)."""


class RewardBuilder(ABC):
    """로봇 타입별 보상 세트 조립 - 켤 항목(구조)은 서브클래스가 정하고, 가중치(숫자)는 전부
    profile.reward_weights[...]에서 그대로 가져온다(코드 쪽 기본값 없음 - yaml에 빠지면
    KeyError로 바로 드러난다)."""

    def build(self, profile: RobotProfile) -> RewardsCfg:
        """모든 타입 공통 항목을 채운 뒤, 타입별로 갈리는 접촉 페널티를 서브클래스에 맡긴다."""
        cfg = RewardsCfg()
        w = profile.reward_weights

        # ---- 속도 추종 (Rudin et al.) ----
        cfg.track_lin_vel_xy_exp = RewTerm(
            func=core_mdp.track_lin_vel_xy_exp,
            weight=w["track_lin_vel_xy_exp"],
            params={"command_name": "base_velocity", "std": 0.5},
        )
        cfg.track_ang_vel_z_exp = RewTerm(
            func=core_mdp.track_ang_vel_z_exp,
            weight=w["track_ang_vel_z_exp"],
            params={"command_name": "base_velocity", "std": 0.5},
        )

        # ---- 에너지·안정성 페널티 (Rudin et al.) ----
        cfg.lin_vel_z_l2 = RewTerm(func=core_mdp.lin_vel_z_l2, weight=w["lin_vel_z_l2"])
        cfg.ang_vel_xy_l2 = RewTerm(func=core_mdp.ang_vel_xy_l2, weight=w["ang_vel_xy_l2"])
        cfg.dof_torques_l2 = RewTerm(func=core_mdp.joint_torques_l2, weight=w["dof_torques_l2"])
        cfg.dof_acc_l2 = RewTerm(func=core_mdp.joint_acc_l2, weight=w["dof_acc_l2"])
        cfg.action_rate_l2 = RewTerm(func=core_mdp.action_rate_l2, weight=w["action_rate_l2"])

        # ---- 발 접촉 (Rudin et al.) ----
        cfg.feet_air_time = RewTerm(
            func=_feet_air_time,
            weight=w["feet_air_time"],
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )

        # ---- 계단 전용 (Extreme Parkour) ----
        cfg.feet_stumble = RewTerm(
            func=_feet_stumble,
            weight=w["feet_stumble"],
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names)},
        )
        cfg.hip_pos = RewTerm(
            func=_hip_pos,
            weight=w["hip_pos"],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=profile.hip_joint_names)},
        )

        self._add_contact_penalty(cfg, profile, w)
        return cfg

    @abstractmethod
    def _add_contact_penalty(self, cfg: RewardsCfg, profile: RobotProfile, w: dict[str, float]) -> None:
        """허벅지·정강이 등 원치 않는 접촉에 대한 페널티 항목 - 로봇 타입에 따라 켤지 말지가 갈린다."""


class LightQuadrupedRewards(RewardBuilder):
    """가볍고 토크가 작은 4족(Go2/A1 등) - undesired_contacts를 아예 안 켠다.

    Isaac Lab 공식 go2/a1 rough_env_cfg.py의 `self.rewards.undesired_contacts = None`과 같은
    판단이다 - 몸집이 작아 정상 보행 중에도 허벅지·정강이가 스치는 접촉이 잦으므로, 이 항목을
    켜두면 접촉 자체를 피하려 웅크린 채 안 움직이는 쪽이 "안전한" local optimum이 되어버린다.
    """

    def _add_contact_penalty(self, cfg: RewardsCfg, profile: RobotProfile, w: dict[str, float]) -> None:
        pass  # 항목 자체를 만들지 않는다 - RewardManager는 cfg에 없는 항목을 그냥 비활성으로 취급한다


class StandardQuadrupedRewards(RewardBuilder):
    """무겁고 토크가 큰 4족(ANYmal-D, Spot 등) - undesired_contacts를 켠다.

    Isaac Lab 공식 anymal_d/rough_env_cfg.py는 이 항목을 포함해 부모 클래스(velocity_env_cfg.py)
    기본값을 아무 것도 오버라이드하지 않는다 - 몸집이 커서 허벅지 스침 없이도 안정적으로 학습된다.
    """

    def _add_contact_penalty(self, cfg: RewardsCfg, profile: RobotProfile, w: dict[str, float]) -> None:
        cfg.undesired_contacts = RewTerm(
            func=core_mdp.undesired_contacts,
            weight=w["undesired_contacts"],
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.undesired_contact_body_names),
                "threshold": 1.0,
            },
        )


# preset의 `rewards:` 키 -> 보상 항목 세트. 각 세트 안에서 robot yaml의 reward_type이 접촉 페널티
# 켤지 말지(light/standard)를 다시 고른다 - 세트는 "어떤 항목들이 있나", reward_type은 "그 중 접촉
# 페널티를 쓰는 몸집인가"로 역할이 갈린다.
REWARD_SETS: dict[str, dict[str, RewardBuilder]] = {
    "rudin_parkour": {
        "light_quadruped": LightQuadrupedRewards(),
        "standard_quadruped": StandardQuadrupedRewards(),
    },
}


def build(profile: RobotProfile, set_name: str) -> RewardsCfg:
    """preset.rewards 세트 안에서 profile.reward_type에 맞는 RewardBuilder를 골라 보상 세트를 조립한다."""
    return REWARD_SETS[set_name][profile.reward_type].build(profile)

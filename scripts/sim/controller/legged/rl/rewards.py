"""보상 함수 - Rudin et al.(CoRL 2021/2022, arXiv:2109.11978) 표준 + Extreme Parkour(ICRA 2024,
chengxuxin/extreme-parkour) 계단 전용 2종 + 발 높이(Isaac Lab 공식 Spot 태스크 포팅).

undesired_contacts(다리 중간 관절 접촉 벌점)는 모든 로봇에 적용하되, 접촉력 임계를 로봇 체중에 맞춰
robot yaml이 정한다(undesired_contact_threshold_n, 대략 체중의 1/4). Isaac Lab 공식은 이 임계가 1N으로
고정이라 정상 보행 중의 스침까지 벌점이 되고, 그래서 가벼운 4족에서는 항목을 아예 끈다 - 접촉을
피하려 웅크린 채 안 움직이는 쪽이 "안전한" local optimum이 되기 때문이다. 임계를 하중 지지 수준으로
올리면 스침은 무시되고 그 부위로 딛고 서는 자세만 걸리므로, 로봇 타입별로 켜고 끄던 분기 없이
임계값 하나로 통일할 수 있다.

가중치(숫자)는 전부 robot yaml의 reward_weights에서 그대로 읽는다 - 코드 쪽에는 숫자 기본값을 두지
않는다(yaml에 빠지면 KeyError로 바로 드러난다).
"""

from __future__ import annotations

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


# ---- 발 높이(Isaac Lab 공식 Spot 태스크 config/spot/mdp/rewards.py의 foot_clearance_reward 포팅) ----


# 스윙발과 디딤발을 가르는 수평 속도 민감도 - 공식 Spot foot_clearance_reward의 tanh_mult 값 그대로
_SWING_SPEED_SENSITIVITY = 2.0

# 발 하나가 받을 수 있는 정규화 높이 오차의 상한. 오차가 (높이-목표)^2/목표^2 이므로 25는 목표 높이의
# 5배만큼 벗어난 지점이다(목표 0.08m면 지면 아래 0.32m - 위 0.48m). 정상 보행의 스윙 높이는 목표의
# 2-3배 안에 들어오므로 이 범위 밖에는 쓸 만한 기울기가 없고, 상한이 없으면 지형 밖으로 떨어지는
# 로봇에서 오차가 수천까지 커져 가치함수가 발산한다. 공식 Spot이 exp()로 감싸 [0,1]에 묶는 것과
# 같은 역할을 포화로 대신한다.
_MAX_CLEARANCE_ERROR = 25.0


def _feet_clearance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, target_height: float
) -> torch.Tensor:
    """스윙 중인 발이 목표 높이에서 벗어난 정도 - 목표 높이의 제곱으로 정규화한 무차원 값.

    공식 Spot의 foot_clearance_reward에서 두 가지를 바꿨다.

    1) 지면 기준. 공식은 발의 월드 절대 z를 쓴다(그 태스크가 평지 전용이라 가능한 단순화). 지형이
       있는 여기서는 씬에 이미 있는 height_scanner의 레이 히트 중 각 발과 xy가 가장 가까운 점의 z를
       그 발 아래 지면으로 삼아 뺀다.
    2) 형태. 공식은 exp(-오차/std)로 감싼 양수 보상인데, 그 태스크는 보상 스케일이 커서(air_time 5.0,
       gait 10.0) 그 안에서 의미 있는 폭을 갖는다. 이 프로젝트는 페널티 중심이라 같은 형태를 쓰면
       학습 시작부터 최댓값의 90%가 나와 기울기가 거의 없다. 그래서 감싸지 않은 오차를 그대로
       돌려주고 가중치를 음수로 준다.

    목표 높이의 제곱으로 나누므로 반환값은 "발이 전혀 안 뜬 채 최대 속도로 스윙" = 다리 수와 같다 -
    로봇 체구가 달라도 같은 가중치를 쓸 수 있다.

    tanh(발의 수평 속도)를 곱하는 이유는 디딤발과 스윙발을 가르기 위해서다 - 멈춰 있는 발은 높이가
    낮아도 벌점을 거의 안 받고, 빠르게 움직이는 발일수록 목표 높이를 지키도록 압력이 커진다.

    지형 밖에서는 두 가지가 깨진다. 레이가 메시를 벗어나면 RayCaster가 히트 좌표를 inf로 돌려주므로
    (isaaclab.utils.warp.ops가 히트 버퍼를 inf로 초기화하고 명중한 레이만 덮어쓴다) 지면 높이를 아예
    모르게 되고, 일부 레이만 살아 있으면 그 발과 상관없는 가장자리 높이를 지면으로 삼게 된다. 앞은
    벌점 0으로, 뒤는 _MAX_CLEARANCE_ERROR 포화로 막는다 - 둘 다 없으면 지형 밖으로 떨어지는 로봇에서
    오차가 발산해 가치함수를 무너뜨린다.
    """
    asset = env.scene[asset_cfg.name]
    scanner = env.scene.sensors[sensor_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    ray_hits = scanner.data.ray_hits_w
    # 발마다 xy가 가장 가까운 레이 히트를 그 발 아래 지면으로 삼는다
    squared_xy_distance = torch.sum((foot_pos[:, :, None, :2] - ray_hits[:, None, :, :2]) ** 2, dim=-1)
    nearest_ray = torch.argmin(squared_xy_distance, dim=-1)
    ground_z = torch.gather(ray_hits[..., 2], 1, nearest_ray)
    # 목표보다 낮을 때만 벌한다 - 높이 드는 것까지 벌하면 단차를 넘는 데 필요한 스윙 자체가 벌점이 된다
    shortfall = (target_height - (foot_pos[..., 2] - ground_z)).clamp(min=0.0)
    clearance_error = torch.square(shortfall) / (target_height**2)
    clearance_error = clearance_error.clamp(max=_MAX_CLEARANCE_ERROR)
    clearance_error = torch.where(torch.isfinite(ground_z), clearance_error, torch.zeros_like(clearance_error))
    foot_speed_xy = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    swing_weight = torch.tanh(_SWING_SPEED_SENSITIVITY * foot_speed_xy)
    return torch.sum(clearance_error * swing_weight, dim=1)


# ---- Extreme Parkour(ICRA 2024, chengxuxin/extreme-parkour) 원본 로직 포팅 ----


# 수평/수직 접촉힘 비의 판정 기준 - Extreme Parkour _reward_feet_stumble의 규약 그대로다.
_STUMBLE_FORCE_RATIO = 4.0
# 스텀블로 셀 최소 수평 접촉힘 - 로봇 체중 대비 비율이다. 무차원 비율로 두면 체구가 다른 로봇들이
# 같은 기준을 공유한다.
_STUMBLE_MIN_FORCE_WEIGHT_RATIO = 0.1
# 체중 계산용 중력 가속도(m/s^2).
_GRAVITY_MAGNITUDE = 9.81


def _feet_stumble(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """발이 수직면을 걷어찬 스텝이면 1 - 계단 챌면을 헛디딘 걸 잡아낸다.

    판정은 두 조건을 함께 요구한다. (1) 수평 접촉힘이 수직의 _STUMBLE_FORCE_RATIO배를 넘을 것,
    (2) 그 수평힘이 체중의 _STUMBLE_MIN_FORCE_WEIGHT_RATIO배를 넘을 것.

    (2)가 없으면 판정이 힘의 비만 보게 되는데, 비는 척도가 없어서 접촉력이 아무리 작아도 성립한다 -
    수직력이 0에 가까운 스침에서는 어떤 수평력이든 4배를 넘는다. 오르막에서는 발이 챌면을 스치는 일이
    정상 보행 중에도 생기므로, 게이트가 없으면 오르는 행동 자체가 상시 벌점을 받아 억제된다. 체중
    비율로 거는 이유는 보행을 실제로 방해하는 크기가 로봇 질량에 비례하기 때문이다.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    horizontal_force = torch.norm(net_forces[..., :2], dim=-1)
    vertical_force = torch.abs(net_forces[..., 2])
    # 전 바디 질량의 합 x 중력 = 체중(N). default_mass는 USD에서 파싱한 값이라 디바이스가 다를 수 있다
    body_weight = asset.data.default_mass.to(net_forces.device).sum(dim=1) * _GRAVITY_MAGNITUDE
    min_horizontal_force = (_STUMBLE_MIN_FORCE_WEIGHT_RATIO * body_weight).unsqueeze(1)
    kicked = (horizontal_force > _STUMBLE_FORCE_RATIO * vertical_force) & (horizontal_force > min_horizontal_force)
    return torch.any(kicked, dim=1).float()


def _hip_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """힙 관절이 기본 자세에서 벗어난 정도의 제곱합 - _reward_hip_pos 원본 그대로.

    다리를 옆으로 벌리지 않고도 지형을 넘도록 유도해, 부자연스러운 스프롤(splay) 자세를 억제한다.
    """
    asset = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(deviation), dim=1)


@configclass
class RewardsCfg:
    """빈 컨테이너 - build()가 항목을 채운다."""


def _build_rudin_parkour(profile: RobotProfile) -> RewardsCfg:
    """Rudin 표준 + Extreme Parkour 계단 항목으로 보상 세트를 조립한다."""
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
    # 임계 0.5초는 4족 스윙 시간(보통 0.2-0.3초)보다 길어 이 항목이 사실상 상시 벌점으로 작동한다
    # (실측: 전 로봇·전 구간에서 음수). 공식 Spot 태스크 air_time_reward의 mode_time 0.3으로 낮춰
    # 제대로 발을 들어 걸으면 양수가 되게 한다.
    cfg.feet_air_time = RewTerm(
        func=_feet_air_time,
        weight=w["feet_air_time"],
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names),
            "command_name": "base_velocity",
            "threshold": 0.3,
        },
    )

    # ---- 발 높이 (Isaac Lab Spot 태스크 포팅) ----
    cfg.feet_clearance = RewTerm(
        func=_feet_clearance,
        weight=w["feet_clearance"],
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=profile.foot_body_names),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": profile.foot_clearance_target_m,
        },
    )

    # ---- 몸통 자세 ----
    cfg.flat_orientation_l2 = RewTerm(func=core_mdp.flat_orientation_l2, weight=w["flat_orientation_l2"])

    # ---- 다리 중간 관절로 딛는 자세 억제 ----
    # 임계가 로봇 체중의 1/4 수준이라 보행 중 스침은 걸리지 않고, 그 부위로 하중을 지지할 때만 걸린다.
    cfg.undesired_contacts = RewTerm(
        func=core_mdp.undesired_contacts,
        weight=w["undesired_contacts"],
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.undesired_contact_body_names),
            "threshold": profile.undesired_contact_threshold_n,
        },
    )

    # ---- 계단 전용 (Extreme Parkour) ----
    cfg.feet_stumble = RewTerm(
        func=_feet_stumble,
        weight=w["feet_stumble"],
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=profile.foot_body_names),
        },
    )
    cfg.hip_pos = RewTerm(
        func=_hip_pos,
        weight=w["hip_pos"],
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=profile.hip_joint_names)},
    )
    return cfg


REWARD_SETS = {
    "rudin_parkour": _build_rudin_parkour,
}


def build(profile: RobotProfile, set_name: str) -> RewardsCfg:
    """preset.rewards 이름으로 보상 세트를 조립한다."""
    return REWARD_SETS[set_name](profile)

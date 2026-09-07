"""관절 부분집합(또는 전체)을 위치제어로 구동한다.

joint_names(선택, 기본 ".*")로 액션에 포함할 관절을 고르고(다리만 액션에 넣고 팔은 액션 밖에 두는
Unitree 공식 h1/g1/h1_2 방식을 재현), scale은 스칼라 1개이거나 관절 패턴별 값이 다른 딕셔너리일 수
있다({".*_hipx_joint": 0.125, "^(?!.*_hipx_joint).*": 0.25} 형태). 이 딕셔너리 지원은 우리가
지어낸 게 아니라 Isaac Lab의 `JointPositionActionCfg.scale`이 원래 두 형태를 그대로 받는 기능이다
(DeepRobotics M20 공식 rough_env_cfg.py에서 실제로 이렇게 쓰는 것을 클론해 확인).

clip=(-100.0, 100.0)은 fan-ziqi/robot_lab·DeepRoboticsLab/rl_training 원본이 전부 거는 안전장치를
그대로 반영한 것이다 - 평소 액션 값(스케일 0.125~0.25대)에는 전혀 걸리지 않는 느슨한 값이지만,
정책이 학습 중 순간적으로 극단값을 내는 드문 경우(원인 미상)에 관절 목표가 무한대로 튀는 것을
막아준다. 이게 빠진 채로 unitree_b2·deeprobotics_lite3를 학습시켰다가 각각 iteration 113·1379·1829
근처에서 가치함수 loss가 폭주(수 iteration 만에 58 -> inf)해 PPO가 멈추는 것을 실측했다 - 관절
목표가 안 잡힌 채 치솟으면 스티프니스가 곱해진 PD 토크가 같이 튀고, 그 토크가 들어가는
dof_torques_l2·joint_power 보상값이 극단적으로 커져 가치함수 학습 타깃을 오염시키는 것으로 추정.
"""

from __future__ import annotations

from isaaclab.envs import mdp as core_mdp
from isaaclab.utils import configclass

_ACTION_CLIP = {".*": (-100.0, 100.0)}


@configclass
class ActionsCfg:
    """빈 컨테이너 - build()가 joint_pos 속성 하나만 채운다."""


def build(action_cfg: dict) -> ActionsCfg:
    """joint_names(문자열 또는 리스트)와 scale(스칼라 또는 딕셔너리)로 위치제어 액션을 만든다."""
    joint_names = action_cfg.get("joint_names", ".*")
    if isinstance(joint_names, str):
        joint_names = [joint_names]
    cfg = ActionsCfg()
    cfg.joint_pos = core_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=joint_names,
        scale=action_cfg["scale"],
        use_default_offset=True,
        clip=_ACTION_CLIP,
    )
    return cfg

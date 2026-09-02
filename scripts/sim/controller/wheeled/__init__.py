from __future__ import annotations

from ..core.base import BaseController, RobotCtrlParams
from .ackermann import AckermannController
from .omni import OmniController
from .unicycle import UnicycleController

WHEELED_TAGS = ("diff", "skid", "ackermann", "omni")

# diff와 skid는 같은 (v, w) 모델이라 한 클래스를 쓰고 wheel_yaw_scale 설정으로만 구분한다
_BY_TAG = {"diff": UnicycleController, "skid": UnicycleController,
           "ackermann": AckermannController, "omni": OmniController}

def make_wheeled_controller(params: RobotCtrlParams, joint_names: list[str],
                            default_pose, num_envs: int, device: str, cfg: dict,
                            physics_dt: float) -> BaseController:
    """base_tag로 wheeled 제어기 구현을 고른다.

    타입 분기를 여기 한 곳에 모아, 상위 make_controller는 wheeled/legged 갈래만 알면 된다.
    wheeled_humanoid는 base_tag가 베이스 타입(diff/skid/omni)으로 풀려 그대로 재사용된다.
    """
    controller_cls = _BY_TAG.get(params.base_tag)
    if controller_cls is None:
        raise ValueError(f"지원하지 않는 wheeled base_tag: {params.base_tag}")
    return controller_cls(params, joint_names, default_pose, num_envs, device,
                          cfg, physics_dt)

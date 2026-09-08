"""legged 로봇 RL 보행 학습 환경(ManagerBasedRLEnvCfg) 조립.

씬 골격·명령은 코드에 고정하고, 관측/보상/종료조건/이벤트/액션 "로직"은 preset(configs/rl/legged/presets/
{preset}.yaml)이 이름으로 고른다 - 각 로직 모듈의 레지스트리에서 그 이름의 빌더를 꺼내 조립한다.
4개 로봇이 같은 preset을 가리키면 embodiment(관절 구동 모델·바디 형상·질량) 비교가 성립한다.
build_loco_rl_env_cfg()가 rl_trainer.py·loco_runner.py의 유일한 진입점이다.
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from scripts.sim.controller.legged.rl import actuator, event, observations, rewards, termination
from scripts.sim.controller.legged.rl import action_interface
from scripts.sim.controller.legged.rl.robot_profile import RLPreset, RobotProfile
from scripts.sim.env.curriculum.stage_env import build_stage_terrain_importer_cfg
from scripts.sim.env.robot_spawn import (
    LOCO_ARTICULATION_PROPS,
    LOCO_RIGID_BODY_PROPS,
    LOCO_SOFT_JOINT_POS_LIMIT_FACTOR,
    resolve_spawn_height,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_USD_ROOT = _REPO_ROOT / "data" / "sim" / "usd" / "real_robot" / "legged" / "multi-legged"

# 학습 명령 각속도 범위의 최댓값 - tools/04_controller_test.py의 LocoRunner가 배포 시 이 범위를 넘는
# 분포 밖 명령을 정책에 주지 않도록 pure pursuit 추종기의 상한을 이 값으로 맞추는 데 쓴다.
MAX_TRAINED_ANG_VEL_Z = 1.0


@configclass
class LocoSceneCfg(InteractiveSceneCfg):
    """지형 - 로봇 - height scanner - contact sensor로 구성된 legged RL 씬."""

    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",  # build_loco_rl_env_cfg()에서 로봇별 base_body_name으로 덮어씀
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class CommandsCfg:
    """목표 이동 속도 명령 - 4개 로봇 전부 동일."""

    base_velocity = core_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=core_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-MAX_TRAINED_ANG_VEL_Z, MAX_TRAINED_ANG_VEL_Z),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class LocoRLEnvCfg(ManagerBasedRLEnvCfg):
    """legged 로봇 보행 정책 학습용 환경 설정.

    scene.robot/observations/actions/rewards/terminations/events는 로봇·preset마다 값이 달라 여기서
    기본값을 못 둔다 - MISSING으로 선언만 해두고 build_loco_rl_env_cfg()가 반드시 채운다.
    """

    scene: LocoSceneCfg = LocoSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: object = MISSING
    actions: object = MISSING
    commands: CommandsCfg = CommandsCfg()
    rewards: object = MISSING
    terminations: object = MISSING
    events: object = MISSING
    # 단계 전환은 rl/curriculum_driver.py의 CurriculumDriver가 stage 번호로 한다(env cfg 안에 승급 term 없음).
    curriculum: object = None

    def __post_init__(self) -> None:
        """공용 시뮬레이션·센서 주기 설정 - 로봇별 세부값은 build_loco_rl_env_cfg()가 덮어쓴다."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # scene.terrain은 build_loco_rl_env_cfg()가 나중에 채우는 지연 필드라(robot/actions와 동일한
        # 이유 - stage에 따라 지형이 달라짐), 여기서는 아직 값이 없다. sim.physics_material은
        # build_loco_rl_env_cfg()가 scene.terrain을 채운 직후에 맞춰준다.
        # num_envs가 수천 개인 rough-terrain legged 학습은 GPU 접촉 패치 수가 기본 버퍼 크기를 쉽게
        # 넘는다 - Isaac Lab 공식 legged locomotion 레퍼런스(velocity_env_cfg.py)가 쓰는 값을 그대로
        # 따른다("Patch buffer overflow" PhysX 에러 방지)
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt


def build_loco_rl_env_cfg(
    robot_id: str, stage: int, num_envs: int | None = None, preset: str | None = None
) -> LocoRLEnvCfg:
    """robot_id + 커리큘럼 stage 하나에 대해 완전히 조립된 LocoRLEnvCfg를 만든다.

    stage는 학습 커리큘럼 단계다(0=평지, 1 이상=계단·경사 난이도 단계) - 씬(지형) 조립에만 쓰인다.
    관측/보상/종료조건/이벤트/액션 로직은 preset이 이름으로 고르고(각 모듈 레지스트리), preset이
    None이면 robot yaml의 rl_preset을 쓴다. num_envs가 None이면 preset.num_envs를 쓴다.
    """
    profile = RobotProfile.load(robot_id)
    preset_cfg = RLPreset.load(preset or profile.rl_preset)
    num_envs = num_envs or preset_cfg.num_envs
    usd_path = _USD_ROOT / profile.usd_path

    # 지면 위로 띄울 베이스 높이 - 안 띄우면 로봇이 지형 표면과 겹친 채로 스폰돼 첫 physics
    # 스텝에서 PhysX가 관통을 강제로 밀어내며 폭발적인 속도가 나온다
    spawn_height = resolve_spawn_height(profile.spawn_height, usd_path)

    env_cfg = LocoRLEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.terrain = build_stage_terrain_importer_cfg(stage, preset_cfg.curriculum["stage"])
    env_cfg.sim.physics_material = env_cfg.scene.terrain.physics_material

    env_cfg.scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        # activate_contact_sensors=True 필수 - scene.contact_forces(ContactSensor)가 이 로봇의
        # 몸체에서 접촉을 읽으려면 스폰 시점에 PhysX 접촉 리포터가 켜져 있어야 한다
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=True,
            rigid_props=LOCO_RIGID_BODY_PROPS,
            articulation_props=LOCO_ARTICULATION_PROPS,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, spawn_height), joint_pos=profile.build_joint_pos_cfg()),
        soft_joint_pos_limit_factor=LOCO_SOFT_JOINT_POS_LIMIT_FACTOR,
        actuators=actuator.build(profile.actuator, profile.controlled_joint_names),
    )
    env_cfg.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{profile.base_body_name}"

    env_cfg.observations = observations.build(preset_cfg.observations)
    env_cfg.actions = action_interface.build(profile, preset_cfg.action)
    env_cfg.rewards = rewards.build(profile, preset_cfg.rewards)
    env_cfg.terminations = termination.build(profile, preset_cfg.terminations)
    env_cfg.terminations.time_out = DoneTerm(func=core_mdp.time_out, time_out=True)
    env_cfg.events = event.build(profile, preset_cfg.events)

    return env_cfg

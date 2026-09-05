"""legged 로봇 RL 보행 학습 환경(ManagerBasedRLEnvCfg) 조립.

로봇마다 별도 파일을 두는 대신(Isaac Lab 예제 방식), configs/robots/legged/{category}/{robot_id}.yaml
하나로 로봇을 특정해 build_loco_rl_env_cfg()가 환경 설정을 조립한다. 관측/액션/이벤트/종료 조건은
모든 legged 로봇에 공통이고, 보상만 형태 그룹(quadruped/humanoid)에 따라 rewards.py에서 골라 쓴다.
"""

from __future__ import annotations

import math
import re
from dataclasses import MISSING, dataclass, field
from pathlib import Path

import yaml

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from scripts.sim.controller.legged.rl.rewards import HumanoidRewardsCfg, QuadrupedRewardsCfg
from scripts.sim.env.robot_spawn import ground_clearance
from scripts.sim.env.rl import build_rl_terrain_importer_cfg, promote_terrain_levels_by_travel_distance

_REPO_ROOT = Path(__file__).resolve().parents[5]
_USD_ROOT = _REPO_ROOT / "data" / "sim" / "usd" / "real_robot" / "legged"
_SIM_RL_CONFIG_PATH = _REPO_ROOT / "configs" / "sim_rl.yaml"
_ROBOT_CONFIG_ROOT = _REPO_ROOT / "configs" / "robots" / "legged"


@dataclass
class LeggedRobotCfg:
    """configs/robots/legged/{category}/{robot_id}.yaml 한 장에 대응하는 로봇별 RL 파라미터."""

    usd_path: str
    morphology_group: str
    base_body_name: str
    foot_body_names: str
    action_scale: float = 0.5
    undesired_contact_body_names: str | None = None
    default_joint_pos: dict[str, float] = field(default_factory=dict)
    # USD에 authored된 PD 게인은 URDF->USD 변환기가 넣은 "정지 유지용" 기본값이라 RL 학습에 못 쓴다
    # (관측됨: unitree_go2 - stiffness 1e7, damping 1e5, 공식 값(25, 0.5)의 40만 배) - 그래서
    # ImplicitActuatorCfg에 usd 값을 그대로 안 쓰고, usd_export_config.py가 관절 effort 한계에서
    # 계산한 값을 쓴다. 여기 25.0/0.5는 그 계산이 없을 때(구버전 config)의 안전한 기본값이다.
    actuator_stiffness: float = 25.0
    actuator_damping: float = 0.5


def load_legged_robot_cfg(category: str, robot_id: str) -> LeggedRobotCfg:
    """configs/robots/legged/{category}/{robot_id}.yaml을 읽어 LeggedRobotCfg로 변환한다."""
    yaml_path = _ROBOT_CONFIG_ROOT / category / f"{robot_id}.yaml"
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return LeggedRobotCfg(
        usd_path=raw["usd_path"],
        morphology_group=raw["morphology_group"],
        base_body_name=raw["base_body_name"],
        foot_body_names=raw["foot_body_names"],
        action_scale=raw.get("action_scale", 0.5),
        undesired_contact_body_names=raw.get("undesired_contact_body_names"),
        default_joint_pos=raw.get("default_joint_pos") or {},
        actuator_stiffness=raw.get("actuator_stiffness", 25.0),
        actuator_damping=raw.get("actuator_damping", 0.5),
    )


def _build_joint_pos_cfg(default_joint_pos: dict[str, float]) -> dict[str, float]:
    """0.0을 기본값으로 하되, 리밋을 벗어나 override가 필요한 관절만 정확한 이름으로 덮어쓴다.

    와일드카드(".*")와 정확한 관절 이름을 같은 딕셔너리에 같이 쓰면 "패턴 두 개에 매칭" 에러가
    나므로, override한 이름을 제외한 나머지에만 적용되는 부정 전방탐색 정규식을 와일드카드 대신
    쓴다(scripts/sim/env/robot_spawn.py의 spawn_robot_safely와 동일한 패턴).
    """
    if not default_joint_pos:
        return {".*": 0.0}
    excluded = "|".join(re.escape(name) for name in default_joint_pos)
    return {f"^(?!({excluded})$).*": 0.0, **default_joint_pos}


@configclass
class LocoSceneCfg(InteractiveSceneCfg):
    """지형 - 로봇 - height scanner - contact sensor로 구성된 legged RL 씬."""

    terrain = build_rl_terrain_importer_cfg(_SIM_RL_CONFIG_PATH)
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
    """목표 이동 속도 명령 - env/rl.py의 지형 커리큘럼이 이 명령 대비 이동 거리로 난이도를 조정한다."""

    base_velocity = core_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=core_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """모든 관절을 위치 목표로 구동 - 스케일은 로봇 config의 action_scale로 build_loco_rl_env_cfg()가 지정."""

    joint_pos = core_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """정책 관측 - 고유 감각(proprioception) + height scan(로컬 지형 굴곡)."""

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
class EventCfg:
    """도메인 랜덤화 - sim-to-real 격차를 줄이기 위한 마찰·질량·초기 상태·외란 이벤트."""

    physics_material = EventTerm(
        func=core_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=core_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),  # build_loco_rl_env_cfg()에서 실제 이름으로 교체
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=core_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=core_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=core_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class TerminationsCfg:
    """시간 초과 종료와, 몸통이 지면에 닿는 낙상 종료."""

    time_out = DoneTerm(func=core_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=core_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),  # build_loco_rl_env_cfg()에서 교체
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """지형 난이도 승급·강등 - 판단 로직은 scripts/sim/env/rl.py에 둔다(씬 레벨 동작이라는 이유)."""

    terrain_levels = CurrTerm(func=promote_terrain_levels_by_travel_distance)


@configclass
class LocoRLEnvCfg(ManagerBasedRLEnvCfg):
    """legged 로봇 보행 정책 학습용 환경 설정."""

    scene: LocoSceneCfg = LocoSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: QuadrupedRewardsCfg = QuadrupedRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """공용 시뮬레이션·센서 주기 설정 - 로봇별 세부값은 build_loco_rl_env_cfg()가 덮어쓴다."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # num_envs가 수천 개인 rough-terrain legged 학습은 GPU 접촉 패치 수가 기본 버퍼 크기를 쉽게
        # 넘는다 - Isaac Lab 공식 legged locomotion 레퍼런스(velocity_env_cfg.py)가 쓰는 값을 그대로
        # 따른다("Patch buffer overflow" PhysX 에러 방지)
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt

        # 커리큘럼 항목이 실제로 설정돼 있을 때만 지형 생성기의 난이도 오름차순 배치를 켠다
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            self.scene.terrain.terrain_generator.curriculum = True
        else:
            self.scene.terrain.terrain_generator.curriculum = False


def build_loco_rl_env_cfg(
    category: str, robot_id: str, num_envs: int = 4096, flat_terrain: bool = False
) -> LocoRLEnvCfg:
    """robot_id 하나에 대해 완전히 조립된 LocoRLEnvCfg를 만든다 - tools/03_controller_rl.py의 유일한 진입점.

    관절 PD 게인은 USD authored 값을 쓰지 않는다 - URDF->USD 변환기가 넣은 값은 정지 자세를 딱딱하게
    붙잡아두기 위한 것이라 RL 학습에 맞지 않을 정도로 크다(관측됨: unitree_go2 stiffness 1e7). 대신
    usd_export_config.py가 관절 effort 한계에서 계산해 config에 저장해 둔 값(robot_cfg의
    actuator_stiffness/damping)을 쓴다.

    flat_terrain=True면 계단·경사 커리큘럼 지형 대신 평지로 바꾼다 - tools/04_controller_test.py의
    기본 경로 추종 테스트(ㄱ자 웨이포인트)는 "정책이 이동 명령을 따라가는가"만 wheeled와 동일한
    조건(평지)으로 확인하는 게 목적이고, 지형 난이도 자체는 학습 커리큘럼에서 이미 검증되므로
    여기서 또 볼 필요가 없다.
    """
    robot_cfg = load_legged_robot_cfg(category, robot_id)
    usd_path = _USD_ROOT / category / robot_cfg.usd_path

    # usd에 저장된 기본 자세 기준 지면 여유 높이 - 안 띄우면 로봇이 지형 표면과 겹친 채로 스폰돼
    # 첫 physics 스텝에서 PhysX가 관통을 강제로 밀어내며 폭발적인 속도가 나온다(관측됨: 스폰 직후
    # ang_vel/lin_vel_z 보상이 물리적으로 불가능한 크기로 튀고 거의 즉시 base_contact 종료됨)
    spawn_height = ground_clearance(usd_path)

    # 공용 골격 위에 이 로봇의 USD·본체 이름·발 이름을 채워 넣는다
    env_cfg = LocoRLEnvCfg()
    env_cfg.scene.num_envs = num_envs
    # 다리 관절이 서로 스치는 접촉을 물리적으로 정확히 풀어내려면 velocity iteration이 더 필요하다
    # (관측됨: 기본값으로 두면 "more than 4 velocity iterations" TGS 경고가 뜸) - Isaac Lab 공식
    # 사족보행/휴머노이드 에셋(A1·Go2·ANYmal·H1·G1)이 전부 쓰는 값을 그대로 따른다
    solver_velocity_iterations = 4 if robot_cfg.morphology_group == "humanoid" else 0
    env_cfg.scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        # activate_contact_sensors=True 필수 - scene.contact_forces(ContactSensor)가 이 로봇의
        # 몸체에서 접촉을 읽으려면 스폰 시점에 PhysX 접촉 리포터가 켜져 있어야 한다
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=solver_velocity_iterations,
            ),
        ),
        # 관절 기본 자세는 0을 기본으로 쓰되, 그게 리밋을 벗어나는 관절(예: 무릎)만 config의
        # default_joint_pos(usd_export_config.py가 usd 리밋에서 미리 계산해 둔 값)로 덮어쓴다 -
        # 안 그러면 ArticulationCfg 검증 단계에서 "기본 자세가 리밋 밖" 예외가 바로 난다
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, spawn_height), joint_pos=_build_joint_pos_cfg(robot_cfg.default_joint_pos)
        ),
        # 리밋 끝까지 밀어붙이면 PPO가 물리적 하드 리밋에 부딪혀 불안정해지므로 10% 여유를 둔다
        # (A1/Go2/H1/G1 공식 설정과 동일)
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=robot_cfg.actuator_stiffness,
                damping=robot_cfg.actuator_damping,
            )
        },
    )
    env_cfg.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{robot_cfg.base_body_name}"
    env_cfg.actions.joint_pos.scale = robot_cfg.action_scale
    env_cfg.events.add_base_mass.params["asset_cfg"].body_names = robot_cfg.base_body_name
    env_cfg.terminations.base_contact.params["sensor_cfg"].body_names = robot_cfg.base_body_name

    if flat_terrain:
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )
        # 평지는 난이도 단계가 없어 지형 커리큘럼 자체가 의미 없다 - 켜둔 채로 두면 이 항목이
        # 참조하는 terrain_generator가 None이라 그대로 에러가 난다
        env_cfg.curriculum.terrain_levels = None

    # 형태 그룹(quadruped/humanoid)에 맞는 보상 세트를 고르고, 로봇별 발/접촉 링크 이름을 채운다
    if robot_cfg.morphology_group == "humanoid":
        env_cfg.rewards = HumanoidRewardsCfg()
        env_cfg.rewards.feet_air_time.params["sensor_cfg"].body_names = robot_cfg.foot_body_names
        env_cfg.rewards.feet_slide.params["sensor_cfg"].body_names = robot_cfg.foot_body_names
        env_cfg.rewards.feet_slide.params["asset_cfg"].body_names = robot_cfg.foot_body_names
    else:
        env_cfg.rewards = QuadrupedRewardsCfg()
        env_cfg.rewards.feet_air_time.params["sensor_cfg"].body_names = robot_cfg.foot_body_names
        if robot_cfg.undesired_contact_body_names:
            env_cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names = (
                robot_cfg.undesired_contact_body_names
            )
        else:
            env_cfg.rewards.undesired_contacts = None

    return env_cfg

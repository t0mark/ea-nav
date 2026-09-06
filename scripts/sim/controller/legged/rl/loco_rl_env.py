"""legged 로봇 RL 보행 학습 환경(ManagerBasedRLEnvCfg) 조립.

로봇마다 별도 파일을 두는 대신, configs/robots/legged/{category}/{robot_id}.yaml 하나로 로봇을 특정해
build_loco_rl_env_cfg()가 환경 설정을 조립한다. 씬 골격(지형·센서)·관측·명령·커리큘럼은 34종 전부
공용이고, 관절 구동 모델(actuator)·액션 구성(action)·보상(rewards)·종료조건(termination)·도메인
랜덤화(domain_randomization)만 로봇 yaml이 직접 선언한 축 값대로 scripts/sim/controller/legged/rl/
{actuators,actions,rewards,terminations,domain_randomization}/ 레지스트리에서 조립한다 - 공개 RL
리포지토리들을 전수 대조한 결과, 이 5개 축이 로봇마다 실제로 갈리는 지점이었고 반대로 씬·관측·명령·
커리큘럼은 로봇 타입과 무관하게 전부 동일했다.
"""

from __future__ import annotations

import math
import re
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from scripts.sim.controller.legged.rl import actions as action_axis
from scripts.sim.controller.legged.rl import actuators as actuator_axis
from scripts.sim.controller.legged.rl import domain_randomization as domain_randomization_axis
from scripts.sim.controller.legged.rl import rewards as reward_axis
from scripts.sim.controller.legged.rl import terminations as termination_axis
from scripts.sim.controller.legged.rl.robot_profile import RobotProfile
from scripts.sim.env.robot_spawn import ground_clearance
from scripts.sim.env.rl import build_rl_terrain_importer_cfg, promote_terrain_levels_by_travel_distance

_REPO_ROOT = Path(__file__).resolve().parents[5]
_USD_ROOT = _REPO_ROOT / "data" / "sim" / "usd" / "real_robot" / "legged"
_SIM_RL_CONFIG_PATH = _REPO_ROOT / "configs" / "sim_rl.yaml"


def _build_joint_pos_cfg(default_joint_pos: dict[str, float], is_complete: bool) -> dict[str, float]:
    """관절 기본 자세 딕셔너리를 만든다. default_joint_pos가 어떤 의미인지는 robot yaml이 미리
    선언해 둔 사실이라(RobotProfile.default_joint_pos_complete), 여기서는 그 선언을 그대로 따를
    뿐 usd를 다시 열어 추론하지 않는다 - usd 구조 분석은 usd_export_config.py의 책임이다.

    - is_complete=False(기본값): default_joint_pos는 "0이 리밋을 벗어나는 관절만" 담은 override
      목록이다(usd_export_config.py가 usd 리밋에서 자동 계산). 이 경우 나머지 관절은 전부 0.0이어야
      하므로, override한 이름을 제외한 나머지에만 적용되는 부정 전방탐색 정규식을 와일드카드로 쓴다
      (와일드카드 ".*"와 정확한 이름을 같은 딕셔너리에 같이 쓰면 "패턴 두 개에 매칭" 에러가 나서
      단순 와일드카드는 못 쓴다).
    - is_complete=True: default_joint_pos가 이미 로봇의 관절 전체에 대한 기본 자세다(예: go2/go1 -
      Isaac Lab 공식 UNITREE_GO2_CFG처럼 12관절 전부를 직접 나열). 이 경우 위 와일드카드를 추가하면
      매칭할 관절이 하나도 안 남아 Isaac Lab이 "패턴이 아무 것도 매칭 안 함" 에러를 던지므로,
      override 목록을 그대로 쓴다.
    """
    if not default_joint_pos:
        return {".*": 0.0}
    if is_complete:
        return dict(default_joint_pos)
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
class CurriculumCfg:
    """지형 난이도 승급·강등 - 판단 로직은 scripts/sim/env/rl.py에 둔다(씬 레벨 동작이라는 이유)."""

    terrain_levels = CurrTerm(func=promote_terrain_levels_by_travel_distance)


@configclass
class LocoRLEnvCfg(ManagerBasedRLEnvCfg):
    """legged 로봇 보행 정책 학습용 환경 설정.

    actions/rewards/terminations/events는 로봇마다 타입 자체가 달라(선택한 축에 따라 필드 구성이
    다름) 여기서 고정 기본값을 못 둔다 - MISSING으로 선언만 해두고 build_loco_rl_env_cfg()가 반드시
    채운다(scene.robot과 동일한 패턴).
    """

    scene: LocoSceneCfg = LocoSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: object = MISSING
    commands: CommandsCfg = CommandsCfg()
    rewards: object = MISSING
    terminations: object = MISSING
    events: object = MISSING
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

    관절 구동 모델·액션 구성·보상·종료조건·도메인 랜덤화는 RobotProfile이 선언한 축 값대로 각 레지스트리
    (actuator_axis/action_axis/reward_axis/termination_axis/domain_randomization_axis)에서 조립한다.

    flat_terrain=True면 계단·경사 커리큘럼 지형 대신 평지로 바꾼다 - 이건 학습 시점에 로봇마다 고르는
    축이 아니라, tools/04_controller_test.py가 학습이 끝난 정책의 구동만 확인할 때 쓰는 테스트 전용
    스위치다(wheeled와 동일한 조건에서 "정책이 이동 명령을 따라가는가"만 보는 게 목적이고, 지형 난이도
    자체는 학습 커리큘럼에서 이미 검증되므로 테스트에서 또 볼 필요가 없다). 학습 지형은 로봇 종류와
    무관하게 항상 커리큘럼 rough-terrain 하나로 통일한다 - 이 프로젝트의 목표 자체가 "모든 로봇이
    동일한 커리큘럼 지형을 통과하는 능력"을 학습시키는 것이라, 로봇마다 학습 지형을 다르게 가져갈
    이유가 없다.
    """
    profile = RobotProfile.load(category, robot_id)
    usd_path = _USD_ROOT / category / profile.usd_path

    # usd에 저장된 기본 자세 기준 지면 여유 높이 - 안 띄우면 로봇이 지형 표면과 겹친 채로 스폰돼
    # 첫 physics 스텝에서 PhysX가 관통을 강제로 밀어내며 폭발적인 속도가 나온다(관측됨: 스폰 직후
    # ang_vel/lin_vel_z 보상이 물리적으로 불가능한 크기로 튀고 거의 즉시 base_contact 종료됨)
    spawn_height = ground_clearance(usd_path)

    env_cfg = LocoRLEnvCfg()
    env_cfg.scene.num_envs = num_envs

    # 다리 관절이 서로 스치는 접촉을 물리적으로 정확히 풀어내려면 velocity iteration이 더 필요하다
    # (관측됨: 기본값으로 두면 "more than 4 velocity iterations" TGS 경고가 뜸) - Isaac Lab 공식
    # 사족보행/휴머노이드 에셋(A1·Go2·ANYmal·H1·G1)이 전부 쓰는 값을 그대로 따른다. 이 값은 형태(다리·팔이
    # 몸통 가까이서 부딪히는 구조인지)에 관한 순수 물리 튜닝값이라, RL 설정 축이 아니라 USD가 놓인
    # 폴더(category)로 그대로 판단한다.
    solver_velocity_iterations = 4 if category == "humanoid" else 0
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
        # default_joint_pos(usd_export_config.py가 usd 리밋에서 미리 계산해 둔 값)로 덮어쓴다
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, spawn_height),
            joint_pos=_build_joint_pos_cfg(profile.default_joint_pos, profile.default_joint_pos_complete),
        ),
        # 리밋 끝까지 밀어붙이면 PPO가 물리적 하드 리밋에 부딪혀 불안정해지므로 10% 여유를 둔다
        soft_joint_pos_limit_factor=0.9,
        actuators=actuator_axis.build(profile.actuator),
    )
    env_cfg.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{profile.base_body_name}"

    # 액션·보상·종료조건·도메인 랜덤화는 로봇 yaml이 고른 축 값대로 각 레지스트리가 조립한다
    env_cfg.actions = action_axis.build(profile.action)
    env_cfg.rewards = reward_axis.build(profile.rewards, profile)
    env_cfg.terminations = termination_axis.build(profile.termination, profile)
    env_cfg.terminations.time_out = DoneTerm(func=core_mdp.time_out, time_out=True)
    env_cfg.events = domain_randomization_axis.build(profile.domain_randomization, profile)

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

    return env_cfg

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

import torch

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
from scripts.sim.env.curriculum.stage_env import build_flat_terrain_importer_cfg, stage_command_speed_limits
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

# 구동 확인용 지형의 서브지형 이름 - 열 인덱스는 test_terrain_column()으로 얻는다.
TEST_TERRAIN_FLAT = "flat"
# 오르막 계단 - 공식 pyramid_stairs_inv(MeshInvertedPyramidStairsTerrainCfg)는 타일 중심이 구덩이
# 바닥이라(origin_z = -(num_steps+1)*step_height) 밖으로 나가는 모든 방향이 올라가는 계단이 된다.
TEST_TERRAIN_STAIRS_UP = "pyramid_stairs_inv"
# 난이도 행 수 - curriculum=True에서 행 인덱스가 곧 난이도이고, difficulty=(row+eta)/num_rows가
# 서브지형 파라미터 구간에 선형 보간된다. 계단 타일에서 어느 행을 쓸지는 로봇이 깬 단차에서
# test_terrain_stairs_up_row()가 역산한다.
TEST_TERRAIN_ROWS = 5
# 시드를 고정하지 않으면(ROUGH_TERRAINS_CFG의 기본값은 None) 실행마다 지형이 달라져 로봇 간
# 비교가 성립하지 않는다.
_TEST_TERRAIN_SEED = 42


def _test_terrain_sub_terrains() -> dict:
    """평지 + Isaac Lab 공식 험지 종류를 균등 비율로 묶는다.

    TerrainGenerator가 열을 비율 누적합으로 가르므로(_generate_curriculum_terrains), 비율이 모두
    같고 열 수가 종류 수와 같으면 열 c가 곧 c번째 종류가 된다 - 어느 열이 무슨 지형인지 별도
    계산 없이 정해진다.
    """
    from isaaclab.terrains import MeshPlaneTerrainCfg
    from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

    merged = {TEST_TERRAIN_FLAT: MeshPlaneTerrainCfg(), **ROUGH_TERRAINS_CFG.sub_terrains}
    proportion = 1.0 / len(merged)
    return {name: cfg.replace(proportion=proportion) for name, cfg in merged.items()}


def test_terrain_column(name: str) -> int:
    """구동 확인용 지형에서 서브지형 이름에 해당하는 열 인덱스."""
    names = list(_test_terrain_sub_terrains())
    if name not in names:
        raise ValueError(f"구동 확인용 지형에 없는 서브지형: {name} (가능: {names})")
    return names.index(name)


def test_terrain_stairs_up_row(cleared_step_height_m: float) -> int:
    """로봇이 깬 단차 높이를 오르막 계단 지형의 난이도 행으로 바꾼다.

    pyramid_stairs 계열은 한 계단의 높이를 step_height_range에서 difficulty로 선형 보간해 그대로
    쓰므로(mesh_terrains.py의 step_height 계산), 목표로 삼을 계단 높이가 곧 그 로봇이 깬 단차다.

    행 인덱스로 바꾸는 식은 curriculum=True의 배치 규약에서 온다: 행 r의 difficulty가
    [r/num_rows, (r+1)/num_rows] 구간이므로(terrain_generator.py의 _generate_curriculum_terrains),
    목표 difficulty에 num_rows를 곱해 내림하면 그 구간을 담는 행이 나온다. 지형이 낼 수 있는 가장
    높은 계단보다 더 높은 단차를 깬 로봇은 마지막 행으로 잘린다.
    """
    stairs_cfg = _test_terrain_sub_terrains()[TEST_TERRAIN_STAIRS_UP]
    lowest_step, highest_step = stairs_cfg.step_height_range
    difficulty = (cleared_step_height_m - lowest_step) / (highest_step - lowest_step)
    return min(max(math.floor(difficulty * TEST_TERRAIN_ROWS), 0), TEST_TERRAIN_ROWS - 1)


def build_test_terrain_importer_cfg() -> TerrainImporterCfg:
    """구동 확인용 지형 - 평지 타일과 랜덤 험지 타일을 한 번에 만들어 둔다.

    구동 확인은 "평지에서 경로 추종"과 "임의 험지에서 보행"을 둘 다 봐야 하는데, 한 프로세스에서
    ManagerBasedRLEnv를 두 번 만들 수 없다(env.close()가 USD prim을 남겨 두 번째 생성이 멈춘다).
    그래서 지형을 하나만 만들고 그 안에서 로봇을 타일 사이로 옮긴다 - Isaac Lab이 커리큘럼 승급을
    처리하는 방식과 같다(TerrainImporter.update_env_origins).
    """
    from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

    sub_terrains = _test_terrain_sub_terrains()
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(
            seed=_TEST_TERRAIN_SEED,
            curriculum=True,
            num_rows=TEST_TERRAIN_ROWS,
            num_cols=len(sub_terrains),
            sub_terrains=sub_terrains,
            # 높이로 색을 입힌다 - 기본값(none)은 전부 같은 회색이라 영상에서 단차가 안 보인다
            color_scheme="height",
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )


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


class LevelScaledVelocityCommand(core_mdp.UniformVelocityCommand):
    """env마다 다른 속도 상한을 갖는 목표 속도 명령.

    지형이 험해질수록 같은 속도를 유지하는 것이 물리적으로 불가능해지므로 명령 속도 상한을 난이도에
    맞춰 낮춰야 하는데, 커리큘럼의 난이도는 env마다 다르다(축도 레벨도 다르다). 상한을 전 env 공통으로
    두면 한 축의 승급이 다른 축의 과제까지 바꿔, 그 축의 점수 이력이 서로 다른 난이도에서 얻은 점수의
    혼합이 된다.

    cfg.ranges는 레벨 0의 기준 상한으로 두고, 커리큘럼이 써 넣은 env별 배율을 샘플에 곱해 실제 상한을
    만든다. 균등분포에서 뽑아 배율을 곱한 것은 그 배율만큼 좁힌 구간에서 뽑은 것과 같다.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._speed_scale = torch.ones(self.num_envs, device=self.device)

    @property
    def speed_scale(self) -> torch.Tensor:
        """env별 명령 속도 배율 - 커리큘럼이 env의 현재 레벨에 맞는 값을 직접 써 넣는다."""
        return self._speed_scale

    def _resample_command(self, env_ids) -> None:
        """기준 상한에서 뽑은 명령에 env별 배율을 곱한다."""
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids] *= self._speed_scale[env_ids].unsqueeze(1)

    def _update_command(self) -> None:
        """heading 제어가 매 스텝 다시 채우는 각속도까지 env별 상한으로 자른다.

        상위 구현은 각속도를 heading 오차에서 계산해 cfg.ranges.ang_vel_z(기준 상한)로만 자르므로,
        배율을 곱해 둔 값이 매 스텝 덮어써진다. 정지 env는 상위 구현이 0으로 두는데, 0을 자르면
        그대로 0이라 그 처리를 깨지 않는다.
        """
        super()._update_command()
        ang_vel_limit = self._speed_scale * self.cfg.ranges.ang_vel_z[1]
        self.vel_command_b[:, 2] = self.vel_command_b[:, 2].clamp(min=-ang_vel_limit, max=ang_vel_limit)


@configclass
class LevelScaledVelocityCommandCfg(core_mdp.UniformVelocityCommandCfg):
    """LevelScaledVelocityCommand를 쓰는 속도 명령 설정."""

    class_type: type = LevelScaledVelocityCommand


@configclass
class CommandsCfg:
    """목표 이동 속도 명령 - 전 로봇 동일. 속도 상한은 env의 난이도 레벨에 따라 런타임에 좁혀진다."""

    base_velocity = LevelScaledVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=LevelScaledVelocityCommandCfg.Ranges(
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
    # 난이도 승급은 rl/rl_trainer.py의 CurriculumTrainer가 지형의 terrain_levels를 바꿔서 한다
    # (env cfg 안에 승급 term을 두지 않는다 - 축별로 따로 올려야 해서 공식 term으로는 표현이 안 된다).
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
        # GPU 접촉 버퍼 두 종류. 넘치면 PhysX가 초과분 접촉을 조용히 버려서("Contacts have been
        # dropped") 발이 지면을 뚫고 접촉 센서도 못 읽으므로, 그 위에서 계산한 보상은 의미가 없다.
        # 기본값은 env가 여러 난이도 행에 흩어진 공식 태스크 기준이라, 4096대가 학습 초반에 삼각형
        # 메시 지형 위로 널브러지는 이 프로젝트에는 모자란다(PhysX가 필요한 크기를 에러로 알려 준다).
        # 몸집이 큰 4족일수록 접촉면이 넓어 더 크게 잡아야 한다.
        self.sim.physx.gpu_max_rigid_patch_count = 2**24
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt


def build_loco_rl_env_cfg(
    robot_id: str,
    num_envs: int | None = None,
    preset: str | None = None,
    terrain_cfg: TerrainImporterCfg | None = None,
    command_level: int = 0,
) -> LocoRLEnvCfg:
    """robot_id에 대해 완전히 조립된 LocoRLEnvCfg를 만든다.

    지형은 호출부가 정한다 - 학습은 전 난이도를 담은 커리큘럼 지형을, 구동 확인은 평면이나 검증용
    타일 격자를 넘긴다. terrain_cfg가 None이면 끝없는 평면이다. 지형을 인자로 받는 이유는 난이도가
    env cfg의 성질이 아니라 커리큘럼의 성질이기 때문이다 - 승급은 지형을 다시 만드는 게 아니라
    이미 만들어 둔 지형 안에서 env를 옮겨서 한다.
    command_level은 명령 속도 상한을 낮추는 데만 쓴다(지형이 험할수록 같은 속도가 불가능해진다).
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
    env_cfg.scene.terrain = terrain_cfg if terrain_cfg is not None else build_flat_terrain_importer_cfg()
    env_cfg.sim.physics_material = env_cfg.scene.terrain.physics_material

    # 지형이 험해질수록 명령 속도 상한을 낮춘다 - 각속도도 같은 비율로 줄여 두 추종 지표의 난이도를 맞춘다
    lin_vel_limit, speed_scale = stage_command_speed_limits(command_level, preset_cfg.curriculum["command"])
    ang_vel_limit = MAX_TRAINED_ANG_VEL_Z * speed_scale
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-lin_vel_limit, lin_vel_limit)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-lin_vel_limit, lin_vel_limit)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-ang_vel_limit, ang_vel_limit)

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

    env_cfg.observations = observations.build(profile, preset_cfg.observations)
    env_cfg.actions = action_interface.build(profile, preset_cfg.action)
    env_cfg.rewards = rewards.build(profile, preset_cfg.rewards)
    env_cfg.terminations = termination.build(profile, preset_cfg.terminations)
    env_cfg.terminations.time_out = DoneTerm(func=core_mdp.time_out, time_out=True)
    env_cfg.events = event.build(profile, preset_cfg.events)

    return env_cfg

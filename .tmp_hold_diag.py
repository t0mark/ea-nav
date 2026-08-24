"""zero-action 기립 홀드 계측: 어떤 관절이 왜 무너지는지 (오차·토크·속도 시계열).

02_train_legged.py와 동일한 로봇 cfg 구성으로 평지 4 env를 만들어 200스텝 홀드,
10스텝마다 base 피치와 관절 그룹별 |오차|·적용 토크·속도를 출력한다.
"""
import json
from pathlib import Path

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (  # noqa: E402
    UnitreeGo2RoughEnvCfg,
)
import sys  # noqa: E402
sys.path.insert(0, "/workspace/eatrav")
from scripts.sim.utils.robot_spawn import _dcmotor_actuators  # noqa: E402

import sys as _sys
USD = Path("/workspace/eatrav/check/01_sim/usd/quad/" + (_sys.argv[1] if len(_sys.argv) > 1 else "quad_0000"))
meta = json.loads((USD / "meta.json").read_text())
joints = json.loads((USD / "joints.json").read_text())["joints"]
st = meta["params"].get("stance_torque", {})
kp = {j["name"]: (5.0 * st[j["name"]] if j["name"] in st else 30.0) for j in joints}
gains = {"stiffness": kp, "damping": {n: 0.025 * v for n, v in kp.items()}}

robot = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD / "robot.usd"), activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False, linear_damping=0.0,
            angular_damping=0.0, max_linear_velocity=1000.0, max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4,
            solver_velocity_iteration_count=0)),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, meta["metrics"]["base_height"] + 0.04),
        joint_pos=dict(meta["standing_pose"]), joint_vel={".*": 0.0}),
    soft_joint_pos_limit_factor=0.9,
    actuators=_dcmotor_actuators(joints, gains, armature=0.0),
)

cfg = UnitreeGo2RoughEnvCfg()
cfg.scene.num_envs = 4
cfg.scene.robot = robot
cfg.scene.terrain.terrain_type = "plane"
cfg.scene.terrain.terrain_generator = None
# 평지에는 지형 커리큘럼 항이 성립하지 않음 (terrain_generator 참조)
cfg.curriculum.terrain_levels = None
BASE = "base" if "go2" in str(USD) else "base_link"
cfg.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + BASE
cfg.terminations.base_contact.params["sensor_cfg"].body_names = BASE
cfg.events.add_base_mass.params["asset_cfg"].body_names = BASE
cfg.events.base_external_force_torque.params["asset_cfg"].body_names = BASE
cfg.rewards.feet_air_time.params["sensor_cfg"].body_names = [f"^{n}$" for n in meta["contact_links"]]
# 리셋 랜덤화 제거 (순수 홀드 관찰)
cfg.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}
cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

env = ManagerBasedRLEnv(cfg=cfg)
obs, _ = env.reset()
art = env.unwrapped.scene["robot"]
names = art.joint_names
default = art.data.default_joint_pos
act = torch.zeros((4, env.unwrapped.action_manager.total_action_dim), device=env.unwrapped.device)

# 관절 그룹: 이름 역할별로 묶어 평균 지표 출력
groups = {}
for i, n in enumerate(names):
    role = n.split("_", 1)[1]
    groups.setdefault(role, []).append(i)

# 지지 기하: 발(접촉 링크)·루트의 월드 좌표로 무게중심-지지 다각형 관계 확인
body_names = art.body_names
foot_idx = [body_names.index(n) for n in meta["contact_links"]]
masses = art.root_physx_view.get_masses().to(env.unwrapped.device)


def geom_report(tag):
    """발 x·z, 루트 x·z, 질량 가중 무게중심 x를 출력한다 (env 0)."""
    bp = art.data.body_pos_w[0]
    com_x = float((bp[:, 0] * masses[0]).sum() / masses[0].sum())
    feet = ", ".join(f"{body_names[i].split('_')[0]}({bp[i,0]:.2f},{bp[i,2]:.2f})" for i in foot_idx)
    print(f"  [{tag}] root=({bp[0,0]:.2f},{bp[0,2]:.2f}) com_x={com_x:.3f} feet={feet}", flush=True)


print("step | pitch(deg) | role: err(rad)/vel(rad/s)/tau(Nm)", flush=True)
geom_report("reset")
for t in range(200):
    obs, _, term, trunc, _ = env.step(act)
    if t in (0, 40, 199):
        geom_report(f"t={t}")
    if t % 20 == 0 or t == 199:
        q = art.data.joint_pos - default
        v = art.data.joint_vel
        tau = art.data.applied_torque
        quat = art.data.root_quat_w[0]
        # 피치 근사: 몸체 z축의 기울기
        import isaaclab.utils.math as math_utils
        g_b = math_utils.quat_apply_inverse(art.data.root_quat_w, torch.tensor([[0.0, 0.0, -1.0]], device=q.device).repeat(4, 1))
        pitch = torch.rad2deg(torch.atan2(g_b[0, 0], -g_b[0, 2]))
        parts = []
        for role, idx in sorted(groups.items()):
            parts.append(f"{role}: {q[0, idx].abs().mean():.2f}/{v[0, idx].abs().mean():.1f}/{tau[0, idx].abs().mean():.0f}")
        print(f"{t:4d} | {pitch:6.1f} | " + " | ".join(parts), flush=True)
    if term.any() or trunc.any():
        print(f"  종료 발생 t={t}", flush=True)

env.close()
app.close()

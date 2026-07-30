"""클래스별 형태 템플릿(위상 문법) + 파라미터 역산.

- 위상은 문법으로 고정, 무작위성은 이산 선택지(다리 수·구동 타입·DOF 구성·센서 구성)와
  연속 파라미터에만 부여한다.
- 목표 외곽 치수 target=(w,l,h)를 먼저 받고 템플릿 파라미터를 역산한다.
- rest pose(관절 0) = 스탠딩 자세. 굽힘 자세는 joint origin rpy에 반영.
- 다관절(2~3 DOF) 관절은 소형 sphere 연결구(ball, connector=True) 체인으로 표현한다.
- 좌우 대칭: 파라미터 공유, y 부호 반전. roll 리밋은 안쪽 작게/바깥쪽 크게(측면 반전).

builder 시그니처: build(rng, target) -> (Robot, meta)
meta: form, base_z(스탠딩 시 base 링크 원점 높이), params, sensors, notes
"""

import math

from common import Robot, Geom, attach_sensor

PI = math.pi


class BuildError(Exception):
    """치수 역산 불능 — 호출부에서 재샘플."""


def _u(rng, a, b):
    return a + (b - a) * rng.random()


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ============================================================ 휴머노이드

def build_humanoid(rng, target):
    w, l, h = target
    p = {}

    # 높이 배분 (지터 후 정규화 → 합 = h 보장)
    fr = {"foot": 0.035, "leg": 0.50, "pelvis": 0.08,
          "chest": 0.24, "neck": 0.025, "head": 0.12}
    fr = {k: v * _u(rng, 0.85, 1.15) for k, v in fr.items()}
    s = sum(fr.values())
    foot_h, leg_h, ph, ch, neck_h, head_h = (fr[k] / s * h for k in
                                             ("foot", "leg", "pelvis", "chest", "neck", "head"))

    torso_d = _clamp(l * _u(rng, 0.82, 0.95), 0.05, l - 0.025)
    leg_r = _clamp(0.028 * h, 0.012, min(0.055, torso_d * 0.45))
    arm_r = _clamp(min(0.8 * leg_r, (w - 0.08) / 4 / 1.15), 0.008, 0.05)
    r_ba = 1.15 * arm_r                      # 어깨 연결구 반지름
    torso_w = _clamp(w - 0.006 - 4 * r_ba, 0.07, w)
    sh_y = torso_w / 2 + 0.003 + r_ba        # 어깨(팔) y 위치

    nb_hip = 2 if rng.random() < 0.4 else 1  # 1: roll+pitch, 2: yaw+roll+pitch
    r_bl = 1.15 * leg_r
    leg_budget = leg_h - 2 * r_bl * nb_hip
    if leg_budget < 0.05:
        raise BuildError("leg budget")
    thigh = leg_budget * 0.52
    shin = leg_budget * 0.48

    # 팔을 위로 들었을 때(shoulder pitch) 머리와 간섭하지 않도록 어깨 안쪽으로 제한
    head_r = min(head_h / 2, 0.45 * l, 0.45 * w, sh_y - arm_r - 0.006)
    if head_r < 0.02:
        raise BuildError("head radius")
    neck_h += head_h - 2 * head_r            # 머리 축소분은 목으로 이월(총높이 유지)
    hip_y = _clamp(0.30 * torso_w, r_bl + 0.002, torso_w / 2 - leg_r * 0.4)

    base_z = foot_h + shin + thigh + 2 * r_bl * nb_hip + ph / 2
    r = Robot("humanoid")
    dens = lambda: _u(rng, 300, 900)

    # 몸통: pelvis(base) + chest, waist는 fixed/yaw/pitch
    r.add_link("base", Geom("box", (torso_d, torso_w, ph)), dens())
    r.add_link("torso_chest", Geom("box", (torso_d, torso_w, ch)),
               dens(), origin=(0, 0, ch / 2))
    waist = rng.choice(["fixed", "yaw", "pitch"])
    if waist == "fixed":
        r.add_joint("fixed", "base", "torso_chest", (0, 0, ph / 2))
    else:
        ax = (0, 0, 1) if waist == "yaw" else (0, 1, 0)
        r.add_joint("revolute", "base", "torso_chest", (0, 0, ph / 2),
                    axis=ax, lower=-0.35, upper=0.35)

    # 목 + 머리
    r.add_link("neck", Geom("cylinder", (max(0.012, 0.3 * head_r), neck_h)),
               dens(), origin=(0, 0, neck_h / 2))
    if rng.random() < 0.5:
        r.add_joint("revolute", "torso_chest", "neck", (0, 0, ch),
                    axis=(0, 0, 1), lower=-1.0, upper=1.0)
    else:
        r.add_joint("fixed", "torso_chest", "neck", (0, 0, ch))
    r.add_link("head", Geom("sphere", (head_r,)), dens())
    r.add_joint("fixed", "neck", "head", (0, 0, neck_h + head_r))

    fl = _clamp(0.15 * h * _u(rng, 0.85, 1.15), 0.06,
                min(0.95 * l, (0.5 * l + 0.02) / 0.7))  # 발끝이 목표 l 초과 금지
    fw = min(2.2 * leg_r, 2 * hip_y - 0.012)
    if fw < 0.015:
        raise BuildError("foot width")

    # 다리 (좌우 미러)
    for sgn, sd in ((1, "l"), (-1, "r")):
        parent, z0 = "base", -ph / 2 - r_bl
        if nb_hip == 2:
            n = f"ball_hipyaw_{sd}"
            r.add_link(n, Geom("sphere", (r_bl,)), dens(), connector=True)
            r.add_joint("revolute", parent, n, (0, sgn * hip_y, z0),
                        axis=(0, 0, 1), lower=-0.2, upper=0.2)
            parent, z0 = n, -2 * r_bl
        n = f"ball_hiproll_{sd}"
        r.add_link(n, Geom("sphere", (r_bl,)), dens(), connector=True)
        xyz = (0, sgn * hip_y, z0) if parent == "base" else (0, 0, z0)
        lo, up = (-0.05, 0.8) if sgn > 0 else (-0.8, 0.05)  # 안쪽(다리 교차) 최소화
        r.add_joint("revolute", parent, n, xyz, axis=(1, 0, 0), lower=lo, upper=up)

        tn = f"leg_thigh_{sd}"
        r.add_link(tn, Geom("cylinder", (leg_r, thigh)), dens(),
                   origin=(0, 0, -(r_bl + thigh / 2)))
        r.add_joint("revolute", n, tn, (0, 0, 0), axis=(0, 1, 0),
                    lower=-1.1, upper=1.1)
        sn = f"leg_shin_{sd}"
        r.add_link(sn, Geom("cylinder", (leg_r * 0.9, shin)), dens(),
                   origin=(0, 0, -shin / 2))
        r.add_joint("revolute", tn, sn, (0, 0, -(r_bl + thigh)),
                    axis=(0, 1, 0), lower=-0.03, upper=2.0)
        fn = f"foot_{sd}"
        r.add_link(fn, Geom("box", (fl, fw, foot_h)), dens(),
                   origin=(fl * 0.2, 0, -foot_h / 2))
        r.add_joint("revolute", sn, fn, (0, 0, -shin),
                    axis=(0, 1, 0), lower=-0.6, upper=0.6)

    # 팔 (좌우 미러)
    nb_sh = 2 if rng.random() < 0.4 else 1
    wrist = rng.random() < 0.5
    lu = 0.155 * h * _u(rng, 0.9, 1.1)
    lf_a = 0.125 * h * _u(rng, 0.9, 1.1)
    r_h = 1.3 * arm_r
    sh_z_world = base_z + ph / 2 + ch - r_ba
    drop = 2 * r_ba * nb_sh + lu + lf_a + 2 * r_h - r_ba
    if sh_z_world - drop < 0.04:             # 팔이 땅에 닿으면 축소
        k = (sh_z_world - 0.04) / drop
        lu, lf_a = lu * k, lf_a * k

    for sgn, sd in ((1, "l"), (-1, "r")):
        n = f"ball_shpitch_{sd}"
        r.add_link(n, Geom("sphere", (r_ba,)), dens(), connector=True)
        r.add_joint("revolute", "torso_chest", n, (0, sgn * sh_y, ch - r_ba),
                    axis=(0, 1, 0), lower=-1.8, upper=1.8)
        parent = n
        if nb_sh == 2:
            n2 = f"ball_shyaw_{sd}"
            r.add_link(n2, Geom("sphere", (r_ba,)), dens(), connector=True)
            r.add_joint("revolute", parent, n2, (0, 0, -2 * r_ba),
                        axis=(0, 0, 1), lower=-0.6, upper=0.6)
            parent = n2
        un = f"arm_upper_{sd}"
        lo, up = (-0.03, 1.3) if sgn > 0 else (-1.3, 0.03)  # 안쪽(몸통 간섭) 차단
        r.add_link(un, Geom("cylinder", (arm_r, lu)), dens(),
                   origin=(0, 0, -(r_ba + lu / 2)))
        r.add_joint("revolute", parent, un, (0, 0, 0), axis=(1, 0, 0),
                    lower=lo, upper=up)
        fn = f"arm_fore_{sd}"
        r.add_link(fn, Geom("cylinder", (arm_r * 0.85, lf_a)), dens(),
                   origin=(0, 0, -lf_a / 2))
        r.add_joint("revolute", un, fn, (0, 0, -(r_ba + lu)),
                    axis=(0, 1, 0), lower=-2.0, upper=0.03)
        hn = f"hand_{sd}"
        r.add_link(hn, Geom("sphere", (r_h,)), dens())
        if wrist:
            r.add_joint("revolute", fn, hn, (0, 0, -lf_a - r_h * 0.8),
                        axis=(0, 0, 1), lower=-0.9, upper=0.9)
        else:
            r.add_joint("fixed", fn, hn, (0, 0, -lf_a - r_h * 0.8))

    # 센서: imu(골반 내장) + rgb(머리 전면) [+depth 가슴/머리, +lidar 가슴]
    sensors = ["sensor_imu", "sensor_rgb"]
    attach_sensor(r, "sensor_imu", "base", (0, 0, 0))
    attach_sensor(r, "sensor_rgb", "head", (head_r * 0.9, 0, head_r * 0.15))
    if rng.random() < 0.5:
        attach_sensor(r, "sensor_depth", "torso_chest",
                      (torso_d / 2 + 0.008, 0, ch * 0.7))
        sensors.append("sensor_depth")
    if rng.random() < 0.3:
        attach_sensor(r, "sensor_lidar", "torso_chest",
                      (torso_d / 2 + 0.018, 0, ch * 0.45))
        sensors.append("sensor_lidar")

    p.update(torso_w=torso_w, torso_d=torso_d, leg_r=leg_r, arm_r=arm_r,
             thigh=thigh, shin=shin, hip_dof=nb_hip + 1, shoulder_dof=nb_sh + 1,
             wrist=wrist, waist=waist, foot=(fl, fw, foot_h), head_r=head_r)
    meta = dict(form="humanoid", base_z=base_z, params=_round(p),
                sensors=sensors, notes=[])
    return r, meta


# ============================================================ 다족 (4/6/8)

def build_multileg(rng, target, n_legs):
    w, l, h = target
    p = {"n_legs": n_legs}
    # 배치 3종: mammal(다리가 몸통 아래, Go2류) / sprawl(좌우 측면, 곤충류)
    #           / radial(원형 몸체 둘레 균등 각도, 육각몸체 헥사포드류 — cylinder로 근사)
    probs = {4: (0.6, 0.2), 6: (0.1, 0.4), 8: (0.1, 0.4)}[n_legs]
    u = rng.random()
    stance = "mammal" if u < probs[0] else \
             "sprawl" if u < probs[0] + probs[1] else "radial"
    # radial은 다리 x-도달거리 한계상 종횡비가 원형에 가까워야 목표 l을 채움
    if stance == "radial" and w < {4: 0.90, 6: 0.55, 8: 0.40}[n_legs] * l:
        stance = "sprawl"
    lidar_top = rng.random() < 0.6
    h_eff = h - 0.047 if lidar_top else h    # 상단 lidar 높이 예약

    leg_r = _clamp(0.030 * max(l, w), 0.008, 0.045)
    r_foot = 1.2 * leg_r
    bh = _clamp(_u(rng, 0.35, 0.55) * h_eff, 0.03, h_eff - 0.06)
    clearance = h_eff - bh
    body_l = _clamp(_u(rng, 0.90, 0.97) * l, 0.08, l - 0.02)
    n_rows = n_legs // 2
    span = body_l * 0.75
    xs = [span * (i / (n_rows - 1) - 0.5) for i in range(n_rows)]

    r = Robot(f"multileg{n_legs}")
    dens = lambda: _u(rng, 300, 900)
    base_z = clearance + bh / 2

    if stance == "radial":
        return _build_radial(rng, r, target, n_legs, bh, clearance,
                             leg_r, r_foot, lidar_top, base_z, p, dens)

    if stance == "mammal":
        body_w = _clamp(w - 4 * leg_r - 0.005, 0.06, w)
        hip_y = body_w / 2 + leg_r + 0.0025
        with_roll = rng.random() < 0.4
        anchor_z = -bh * 0.25
        drop = clearance + bh * 0.25 - r_foot
        if drop < 0.04:
            raise BuildError("mammal drop")
        alpha = _u(rng, 0.15, 0.55)
        lf = drop * _u(rng, 0.45, 0.6) / math.cos(alpha)
        V = drop - lf * math.cos(alpha)
        H = lf * math.sin(alpha)
        lt = math.hypot(V, H)
        gamma = math.atan2(H, V)
        if min(lf, lt) < max(0.03, 2.2 * leg_r):
            raise BuildError("mammal leg seg")
        hp_lim = 0.8 if n_legs == 4 else 0.5
        p.update(stance=stance, alpha=alpha, gamma=gamma, lf=lf, lt=lt,
                 with_roll=with_roll, body=(body_l, body_w, bh))
    else:
        body_w = _clamp(_u(rng, 0.38, 0.52) * w, 0.05, w - 0.1)
        A_tot = (w - body_w) / 2
        V = clearance + bh / 2 - r_foot
        if V < 0.03 or A_tot < 0.05:
            raise BuildError("sprawl budget")
        cx = _clamp(_u(rng, 0.25, 0.4) * A_tot, 0.02, A_tot - 0.03)
        A = A_tot - cx - r_foot          # 발 구체 반지름만큼 측면폭 예산에서 차감
        if A < 0.02:
            raise BuildError("sprawl lateral")
        lf = lt = th1 = th2 = None
        seg_min = max(0.03, 2.2 * leg_r)     # tibia 상단이 coxa에 닿지 않을 최소 길이
        for _ in range(30):                  # 측면폭 A·낙차 V 동시 만족 해 탐색
            t1, t2 = _u(rng, 0.15, 0.5), _u(rng, 1.0, 1.45)
            d = math.sin(t2 - t1)
            f = (A * math.sin(t2) - V * math.cos(t2)) / d
            t = (V * math.cos(t1) - A * math.sin(t1)) / d
            if seg_min < f < 1.5 and seg_min < t < 1.5:
                lf, lt, th1, th2 = f, t, t1, t2
                break
        if lf is None:
            raise BuildError("sprawl IK")
        yaw_lim = {4: 0.4, 6: 0.3, 8: 0.25}[n_legs]
        p.update(stance=stance, cx=cx, lf=lf, lt=lt, th1=th1, th2=th2,
                 body=(body_l, body_w, bh))

    r.add_link("base", Geom("box", (body_l, body_w, bh)), dens())

    for i, x in enumerate(xs):
        for sgn, sd in ((1, "l"), (-1, "r")):
            tag = f"{i}{sd}"
            if stance == "mammal":
                parent = "base"
                anchor = (x, sgn * hip_y, anchor_z)
                if with_roll:
                    bn = f"ball_hip_{tag}"
                    r.add_link(bn, Geom("sphere", (1.15 * leg_r,)), dens(),
                               connector=True)
                    lo, up = (-0.1, 0.5) if sgn > 0 else (-0.5, 0.1)
                    r.add_joint("revolute", "base", bn, anchor,
                                axis=(1, 0, 0), lower=lo, upper=up)
                    parent, anchor = bn, (0, 0, 0)
                fn = f"leg_femur_{tag}"
                r.add_link(fn, Geom("cylinder", (leg_r, lf)), dens(),
                           origin=(0, 0, -lf / 2))
                r.add_joint("revolute", parent, fn, anchor, rpy=(0, -alpha, 0),
                            axis=(0, 1, 0), lower=-hp_lim, upper=hp_lim)
                tn = f"leg_tibia_{tag}"
                r.add_link(tn, Geom("cylinder", (leg_r * 0.85, lt)), dens(),
                           origin=(0, 0, -lt / 2))
                r.add_joint("revolute", fn, tn, (0, 0, -lf),
                            rpy=(0, gamma + alpha, 0), axis=(0, 1, 0),
                            lower=-0.9, upper=0.9)
            else:
                cn = f"leg_coxa_{tag}"
                r.add_link(cn, Geom("cylinder", (leg_r, cx), rpy=(PI / 2, 0, 0)),
                           dens(), origin=(0, sgn * cx / 2, 0))
                r.add_joint("revolute", "base", cn, (x, sgn * body_w / 2, 0),
                            axis=(0, 0, 1), lower=-yaw_lim, upper=yaw_lim)
                fn = f"leg_femur_{tag}"
                phi1 = sgn * (PI / 2 - th1)
                r.add_link(fn, Geom("cylinder", (leg_r, lf)), dens(),
                           origin=(0, 0, -lf / 2))
                r.add_joint("revolute", cn, fn, (0, sgn * cx, 0),
                            rpy=(phi1, 0, 0), axis=(1, 0, 0),
                            lower=-0.5, upper=0.5)
                tn = f"leg_tibia_{tag}"
                dphi = sgn * (th1 - th2)
                r.add_link(tn, Geom("cylinder", (leg_r * 0.85, lt)), dens(),
                           origin=(0, 0, -lt / 2))
                r.add_joint("revolute", fn, tn, (0, 0, -lf), rpy=(dphi, 0, 0),
                            axis=(1, 0, 0), lower=-0.6, upper=0.6)
            r.add_link(f"foot_{tag}", Geom("sphere", (r_foot,)), dens())
            r.add_joint("fixed", tn, f"foot_{tag}", (0, 0, -lt))

    # 센서
    sensors = ["sensor_imu", "sensor_rgb"]
    attach_sensor(r, "sensor_imu", "base", (0, 0, bh / 2 - 0.005))
    attach_sensor(r, "sensor_rgb", "base", (body_l / 2 + 0.008, 0, bh * 0.2))
    if bh >= 0.10 and rng.random() < 0.5:    # 얇은 몸통은 rgb와 겹침 → 생략
        attach_sensor(r, "sensor_depth", "base",
                      (body_l / 2 + 0.009, 0, -bh * 0.15))
        sensors.append("sensor_depth")
    if lidar_top:
        attach_sensor(r, "sensor_lidar", "base", (0, 0, bh / 2 + 0.022))
        sensors.append("sensor_lidar")

    meta = dict(form=f"{'quad' if n_legs == 4 else 'hex' if n_legs == 6 else 'oct'}",
                base_z=base_z, params=_round(p), sensors=sensors,
                notes=["foot=fixed sphere"])
    return r, meta


def _build_radial(rng, r, target, n_legs, bh, clearance, leg_r, r_foot,
                  lidar_top, base_z, p, dens):
    """방사형 다족: cylinder 몸체 둘레에 다리를 균등 각도로 배치.

    다리 각도는 π/n 오프셋(정면 +x축에 다리가 오지 않게 → 전면 카메라 확보).
    다리별 수평 예산은 목표 AABB의 직사각형 경계 min(a/|cosθ|, b/|sinθ|) 기준으로
    잡아 w·l이 모두 채워지게 한다(몸체가 원형이어도 치수 검증 통과).
    """
    w, l, h = target
    a, b = l / 2, w / 2
    # 몸체 둘레에서 이웃 coxa끼리 닿지 않을 최소 반지름(현 크기의 다리 두께 기준)
    rb_min = max(0.03, (leg_r + 0.003) / math.sin(PI / n_legs))
    rb_max = min(a, b) - 0.06
    if rb_min > rb_max:
        raise BuildError("radial body")
    rb = _clamp(min(a, b) * _u(rng, 0.35, 0.5), rb_min, rb_max)
    # 다리는 몸체 하단(-0.25bh), 전면 센서는 상단(+0.25bh)에 붙여 z로 분리
    anchor_z = -0.25 * bh
    if 0.5 * bh < leg_r + 0.015:
        raise BuildError("radial sensor gap")
    V = clearance + 0.25 * bh - r_foot
    if V < 0.03:
        raise BuildError("radial drop")

    angles = [PI / n_legs + 2 * PI * i / n_legs for i in range(n_legs)]
    A_list = []
    for th in angles:
        R = min(a / max(abs(math.cos(th)), 1e-6),
                b / max(abs(math.sin(th)), 1e-6))
        A = R - rb - r_foot
        if A < 0.05:
            raise BuildError("radial lateral")
        A_list.append(A)

    sol = None
    for _ in range(30):                      # 전 다리 공통 각도(th1,th2)로 IK 해 탐색
        t1, t2 = _u(rng, 0.15, 0.5), _u(rng, 1.0, 1.45)
        legs = []
        seg_min = max(0.03, 2.2 * leg_r)     # tibia 상단이 coxa에 닿지 않을 최소 길이
        for A in A_list:
            cx = _clamp(0.3 * A, 0.015, A - 0.03)
            Ar = A - cx
            d = math.sin(t2 - t1)
            lf = (Ar * math.sin(t2) - V * math.cos(t2)) / d
            lt = (V * math.cos(t1) - Ar * math.sin(t1)) / d
            if not (seg_min < lf < 1.5 and seg_min < lt < 1.5):
                break
            legs.append((cx, lf, lt))
        else:
            sol = (t1, t2, legs)
            break
    if sol is None:
        raise BuildError("radial IK")
    t1, t2, legs = sol
    yaw_lim = {4: 0.35, 6: 0.25, 8: 0.2}[n_legs]

    r.add_link("base", Geom("cylinder", (rb, bh)), dens())
    for i, (th, (cx, lf, lt)) in enumerate(zip(angles, legs)):
        psi = th - PI / 2                    # 로컬 +y = 바깥 방향
        cn = f"leg_coxa_{i}"
        r.add_link(cn, Geom("cylinder", (leg_r, cx), rpy=(PI / 2, 0, 0)),
                   dens(), origin=(0, cx / 2, 0))
        r.add_joint("revolute", "base", cn,
                    (rb * math.cos(th), rb * math.sin(th), anchor_z),
                    rpy=(0, 0, psi), axis=(0, 0, 1),
                    lower=-yaw_lim, upper=yaw_lim)
        fn = f"leg_femur_{i}"
        r.add_link(fn, Geom("cylinder", (leg_r, lf)), dens(),
                   origin=(0, 0, -lf / 2))
        r.add_joint("revolute", cn, fn, (0, cx, 0), rpy=(PI / 2 - t1, 0, 0),
                    axis=(1, 0, 0), lower=-0.5, upper=0.5)
        tn = f"leg_tibia_{i}"
        r.add_link(tn, Geom("cylinder", (leg_r * 0.85, lt)), dens(),
                   origin=(0, 0, -lt / 2))
        r.add_joint("revolute", fn, tn, (0, 0, -lf), rpy=(t1 - t2, 0, 0),
                    axis=(1, 0, 0), lower=-0.6, upper=0.6)
        r.add_link(f"foot_{i}", Geom("sphere", (r_foot,)), dens())
        r.add_joint("fixed", tn, f"foot_{i}", (0, 0, -lt))

    sensors = ["sensor_imu", "sensor_rgb"]
    attach_sensor(r, "sensor_imu", "base", (0, 0, bh / 2 - 0.005))
    attach_sensor(r, "sensor_rgb", "base", (rb + 0.008, 0, 0.25 * bh))
    if bh >= 0.18 and rng.random() < 0.5:    # 위로 쌓을 여유 있을 때만
        attach_sensor(r, "sensor_depth", "base",
                      (rb + 0.009, 0, 0.25 * bh + 0.032))
        sensors.append("sensor_depth")
    if lidar_top:
        attach_sensor(r, "sensor_lidar", "base", (0, 0, bh / 2 + 0.022))
        sensors.append("sensor_lidar")

    p.update(stance="radial", body_r=rb, body_h=bh, th1=t1, th2=t2,
             legs=[(round(c, 4), round(f, 4), round(t, 4)) for c, f, t in legs])
    meta = dict(form={4: "quad", 6: "hex", 8: "oct"}[n_legs],
                base_z=base_z, params=_round(p), sensors=sensors,
                notes=["radial: 육각몸체는 cylinder로 근사", "foot=fixed sphere"])
    return r, meta


# ============================================================ 휠

def build_wheeled(rng, target, drive):
    w, l, h = target
    p = {"drive": drive}
    lidar_top = rng.random() < 0.7
    reserve = 0.047 if lidar_top else 0.0

    wheel_r = _clamp(_u(rng, 0.10, 0.22) * l, 0.03, min(0.30, 0.45 * (h - reserve)))
    chassis_l = _clamp(_u(rng, 0.92, 0.98) * l, 0.08, l - 0.01)
    if drive == "diff":
        wheel_r = min(wheel_r, 0.5 * chassis_l - 0.06)   # 캐스터와 간섭 방지
    else:
        wheel_r = min(wheel_r, (chassis_l - 0.05) / 4)   # 앞뒤 바퀴 간섭 방지
    if wheel_r < 0.03:
        raise BuildError("wheel radius")
    wheel_t = wheel_r * _u(rng, 0.35, 0.6)
    cb = _clamp(wheel_r * _u(rng, 0.55, 0.9), 0.02, wheel_r * 1.2)  # 샤시 지상고
    avail = h - reserve - cb
    max_ch = min(0.65 * l, 0.6)
    if avail <= max_ch:
        ch_h, mast_h = max(avail, 0.04), 0.0
    else:
        ch_h = _clamp(_u(rng, 0.35, 0.55) * avail, 0.05, max_ch)
        mast_h = avail - ch_h
    # 조향(ackermann)은 최대 조향각에서 바퀴 디스크가 샤시 측면을 침범하지 않게
    # 샤시-바퀴 간격을 조향 스윕만큼 확보한다
    gap_in = wheel_r * math.sin(0.5) + 0.008 if drive == "ackermann" else 0.002
    chassis_w = _clamp(w - 2 * wheel_t - 2 * gap_in - 0.004, 0.05, w)

    r = Robot(f"wheeled_{drive}")
    dens = lambda: _u(rng, 300, 900)
    wdens = lambda: _u(rng, 600, 1200)
    base_z = cb + ch_h / 2
    r.add_link("base", Geom("box", (chassis_l, chassis_w, ch_h)), dens())
    wy = chassis_w / 2 + wheel_t / 2 + gap_in
    wz = wheel_r - base_z
    xw = 0.5 * chassis_l - wheel_r - 0.01

    def add_wheel(name, x, sgn, parent="base", rel=None):
        r.add_link(name, Geom("cylinder", (wheel_r, wheel_t), rpy=(PI / 2, 0, 0)),
                   wdens())
        xyz = rel if rel is not None else (x, sgn * wy, wz)
        r.add_joint("continuous", parent, name, xyz, axis=(0, 1, 0))

    notes = []
    if drive == "diff":
        for sgn, sd in ((1, "l"), (-1, "r")):
            add_wheel(f"wheel_{sd}", 0, sgn)
        r_c = max(0.015, cb * 0.6)
        n_c = 2 if rng.random() < 0.5 else 1
        cx_pos = [0.5 * chassis_l - r_c - 0.01]
        if n_c == 2:
            cx_pos.append(-cx_pos[0])
        for i, x in enumerate(cx_pos):
            r.add_link(f"caster_{i}", Geom("sphere", (r_c,)), wdens())
            r.add_joint("fixed", "base", f"caster_{i}", (x, 0, r_c - base_z))
        notes.append("caster=fixed sphere(단순화)")
        p.update(n_casters=n_c, caster_r=r_c)
    elif drive in ("skid4", "mecanum"):
        for x, xf in ((xw, "f"), (-xw, "b")):
            for sgn, sd in ((1, "l"), (-1, "r")):
                add_wheel(f"wheel_{xf}{sd}", x, sgn)
        if drive == "mecanum":
            notes.append("mecanum: 롤러 미모델링, meta.wheel_subtype로 구분")
            p["wheel_subtype"] = "mecanum"
    else:  # ackermann: 후륜 구동 + 전륜 조향
        for sgn, sd in ((1, "l"), (-1, "r")):
            add_wheel(f"wheel_b{sd}", -xw, sgn)
        for sgn, sd in ((1, "l"), (-1, "r")):
            sn = f"steer_f{sd}"
            r.add_link(sn, Geom("cylinder", (0.015, 0.05)), dens())
            r.add_joint("revolute", "base", sn,
                        (xw, sgn * wy, wz + wheel_r * 0.6),
                        axis=(0, 0, 1), lower=-0.45, upper=0.45)
            add_wheel(f"wheel_f{sd}", 0, sgn, parent=sn,
                      rel=(0, 0, -wheel_r * 0.6))

    if mast_h > 0:
        r_m = _clamp(0.03, 0.015, chassis_w * 0.3)
        mast_x = _u(rng, -0.25, 0.0) * chassis_l
        r.add_link("mast", Geom("cylinder", (r_m, mast_h)), dens(),
                   origin=(0, 0, mast_h / 2))
        r.add_joint("fixed", "base", "mast", (mast_x, 0, ch_h / 2))

    # 센서
    sensors = ["sensor_imu", "sensor_rgb"]
    attach_sensor(r, "sensor_imu", "base", (0, 0, 0))
    if mast_h > 0:
        attach_sensor(r, "sensor_rgb", "mast", (r_m + 0.008, 0, mast_h - 0.03))
    else:
        attach_sensor(r, "sensor_rgb", "base",
                      (chassis_l / 2 + 0.008, 0, ch_h * 0.25))
    if rng.random() < 0.5:
        attach_sensor(r, "sensor_depth", "base",
                      (chassis_l / 2 + 0.009, 0, -ch_h * 0.1))
        sensors.append("sensor_depth")
    if lidar_top:
        if mast_h > 0:
            attach_sensor(r, "sensor_lidar", "mast", (0, 0, mast_h + 0.022))
        else:
            attach_sensor(r, "sensor_lidar", "base", (0, 0, ch_h / 2 + 0.022))
        sensors.append("sensor_lidar")

    p.update(wheel_r=wheel_r, wheel_t=wheel_t, clearance=cb,
             chassis=(chassis_l, chassis_w, ch_h), mast_h=mast_h)
    meta = dict(form=drive, base_z=base_z, params=_round(p),
                sensors=sensors, notes=notes)
    return r, meta


# ============================================================ 등록

def _round(d):
    def rr(v):
        if isinstance(v, float):
            return round(v, 4)
        if isinstance(v, (tuple, list)):
            return [rr(x) for x in v]
        return v
    return {k: rr(v) for k, v in d.items()}


CLASS_FORMS = {
    "humanoid": ["humanoid"],
    "multileg": ["quad", "hex", "oct"],
    "wheeled": ["diff", "skid4", "ackermann", "mecanum"],
}

BUILDERS = {
    "humanoid": lambda rng, t: build_humanoid(rng, t),
    "quad": lambda rng, t: build_multileg(rng, t, 4),
    "hex": lambda rng, t: build_multileg(rng, t, 6),
    "oct": lambda rng, t: build_multileg(rng, t, 8),
    "diff": lambda rng, t: build_wheeled(rng, t, "diff"),
    "skid4": lambda rng, t: build_wheeled(rng, t, "skid4"),
    "ackermann": lambda rng, t: build_wheeled(rng, t, "ackermann"),
    "mecanum": lambda rng, t: build_wheeled(rng, t, "mecanum"),
}


def sample_target(rng, cls):
    """클래스별 목표 외곽 치수 (w,l,h) 샘플. 주치수는 호출부에서 층화(bin) 지정."""
    if cls == "humanoid":
        h = _u(rng, 0.5, 1.9)
        w = _clamp(h * _u(rng, 0.28, 0.42), 0.2, 0.7)
        l = _clamp(h * _u(rng, 0.16, 0.28), 0.15, 0.45)
    elif cls == "multileg":
        l = _u(rng, 0.3, 1.2)
        w = _clamp(l * _u(rng, 0.55, 0.95), 0.25, 0.9)
        h = _clamp(l * _u(rng, 0.4, 0.75), 0.15, 0.8)
    else:  # wheeled
        l = _u(rng, 0.3, 1.2)
        w = _clamp(l * _u(rng, 0.55, 0.95), 0.3, 1.0)
        if rng.random() < 0.6:
            h = _clamp(l * _u(rng, 0.35, 0.9), 0.15, 0.9)   # 저상형
        else:
            h = _u(rng, 0.7, 1.4)                            # 마스트형
    return (w, l, h)


PRIMARY_DIM = {"humanoid": ("h", 0.5, 1.9),
               "multileg": ("l", 0.3, 1.2),
               "wheeled": ("l", 0.3, 1.2)}


def sample_target_binned(rng, cls, bin_idx, n_bins):
    """주치수를 bin 내 균등 샘플로 강제해 층화 커버리지 확보."""
    for _ in range(200):
        t = sample_target(rng, cls)
        dim, lo, hi = PRIMARY_DIM[cls]
        v = dict(zip("wlh", t))[dim]
        b = min(int((v - lo) / (hi - lo) * n_bins), n_bins - 1)
        if b == bin_idx:
            return t
    return sample_target(rng, cls)

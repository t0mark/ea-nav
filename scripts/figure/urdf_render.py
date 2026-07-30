"""URDF에서 로봇 그림을 뽑아 씬 렌더 위에 얹을 RGBA로 만든다.

MANSION 풀의 URDF(`/data/URDF/real_robots`)는 visual이 자리표시자라 메시가 없다.
그래서 같은 로봇의 공개 description 패키지에서 메시째 가져와 렌더한다 — 형상은
그림용이고, 통과성 판정에 쓰이는 치수(w_eff·h·stairs_ok)는 여전히 풀 쪽 값이다.
두 URDF의 외곽 치수가 같은지는 check_dims()로 확인할 수 있다.

배경은 세그멘테이션 마스크로 지워 알파를 만들고, 픽셀당 미터를 함께 돌려주어
합성 쪽에서 씬의 원근 배율에 맞춰 크기를 정할 수 있게 한다.
"""

import math
import os
import re
import tempfile

import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
MESH_URDF = {
    "unitree_go2": f"{ASSETS}/go2_description/urdf/go2_description.urdf",
    "turtlebot3_waffle": (f"{ASSETS}/turtlebot3_description/urdf/"
                          f"turtlebot3_waffle.urdf"),
}

# 네 다리를 살짝 굽힌 기본 기립 자세. 모두 0으로 두면 다리가 일직선으로 뻗어
# 서 있는 로봇처럼 보이지 않는다.
GO2_STANCE = {"thigh": 0.72, "calf": -1.45}

# STL만 있는 링크는 색이 없어 pybullet 기본색으로 나온다. 부위별로 칠해 몸통·
# 구동부·센서가 구분되게 한다. 키는 링크 이름에 들어가는 조각이다.
TINT = {
    "turtlebot3_waffle": {"wheel": (0.10, 0.10, 0.12, 1.0),
                          "caster": (0.10, 0.10, 0.12, 1.0),
                          "scan": (0.05, 0.65, 0.91, 1.0),
                          "camera": (0.05, 0.65, 0.91, 1.0),
                          "base": (0.82, 0.84, 0.86, 1.0)},
}


def resolved_urdf(path):
    """`package://` 참조를 실제 경로로 바꾼 임시 URDF를 만든다. 임시 파일은
    메시 상대경로가 깨지지 않도록 원본과 같은 디렉터리에 둔다."""
    txt = re.sub(r"package://([^/]+)/", ASSETS + r"/\1/", open(path).read())
    txt = txt.replace("${namespace}", "")
    fd, tmp = tempfile.mkstemp(suffix=".urdf", dir=os.path.dirname(path))
    os.close(fd)
    with open(tmp, "w") as f:
        f.write(txt)
    return tmp


def _tint(p, cid, body, rules):
    """링크 이름 규칙에 따라 색을 덮어쓴다. 규칙에 없으면 메시 원래 색 유지."""
    for j in range(-1, p.getNumJoints(body, physicsClientId=cid)):
        nm = ("base" if j < 0 else
              p.getJointInfo(body, j, physicsClientId=cid)[12].decode())
        for key, rgba in rules.items():
            if key in nm:
                p.changeVisualShape(body, j, rgbaColor=rgba,
                                    physicsClientId=cid)
                break


def render(robot_id, size=900, pitch_deg=35.0, yaw_deg=210.0, fov=16.0,
           stance=None, tint=None):
    """로봇 한 대를 투명 배경 RGBA로 렌더하고 (이미지, 픽셀당 미터)를 돌려준다.

    화각을 좁게 잡고 멀리서 당겨 찍어 정사영에 가깝게 만든다. 그래야 씬에
    합성했을 때 로봇만 원근이 달라 보이는 어색함이 없다.
    """
    import pybullet as p

    cid = p.connect(p.DIRECT)
    tmp = resolved_urdf(MESH_URDF[robot_id])
    try:
        body = p.loadURDF(tmp, useFixedBase=True, physicsClientId=cid)
        n = p.getNumJoints(body, physicsClientId=cid)
        for j in range(n):
            info = p.getJointInfo(body, j, physicsClientId=cid)
            if info[2] == p.JOINT_FIXED:
                continue
            ang = 0.0
            for key, val in (stance or {}).items():
                if key in info[1].decode():
                    ang = val
            p.resetJointState(body, j, ang, physicsClientId=cid)
        rules = tint if tint is not None else TINT.get(robot_id)
        if rules:
            _tint(p, cid, body, rules)

        lo, hi = p.getAABB(body, -1, physicsClientId=cid)
        for j in range(n):
            a, b = p.getAABB(body, j, physicsClientId=cid)
            lo = [min(lo[i], a[i]) for i in range(3)]
            hi = [max(hi[i], b[i]) for i in range(3)]
        ctr = [(lo[i] + hi[i]) / 2 for i in range(3)]
        span = max(hi[i] - lo[i] for i in range(3))
        dist = span / (2 * math.tan(math.radians(fov) / 2)) * 2.2

        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=ctr, distance=dist, yaw=yaw_deg,
            pitch=-pitch_deg, roll=0, upAxisIndex=2, physicsClientId=cid)
        proj = p.computeProjectionMatrixFOV(fov, 1.0, dist * 0.1, dist * 5,
                                            physicsClientId=cid)
        w, h, rgb, _, seg = p.getCameraImage(
            size, size, view, proj, renderer=p.ER_TINY_RENDERER,
            lightDirection=[-0.5, -0.8, 1.0], shadow=0, physicsClientId=cid)
        rgb = np.reshape(np.array(rgb, np.uint8), (h, w, 4))
        seg = np.reshape(np.array(seg, np.int32), (h, w))
        out = rgb.copy()
        out[..., 3] = np.where(seg >= 0, 255, 0).astype(np.uint8)
        ppm = (h / 2.0) / (dist * math.tan(math.radians(fov) / 2.0))
        return crop_alpha(out), ppm
    finally:
        p.disconnect(cid)
        os.unlink(tmp)


def check_dims(robot_id):
    """메시 URDF의 외곽 치수를 재서 풀 쪽 치수와 견줄 수 있게 돌려준다."""
    import pybullet as p

    cid = p.connect(p.DIRECT)
    tmp = resolved_urdf(MESH_URDF[robot_id])
    try:
        body = p.loadURDF(tmp, useFixedBase=True, physicsClientId=cid)
        lo, hi = p.getAABB(body, -1, physicsClientId=cid)
        for j in range(p.getNumJoints(body, physicsClientId=cid)):
            a, b = p.getAABB(body, j, physicsClientId=cid)
            lo = [min(lo[i], a[i]) for i in range(3)]
            hi = [max(hi[i], b[i]) for i in range(3)]
        return {"l": hi[0] - lo[0], "w": hi[1] - lo[1], "h": hi[2] - lo[2]}
    finally:
        p.disconnect(cid)
        os.unlink(tmp)


def crop_alpha(img, pad=2):
    """알파가 있는 영역만 남긴다. 여백째 합성하면 위치 계산이 어긋난다."""
    ys, xs = np.where(img[..., 3] > 0)
    if not len(ys):
        return img
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0] - 1)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1] - 1)
    return img[y0:y1 + 1, x0:x1 + 1]


def outlined(img, color=(255, 255, 255), width=3):
    """실루엣 둘레에 테두리를 둘러 배경이 복잡해도 형태가 끊기지 않게 한다."""
    import cv2

    a = (img[..., 3] > 0).astype(np.uint8)
    ring = cv2.dilate(a, np.ones((width * 2 + 1,) * 2, np.uint8)) - a
    out = img.copy()
    out[ring > 0, :3] = color
    out[ring > 0, 3] = 255
    return out

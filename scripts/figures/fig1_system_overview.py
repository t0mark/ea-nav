"""Fig 1 시스템 개요 — 같은 지시를 받은 두 로봇이 계단·엘베로 갈라지는 그림.

세 단계로 나뉘며 각각 따로 실행한다. render는 GPU와 AI2-THOR가 필요하고
plan은 플래너만, compose는 캐시만 쓰므로 반복 수정이 싸다.
  render  — 계단·엘베가 함께 보이는 대각선 BEV를 실사 렌더하고 카메라 메타 저장
  plan    — 두 로봇의 실제 경로를 계획해 F1 구간 폴리라인을 월드 좌표로 저장
  compose — 캐시를 읽어 경로선·로봇·말풍선·URDF 패널을 합성

배경 씬은 오피스 1층이다. 계단(x5~7)과 엘베(x7~9)가 z0~2에 나란히 붙어 있어
한 화면에 담기고, 출발점인 리셉션(z6~11)에서 복도를 따라 내려오는 접근로가
그대로 보인다. MANSION 층은 천장이 없어 위에서 그대로 내부가 보인다 —
proceduralParameters의 ceilingMaterial을 건드리면 CreateHouse가 죽으므로
씬은 원본 그대로 로드한다.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "datasets", "MANSION"))

BUILDING = ("/data/MansionWorld/mansionworld/"
            "office_corporate_hq_6f_300_fp001#0")
OUT_DIR = "/workspace/research/check/figure"
CACHE = os.path.join(OUT_DIR, "_cache")

# 카메라는 복도에 낮게 서서 계단·엘베를 살짝 올려다본다(부각 -3.6°).
# 높이 2.2m는 아래에서 정해진 하한이다 — 라운지 가구가 z 4~7을 채우고 로봇은
# z=2.5·높이 0.3m라, 2.0m 아래로 내리면 시선이 의자 등받이에 걸려 로봇이
# 가린다(1.7m·거리 6m에서 가구 지점 시선 높이 0.9m). x를 8.6까지 밀어 화장실
# 타일벽(x=5)을 화면 밖으로 보내되, 계단실이 그 벽에 붙어 있어 모서리는 남는다.
CAM_POS = (8.6, 2.35, 9.0)
CAM_TARGET = (7.0, 2.90, 0.8)
CAM_FOV = 64.0
RENDER_W, RENDER_H = 2100, 1400

# 두 로봇은 리셉션 중앙에서 함께 출발해 3층 팬트리로 향한다. 계단 통행 가능
# 여부(stairs_ok)만 다르고 나머지 조건은 같다.
ROBOTS = ("turtlebot3_waffle", "unitree_go2")
START_ROOM = ("reception", 1)
GOAL_ROOM = ("pantry", 3)


def cam_basis(pos, target):
    """카메라 위치·목표에서 Unity 오일러각과 정규직교 기저를 만든다.

    Unity는 y가 상방, rotation.x가 아래로 기울이는 각이다. roll은 0으로 두므로
    회전은 Ry(yaw)·Rx(pitch) 하나로 결정되고, 기저는 그 행렬의 열이 된다.
    """
    d = np.array(target, float) - np.array(pos, float)
    yaw = math.degrees(math.atan2(d[0], d[2]))
    pitch = math.degrees(math.atan2(-d[1], math.hypot(d[0], d[2])))
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    right = np.array([cy, 0.0, -sy])
    up = np.array([sy * sp, cp, cy * sp])
    fwd = np.array([sy * cp, -sp, cy * cp])
    return (pitch, yaw), right, up, fwd


def projector(meta):
    """월드 좌표 → 픽셀 좌표 변환기. fieldOfView는 수직 화각이라 초점거리는
    이미지 높이에서 얻는다. 카메라 뒤쪽 점은 None으로 걸러낸다."""
    pos = np.array(meta["pos"], float)
    _, right, up, fwd = cam_basis(meta["pos"], meta["target"])
    w, h = meta["width"], meta["height"]
    f = (h / 2.0) / math.tan(math.radians(meta["fov"]) / 2.0)

    def w2p(x, y, z):
        rel = np.array([x, y, z], float) - pos
        zc = float(rel @ fwd)
        if zc <= 1e-3:
            return None
        return (w / 2.0 + f * float(rel @ right) / zc,
                h / 2.0 - f * float(rel @ up) / zc)

    return w2p


def stage_render(args):
    """천장을 뺀 오피스 1층을 대각선 BEV로 렌더하고 깊이·메타를 함께 저장."""
    import cv2
    import common

    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(BUILDING, "floor_1.json")) as f:
        scene = json.load(f)
    scene = plain_side_wall(scene)

    (pitch, yaw), _, _, _ = cam_basis(CAM_POS, CAM_TARGET)
    ctrl = common.launch_controller(width=RENDER_W, height=RENDER_H,
                                    render=True)
    try:
        common.load_floor_scene(ctrl, scene)
        print("엘베 문틀 제거:", hide_elevator_frame(ctrl))
        gone = clear_furniture(ctrl)
        print(f"복도 집기 제거 {len(gone)}개: {gone[:12]}")
        ev = ctrl.step(action="AddThirdPartyCamera",
                       position=dict(x=CAM_POS[0], y=CAM_POS[1], z=CAM_POS[2]),
                       rotation=dict(x=pitch, y=yaw, z=0.0),
                       fieldOfView=CAM_FOV)
        if not ev.metadata["lastActionSuccess"]:
            raise RuntimeError(ev.metadata.get("errorMessage"))
        rgb = ev.third_party_camera_frames[0]
        dep = ev.third_party_depth_frames[0]
        cv2.imwrite(os.path.join(CACHE, "f1_bev.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(os.path.join(CACHE, "f1_bev_depth.npy"), dep.astype(np.float32))
        meta = {"pos": list(CAM_POS), "target": list(CAM_TARGET),
                "fov": CAM_FOV, "width": RENDER_W, "height": RENDER_H,
                "pitch": pitch, "yaw": yaw}
        with open(os.path.join(CACHE, "f1_bev_cam.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"렌더 완료 {rgb.shape} pitch={pitch:.1f} yaw={yaw:.1f}")
        verify_projection(meta, dep)
    finally:
        ctrl.stop()


# 복도(x5~15, z1.5~12) 안의 집기는 전부 끈다. 시선에 걸리는 것만 골라 끄면
# 남은 가구가 로봇·경로와 계속 겹쳐 어수선하다.
CLEAR_BOX = (5.2, 15.0, 1.5, 12.0)


def clear_furniture(ctrl, box=CLEAR_BOX):
    """복도에 놓인 집기를 전부 끈다.

    라운지 의자·소파·탁자가 카메라와 로봇 사이에 놓여 로봇이 가구에 겹쳐
    보이고 경로선도 토막 난다. 벽·문·바닥은 건드리지 않는다 — 뚫으면 화장실
    내부가 드러나 더 산만해진다.
    """
    keep = ("wall|", "door|", "room|", "Floor", "Ceiling", "elevator_doors|")
    x0, x1, z0, z1 = box
    dropped = []
    for o in ctrl.step(action="Pass").metadata["objects"]:
        oid = o["objectId"]
        if any(oid.startswith(k) for k in keep):
            continue
        p = o["position"]
        if not (x0 <= p["x"] <= x1 and z0 <= p["z"] <= z1):
            continue
        if ctrl.step(action="DisableObject",
                     objectId=oid).metadata["lastActionSuccess"]:
            dropped.append(oid.split(" ")[0])
    return dropped


def hide_elevator_frame(ctrl):
    """엘베 문틀(Doorframe_Double_9)을 끈다.

    common.load_floor_scene은 엘베 문에 문틀과 커스텀 슬라이딩 도어를 **둘 다**
    붙이는데 높이가 맞지 않아 흰 문이 문틀 위로 튀어나온다. 그림에서는 문틀만
    빼고 슬라이딩 도어를 남긴다(데이터셋 파이프라인은 건드리지 않는다).
    """
    out = []
    for o in ctrl.step(action="Pass").metadata["objects"]:
        oid = o["objectId"]
        if oid.startswith("door|") and "elevator" in oid.lower():
            if ctrl.step(action="DisableObject",
                         objectId=oid).metadata["lastActionSuccess"]:
                out.append(oid)
    return out


def plain_side_wall(scene, x_plane=5.0, material="LightWhite"):
    """화면 오른쪽을 채우는 화장실 경계벽(x=5)의 재질을 무늬 없는 흰색으로.

    파란 세라믹 타일이 화면의 1/4을 차지하며 시선을 끌어 두 문에서 주의를
    분산시킨다. 벽을 지우는 안은 불채택 — 그 벽에 달린 문이 참조를 잃어
    CreateHouse가 죽고, 억지로 지우면 화장실 내부(변기·칸막이)가 드러나
    오히려 더 산만해진다. 기하는 그대로 두고 재질만 바꾼다.
    """
    import copy

    out = copy.deepcopy(scene)
    for w in out["walls"]:
        p = np.array([[q["x"], q["z"]] for q in w["polygon"]], float)
        if abs(p[:, 0].mean() - x_plane) < 0.2 and np.ptp(p[:, 0]) < 0.3:
            w["material"] = {"name": material}
    return out


def verify_projection(meta, dep):
    """투영식 자체 검증 — 바닥 격자점을 투영한 픽셀의 깊이가 카메라까지의
    실제 거리와 맞는지 본다. 어긋나면 경로선이 바닥에 안 붙는다."""
    w2p = projector(meta)
    pos = np.array(meta["pos"], float)
    _, _, _, fwd = cam_basis(meta["pos"], meta["target"])
    errs = []
    for x in np.arange(5.5, 14.5, 1.5):
        for z in np.arange(0.5, 11.0, 1.5):
            uv = w2p(x, 0.05, z)
            if uv is None:
                continue
            u, v = int(round(uv[0])), int(round(uv[1]))
            if not (0 <= u < meta["width"] and 0 <= v < meta["height"]):
                continue
            rel = np.array([x, 0.05, z]) - pos
            errs.append(abs(float(dep[v, u]) - float(rel @ fwd)))
    if errs:
        print(f"투영 검증: 표본 {len(errs)}개 깊이오차 "
              f"중앙 {np.median(errs):.3f}m 최대 {max(errs):.3f}m")


def stage_plan(args):
    """두 로봇의 3층 팬트리행 경로를 계획해 층별 폴리라인을 월드로 저장."""
    import common

    os.makedirs(CACHE, exist_ok=True)
    fl = common.load_building(BUILDING)
    pool = {r["id"]: r for r in common.robot_pool("/data/URDF/real_robots")}
    nav = common.BuildingNav(fl)

    def room_center_cell(geom, key):
        room = next(r for r in geom.scene["rooms"]
                    if key in r["roomType"].lower())
        v = np.array(room["vertices"], float)
        cx, cz = v[:, 0].mean(), v[:, 1].mean()
        return geom.to_idx(cx, cz), (float(cx), float(cz)), room["id"]

    s_cell, s_world, s_id = room_center_cell(fl[START_ROOM[1]], START_ROOM[0])
    g_cell, g_world, g_id = room_center_cell(fl[GOAL_ROOM[1]], GOAL_ROOM[0])

    out = {"start": {"world": s_world, "room": s_id, "floor": START_ROOM[1]},
           "goal": {"world": g_world, "room": g_id, "floor": GOAL_ROOM[1]},
           "robots": {}}
    for rid in ROBOTS:
        rb = pool[rid]
        res = nav.plan(rb, (START_ROOM[1], *s_cell), (GOAL_ROOM[1], *g_cell))
        segs = []
        for fno, cells in res.get("segments", []):
            geom = fl[fno]
            segs.append({"floor": fno, "pts": [
                [float(a) for a in geom.to_world(iz, ix)] for iz, ix in cells]})
        out["robots"][rid] = {
            "stairs_ok": bool(rb["stairs_ok"]),
            "success": bool(res.get("success")),
            "cost": float(res.get("cost", -1)),
            "modes": [t["mode"] for t in res.get("transitions", [])],
            "segments": segs}
        print(f"{rid}: 성공={res.get('success')} 비용={res.get('cost', -1):.1f} "
              f"전환={[t['mode'] for t in res.get('transitions', [])]}")
    with open(os.path.join(CACHE, "paths.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"경로 저장: {os.path.join(CACHE, 'paths.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("render", "plan", "compose"))
    args = ap.parse_args()
    if args.stage == "render":
        stage_render(args)
    elif args.stage == "plan":
        stage_plan(args)
    else:
        import fig1_compose
        fig1_compose.run(CACHE, OUT_DIR)


if __name__ == "__main__":
    main()

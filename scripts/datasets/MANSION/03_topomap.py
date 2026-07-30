"""토폴로지 맵(장기 기억 그래프) 생성기 파일럿 (Phase 3 — rule-based 무학습).

기억 그래프는 (씬, 탐사 궤적, 로봇) 단위: 통과 가능 영역이 embodiment마다
달라 같은 씬이라도 로봇별 기억이 다르다. 생성 절차(전부 규칙):
  ① 표준 탐사 궤적 — 층별 방(zone) 앵커(방 안 가장 개활한 지점)를 최근접
     순회, 층은 오름차순. 구간 주행은 관측 기반(drive_expert) expert 재사용.
     도달 불가 방은 스킵하고 기록(그 로봇 기억에는 없는 방).
  ② 노드 — 일정 간격(NODE_SPACING) + 게이트 통과 지점 + 방 앵커.
     **정합(merge)**: 후보가 기존 노드 반경 MERGE_R 안이면 새 노드를 만들지
     않고 그 노드로 재방문 처리(사용자 지적: 재방문 복도에 중복 노드가 쌓여
     그래프가 궤적 체인이 됨). 재방문은 루프 클로저 엣지를 만들고, 도착
     방향이 기존과 60° 이상 다르면 그 노드에 스냅샷 heading을 추가(전방
     단일 뷰라 노드 관측이 방향 의존 — 노드 = 장소, 스냅샷 = 방향별 최대 4).
     **제자리 회전은 쓰지 않는다**(plan 2026-07-27) — 기억에는 주행하며
     실제로 본 것만 담긴다. 방은 진입·통과·진출 heading이 자연히 갈리므로
     앵커 한 곳으로도 방 안이 여러 방향에서 관측된다. 계단/엘베 내부 셀에는
     노드를 두지 않음.
  ③ 엣지 — 궤적상 인접 노드(mode=walk) + 층 전환(stairs/elevator, 엘베는
     wait_steps). (노드쌍, 모드) 중복 제거.
  ④ 프리픽스 절단 — 노드·엣지에 발견 순서 t 기록. 컷 = 앞 k개 노드 +
     "양 끝이 살아있고 t가 컷 시점 이전"인 엣지(늦게 발견된 루프 클로저가
     이른 부분 기억에 새는 것 방지). 결합형 지시 에피소드의 초기 기억 상태.

파일럿 실행(로봇 단위 병렬, GPU 0~3 순환):
  docker exec airlab_hw_mansion bash -c 'cd /workspace/research/scripts/datasets/MANSION \
    && python 03_topomap.py --building "/data/MansionWorld/mansionworld/public_hotel_dormitory_4f_300_fp001#0" \
    --out /workspace/research/check/MANSION/03_topomap'
"""

import argparse
import json
import math
import os
import time

import cv2
import numpy as np

import common


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

NODE_SPACING = 2.5   # 일반 구간 노드 간격 (m)
GATE_NODE_R = 0.5    # 게이트 중심 이내면 게이트 노드 (m)
MERGE_R = 1.0        # 기존 노드와 이 거리 안이면 병합(재방문) (m)
HEAD_SEP = 60.0      # 노드당 스냅샷 heading 최소 각도 차 (°)
MAX_HEADINGS = 4     # 노드당 스냅샷 방향 수 상한
HEAD_AHEAD = 1.0     # heading = 궤적 1m 전방 방향
PREFIX_CUTS = (0.3, 0.5, 0.7)
MIN_ROOM_M2 = 1.5    # 이보다 작은 방은 앵커 제외
SNAP_COLS = 8        # 스냅샷 몽타주 열 수
KIND_PRIO = {"space": 0, "anchor": 1, "gate": 2}  # 병합 시 상위 종류 유지
KIND_COLOR = {"space": (80, 220, 80), "gate": (60, 60, 230),
              "anchor": (230, 160, 60)}


def pick_robots(robots):
    """대표 2대: 최소 폭 wheeled(엘베 사용) + 중간 multileg(계단 사용)."""
    wh = sorted([r for r in robots if r["cls"] == "wheeled"],
                key=lambda r: r["w_eff"])
    ml = sorted([r for r in robots if r["cls"] == "multileg"],
                key=lambda r: r["w_eff"])
    return [wh[0], ml[len(ml) // 2]]


def ang_diff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def room_anchors(geom, robot):
    """방별 앵커 = 방 안에서 가장 개활한 주행 가능 셀. [(room_id, cell)]."""
    ok = geom.passable(robot["w_eff"], robot["h"])
    wm = geom.width_map(robot["h"])
    out, skipped = [], []
    for i, r in enumerate(geom.rooms):
        rid = r["id"]
        if "stair" in rid.lower() or "elev" in rid.lower():
            continue
        mask = (geom.room_grid == i) & ok
        if mask.sum() * common.GRID ** 2 < MIN_ROOM_M2:
            skipped.append(rid)
            continue
        w = np.where(mask, wm, -1.0)
        iz, ix = np.unravel_index(int(np.argmax(w)), w.shape)
        out.append((rid, (int(iz), int(ix))))
    return out, skipped


def nn_tour(geom, anchors, start_cell=None):
    """앵커 최근접 이웃 순회 순서 (결정적)."""
    rest = list(anchors)
    cur = start_cell
    if cur is None:  # 가장 개활한 앵커에서 시작
        rest.sort(key=lambda a: -geom.width_map(0.2)[a[1]])
        cur = rest[0][1]
    order = []
    while rest:
        k = min(range(len(rest)),
                key=lambda j: (rest[j][1][0] - cur[0]) ** 2
                + (rest[j][1][1] - cur[1]) ** 2)
        order.append(rest.pop(k))
        cur = order[-1][1]
    return order


def explore_route(nav, floors, robot):
    """표준 탐사 궤적: 층 오름차순 × 방 앵커 NN 순회. plan 결과 리스트 반환."""
    legs, visited, skipped_rooms, failed = [], [], [], []
    cur = None
    for no in sorted(floors):
        geom = floors[no]
        anchors, small = room_anchors(geom, robot)
        skipped_rooms += [(no, rid) for rid in small]
        if not anchors:
            continue
        start_cell = cur[1:] if (cur and cur[0] == no) else None
        for rid, cell in nn_tour(geom, anchors, start_cell):
            if cur is None:
                cur = (no, *cell)
                visited.append((no, rid))
                continue
            res = nav.plan(robot, cur, (no, *cell))
            if not res.get("success"):
                failed.append((no, rid, res.get("reason")))
                continue  # 도달 불가 방 — 이 로봇 기억에 없음
            legs.append(res)
            visited.append((no, rid))
            cur = (no, *cell)
    return legs, visited, skipped_rooms, failed


def _walkable_seg(geom, cells):
    """세그먼트가 일반 주행 구간인가 (과반이 계단/엘베 밖)."""
    inz = sum(1 for iz, ix in cells
              if geom.stair_mask[iz, ix] or geom.elev_mask[iz, ix])
    return inz < len(cells) / 2


class GraphBuilder:
    """탐사 궤적 → 정합된 노드/엣지 (발견 순서 t 포함)."""

    def __init__(self, floors, robot, anchor_scan=False):
        self.floors = floors
        self.robot = robot
        # 제자리 회전 불채택(plan 2026-07-27)이라 기본은 스캔 없음.
        # True는 구 규약 재현 — 파일럿 전·후 비교 전용.
        self.anchor_scan = anchor_scan
        self.gates = {no: common.floor_gates(g) for no, g in floors.items()}
        self.nodes, self.edges = [], []
        self.edge_seen = set()   # (min_id, max_id, mode)
        self.last = None         # 직전 방문 노드 id
        self.tick = 0            # 발견 순서 (노드·엣지 공용)
        self.pend_mode = None    # 다음 엣지 모드 (층 전환 직후)
        self.pend_attr = {}

    def _near(self, fl, x, z):
        for nd in self.nodes:  # 수백 개 — 선형 탐색으로 충분
            if nd["floor"] == fl and math.hypot(nd["x"] - x,
                                                nd["z"] - z) <= MERGE_R:
                return nd
        return None

    def visit(self, fl, x, z, heading, kind):
        """노드 방문: 근처 기존 노드에 병합하거나 새로 만들고 엣지 연결."""
        self.tick += 1
        nd = self._near(fl, x, z)
        if nd is None:
            nd = {"id": len(self.nodes), "t": self.tick, "floor": fl,
                  "x": round(x, 3), "z": round(z, 3),
                  "headings": [round(heading, 1)], "kind": kind or "space"}
            self.nodes.append(nd)
        else:  # 병합(재방문) — 새 방향이면 스냅샷 heading 추가, 종류 승격
            if (len(nd["headings"]) < MAX_HEADINGS
                    and all(ang_diff(heading, h) >= HEAD_SEP
                            for h in nd["headings"])):
                nd["headings"].append(round(heading, 1))
            if kind and KIND_PRIO[kind] > KIND_PRIO[nd["kind"]]:
                nd["kind"] = kind
        if self.anchor_scan and nd["kind"] == "anchor" and not nd.get("scan"):
            nd["headings"] = [round((heading + o) % 360.0, 1)
                              for o in (0.0, 90.0, 180.0, 270.0)]
            nd["scan"] = True
        if self.last is not None and self.last != nd["id"]:
            key = (min(self.last, nd["id"]), max(self.last, nd["id"]),
                   self.pend_mode or "walk")
            if key not in self.edge_seen:
                self.edge_seen.add(key)
                a, b = self.nodes[self.last], nd
                e = {"a": self.last, "b": nd["id"], "t": self.tick,
                     "mode": self.pend_mode or "walk",
                     "length_m": round(math.hypot(a["x"] - b["x"],
                                                  a["z"] - b["z"]), 2)}
                e.update(self.pend_attr)
                self.edges.append(e)
        self.last = nd["id"]
        self.pend_mode, self.pend_attr = None, {}
        return nd

    def add_leg(self, res):
        trans = list(res.get("transitions", []))
        prev_fl = None
        for fl, cells in res["segments"]:
            geom = self.floors[fl]
            if not _walkable_seg(geom, cells):
                continue  # 계단 등반/엘베 진입·진출 — 엣지로 추상화
            if prev_fl is not None and fl != prev_fl:
                t = next((t for t in trans
                          if {t["floor"], t["to"]} == {prev_fl, fl}), None)
                self.pend_mode = t["mode"] if t else "transit"
                if t and t["mode"] == "elevator":
                    self.pend_attr = {"wait_steps": t.get("wait_steps", 3)}
            prev_fl = fl
            self._add_walk(geom, fl, cells)

    def _add_walk(self, geom, fl, cells):
        sm = common.natural_path(geom, self.robot, cells)
        pts = [geom.to_world(iz, ix) for iz, ix in sm]
        seglens = [math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(pts[:-1], pts[1:])]
        total = sum(seglens)
        if total < 1e-6:
            return

        def at(t):
            k, acc = 0, 0.0
            while k < len(seglens) - 1 and acc + seglens[k] < t:
                acc += seglens[k]
                k += 1
            f = (t - acc) / max(seglens[k], 1e-9)
            return (pts[k][0] + (pts[k + 1][0] - pts[k][0]) * f,
                    pts[k][1] + (pts[k + 1][1] - pts[k][1]) * f)

        def head(t):
            a, b = at(t), at(min(t + HEAD_AHEAD, total))
            if math.hypot(b[0] - a[0], b[1] - a[1]) < 0.05:
                a = at(max(0.0, t - HEAD_AHEAD))
            return common.yaw_deg(b[0] - a[0], b[1] - a[1])

        def in_zone(x, z):
            iz, ix = geom.to_idx(x, z)
            iz, ix = min(iz, geom.nz - 1), min(ix, geom.nx - 1)
            return geom.stair_mask[iz, ix] or geom.elev_mask[iz, ix]

        gates = self.gates[fl]
        armed = [True] * len(gates)
        carry = NODE_SPACING if self.last is None else 0.0
        t, step = 0.0, 0.25
        while t <= total:
            x, z = at(t)
            if in_zone(x, z):
                t += step
                carry += step
                continue
            near = self._near(fl, x, z)
            if near is not None:
                if near["id"] != self.last:  # 재방문 — 병합 + 루프 클로저
                    self.visit(fl, near["x"], near["z"], head(t), None)
                    carry = 0.0
                t += step
                carry += step
                continue
            kind = None
            for gi, g in enumerate(gates):
                d = math.hypot(g["x"] - x, g["z"] - z)
                if d <= GATE_NODE_R and armed[gi]:
                    kind, armed[gi] = "gate", False
                    break
                if d > GATE_NODE_R * 2:
                    armed[gi] = True
            if kind is None and carry >= NODE_SPACING:
                kind = "space"
            if kind:
                self.visit(fl, x, z, head(t), kind)
                carry = 0.0
            t += step
            carry += step
        ex, ez = pts[-1]
        if not in_zone(ex, ez):  # 방 앵커 (병합되면 종류 승격)
            self.visit(fl, ex, ez, head(total), "anchor")


def scene_objects(geom):
    """방에 속한 가구 오브젝트 [(방 id, 셀)] — 커버리지 판정 대상."""
    out = []
    for o in geom.scene.get("objects", []):
        p, rid = o.get("position"), o.get("roomId")
        if not p or not rid:
            continue
        iz, ix = geom.to_idx(p["x"], p["z"])
        if 0 <= iz < geom.nz and 0 <= ix < geom.nx:
            out.append((rid, (iz, ix)))
    return out


def object_coverage(floors, robot, nodes):
    """기억 스냅샷이 실제로 담은 가구 비율 (방 단위).

    판정 = 노드 위치·저장 heading에서 FOV 90°·10m·LOS (04의 결합형 goal
    관측 판정과 같은 규격). 목표 지시형 goal은 기억에 잡힌 대상에서만
    뽑으므로, 이 값이 곧 확보 가능한 goal 수의 상한이다.
    """
    torch, dev = common.torch_dev()
    seen, total, rooms_hit, rooms_all = 0, 0, set(), set()
    for fl, geom in sorted(floors.items()):
        objs = scene_objects(geom)
        here = [nd for nd in nodes if nd["floor"] == fl]
        rooms_all |= {(fl, r) for r, _ in objs}
        total += len(objs)
        if not objs or not here:
            continue
        block_t = torch.from_numpy(
            common.sight_block(geom, robot["cam_h"])).to(dev)
        cells = [c for _, c in objs]
        hit = [False] * len(objs)
        for nd in here:
            for hd in nd["headings"]:
                vis = common.gates_in_view(
                    block_t, geom.to_idx(nd["x"], nd["z"]),
                    math.radians(hd), cells)
                hit = [a or b for a, b in zip(hit, vis)]
        seen += sum(hit)
        rooms_hit |= {(fl, objs[i][0]) for i, v in enumerate(hit) if v}
    return {"objects_seen": seen, "objects_total": total,
            "rooms_with_any": len(rooms_hit), "rooms_total": len(rooms_all)}


def prefix_keep(nodes, edges, k):
    """앞 k개 노드의 부분 기억: 노드 id<k + t<컷시점·양끝 생존 엣지."""
    t_cut = nodes[k]["t"] if k < len(nodes) else float("inf")
    keep_n = {nd["id"] for nd in nodes[:k]}
    keep_e = [e for e in edges
              if e["a"] in keep_n and e["b"] in keep_n and e["t"] < t_cut]
    return keep_n, keep_e


def render_snapshots(c, floors, robot, nodes, out_prefix):
    """노드 × 방향별 전방 RGB 스냅샷 → 몽타주 페이지들."""
    tiles, cur_fl = [], None
    for nd in nodes:
        geom = floors[nd["floor"]]
        if nd["floor"] != cur_fl:
            common.load_floor_scene(c, geom.scene)
            c.step(action="AddThirdPartyCamera",
                   position=dict(x=0, y=1, z=0),
                   rotation=dict(x=0, y=0, z=0), fieldOfView=90)
            cur_fl = nd["floor"]
        cam_y = min(robot["cam_h"], geom.ceil_h - 0.15)
        for hi, hd in enumerate(nd["headings"]):
            rgb, _ = common.update_cam(c, nd["x"], cam_y, nd["z"], hd)
            cv2.rectangle(rgb, (0, 0), (common.RES - 1, common.RES - 1),
                          KIND_COLOR[nd["kind"]], 4)
            cv2.putText(rgb, f'n{nd["id"]}.{hi} F{nd["floor"]} {nd["kind"]}',
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            tiles.append(rgb)
    pages = []
    per_page = SNAP_COLS * 4
    for p0 in range(0, len(tiles), per_page):
        chunk = tiles[p0:p0 + per_page]
        while len(chunk) % SNAP_COLS:
            chunk.append(np.zeros_like(chunk[0]))
        rows = [np.concatenate(chunk[i:i + SNAP_COLS], axis=1)
                for i in range(0, len(chunk), SNAP_COLS)]
        fp = f"{out_prefix}_snaps_p{len(pages)}.png"
        cv2.imwrite(fp, np.concatenate(rows, axis=0))
        pages.append(fp)
    return pages


def draw_map(floors, building_dir, nodes, edges, out_png,
             keep_n=None, keep_e=None, title=""):
    """층별 탑뷰에 그래프 오버레이. keep_*이 있으면 나머지는 회색(절단)."""
    panels = {}
    for no, geom in sorted(floors.items()):
        img, w2p = common.sim_view(geom, building_dir)
        cv2.putText(img, f"F{no}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        panels[no] = (img, w2p)
    live_e = (lambda e: keep_e is None or e in keep_e)
    for e in edges:
        a, b = nodes[e["a"]], nodes[e["b"]]
        col = (255, 255, 255) if live_e(e) else (110, 110, 110)
        if a["floor"] == b["floor"]:
            img, w2p = panels[a["floor"]]
            cv2.line(img, w2p(a["x"], a["z"]), w2p(b["x"], b["z"]), col,
                     2 if live_e(e) else 1, cv2.LINE_AA)
        else:  # 층 전환 엣지 — 양 층에 모드 라벨
            for nd in (a, b):
                img, w2p = panels[nd["floor"]]
                u, v = w2p(nd["x"], nd["z"])
                cv2.putText(img, e["mode"], (u + 6, v + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    for nd in nodes:
        img, w2p = panels[nd["floor"]]
        live = keep_n is None or nd["id"] in keep_n
        col = KIND_COLOR[nd["kind"]] if live else (120, 120, 120)
        u, v = w2p(nd["x"], nd["z"])
        cv2.circle(img, (u, v), 6, col, -1)
        for hd in nd["headings"]:
            hx = nd["x"] + 0.5 * math.sin(math.radians(hd))
            hz = nd["z"] + 0.5 * math.cos(math.radians(hd))
            cv2.line(img, (u, v), w2p(hx, hz), col, 2)
        if nd["id"] % 5 == 0:
            cv2.putText(img, str(nd["id"]), (u + 6, v - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    imgs = [panels[no][0] for no in sorted(panels)]
    hmax = max(i.shape[0] for i in imgs)
    imgs = [cv2.copyMakeBorder(i, 0, hmax - i.shape[0], 0, 8,
                               cv2.BORDER_CONSTANT) for i in imgs]
    img = np.concatenate(imgs, axis=1)
    bar = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    cv2.imwrite(out_png, np.concatenate([bar, img], axis=0))


def run_robot(robot, args):
    """로봇 1대의 기억 그래프 생성 (서브프로세스 단위)."""
    t0 = time.time()
    log(f"{robot['id']} 시작 — 층 로드")
    floors = common.load_building(args.building)
    nav = common.BuildingNav(floors)
    log(f"{robot['id']} 탐사 궤적 계획 ({time.time() - t0:.0f}s)")
    legs, visited, small, failed = explore_route(nav, floors, robot)
    gb = GraphBuilder(floors, robot)
    for res in legs:
        gb.add_leg(res)
    nodes, edges = gb.nodes, gb.edges
    n = len(nodes)
    n_snap = sum(len(nd["headings"]) for nd in nodes)
    modes = [e["mode"] for e in edges if e["mode"] != "walk"]
    loops = len(edges) - (n - 1)  # 트리 대비 초과 엣지 = 루프 클로저 수
    cuts = {str(f): int(math.ceil(n * f)) for f in PREFIX_CUTS}
    log(f"{robot['id']} 그래프 완료 — 노드 {n} 스냅샷 {n_snap} "
        f"({time.time() - t0:.0f}s), 커버리지 판정 시작")

    cov = object_coverage(floors, robot, nodes)
    # 구 규약(앵커 360° 스캔) 재현본과 나란히 — 스캔 제거로 잃은 양을 잰다
    gb_s = GraphBuilder(floors, robot, anchor_scan=True)
    for res in legs:
        gb_s.add_leg(res)
    cov_s = object_coverage(floors, robot, gb_s.nodes)
    n_snap_s = sum(len(nd["headings"]) for nd in gb_s.nodes)

    print(f"== {robot['id']} ({robot['cls']} w_eff={robot['w_eff']:.2f}"
          f" h={robot['h']:.2f})")
    print(f"  방 방문 {len(visited)} / 스킵(소형) {len(small)}"
          f" / 실패 {len(failed)} {failed if failed else ''}")
    print(f"  노드 {n} (gate {sum(1 for x in nodes if x['kind'] == 'gate')}"
          f", anchor {sum(1 for x in nodes if x['kind'] == 'anchor')})"
          f" 스냅샷 {n_snap} 엣지 {len(edges)} (루프 {loops}) 전환 {modes}")
    print(f"  프리픽스 절단(노드 수): {cuts}")
    print(f"  가구 관측 커버리지: {cov['objects_seen']}/{cov['objects_total']}"
          f" ({cov['objects_seen'] / max(cov['objects_total'], 1):.0%})"
          f" | 구 스캔 규약 {cov_s['objects_seen']}/{cov_s['objects_total']}"
          f" ({cov_s['objects_seen'] / max(cov_s['objects_total'], 1):.0%})"
          f" — 스냅샷 {n_snap} vs {n_snap_s}장")
    print(f"  가구가 하나라도 잡힌 방: {cov['rooms_with_any']}/"
          f"{cov['rooms_total']} | 구 규약 {cov_s['rooms_with_any']}/"
          f"{cov_s['rooms_total']}")

    prefix = os.path.join(args.out, common.out_name(args.building,
                                                    robot["id"]))
    pages = []
    if not args.no_render:
        c = common.launch_controller(width=common.RES, height=common.RES,
                                     render=True)  # 관측 규격 256²
        pages = render_snapshots(c, floors, robot, nodes, prefix)
        c.stop()
    with open(prefix + ".json", "w") as fh:
        json.dump({
            "building": os.path.basename(args.building.rstrip("/")),
            "robot": robot["id"], "cls": robot["cls"],
            "node_spacing_m": NODE_SPACING, "merge_r_m": MERGE_R,
            "visited_rooms": visited, "failed_rooms": failed,
            # 컷 규칙: 노드 id<k + (양 끝 생존 & t<nodes[k].t) 엣지
            "prefix_cuts": cuts,
            "anchor_scan": False,
            "coverage": cov, "coverage_legacy_scan": cov_s,
            "nodes": nodes, "edges": edges,
            "snapshot_pages": [os.path.basename(p) for p in pages],
        }, fh, ensure_ascii=False, indent=1)
    title = (f"{robot['id']} {robot['cls']}  nodes={n} snaps={n_snap}"
             f" edges={len(edges)} loops={loops}"
             f"  transitions={','.join(modes) or '-'}")
    draw_map(floors, args.building, nodes, edges, prefix + "_map.png",
             title=title)
    keep_n, keep_e = prefix_keep(nodes, edges, cuts["0.5"])
    draw_map(floors, args.building, nodes, edges, prefix + "_prefix50.png",
             keep_n=keep_n, keep_e=keep_e,
             title=title + "  [prefix 50% - gray = cut]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="로봇 id 하나만 (서브프로세스 모드)")
    ap.add_argument("--no-render", action="store_true",
                    help="스냅샷 렌더 생략 (그래프/카드만)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    reps = pick_robots(common.robot_pool())

    if args.only:
        robot = next(r for r in reps if r["id"] == args.only)
        run_robot(robot, args)
        return

    # 병렬(규칙): 로봇 단위 서브프로세스, GPU 0~3 순환
    import subprocess
    import time as _time

    pending = [r["id"] for r in reps]
    running = []
    gpu_i, fail = 0, False
    while pending or running:
        while pending and len(running) < max(1, args.workers):
            rid = pending.pop(0)
            env = dict(os.environ, MANSION_GPU=str(gpu_i % 4))
            gpu_i += 1
            cmd = ["python", os.path.abspath(__file__),
                   "--building", args.building, "--out", args.out,
                   "--only", rid]
            if args.no_render:
                cmd.append("--no-render")
            running.append((subprocess.Popen(cmd, env=env), rid))
            print(f"[병렬] {rid} 시작 (GPU {env['MANSION_GPU']})")
        for p, rid in running[:]:
            if p.poll() is not None:
                print(f"[병렬] {rid} 종료 (rc={p.returncode})")
                fail |= p.returncode != 0
                running.remove((p, rid))
        _time.sleep(2)
    print(f"저장: {args.out}/")
    if fail:  # 워커 실패를 rc=0으로 삼키면 하류(04)가 빈 산출물을 읽음
        raise SystemExit(1)


if __name__ == "__main__":
    main()

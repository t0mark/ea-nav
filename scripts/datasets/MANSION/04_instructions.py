"""지시 생성 파일럿 (Phase 4 — LLM 직접 생성 + 역파싱 자동 검증, 사용자 확정).

3형태(전부 goal 우선 파이프라인: goal 샘플 → 접근 셀 → expert 궤적 → 레코드에
goal 메타+랜드마크를 함께 담아 프롬프트 모듈이 "이 궤적이 어떤 goal로
만들어졌는지"를 항상 알게 함):
  · 목표형  — (지시, 목표 오브젝트 GT id). 역파싱 = 후보 목록에서 목표 선택.
  · R2R형   — (지시, 서브지시↔랜드마크 구간 정렬). 같은 (시작,목표)에서
              wheeled(엘베)/legged(계단) 지시가 갈리는지 확인. 역파싱 =
              지시에서 랜드마크·전환 시퀀스 복원 → GT와 LCS 순서 일치율.
  · 결합형  — (지시, 목표 GT + 03 프리픽스 부분 기억). 기억에 있는 구간(층
              전환 수단 포함) + 미탐색 목적지 접근을 한 지시로. 역파싱 =
              목표 선택 + 언급된 전환 수단 대조.

목표 후보 = 씬 JSON 오브젝트를 규칙 필터(장식·벽부착물 블랙리스트)로 선정 —
LLM 불사용. 지시는 영어(R2R 표준). LLM: 생성 = gpt-5.6-luna, 역파싱 =
gpt-5-mini (파일럿에서 A/B 후 본생성 모델 확정 예정).

파일럿 실행(키는 OPENAI_API_KEY 환경변수로 주입, 코드·로그에 노출 금지):
  docker exec -e OPENAI_API_KEY=... airlab_hw_mansion bash -c \
    'cd /workspace/research/scripts/datasets/MANSION && python 04_instructions.py \
     --building "/data/MansionWorld/mansionworld/public_hotel_dormitory_4f_300_fp001#0" \
     --topomap /workspace/research/check/MANSION/03_topomap \
     --out /workspace/research/check/MANSION/04_instructions'
"""

import argparse
import json
import math
import os
import re
import textwrap

import cv2
import numpy as np

import common

GEN_MODEL = "gpt-5.6-luna"
PARSE_MODEL = "gpt-5-mini"
NAME_BLACKLIST = ("wall", "floor", "framed", "paper", "waste", "cleaning",
                  "ceiling", "curtain", "rug", "doormat")
LM_RADIUS = 2.0      # 경로 주변 랜드마크 반경 (m)
LM_MAX = 8           # R2R 랜드마크 수 상한
APPROACH_R = 1.6     # goal 오브젝트 접근 셀 탐색 반경 (m)


# ---------------------------------------------------------------- 목표 후보

def pretty_room(rid, floor=None):
    """'F2_guest_room_204' → ('guest room 204', 2). 접두어 없으면 floor 폴백."""
    m = re.match(r"F(\d+)_(.+)", rid)
    if m:
        return m.group(2).replace("_", " "), int(m.group(1))
    return rid.replace("_", " "), floor


def pretty_name(o):
    """오브젝트 이름 정돈: 방 접두어·인덱스 제거, '_'→' '."""
    name = o["object_name"].split("-")[0]
    room_word = o["roomId"].split("_")[1].lower()
    parts = name.split("_")
    if parts and parts[0].lower() == room_word:
        parts = parts[1:]
    return " ".join(parts) or name


def goal_registry(floors):
    """씬 오브젝트 → 목표 후보 [(gid, name, room, floor, x, z)]. 규칙 필터."""
    out = []
    for no, geom in sorted(floors.items()):
        for o in geom.scene.get("objects", []):
            nm = o.get("object_name", "")
            if not nm or nm.startswith("_"):
                continue
            if any(b in nm.lower() for b in NAME_BLACKLIST):
                continue
            rid = o.get("roomId", "")
            if not rid or "stair" in rid.lower() or "elev" in rid.lower():
                continue
            room, rno = pretty_room(rid, no)
            # 방 번호는 렌더에 문패가 없어 관측으로 확인 불가(사용자 지적)
            # → 지시 어휘는 방 종류(rtype)+층만 사용
            rtype = re.sub(r"\s*\d+$", "", room)
            out.append({"gid": f"F{no}:{nm}", "name": pretty_name(o),
                        "room": room, "rtype": rtype, "rid": rid,
                        "floor": no,
                        "x": round(o["position"]["x"], 3),
                        "z": round(o["position"]["z"], 3)})
    # 유일성 = (오브젝트명, 방 종류, 층): 동일 객실이 여럿인 호텔에서 방
    # 번호 없이 시각적으로 특정 가능한 목표만 goal 후보(unique=True).
    # 비유일 인스턴스도 랜드마크로는 사용.
    from collections import Counter
    cnt = Counter((g["name"], g["rtype"], g["floor"]) for g in out)
    for g in out:
        g["unique"] = cnt[(g["name"], g["rtype"], g["floor"])] == 1
    return out


def approach_cell(geom, robot, gx, gz):
    """goal 오브젝트 앞 주행 가능 셀 (반경 내 최근접). 없으면 None."""
    ok = geom.passable(robot["w_eff"], robot["h"])
    giz, gix = geom.to_idx(gx, gz)
    r = int(APPROACH_R / common.GRID)
    z0, z1 = max(0, giz - r), min(geom.nz, giz + r + 1)
    x0, x1 = max(0, gix - r), min(geom.nx, gix + r + 1)
    sub = ok[z0:z1, x0:x1]
    if not sub.any():
        return None
    zz, xx = np.where(sub)
    k = int(np.argmin((zz + z0 - giz) ** 2 + (xx + x0 - gix) ** 2))
    return (int(zz[k] + z0), int(xx[k] + x0))


# ---------------------------------------------------------------- 랜드마크

def route_landmarks(floors, registry, res, goal):
    """expert 궤적 주변 랜드마크·이벤트의 순서 시퀀스 (R2R 소재)."""
    seq, seen = [], set()
    trans = list(res.get("transitions", []))
    prev_fl = None
    for fl, cells in res["segments"]:
        geom = floors[fl]
        if prev_fl is not None and fl != prev_fl:
            t = next((t for t in trans
                      if {t["floor"], t["to"]} == {prev_fl, fl}), None)
            seq.append({"event": t["mode"] if t else "transit",
                        "to_floor": fl})
        prev_fl = fl
        cand = [g for g in registry if g["floor"] == fl
                and g["gid"] != goal["gid"]]
        step = max(1, int(0.5 / common.GRID))
        for i in range(0, len(cells), step):
            iz, ix = cells[i]
            x, z = geom.to_world(iz, ix)
            j = min(i + step, len(cells) - 1)
            hx, hz = geom.to_world(*cells[j])
            hdx, hdz = hx - x, hz - z
            for g in cand:
                if g["gid"] in seen:
                    continue
                dx, dz = g["x"] - x, g["z"] - z
                if math.hypot(dx, dz) > LM_RADIUS:
                    continue
                # 우측 성분 = dx·hdz − dz·hdx (Unity 좌표: 진행 방향의
                # 오른쪽 벡터 = (hdz, −hdx))
                side = "right" if dx * hdz - dz * hdx > 0 else "left"
                seen.add(g["gid"])
                seq.append({"landmark": g["name"], "room": g["rtype"],
                            "side": side, "floor": fl,
                            "x": g["x"], "z": g["z"]})
    # 같은 이름 연속 중복 제거 + 상한
    out = []
    for s in seq:
        if out and "landmark" in s and out[-1].get("landmark") == s["landmark"]:
            continue
        out.append(s)
    lms = [s for s in out if "landmark" in s]
    if len(lms) > LM_MAX:  # 균등 솎기 (이벤트는 유지)
        keep = {id(lms[i]) for i in
                np.linspace(0, len(lms) - 1, LM_MAX).astype(int)}
        out = [s for s in out if "landmark" not in s or id(s) in keep]
    return out


# ---------------------------------------------------------------- 카드
# 검증 방식(사용자 지시): 자연어 JSON만으로는 검수 불가 — 지시가 어떤
# 목적지·어떤 경로를 위한 것인지 탑뷰에 표시한 카드를 형태별로 저장.

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text_bar(width, lines):
    rows = []
    per_line = max(24, width // 10)  # 0.55 스케일 글자폭 ≈ 10px — 폭 기준 줄바꿈
    for t in lines:
        rows += textwrap.wrap(t, per_line) or [""]
    bar = np.zeros((24 * len(rows) + 12, width, 3), dtype=np.uint8)
    for i, ln in enumerate(rows):
        cv2.putText(bar, ln, (8, 22 + 24 * i), FONT, 0.55,
                    (255, 255, 255), 1)
    return bar


def _goal_marker(img, w2p, g):
    u, v = w2p(g["x"], g["z"])
    cv2.circle(img, (u, v), 14, (0, 0, 255), 3)
    cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
    cv2.putText(img, f'GOAL: {g["name"]}', (u + 16, v - 10), FONT, 0.6,
                (0, 0, 255), 2)


def _room_box(img, w2p, geom, rid):
    ri = next((i for i, r in enumerate(geom.rooms) if r["id"] == rid), None)
    if ri is None:
        return
    zz, xx = np.where(geom.room_grid == ri)
    if not len(zz):
        return
    u0, v0 = w2p(*geom.to_world(int(zz.min()), int(xx.min())))
    u1, v1 = w2p(*geom.to_world(int(zz.max()), int(xx.max())))
    cv2.rectangle(img, (min(u0, u1), min(v0, v1)),
                  (max(u0, u1), max(v0, v1)), (0, 200, 255), 2)


def _finish_card(panels_imgs, lines, out_png, scale=0.55):
    hmax = max(i.shape[0] for i in panels_imgs)
    imgs = [cv2.copyMakeBorder(i, 0, hmax - i.shape[0], 0, 8,
                               cv2.BORDER_CONSTANT) for i in panels_imgs]
    img = cv2.resize(np.concatenate(imgs, axis=1), None, fx=scale, fy=scale)
    cv2.imwrite(out_png, np.concatenate(
        [_text_bar(img.shape[1], lines), img], axis=0))


def card_goal_form(floors, bdir, g, rec, out_png):
    """목표형: goal 층 탑뷰 + 방 박스 + 목표 마커."""
    geom = floors[g["floor"]]
    img, w2p = common.sim_view(geom, bdir)
    cv2.putText(img, f'F{g["floor"]}', (8, 30), FONT, 1.0,
                (255, 255, 255), 2)
    _room_box(img, w2p, geom, g["rid"])
    _goal_marker(img, w2p, g)
    _finish_card([img], [
        f'[{rec["form"]}] robot={rec["robot"]}  '
        + ("PASS" if rec["pass"] else "FAIL"),
        f'goal: {g["name"]} @ {g["rtype"]} (floor {g["floor"]})'
        f'  [GT id: {g["gid"]}]',
        f'instruction: {rec["instruction"]}'], out_png)


def card_r2r_form(floors, bdir, g, res, seq, rec, robot, start, out_png):
    """R2R형: 경로 층 패널 + 궤적 + 랜드마크 번호 + 전환 + START/GOAL."""
    fls = sorted({fl for fl, _ in res["segments"]})
    panels = {}
    for fl in fls:
        img, w2p = common.sim_view(floors[fl], bdir)
        cv2.putText(img, f"F{fl}", (8, 30), FONT, 1.0, (255, 255, 255), 2)
        panels[fl] = (img, w2p)
    for fl, cells in res["segments"]:
        img, w2p = panels[fl]
        sm = common.natural_path(floors[fl], robot, cells)
        pts = common.cells_to_px(floors[fl], sm, w2p)
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False,
                      (200, 60, 200), 3, cv2.LINE_AA)
        # 중간 waypoint 점 — 경로 호길이 1.5m 간격(사용자 요청)
        wpts = [floors[fl].to_world(iz, ix) for iz, ix in sm]
        acc = 0.0
        for a, b in zip(wpts[:-1], wpts[1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            while acc + seg >= 1.5:
                f = (1.5 - acc) / max(seg, 1e-9)
                wx, wz = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
                cv2.circle(img, w2p(wx, wz), 5, (255, 255, 255), -1)
                cv2.circle(img, w2p(wx, wz), 5, (0, 0, 0), 1)
                a, seg, acc = (wx, wz), seg - (1.5 - acc), 0.0
            acc += seg
    for fl in fls:
        img, w2p = panels[fl]
        common.draw_transitions(img, floors[fl], w2p,
                                res.get("transitions", []), (200, 60, 200))
        common.draw_climb(img, floors[fl], w2p, res.get("transitions", []),
                          (200, 60, 200))
    sfl = start[0]
    u, v = panels[sfl][1](*floors[sfl].to_world(*start[1:]))
    cv2.drawMarker(panels[sfl][0], (u, v), (255, 255, 255),
                   cv2.MARKER_TILTED_CROSS, 22, 3)
    cv2.putText(panels[sfl][0], "START", (u + 12, v - 10), FONT, 0.6,
                (255, 255, 255), 2)
    for k, sx in enumerate(seq):
        if "landmark" not in sx:
            continue
        img, w2p = panels[sx["floor"]]
        u, v = w2p(sx["x"], sx["z"])
        cv2.circle(img, (u, v), 11, (60, 200, 230), 2)
        cv2.putText(img, str(k), (u - 5, v + 5), FONT, 0.45,
                    (60, 200, 230), 2)
    _goal_marker(*panels[g["floor"]], g)
    seq_txt = " > ".join(
        (f'{k}:{sx["landmark"]}({sx["side"][0].upper()})'
         if "landmark" in sx else f'[{sx["event"]}->F{sx["to_floor"]}]')
        for k, sx in enumerate(seq))
    _finish_card([panels[fl][0] for fl in fls], [
        f'[r2r] robot={rec["robot"]} transitions={rec["transitions"]}  '
        + ("PASS" if rec["pass"] else "FAIL") + f'  lcs={rec["lcs"]}',
        f'goal: {g["name"]} @ {g["rtype"]} (floor {g["floor"]})'
        f'  [GT id: {g["gid"]}]',
        f'sequence: {seq_txt}',
        f'instruction: {rec["instruction"]}'], out_png)


def card_combined_form(floors, bdir, g, tm, k, rec, out_png):
    """결합형: 전 층 패널 + 기억 노드(초록, 프리픽스) + 미탐색 goal 마커."""
    panels = {}
    for no in sorted(floors):
        img, w2p = common.sim_view(floors[no], bdir)
        cv2.putText(img, f"F{no}", (8, 30), FONT, 1.0, (255, 255, 255), 2)
        panels[no] = (img, w2p)
    for nd in tm["nodes"][:k]:
        img, w2p = panels[nd["floor"]]
        cv2.circle(img, w2p(nd["x"], nd["z"]), 5, (80, 220, 80), -1)
    # goal을 관측한 기억 노드 강조 + 관측 시선 연결선(검수용: "이 노드
    # 스냅샷에 goal이 잡혀 있음")
    ond = next(nd for nd in tm["nodes"]
               if nd["id"] == rec["observed_from_node"])
    img, w2p = panels[ond["floor"]]
    cv2.circle(img, w2p(ond["x"], ond["z"]), 12, (0, 255, 255), 3)
    cv2.line(img, w2p(ond["x"], ond["z"]), w2p(g["x"], g["z"]),
             (0, 255, 255), 2, cv2.LINE_AA)
    _goal_marker(*panels[g["floor"]], g)
    _finish_card([panels[no][0] for no in sorted(floors)], [
        f'[combined] robot={rec["robot"]} memory=prefix50 '
        f'(green nodes = known map)  ' + ("PASS" if rec["pass"] else "FAIL"),
        f'goal (seen from memory node n{ond["id"]}, yellow): '
        f'{g["name"]} @ {g["rtype"]} (floor {g["floor"]})'
        f'  [GT id: {g["gid"]}]',
        f'instruction: {rec["instruction"]}'], out_png, scale=0.4)


# ---------------------------------------------------------------- LLM

def llm_json(client, model, system, user):
    """JSON 모드 호출 → dict. 실패 시 1회 재시도."""
    for _ in range(2):
        r = client.chat.completions.create(
            model=model, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        try:
            return json.loads(r.choices[0].message.content)
        except json.JSONDecodeError:
            continue
    return {}


GEN_GOAL_SYS = (
    "You write one concise English navigation instruction for a robot in a "
    "multi-floor building, like a person casually asking it to go somewhere. "
    "Mention the goal object and enough location context (room type, floor) "
    "to disambiguate. Room numbers do not exist in this building — never "
    "invent one. Do not describe a route. Return JSON "
    '{"instruction": "..."}')

GEN_R2R_SYS = (
    "You write one English route-following instruction for a robot, in the "
    "style of the R2R dataset: describe the route step by step using the "
    "given ordered landmarks/events, ending at the goal. Use floor-change "
    "events exactly as given (stairs vs elevator matters). Room numbers do "
    "not exist in this building — never invent one. Also split it "
    "into sub-instructions aligned to the sequence. Return JSON "
    '{"instruction": "...", "sub_instructions": [{"text": "...", '
    '"seq_from": int, "seq_to": int}]} where seq indices refer to the given '
    "sequence items (0-based, inclusive).")

GEN_COMBO_SYS = (
    "You write one concise English instruction for a robot that partially "
    "explored a building. During that exploration it observed the goal "
    "object (it is in the robot's memory). Write a goal-directed "
    "instruction sending it back to that object; you may use hints from "
    "its known map (e.g. which floor transition it used), but do not "
    "invent landmarks that are not given. Room numbers do not exist — "
    "never mention one. Return JSON "
    '{"instruction": "..."}')

# 어려운 표현 3종 — goal과 정답 상태는 그대로 두고 문장만 까다롭게 만든다.
# 거짓 전제·능력 불일치는 정답 상태 자체가 달라지므로 여기 넣지 않고
# 기존 notfound·instr_conflict 평가 세트가 담당한다.
# goal은 정의상 그 층에서 유일하므로 "두 번째 X" 같은 서수는 참일 수 없다
# (ordinal 종류를 넣었다가 재생성해도 같은 거짓 서수가 반복돼 폐기).
# 대신 간접 지시(relational)를 난이도 축으로 쓴다.
HARD_KINDS = ("negation", "relational", "conditional")
GEN_RETRY = 2   # 서수 모순·역파싱 불일치 시 재생성 횟수

GEN_HARD_SYS = {
    "negation": "Phrase it with a negation or exclusion that still points to "
                "the same single goal (e.g. tell it which similar place NOT "
                "to go to first).",
    "relational": "Refer to the goal indirectly through one of the given "
                  "other_objects_in_room (e.g. the thing right next to it) "
                  "instead of stating the goal name first.",
    "conditional": "Phrase it with a conditional clause that does not change "
                   "the goal (e.g. what to do if a route is unavailable).",
}

# 층 서수("second floor", "second-floor")는 정상이므로 제외
ORDINAL_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|last|1st|2nd|3rd|4th|5th)\b"
    r"(?![-\s]*floor)", re.I)


def ordinal_conflict(instr, goal, registry):
    """서수 표현이 실제 개수와 어긋나는지 — 어긋나면 사유 문자열.

    "지나가다 만나는 두 번째 X"처럼 경로에 따라 달라지는 서수는, 같은 이름
    대상이 그 층에 하나뿐이면 지시가 GT와 다른 것을 가리키게 된다. 층 서수
    ("second floor")는 정상이라 제외한다. 눈으로는 잡기 어려워 수치로 검출.
    """
    m = ORDINAL_RE.search(instr or "")
    if not m:
        return None
    same = sum(1 for x in registry
               if x["name"] == goal["name"] and x["floor"] == goal["floor"])
    if same <= 1:
        return f"'{m.group(0)}' but only {same} '{goal['name']}' on this floor"
    return None


PARSE_GOAL_SYS = (
    "You are given a navigation instruction and a numbered list of candidate "
    "goal objects. Pick the single candidate the instruction refers to. "
    'Return JSON {"choice": <number>}')

PARSE_R2R_SYS = (
    "You are given a route instruction and a vocabulary of landmark names "
    "and floor-change events. Extract the ordered sequence the instruction "
    "follows, using only vocabulary items. Return JSON "
    '{"sequence": ["item", ...]} in the order mentioned.')


def lcs_ratio(gt, pred):
    """순서 일치율 = LCS(gt, pred) / len(gt)."""
    n, m = len(gt), len(pred)
    if n == 0:
        return 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            dp[i + 1][j + 1] = (dp[i][j] + 1 if gt[i] == pred[j]
                                else max(dp[i][j + 1], dp[i + 1][j]))
    return dp[n][m] / n


# ---------------------------------------------------------------- 파일럿

def observed_gids(floors, registry, nodes, cam_h):
    """기억 노드에서 실제로 관측된 목표 후보 {gid: 최초 관측 노드 id}.

    판정 = FOV 90°·10m·LOS(top 맵). 목표형·R2R·결합형 goal은 모두 이 결과에서
    뽑는다 — 기억 스냅샷에 없는 대상을 goal로 두면 스냅샷 랭킹 grounding이
    성립하지 않고, 커버리지가 낮아진 만큼 미탐색 오분류로 샌다.
    """
    torch, dev = common.torch_dev()
    out = {}
    for fl in sorted(floors):
        geom = floors[fl]
        objs = [g for g in registry if g["floor"] == fl]
        here = [nd for nd in nodes if nd["floor"] == fl]
        if not objs or not here:
            continue
        cells = [geom.to_idx(g["x"], g["z"]) for g in objs]
        block_t = torch.from_numpy(common.sight_block(geom, cam_h)).to(dev)
        for nd in here:
            for hd in nd["headings"]:
                vis = common.gates_in_view(
                    block_t, geom.to_idx(nd["x"], nd["z"]),
                    math.radians(hd), cells)
                for g, v in zip(objs, vis):
                    if v:
                        out.setdefault(g["gid"], nd["id"])
    return out


def sample_goals(registry, rng, n, floor=None, unique_only=True,
                 observed=None):
    """목표 후보 표집. observed를 주면 기억에 관측된 대상으로 한정."""
    cand = [g for g in registry
            if (floor is None or g["floor"] == floor)
            and (g["unique"] or not unique_only)
            and (observed is None or g["gid"] in observed)]
    if not cand:
        return []
    idx = rng.choice(len(cand), size=min(n, len(cand)), replace=False)
    return [cand[i] for i in idx]


def start_cell(floors, robot, rng, fl):
    ok = floors[fl].passable(robot["w_eff"], robot["h"])
    zz, xx = np.where(ok)
    k = rng.integers(len(zz))
    return (fl, int(zz[k]), int(xx[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", required=True)
    ap.add_argument("--topomap", required=True,
                    help="03_topomap 출력 폴더 (결합형 부분 기억)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-goal", type=int, default=3)
    ap.add_argument("--n-r2r", type=int, default=1,
                    help="R2R 갈림 쌍 수 (쌍당 wheeled+multileg 2건)")
    ap.add_argument("--n-combo", type=int, default=3)
    ap.add_argument("--hard-ratio", type=float, default=0.2,
                    help="어려운 표현 비율 (plan 확정 0.2)")
    ap.add_argument("--gen-model", default=GEN_MODEL)
    ap.add_argument("--parse-model", default=PARSE_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from openai import OpenAI
    client = OpenAI()  # OPENAI_API_KEY 환경변수
    rng = np.random.default_rng(args.seed)

    floors = common.load_building(args.building)
    nav = common.BuildingNav(floors)
    robots = common.robot_pool()
    wh = sorted([r for r in robots if r["cls"] == "wheeled"],
                key=lambda r: r["w_eff"])[0]
    ml = sorted([r for r in robots if r["cls"] == "multileg"],
                key=lambda r: r["w_eff"])
    ml = ml[len(ml) // 2]
    registry = goal_registry(floors)
    print(f"목표 후보 {len(registry)}개 (층별:",
          {no: sum(1 for g in registry if g['floor'] == no)
           for no in sorted(floors)}, ")")

    # 기억에 실제로 잡힌 목표만 goal이 된다 (03 스캔 제거로 커버리지가 줄어든
    # 만큼 여기서 걸러진다). 미관측 대상은 notfound 평가 세트 전용.
    tms = {}
    for r in (wh, ml):
        p = os.path.join(args.topomap,
                         common.out_name(args.building, r["id"] + ".json"))
        tms[r["id"]] = json.load(open(p))
    obs = {r["id"]: observed_gids(floors, registry, tms[r["id"]]["nodes"],
                                  r["cam_h"]) for r in (wh, ml)}
    for r in (wh, ml):
        print(f"  {r['id']} 관측 목표 {len(obs[r['id']])}/{len(registry)}개")
    obs_both = obs[wh["id"]].keys() & obs[ml["id"]].keys()
    print(f"  두 로봇 공통 관측 {len(obs_both)}개 (R2R 갈림 쌍용)")

    records, n_pass = [], 0

    def check(rec):
        nonlocal n_pass
        records.append(rec)
        n_pass += bool(rec["pass"])
        tag = f"[{rec['form']}" + (f"/{rec['hard']}]" if rec.get("hard")
                                   else "]")
        why = f" CONFLICT({rec['conflict']})" if rec.get("conflict") else ""
        print(f"  {tag} {'PASS' if rec['pass'] else 'FAIL'}{why} "
              f"goal={rec['goal']['gid']}\n    지시: {rec['instruction']}")

    # ---- 목표형 -------------------------------------------------------
    # HARD_RATIO 만큼은 어려운 표현으로 생성 (goal·정답 상태는 동일)
    print("\n== 목표형")
    goals = sample_goals(registry, rng, args.n_goal, observed=obs[wh["id"]])
    n_hard = int(round(len(goals) * args.hard_ratio))
    hard_of = {i: HARD_KINDS[i % len(HARD_KINDS)]
               for i in range(len(goals) - n_hard, len(goals))}
    for i, g in enumerate(goals):
        ctx = [x["name"] for x in registry
               if x["rid"] == g["rid"] and x["gid"] != g["gid"]][:5]
        kind = hard_of.get(i)
        sys_p = GEN_GOAL_SYS + (" " + GEN_HARD_SYS[kind] if kind else "")
        # 역파싱: 정답 + 방해 후보 9개
        idx = rng.choice(len(registry), size=9, replace=False)
        cands = [registry[i] for i in idx if registry[i]["gid"] != g["gid"]][:9]
        cands.append(g)
        rng.shuffle(cands)
        gt_i = next(i for i, c in enumerate(cands) if c["gid"] == g["gid"])
        # 서수 모순은 폐기·재생성 (역파싱 불일치와 같은 처리)
        for attempt in range(GEN_RETRY + 1):
            out = llm_json(client, args.gen_model, sys_p, json.dumps(
                {"goal": {"object": g["name"], "room_type": g["rtype"],
                          "floor": g["floor"]},
                 "other_objects_in_room": ctx}))
            instr = out.get("instruction", "")
            conf = ordinal_conflict(instr, g, registry)
            p = llm_json(client, args.parse_model, PARSE_GOAL_SYS, json.dumps(
                {"instruction": instr,
                 "candidates": [{"number": i, "object": c["name"],
                                 "room": c["rtype"], "floor": c["floor"]}
                                for i, c in enumerate(cands)]}))
            if conf is None and p.get("choice") == gt_i:
                break
            if attempt < GEN_RETRY:
                print(f"    재생성 {attempt + 1}/{GEN_RETRY} "
                      f"({conf or '역파싱 불일치'})")
        check({"form": "goal", "goal": g, "robot": wh["id"],
               "hard": kind, "instruction": instr, "parse": p,
               "conflict": conf, "retries": attempt,
               "pass": p.get("choice") == gt_i and conf is None})
        card_goal_form(floors, args.building, g, records[-1],
                       os.path.join(args.out, common.out_name(
                           args.building, f"card_goal_{i}.png")))

    # ---- R2R형: 같은 (시작,목표), wheeled vs legged ------------------
    # 갈림 쌍이 성립하려면 goal이 두 로봇 모두 도달 가능해야 함 — 접근/계획
    # 실패 시 goal 재샘플 (multileg는 좁은 문 방 다수가 불가라 스킵 잦음)
    print("\n== R2R형 (갈림 확인)")
    pairs, tried = [], set()
    for _ in range(40):
        if len(pairs) >= args.n_r2r:
            break
        fl_g = int(rng.choice(sorted(floors)[1:]))  # 층 이동 강제
        gs = sample_goals(registry, rng, 1, floor=fl_g, observed=obs_both)
        if not gs:
            continue
        g = gs[0]
        if g["gid"] in tried:
            continue
        tried.add(g["gid"])
        s = start_cell(floors, wh, rng, min(floors))
        legs = []
        for robot in (wh, ml):
            ap_cell = approach_cell(floors[g["floor"]], robot, g["x"], g["z"])
            res = (nav.plan(robot, s, (g["floor"], *ap_cell))
                   if ap_cell else {})
            if not res.get("success"):
                break
            legs.append((robot, res))
        if len(legs) == 2:
            pairs.append((g, s, legs))
    print(f"  성립 쌍 {len(pairs)}/{args.n_r2r}"
          f" (시도 {len(tried)} goal)")
    flat = [(pi, g, s, robot, res)
            for pi, (g, s, legs) in enumerate(pairs)
            for robot, res in legs]
    for pi, g, s, robot, res in flat:
        seq = route_landmarks(floors, registry, res, g)
        seq_full = seq + [{"goal": g["name"], "room": g["rtype"]}]
        # LLM에는 좌표 필드 제외(카드 렌더 전용)
        seq_llm = [{k: v for k, v in sx.items()
                    if k not in ("x", "z", "floor")} for sx in seq_full]
        out = llm_json(client, args.gen_model, GEN_R2R_SYS,
                       json.dumps({"sequence": seq_llm}))
        instr = out.get("instruction", "")
        vocab = sorted({x["landmark"] for x in seq if "landmark" in x}
                       | {"stairs", "elevator", g["name"]})
        p = llm_json(client, args.parse_model, PARSE_R2R_SYS, json.dumps(
            {"instruction": instr, "vocabulary": vocab}))
        gt_seq = [x.get("landmark") or x.get("event") for x in seq]
        gt_seq = [("stairs" if x == "stairs" else
                   "elevator" if x == "elevator" else x) for x in gt_seq]
        ratio = lcs_ratio(gt_seq, [str(x) for x in p.get("sequence", [])])
        modes = [t["mode"] for t in res.get("transitions", [])]
        check({"form": "r2r", "goal": g, "robot": robot["id"],
               "transitions": modes, "sequence": seq_full,
               "instruction": instr,
               "sub_instructions": out.get("sub_instructions", []),
               "parse": p, "lcs": round(ratio, 2), "pass": ratio >= 0.6})
        card_r2r_form(floors, args.building, g, res, seq, records[-1],
                      robot, s, os.path.join(
                          args.out, common.out_name(
                              args.building,
                              f"card_r2r_{pi}_{robot['cls']}.png")))

    # ---- 결합형: 03 프리픽스 부분 기억 + 미탐색 목적지 ----------------
    print("\n== 결합형")
    tm = tms[wh["id"]]
    k = tm["prefix_cuts"]["0.5"]
    t_cut = (tm["nodes"][k]["t"] if k < len(tm["nodes"]) else float("inf"))
    known_rooms, known_modes = set(), set()
    for nd in tm["nodes"][:k]:
        geom = floors[nd["floor"]]
        iz, ix = geom.to_idx(nd["x"], nd["z"])
        ri = geom.room_grid[min(iz, geom.nz - 1), min(ix, geom.nx - 1)]
        if ri >= 0:
            known_rooms.add((nd["floor"], geom.rooms[ri]["id"]))
    for e in tm["edges"]:
        if e["mode"] != "walk" and e["t"] < t_cut \
                and e["a"] < k and e["b"] < k:
            known_modes.add(e["mode"])
    # 알려진 방 요약(방 번호 제거, 종류별 개수)
    from collections import Counter as _Counter
    kc = _Counter(
        (no, re.sub(r"\s*\d+$", "", pretty_room(rid, no)[0]))
        for no, rid in known_rooms)
    known = sorted(f"{rt} (floor {no})" + (f" x{c}" if c > 1 else "")
                   for (no, rt), c in kc.items())
    # goal = 프리픽스 주행 동안 실제 관측된 오브젝트만 — 목표형·R2R과 같은
    # 판정을 프리픽스 노드에만 적용한다.
    obs_from = observed_gids(floors, [g for g in registry if g["unique"]],
                             tm["nodes"][:k], wh["cam_h"])
    combo_pool = [g for g in registry if g["gid"] in obs_from]
    print(f"  프리픽스 관측 오브젝트 {len(combo_pool)}개 "
          f"(유일 후보 {sum(1 for g in registry if g['unique'])}개 중)")
    idxs = rng.choice(len(combo_pool),
                      size=min(args.n_combo, len(combo_pool)), replace=False)
    for i, g in enumerate(combo_pool[j] for j in idxs):
        out = llm_json(client, args.gen_model, GEN_COMBO_SYS, json.dumps(
            {"known_rooms": known,
             "known_floor_transitions": sorted(known_modes),
             "goal": {"object": g["name"], "room_type": g["rtype"],
                      "floor": g["floor"]}}))
        instr = out.get("instruction", "")
        idx = rng.choice(len(registry), size=9, replace=False)
        cands = [registry[i] for i in idx if registry[i]["gid"] != g["gid"]][:9]
        cands.append(g)
        rng.shuffle(cands)
        gt_i = next(i for i, c in enumerate(cands) if c["gid"] == g["gid"])
        p = llm_json(client, args.parse_model, PARSE_GOAL_SYS, json.dumps(
            {"instruction": instr,
             "candidates": [{"number": i, "object": c["name"],
                             "room": c["rtype"], "floor": c["floor"]}
                            for i, c in enumerate(cands)]}))
        check({"form": "combined", "goal": g, "robot": wh["id"],
               "memory_prefix": "0.5", "known_rooms_n": len(known),
               "observed_from_node": obs_from[g["gid"]],
               "instruction": instr, "parse": p,
               "pass": p.get("choice") == gt_i})
        card_combined_form(floors, args.building, g, tm, k, records[-1],
                           os.path.join(args.out, common.out_name(
                           args.building, f"card_combined_{i}.png")))

    with open(os.path.join(args.out, common.out_name(
            args.building, "instructions.json")), "w") as fh:
        json.dump({"gen_model": args.gen_model,
                   "parse_model": args.parse_model,
                   "hard_ratio": args.hard_ratio,
                   "n": len(records), "n_pass": n_pass,
                   "records": records}, fh, ensure_ascii=False, indent=1)
    # 종류별 집계 — 어려운 표현이 GT와 어긋나는 빈도를 수치로 본다
    print("\n검증 집계 (통과 = 역파싱 정답 + 서수 모순 없음)")
    for form in ("goal", "r2r", "combined"):
        rs = [r for r in records if r["form"] == form]
        if rs:
            print(f"  {form:9s} {sum(1 for r in rs if r['pass'])}/{len(rs)}")
    hard = [r for r in records if r.get("hard")]
    if hard:
        print("  어려운 표현:")
        for k in HARD_KINDS:
            rs = [r for r in hard if r["hard"] == k]
            if rs:
                nc = sum(1 for r in rs if r.get("conflict"))
                print(f"    {k:12s} {sum(1 for r in rs if r['pass'])}/{len(rs)}"
                      f" 통과, 서수 모순 {nc}건")
    print(f"\n역파싱 통과 {n_pass}/{len(records)}  저장: "
          f"{args.out}/instructions.json")


if __name__ == "__main__":
    main()

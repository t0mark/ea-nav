"""MANSION 시뮬 벤치마크 — 감독자·실행자 구조.

실패 처리는 코드 곳곳의 try/except가 아니라 프로세스 수명주기 하나로 다룬다:

  --role exec  실행자. THOR와 상호작용하는 단순 선형 코드 — 복구 없음,
               어떤 실패(웨지·크래시)든 그대로 죽는다. 항목마다 장부
               (mansion_sim_{tag}.items.jsonl)에 즉시 기록, 시작 시 장부의
               완료 항목은 건너뜀(재개).
  --role super 감독자(기본). THOR 무접촉. 작은 유닛(에피소드 8개 샤드)
               큐를 GPU 슬롯에 배분하고 실행자를 시차 기동(STAGGER_S,
               동시 에셋 로드 레이스 회피). 로그 심박이 STALL_S 끊기거나
               비정상 종료면 프로세스 그룹 kill 후 같은 유닛 재큐잉
               (최대 MAX_RETRY). 전 유닛 완료 시 집계 실행.
  --role agg   집계. 장부(항목 단위) + 레거시 완료 샤드 json(항목 장부
               이전 형식, 집계 단위)을 n-가중 병합해 최종 보고.
               태그 파일명은 경계 매칭 — 접두사 충돌(hotel ⊂ elevonly
               변형) 방지.

주행 = 홉 단위 전역 선택 루프(HopDriver, scripts/models/mansion_adapter).
홉마다 관측→제안기 히트맵→월드 리프트→그래프 갱신→전역 선택 1회이고,
행선까지의 저수준 이동과 층 전환 스킬 구간에서는 선택을 호출하지 않는다.
규칙 다익스트라는 실행에서 빠지고 상태 판정 전용으로만 남는다(judge_state).

변형(--variant):
  learned  학습 전역 선택기 — 기본 실행 경로
  rule     규칙 다익스트라 전역 — ablation 베이스라인. 저수준 이동·스킬은
           learned와 같은 것을 써서 두 팔의 차이를 전역 선택 하나로 좁힌다

판정 체계: 상태 축 = 불가능(미탐색: goal GOAL_NODE_M 내 기억 노드
없음)/불가능(embodiment: 게이트 차단 단절) ↔ GT 대조(state_axis).
실행 축 = 도착(전역 STOP + 반경 ARRIVE_M — 벤치 SR과 동일값)/
arrival_error(STOP했으나 반경 밖)/stalled(연속 홉 실패)/budget.
BUDGET=지표 비관여 가드. τ=3.0 고정. LiDAR 가정 로컬 회피는 제어 층 전용.

사용:
  python mansion.py --envs <tag ...>            # 감독자
  python mansion.py --role agg                  # 집계만
결과: {out}/mansion_sim_{tag}.items.jsonl(장부) + mansion_sim_report.json
"""
import argparse
import glob
import json
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "models"))
sys.path.insert(0, _HERE)

REGIME = {"public_hotel_dormitory_4f_300_fp001": "stairs+elev",
          "office_corporate_hq_6f_300_fp001": "stairs+elev",
          "residential_duplex_townhouse_3f_150_fp001": "stairs_only",
          "public_hotel_dormitory_4f_300_fp001_custom_elevonly":
              "elev_only"}
ALL_TAGS = list(REGIME)
# BUDGET: 러너 폭주 방지 가드(지표 비관여 — 실패 판정은 stalled가 담당)
BUDGET = 2000
# EP_PER_UNIT: 실행 유닛 크기 기본값(에피소드 수) — 작을수록 재큐잉 손실이
# 작고 병렬 폭이 넓어지는 대신 실행자 초기화 횟수가 늘어난다 (--ep-per-unit)
EP_PER_UNIT = 8
# STALL_S: 실행자 심박 정체 한계(초) — 초기화 소요(~5분)보다 길게
STALL_S = 420
# STAGGER_S: 실행자 시차 기동 간격(초)
STAGGER_S = 20
MAX_RETRY = 3
RR_DIR = "/data/URDF/real_robots"
# INSTR_LEN: 지시 토큰 길이, GOAL_NODE_M: goal 인근 기억 노드 탐색 반경
# (이 반경 안에 기억 노드가 없으면 불가능-미탐색)
INSTR_LEN = 80
GOAL_NODE_M = 4.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def robot_ids():
    return sorted(os.path.basename(os.path.dirname(p)) for p in
                  glob.glob(os.path.join(RR_DIR, "*", "*", "robot.urdf")))


def ledger_files(out, tag):
    """태그 경계 매칭 장부 목록 — 접두사 충돌 방지."""
    outs = []
    for f in glob.glob(os.path.join(out, f"mansion_sim_{tag}*.items.jsonl")):
        rest = os.path.basename(f)[len(f"mansion_sim_{tag}"):]
        if rest == ".items.jsonl" or rest.startswith("_shard"):
            outs.append(f)
    return outs


def ledger_rows(out, tag):
    """장부 파싱 — (ep,rid,변형) 중복은 첫 기록 우선, 잘린 줄 무시.

    중복 키에 변형이 들어가야 한 장부에 두 팔을 쌓을 수 있다. 변형을 빼면
    먼저 쓰인 팔이 다른 팔의 같은 (ep,rid)를 가려 집계에서 통째로 사라진다.
    """
    seen, rows = set(), []
    for f in sorted(ledger_files(out, tag)):
        for ln in open(f):
            try:
                it = json.loads(ln)
            except json.JSONDecodeError:
                continue
            k = (it["ep"], it["rid"], it.get("variant", "rule_legacy"))
            if k in seen:
                continue
            seen.add(k)
            rows.append(it)
    return rows


# ------------------------------------------------------------- 주행 부품

def goal_instructions(tag, ds):
    """에피소드 id → 목표형 지시 텍스트.

    전역 선택기는 지시 조건부다 — 지시가 없으면 무지시 토큰으로 주행한다
    (학습에서 지시 없는 프레임을 같은 토큰으로 다룬 것과 같은 규약).
    """
    p = os.path.join(ds, f"{tag}_instructions.json")
    if not os.path.exists(p):
        return {}
    out = {}
    for r in json.load(open(p)).get("records", []):
        if r.get("form") == "goal" and r.get("pass") and r.get("episode"):
            out.setdefault(r["episode"], r["instruction"])
    return out


def encode_instruction(policy, tok, text, device):
    """지시 → (txt_embeds, masks). null_instruction과 같은 자리에 들어간다."""
    import torch
    enc = tok(text, padding="max_length", truncation=True,
              max_length=INSTR_LEN, return_tensors="pt")
    ids, msk = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    with torch.no_grad():
        return policy("language", txt_ids=ids, txt_masks=msk), msk


def load_stage(policy, lp, args):
    """홉 루프 학습분 로드 — 제안기(wp)·전역 선택기(nav).

    학습 4단계 완료 전에는 두 ckpt가 없다. 그때는 무작위 초기값으로
    파이프라인 자체만 돌 수 있게 경고만 남긴다(산출 수치는 무효).
    """
    import torch
    for path, mod, key in ((args.ckpt02, policy.wp, "wp"),
                           (args.ckpt03, policy.nav, "nav")):
        if not path or not os.path.exists(path):
            log(f"경고: {key} ckpt 없음({path}) — 무작위 초기값이므로 "
                f"수치 무효(파이프라인 스모크 전용)")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        lp.load_compat(mod, ck[key] if key in ck else ck)
        log(f"로드: {key} ← {os.path.basename(path)}")


def save_viz(mc, ma, tag, env, mem, path_nodes, gtr, e, rid, reached,
             reason, out_dir):
    """탑뷰 궤적 카드 — GT(초록)/계획(하늘, 규칙 팔만)/실주행(빨강) +
    기억 그래프 + 범례. 파일명 = {tag}_{episode}_{robot}.png."""
    import cv2
    bld, _var = ma.building_of(tag)
    ref = ([tuple(q) for q in gtr["ref"]]
           if gtr.get("ref") and len(gtr["ref"]) >= 2 else [])
    fls = sorted({p[0] for p in env.trajectory}
                 | {p[0] for p in ref} | {env.goal[0]})
    panels = []
    for fl in fls:
        img, w2p = mc.sim_view(env.floors[fl], bld)
        for (u, v, _m, _L) in mem.edges:
            nu, nv = mem.nodes[u], mem.nodes[v]
            if nu["floor"] == fl and nv["floor"] == fl:
                cv2.line(img, w2p(nu["x"], nu["z"]),
                         w2p(nv["x"], nv["z"]), (150, 150, 150), 1)
        for n in mem.nodes:
            if n["floor"] == fl:
                col = (0, 140, 255) if n["kind"] == "gate" \
                    else (150, 150, 150)
                cv2.circle(img, w2p(n["x"], n["z"]), 3, col, -1)
        pts = [w2p(mem.nodes[u]["x"], mem.nodes[u]["z"])
               for u in (path_nodes or []) if mem.nodes[u]["floor"] == fl]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (255, 200, 0), 2)
        rp = [w2p(x, z) for (f, x, z) in ref if f == fl]
        for a, b in zip(rp[:-1], rp[1:]):
            cv2.line(img, a, b, (0, 200, 0), 2)
        tp = [w2p(x, z) for (f, x, z) in env.trajectory if f == fl]
        for a, b in zip(tp[:-1], tp[1:]):
            cv2.line(img, a, b, (0, 0, 255), 2)
        if env.trajectory and env.trajectory[0][0] == fl:
            cv2.circle(img, w2p(env.trajectory[0][1],
                                env.trajectory[0][2]), 8, (255, 0, 0), -1)
        if env.goal[0] == fl:
            cv2.circle(img, w2p(env.goal[1], env.goal[2]),
                       10, (0, 255, 255), 3)
        img = cv2.flip(img, 0)
        cv2.putText(img, f"F{fl}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
        panels.append(img)
    H = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, H - p.shape[0], 0, 4,
                                 cv2.BORDER_CONSTANT) for p in panels]
    card = cv2.hconcat(panels)
    bar = np.zeros((64, card.shape[1], 3), np.uint8)
    cv2.putText(bar, f"{e['id']} {rid} SR={int(bool(reached))} "
                f"reason={reason}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    legend = [((0, 200, 0), "GT expert"), ((255, 200, 0), "plan(rule)"),
              ((0, 0, 255), "agent"), ((150, 150, 150), "memory"),
              ((0, 140, 255), "gate node"), ((255, 0, 0), "start"),
              ((0, 255, 255), "goal")]
    x = 8
    for col, name in legend:
        cv2.line(bar, (x, 48), (x + 22, 48), col, 3)
        cv2.putText(bar, name, (x + 27, 53), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)
        x += 27 + 9 * len(name) + 18
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{tag}_{e['id']}_{rid}.png"),
                cv2.vconcat([bar, card]))


# ---------------------------------------------------------------- 실행자

def run_exec(args):
    import common
    common.pick_gpu(args.gpu)
    import pickle

    import torch

    import mansion_adapter as ma
    import vln_metrics as vm
    import weights as lp
    from EA_Nav import EANav, tokenizer
    from mansion_sim_env import MansionSimEnv
    from modules.URDFEncoder import collate_graphs, urdf_to_graph
    device = "cuda"
    tag = args.envs[0]

    log("초기화: 모델 로드 시작")
    policy = EANav(clip_device=device).to(device)
    policy.load_pretrained(finetuned=True, verbose=False)
    common.load_into(args.ckpt01, urdf_enc=policy.urdf_enc, g=policy.g)
    load_stage(policy, lp, args)
    policy.eval()
    tok = tokenizer()

    mc = ma._mc()
    real = {r["id"]: r for r in mc.robot_pool(RR_DIR)}
    zs = {}
    for rid, r in real.items():
        gb = collate_graphs([urdf_to_graph(
            os.path.join(RR_DIR, r["cls"], rid, "robot.urdf"))])
        gb = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in gb.items()}
        with torch.no_grad():
            z, aux = policy.urdf_enc(gb)
        zs[rid] = (z, float(torch.sigmoid(aux["stairs_ok_logit"])[0]) > .5)

    log("초기화: 기억·게이트 준비")
    mem = ma.UnionMemory(tag)
    feats = ma.MemoryFeatures(policy, mem, device)
    bank = ma.GateBank(tag)
    eg_map = ma.edge_gate_map(mem, bank)
    env = MansionSimEnv(tag, gpu=args.gpu, tol_cells=1,
                        render=(args.variant == "learned") or args.render)
    ej = json.load(open(os.path.join(
        ma.DS, "episodes", f"{tag}_episodes.json")))
    with open(f"/data/EVLN_ckpt/benchmark/realgt_{tag}.pkl", "rb") as fh:
        realgt = pickle.load(fh)
    instr = goal_instructions(tag, ma.DS)
    log(f"초기화: 목표형 지시 {len(instr)}건")
    # 판정 입력(노드 단위 게이트 점수 + z-차등 엣지 차단) — 상태 축 전용
    bank_sc = {}
    for rid, r in real.items():
        log(f"초기화: 게이트 점수 {rid}")
        bank_sc[rid] = bank.scores(policy, zs[rid][0], r["cam_h"], device)
    eblock = {rid: ma.differential_block(bank_sc, eg_map, rid, args.tau)
              for rid in real}

    # 재개 키에 변형 포함 — 한 장부에 학습·규칙 두 팔을 쌓으려면 자기 팔의
    # 기록만 완료로 봐야 한다(빼면 다른 팔 기록을 자기 것으로 오인해 통째로
    # 건너뛴다 — 실측: 규칙 팔이 learned 363항목을 보고 0항목 실행)
    done = {(it["ep"], it["rid"]) for it in ledger_rows(args.out, tag)
            if it.get("variant", "rule_legacy") == args.variant}
    if done:
        log(f"재개: 완료 {len(done)}항목 건너뜀 (변형 {args.variant})")
    ledger = os.path.join(args.out, f"mansion_sim_{tag}.items.jsonl")
    os.makedirs(args.out, exist_ok=True)

    def persist(kind, ep_id, rid, **kw):
        with open(ledger, "a") as fh:
            fh.write(json.dumps({"kind": kind, "tag": tag, "ep": ep_id,
                                 "rid": rid, "variant": args.variant,
                                 **kw},
                                default=float) + "\n")

    rng = np.random.default_rng(7)
    vized = 0
    episodes = ej["episodes"][args.shard::args.nshards]
    log(f"유닛 {tag} 샤드 {args.shard}/{args.nshards}: "
        f"에피소드 {len(episodes)} (변형 {args.variant})")
    for e in episodes:
        st = e["start"]
        sx, sz = env.floors[st["floor"]].to_world(*st["cell"])
        gl = e["goal"]
        goal = (int(gl["floor"]), float(gl["x"]), float(gl["z"]))
        txt = None
        if instr.get(e["id"]):
            txt = encode_instruction(policy, tok, instr[e["id"]], device)
        for rid, r in real.items():
            if (e["id"], rid) in done:
                continue
            gtr = realgt[e["id"]][rid]
            expect_imp = not gtr["success"] or len(gtr["ref"]) < 2
            z, sok = zs[rid]
            su = mem.nearest(st["floor"], sx, sz)
            gu = mem.nodes_near(*goal, GOAL_NODE_M)
            if gu:
                gu = [min(gu, key=lambda i:
                          (mem.nodes[i]["x"] - goal[1]) ** 2
                          + (mem.nodes[i]["z"] - goal[2]) ** 2)]
            # 상태 축 = 판정 전용 경로. 실행(홉 루프)은 이 결과를 쓰지 않는다.
            # 차단은 z-차등 엣지 하나로 통일(judge_state 규약, eval.py 동일)
            jd = ma.judge_state(mem, su, gu, sok, tau=args.tau,
                                edge_block=eblock[rid])
            if jd["state"] != "ok":
                ok = bool(expect_imp)
                log(f"  [item] {e['id']} {rid} state=impossible_"
                    f"{jd['state']} gt_impossible={int(expect_imp)} "
                    f"-> {'정답' if ok else '오판'}")
                persist("state", e["id"], rid,
                        sys_state=jd["state"], ok=int(ok))
                continue
            env.reset(dict(w_eff=r["w_eff"], h=r["h"], cam_h=r["cam_h"],
                           stairs_ok=r["stairs_ok"]),
                      (st["floor"], sx, sz, float(rng.uniform(0, 360))),
                      goal, wait_steps=3 + int(rng.integers(4)))
            drv = ma.HopDriver(policy, device, mem, feats=feats,
                               tau=args.tau, stairs_ok=sok,
                               cam_h=r["cam_h"], budget=BUDGET, log=log)
            if args.variant == "rule":
                drv.start(seed=False)
                res = drv.follow(env, jd["route"]["path"][1:], goal)
            else:
                drv.start(z, txt)
                res = drv.run(env, goal)
            reached = res["state"] == "arrived"
            log(f"  [item] {e['id']} {rid} reach={int(reached)} "
                f"state={res['state']} reason={res['reason']} "
                f"hops={res['hops']} steps={env.steps} "
                f"fl={env.fl}→goal{goal[0]}")
            if args.viz and vized < args.viz:
                save_viz(mc, ma, tag, env, mem,
                         jd["route"]["path"] if args.variant == "rule"
                         else [], gtr, e, rid, reached, res["reason"],
                         args.viz_dir)
                vized += 1
            if expect_imp:
                persist("missed", e["id"], rid, reached=int(reached))
                continue
            ref = [tuple(q) for q in gtr["ref"]]
            m = vm.episode_metrics(env.trajectory, ref, goal)
            m["SR"] = float(reached)
            m["SPL"] = m["SR"] * m["L_ref"] / max(m["L_ref"], m["TL"])
            task = ("elevator" if "elevator" in gtr["modes"] else
                    "stairs" if "stairs" in gtr["modes"] else "same_floor")
            groups = dict(env=tag, regime=REGIME.get(tag, "?"), robot=rid,
                          task=task, variant=args.variant,
                          div="divergent" if e.get("divergent")
                          else "nondiv")
            # 실제로 쓴 층 전환 수단 — 폐루프 "경로 갈림"의 측정 근거.
            # 판정 모듈이 고른 수단이 아니라 선택기가 주행으로 밟은 결과다.
            used = []
            for ev_ in env.events:
                if ev_[0] == "board" and "elevator" not in used:
                    used.append("elevator")
                elif ev_[0] == "stairs_transit" and "stairs" not in used:
                    used.append("stairs")
            persist("sr", e["id"], rid, m=m, groups=groups,
                    reason=res["reason"], hops=res["hops"],
                    modes_used=used, modes_gt=list(gtr["modes"]))
    env.close()
    log("유닛 완료")


# ---------------------------------------------------------------- 감독자

def unit_expected(ej, shard, nshards, rids):
    return {(e["id"], rid) for e in ej["episodes"][shard::nshards]
            for rid in rids}


def sweep_orphan_thor():
    """고아 THOR 렌더러(PPID=1) 정리.

    렌더러는 자체 세션으로 분리돼 실행자 프로세스 그룹 kill에 잡히지
    않는다 — 부모(실행자)가 죽어 init에 재부모화된 렌더러를 회수한다.
    """
    out = subprocess.run(["ps", "-eo", "pid,ppid,args"],
                         capture_output=True, text=True).stdout
    for ln in out.splitlines():
        f = ln.split(None, 2)
        if len(f) == 3 and f[1] == "1" and "thor-CloudRendering" in f[2]:
            try:
                os.kill(int(f[0]), signal.SIGKILL)
            except OSError:
                pass


def run_super(args):
    import mansion_adapter as ma
    sweep_orphan_thor()
    rids = robot_ids()
    gpus = [int(g) for g in args.gpus.split(",")]
    units = []
    for tag in args.envs:
        ej = json.load(open(os.path.join(
            ma.DS, "episodes", f"{tag}_episodes.json")))
        nsh = max(1, math.ceil(len(ej["episodes"]) / args.ep_per_unit))
        done = {(it["ep"], it["rid"]) for it in ledger_rows(args.out, tag)
                if it.get("variant", "rule_legacy") == args.variant}
        for s in range(nsh):
            exp = unit_expected(ej, s, nsh, rids)
            if exp - done:
                units.append({"tag": tag, "shard": s, "nshards": nsh,
                              "retry": 0, "n_left": len(exp - done)})
    total_left = sum(u["n_left"] for u in units)
    base_n = sum(len(ledger_rows(args.out, t)) for t in set(args.envs))
    log(f"감독자: 유닛 {len(units)}개(잔여 {total_left}항목), "
        f"슬롯 {len(gpus)}×{args.workers_per_gpu}")
    # GPU 우선 인터리브 — 유닛이 슬롯보다 적어도 GPU에 고르게 분산
    slots = [{"gpu": g, "proc": None, "unit": None, "logf": None}
             for _ in range(args.workers_per_gpu) for g in gpus]
    queue = list(units)
    last_spawn = 0.0
    failed = []
    t0 = time.time()

    def spawn(slot, unit):
        lf = os.path.join(args.logdir,
                          f"bench_{unit['tag']}_u{unit['shard']}"
                          f"_try{unit['retry']}.log")
        cmd = [sys.executable, os.path.abspath(__file__), "--role", "exec",
               "--envs", unit["tag"], "--shard", str(unit["shard"]),
               "--nshards", str(unit["nshards"]),
               "--gpu", str(slot["gpu"]), "--tau", str(args.tau),
               "--variant", args.variant, "--out", args.out,
               "--ckpt01", args.ckpt01, "--ckpt02", args.ckpt02,
               "--ckpt03", args.ckpt03]
        if args.render:
            cmd.append("--render")
        fh = open(lf, "w")
        slot["proc"] = subprocess.Popen(cmd, stdout=fh, stderr=fh,
                                        start_new_session=True)
        slot["unit"], slot["logf"] = unit, lf
        log(f"기동: {unit['tag'][-20:]} 샤드{unit['shard']} "
            f"(잔여 {unit['n_left']}) → GPU{slot['gpu']} "
            f"(재시도 {unit['retry']})")

    def kill(slot):
        try:
            os.killpg(os.getpgid(slot["proc"].pid), signal.SIGKILL)
        except Exception:
            pass
        sweep_orphan_thor()

    while queue or any(s["proc"] is not None for s in slots):
        now = time.time()
        for s in slots:
            if s["proc"] is None:
                if queue and now - last_spawn >= STAGGER_S:
                    spawn(s, queue.pop(0))
                    last_spawn = time.time()
                continue
            rc = s["proc"].poll()
            stale = now - os.path.getmtime(s["logf"]) \
                if os.path.exists(s["logf"]) else 0.0
            if rc is None and stale < STALL_S:
                continue
            u = s["unit"]
            if rc == 0:
                log(f"완료: {u['tag'][-20:]} 샤드{u['shard']}")
            else:
                kill(s)
                u["retry"] += 1
                why = f"exit={rc}" if rc is not None \
                    else f"정체 {int(stale)}s"
                if u["retry"] <= MAX_RETRY:
                    log(f"재큐잉: {u['tag'][-20:]} 샤드{u['shard']} "
                        f"({why}, 재시도 {u['retry']}/{MAX_RETRY})")
                    queue.append(u)
                else:
                    log(f"포기: {u['tag'][-20:]} 샤드{u['shard']} ({why})")
                    failed.append(u)
            s["proc"] = s["unit"] = s["logf"] = None
        # 진행 로그(심박) — 완료 페이스 기반 ETA 포함
        if int(now - t0) % 60 < 5:
            done_n = sum(len(ledger_rows(args.out, t))
                         for t in set(args.envs))
            gained = done_n - base_n
            left = total_left - gained
            eta = (f"{(now - t0) / gained * left / 60:.0f}분" if gained > 0
                   else "측정 중")
            log(f"진행: 장부 {done_n}항목(+{gained}/{total_left}) | "
                f"대기 {len(queue)} | "
                f"실행 {sum(1 for s in slots if s['proc'])} | "
                f"실패 {len(failed)} | {int(now - t0) // 60}분 경과 | "
                f"ETA {eta}")
        time.sleep(5)
    log(f"감독자 종료 — 실패 유닛 {len(failed)}개")
    run_agg(args)


# ------------------------------------------------------------------ 집계

def _wmerge(rows_tables, legacy_tables):
    """(SR·SPL·TL·nDTW·NE) n-가중 병합 — NE는 None 제외."""
    out = {}
    for src in (rows_tables + legacy_tables):
        for grp, t in src.items():
            o = out.setdefault(grp, {"n": 0, "_ne_n": 0})
            n = t.get("n", 0)
            for k in ("SR", "SPL", "TL", "nDTW"):
                if k in t and t[k] is not None:
                    o[k] = (o.get(k, 0.0) * o["n"] + t[k] * n)
            ne = t.get("NE")
            if ne is not None:
                o["NE"] = o.get("NE", 0.0) * o["_ne_n"] + ne * n
                o["_ne_n"] += n
            o["n"] += n
            for k in ("SR", "SPL", "TL", "nDTW"):
                if k in o:
                    o[k] /= max(o["n"], 1)
            if "NE" in o:
                o["NE"] /= max(o["_ne_n"], 1)
    for o in out.values():
        o.pop("_ne_n", None)
    return out


def run_agg(args):
    import vln_metrics as vm
    state_keys = ("n", "correct", "notfound", "embodiment",
                  "false_impossible", "missed_impossible",
                  "gt_mismatch_reached")
    agg = vm.Agg()
    state = {k: 0 for k in state_keys}
    per_env_state = {}
    for tag in args.envs:
        for it in ledger_rows(args.out, tag):
            es = per_env_state.setdefault(tag, {k: 0 for k in state_keys})
            if it["kind"] == "sr":
                gr = dict(it["groups"])
                # 변형 태그 이전 장부 = 규칙 전역 실행분(폐기 아님,
                # "규칙 전역" ablation 베이스라인 행으로 유지)
                gr.setdefault("variant", it.get("variant", "rule_legacy"))
                agg.add(it["m"], **gr)
            elif it["kind"] == "state":
                for d in (state, es):
                    d["n"] += 1
                    d[it["sys_state"]] += 1
                    d["correct"] += it["ok"]
                    d["false_impossible"] += 0 if it["ok"] else 1
            elif it["kind"] == "missed":
                for d in (state, es):
                    d["n"] += 1
                    d["missed_impossible"] += 1
                    d["gt_mismatch_reached"] += it["reached"]
    # 레거시 완료 샤드 json(항목 장부 이전 형식, 집계 단위) 병합
    legacy = {}
    for pat in args.legacy:
        for f in glob.glob(pat):
            j = json.load(open(f))
            legacy[f] = j
            sa = j.get("state_axis")
            if sa:
                for k in state_keys:
                    state[k] += sa.get(k, 0)
    def _leg_tag(f):
        return os.path.basename(f).split("mansion_sim_")[1] \
            .split(".json")[0].rsplit("_shard", 1)[0]

    tables = {}
    for by in (None, "env", "task", "robot", "div", "variant"):
        key = by or "전체"
        rows_t = [agg.table(by)]
        if by is None:
            leg_t = [j.get("overall", {}) for j in legacy.values()]
        elif by == "env":
            leg_t = [{_leg_tag(f): j.get("overall", {}).get("전체", {})}
                     for f, j in legacy.items()]
        elif by == "variant":
            # 레거시 샤드 json에는 변형 태그가 없다 — 전부 규칙 전역 실행분
            leg_t = [{"rule_legacy": j.get("overall", {}).get("전체", {})}
                     for j in legacy.values()]
        else:
            leg_t = [j.get(f"by_{by}", {}) for j in legacy.values()]
        tables[key] = _wmerge(rows_t, leg_t)
    # regime = env의 함수 — env 표에서 정확 파생(레거시 by_regime 부재 대응)
    reg_src = [{REGIME.get(t, "?"): d} for t, d in tables["env"].items()]
    tables["regime"] = _wmerge(reg_src, [])
    report = {"tables": tables, "state_axis": state,
              "state_axis_by_env_ledger": per_env_state,
              "legacy_sources": sorted(legacy),
              "variant_note": "learned=학습 전역 선택기, rule/rule_legacy="
                              "규칙 다익스트라(ablation 베이스라인)",
              "date": time.strftime("%Y-%m-%d %H:%M")}
    out = os.path.join(args.out, "mansion_sim_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    log(f"보고 저장: {out}")
    print("\n== MANSION 시뮬 벤치마크 종합 (실로봇) ==")
    for by, tab in tables.items():
        for g_, t in sorted(tab.items()):
            print(f"  [{by}={g_}] SR {t.get('SR', 0):.3f} "
                  f"SPL {t.get('SPL', 0):.3f} NE {t.get('NE', 0):.2f} "
                  f"TL {t.get('TL', 0):.1f} nDTW {t.get('nDTW', 0):.3f} "
                  f"(n={t['n']})")
    sa = state
    print(f"  [상태판정] {sa['correct']}/{sa['n']} (미탐색 {sa['notfound']}"
          f"·embodiment {sa['embodiment']}·false {sa['false_impossible']}"
          f"/missed {sa['missed_impossible']}, GT불일치도달 "
          f"{sa['gt_mismatch_reached']})")


# ==================================================================
# 도구 롤: validate — 시뮬 벤치 검증 게이트 — 벤치 실행 전 반드시 통과해야 하는 기본 검증.
# ==================================================================
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, _HERE)
import loaders
import mansion_adapter as ma
from mansion_sim_env import MansionSimEnv

POOL = "/data/URDF/real_robots"


def replay_rate(env, tag, meta_r, n_traj=200):
    tot = bad = 0
    for f in sorted(glob.glob(f"{ma.DS}/episodes/{tag}_ep*.npz"))[:n_traj]:
        base = os.path.basename(f)[len(tag) + 1:-4]
        _, rid = base.split("_", 1)
        env.robot = {"w_eff": meta_r[rid]["w_eff"],
                     "h": meta_r[rid]["h"], "cam_h": 1.0,
                     "stairs_ok": True}
        env._pass_cache = {}
        with np.load(f) as z:
            pose, special = z["pose"], z["special"]
        for t in range(len(pose)):
            if special[t] in (1, 2, 3, 4):
                continue
            fl = int(pose[t, 0])
            g = env.floors[fl]
            iz, ix = g.to_idx(pose[t, 1], pose[t, 2])
            if (0 <= iz < g.stair_mask.shape[0]
                    and 0 <= ix < g.stair_mask.shape[1]
                    and g.stair_mask[iz, ix]):
                # 계단 등반 프레임은 주행 그리드 밖
                continue
            tot += 1
            if not env._free(fl, pose[t, 1], pose[t, 2]):
                bad += 1
    return 1.0 - bad / max(tot, 1)


def main_validate():
    mc = ma._mc()
    real = mc.robot_pool(POOL)
    all_pass = True
    tol_pick = {}
    for tag in loaders.tags():
        meta_r = loaders.robots_meta(tag)
        # G1: tol_cells 보정
        rate, tol = 0.0, None
        for k in (0, 1, 2, 3, 4, 5):
            env = MansionSimEnv(tag, render=False, tol_cells=k)
            rate = replay_rate(env, tag, meta_r)
            if rate >= 0.99:
                tol = k
                break
        tol_pick[tag] = tol
        ok1 = tol is not None
        print(f"[{tag[:22]}] G1 expert 리플레이: tol_cells={tol} "
              f"통과율 {rate:.3%} {'PASS' if ok1 else 'FAIL'}")
        all_pass &= ok1
        # G2: 존 도달성 (로봇별 anchor + 완화 충돌)
        env = MansionSimEnv(tag, render=False,
                            tol_cells=tol if tol is not None else 3)
        for r in real:
            env.robot = {"w_eff": r["w_eff"], "h": r["h"],
                         "cam_h": r["cam_h"],
                         "stairs_ok": r["stairs_ok"]}
            env._pass_cache = {}
            env._find_transit_zones(r["w_eff"], r["h"])
            env._zone_w = round(r["w_eff"], 3)
            bad = []
            for fl in env.floors:
                if fl in env.elev:
                    ex, ez, nx, nz, cx, cz = env.elev[fl]
                    if not env._free(fl, ex + nx * 0.8, ez + nz * 0.8):
                        bad.append(f"elevF{fl}")
                if fl in env.stairs and r["stairs_ok"]:
                    sx, sz = env.stairs[fl]
                    if not env._free(fl, sx, sz):
                        bad.append(f"stairF{fl}")
            ok2 = not bad
            all_pass &= ok2
            print(f"   G2 {r['id']:18} 존 도달성 "
                  f"{'PASS' if ok2 else 'FAIL ' + ','.join(bad)}")
    print(f"\n검증 게이트: {'전체 PASS' if all_pass else 'FAIL — 벤치 실행 금지'}")
    with open("/data/EVLN_ckpt/benchmark/sim_validation.json", "w") as fh:
        json.dump({"tol_cells": tol_pick, "pass": all_pass}, fh)



# ==================================================================
# 도구 롤: prep_gt — 실로봇(v2)별 에피소드 GT 재생성 — 기대 가능성·수단·참조 경로.
# ==================================================================
import argparse
import importlib.util
import json
import os
import pickle
import sys
import time

import mansion_adapter as ma

POOL = "/data/URDF/real_robots"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_m04():
    md = os.path.join(_HERE, "..", "datasets", "MANSION")
    # 04 스크립트의 'import common'(MANSION 쪽) 해석 경로
    sys.path.insert(0, md)
    p = os.path.join(md, "04_instructions.py")
    spec = importlib.util.spec_from_file_location("m04", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["m04"] = m
    spec.loader.exec_module(m)
    return m


def main_prep_gt():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--envs", nargs="*", default=None)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MANSION_GPU"] = "0"   # 가시 장치 제한 후엔 항상 0

    import loaders
    mc = ma._mc()
    m04 = load_m04()
    real = mc.robot_pool(POOL)
    for tag in (args.envs or loaders.tags()):
        floors = ma.load_floors(tag)
        nav = mc.BuildingNav(floors)
        ej = json.load(open(os.path.join(
            ma.DS, "episodes", f"{tag}_episodes.json")))
        out = {}
        t0 = time.time()
        for i, e in enumerate(ej["episodes"]):
            st, gl = e["start"], e["goal"]
            gf = int(gl["floor"])
            row = {}
            for r in real:
                ap_ = m04.approach_cell(floors[gf], r, gl["x"], gl["z"])
                if ap_ is None:
                    row[r["id"]] = {"success": False, "modes": [],
                                    "ref": [], "reason": "no_approach"}
                    continue
                res = nav.plan(r, (st["floor"], *st["cell"]),
                               (gf, *ap_))
                ref = []
                for fl, cells in res.get("segments", []):
                    g = floors[fl]
                    # 10cm 간격 다운샘플
                    for c in cells[::4]:
                        x, z = g.to_world(int(c[0]), int(c[1]))
                        ref.append((int(fl), float(x), float(z)))
                row[r["id"]] = {
                    "success": bool(res.get("success")),
                    "modes": sorted({t["mode"] for t in
                                     res.get("transitions", [])}),
                    "ref": ref,
                    "reason": res.get("reason", "")}
            out[e["id"]] = row
            if (i + 1) % 20 == 0:
                log(f"{tag[:20]} {i+1}/{len(ej['episodes'])} "
                    f"({time.time()-t0:.0f}s)")
        os.makedirs("/data/EVLN_ckpt/benchmark", exist_ok=True)
        with open(f"/data/EVLN_ckpt/benchmark/realgt_{tag}.pkl",
                  "wb") as fh:
            pickle.dump(out, fh)
        n_ok = sum(1 for row in out.values() for v in row.values()
                   if v["success"])
        log(f"{tag}: 저장 — 성공 GT {n_ok} / "
            f"{len(out) * len(real)}")



# ==================================================================
# 도구 롤: prep_robots — 실로봇(G1·Spot·Husky) URDF → 벤치마크 표준 embodiment 준비.
# ==================================================================
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, "/workspace/research/models")

# (이름, 클래스, 소스, 스펙 치수 w,l,h[m], base_z[m], 센서 부착: (부모 후보, xyz))
# 실로봇 풀 구성: Spot·Husky → 연구 표준·벤치 호환 폭의
# Go2·TurtleBot3로 교체. v1 산출물(/data/URDF/real_robots)은 기록 보존.
OUT_ROOT = "/data/URDF/real_robots"
ROBOTS = [
    ("unitree_g1", "humanoid", "/tmp/g1.urdf",
     (0.45, 0.30, 1.32), 0.793,
     (["head", "torso", "pelvis", "base"], (0.06, 0.0, 0.42))),
    ("unitree_go2", "multileg", "/tmp/go2.urdf",
     (0.31, 0.70, 0.40), 0.32,
     (["head", "base"], (0.28, 0.0, 0.05))),
    ("turtlebot3_waffle", "wheeled", "/tmp/tb3.urdf",
     (0.306, 0.281, 0.141), 0.010,
     (["base"], (0.07, 0.0, 0.11))),
]

# (정규식, NODE_TYPES prefix) 쌍
RULES = [
    (r"wheel", "wheel"), (r"caster", "caster"),
    (r"hip|thigh|upper_leg|knee|calf|lower_leg|shin|ankle|hind|front_l|"
     r"rear_l|_uleg|_lleg", "leg"),
    (r"foot|toe", "foot"),
    (r"shoulder|elbow|arm|bicep", "arm"),
    (r"wrist|hand|gripper|finger", "hand"),
    (r"head|skull", "head"), (r"neck", "neck"),
    (r"torso|trunk|spine|waist", "torso"),
    (r"camera|realsense|d435|blackfly|frontleft_fisheye|rgb", "sensor_rgb"),
    (r"imu", "sensor_imu"), (r"lidar|velodyne|laser|lms|vlp", "sensor_lidar"),
    (r"base|pelvis|body|chassis", "base"),
]


def norm_name(name, used):
    low = name.lower()
    for pat, pre in RULES:
        if re.search(pat, low):
            break
    else:
        pre = "base"
    k = 0
    while f"{pre}_{k:02d}" in used:
        k += 1
    used.add(f"{pre}_{k:02d}")
    return f"{pre}_{k:02d}"


def prep(name, cls, src, dims, base_z, sensor):
    tree = ET.parse(src)
    root = tree.getroot()
    used, ren = set(), {}
    for link in root.iter("link"):
        old = link.get("name")
        ren[old] = norm_name(old, used)
        link.set("name", ren[old])
    for j in root.iter("joint"):
        for tag in ("parent", "child"):
            el = j.find(tag)
            if el is not None and el.get("link") in ren:
                el.set("link", ren[el.get("link")])
    # 메시 참조 제거(yourdfpy load_meshes=False라 무해하지만 경로 청소)
    for mesh in root.iter("mesh"):
        mesh.set("filename", "package://none/none.stl")

    has_cam = any(n.startswith("sensor_rgb") for n in used)
    if not has_cam:
        cands, xyz = sensor
        parent = None
        for c in cands:
            hits = [n for o, n in ren.items() if c in o.lower()]
            if hits:
                parent = hits[0]
                break
        parent = parent or list(ren.values())[0]
        li = ET.SubElement(root, "link", name="sensor_rgb_00")
        jo = ET.SubElement(root, "joint", name="sensor_rgb_00_joint",
                           type="fixed")
        ET.SubElement(jo, "parent", link=parent)
        ET.SubElement(jo, "child", link="sensor_rgb_00")
        ET.SubElement(jo, "origin",
                      xyz=f"{xyz[0]} {xyz[1]} {xyz[2]}", rpy="0 0 0")
        note = f"sensor_rgb 스펙 근사 부착: parent={parent}, xyz={xyz}"
    else:
        note = "원본 카메라 링크 사용"

    out = f"{OUT_ROOT}/{cls}/{name}"
    os.makedirs(out, exist_ok=True)
    tree.write(f"{out}/robot.urdf")
    w, l, h = dims
    meta = {"id": name, "robot_class": cls, "form": name,
            "source": os.path.basename(src),
            "measured": {"w": w, "l": l, "h": h},
            "base_z": base_z, "notes": f"실로봇 스펙 치수(공식 제원). {note}",
            "sensors": ["sensor_rgb"]}
    with open(f"{out}/meta.json", "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    return out


def main_prep_robots():
    outs = [prep(*r) for r in ROBOTS]
    # 검증: 풀 로더 + 그래프 + z
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mansion_common",
        "/workspace/research/scripts/datasets/MANSION/common.py")
    mc = importlib.util.module_from_spec(spec)
    sys.modules["mansion_common"] = mc
    spec.loader.exec_module(mc)
    pool = mc.robot_pool(OUT_ROOT)
    import torch
    from modules.URDFEncoder import URDFEncoder, collate_graphs, urdf_to_graph
    enc = URDFEncoder(z_dim=128)
    ck = torch.load("/data/EVLN_ckpt/traversability.pt", map_location="cpu",
                    weights_only=False)
    enc.load_state_dict(ck["urdf_enc"])
    enc.eval()
    gb = collate_graphs([urdf_to_graph(
        f"{OUT_ROOT}/{r['cls']}/{r['id']}/robot.urdf")
        for r in pool])
    with torch.no_grad():
        z, aux = enc(gb)
    print("\n== 실로봇 3종 준비 결과")
    for i, r in enumerate(pool):
        print(f"  {r['id']:18} cls={r['cls']:9} w_eff={r['w_eff']:.2f} "
              f"h={r['h']:.2f} cam_h(FK)={r['cam_h']:.3f} "
              f"stairs_ok={r['stairs_ok']} | 프로브 cam_h="
              f"{float(aux['cam_h'][i]):.3f} stairs="
              f"{float(torch.sigmoid(aux['stairs_ok_logit'][i])):.2f}")



def main():
    # 도구 롤은 자체 argparse 사용 — --role 만 떼어내고 위임
    for tool in ("validate", "prep_gt", "prep_robots"):
        if "--role" in sys.argv and \
                sys.argv[sys.argv.index("--role") + 1] == tool:
            i = sys.argv.index("--role")
            del sys.argv[i:i + 2]
            return globals()[f"main_{tool}"]()
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("super", "exec", "agg",
                    "validate", "prep_gt", "prep_robots"),
                    default="super")
    ap.add_argument("--envs", nargs="*", default=ALL_TAGS)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)      # exec 전용
    ap.add_argument("--shard", type=int, default=0)    # exec 전용
    # --shard/--nshards는 exec 역할 전용
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--ep-per-unit", type=int, default=EP_PER_UNIT,
                    help="유닛당 에피소드 수 — 잔여가 적을 때 낮추면 "
                         "병렬 폭 확보")
    # learned = 학습 전역 선택기(기본 실행), rule = 규칙 다익스트라 ablation
    ap.add_argument("--variant", choices=("learned", "rule"),
                    default="learned")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--viz", type=int, default=0)
    ap.add_argument("--viz-dir", default="/workspace/research/check/benchmark")
    ap.add_argument("--ckpt01", default="/data/EVLN_ckpt/traversability.pt")
    ap.add_argument("--ckpt02", default="/data/EVLN_ckpt/waypoint.pt",
                    help="제안기(히트맵) — 학습 2단계 산출")
    ap.add_argument("--ckpt03", default="/data/EVLN_ckpt/global.pt",
                    help="전역 선택기 — 학습 3단계 산출")
    ap.add_argument("--out", default="/data/EVLN_ckpt/benchmark")
    ap.add_argument("--logdir", default="/data/EVLN_ckpt/logs")
    ap.add_argument("--legacy", nargs="*", default=
                    ["/data/EVLN_ckpt/benchmark/mansion_sim_*_shard*.json"],
                    help="증분 저장 도입 전 완료 샤드 json(집계 병합용)")
    args = ap.parse_args()
    if args.role == "exec":
        run_exec(args)
    elif args.role == "agg":
        run_agg(args)
    else:
        args.render = True if args.variant == "learned" else args.render
        run_super(args)


if __name__ == "__main__":
    main()

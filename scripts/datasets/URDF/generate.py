"""랜덤 URDF 생성 CLI.

흐름: 클래스 → 형태(round-robin) → 목표 치수 층화 샘플 → 템플릿 빌드 → 검증 필터
      → (관절 컨피그 충돌 시 리밋 30% 축소 후 1회 재검증) → 통과분만 저장.

파일럿과 본 생성의 저장 위치가 다르다:
  파일럿  : --out /tmp/urdf_pilot (임시, 카드 확인 후 삭제) — data/에 저장 금지
  본 생성 : --out /data/URDF/<split> (사용자 승인 후에만)

사용 예 (컨테이너 내부, 파일럿):
  cd /workspace/research/scripts/datasets/URDF
  python generate.py --n 20 --seed 42 --out /tmp/urdf_pilot \
                     --check-dir /workspace/research/check/URDF

산출: <out>/<class>/<id>/robot.urdf + meta.json
      <check-dir>/<class>.png (로봇당 BEV 1뷰 몽타주), 통계는 stdout 로그
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pybullet as p

from common import node_type
from templates import BUILDERS, CLASS_FORMS, PRIMARY_DIM, BuildError, \
    sample_target_binned
import validate as V

N_BINS = 5


def generate_class(cls, n, seed, out_dir, k_configs, forms_filter=None):
    forms = CLASS_FORMS[cls]
    if forms_filter:  # 홀드아웃 스플릿용 폼 필터 (본생성: unseen-form 분리)
        forms = [f for f in forms if f in forms_filter]
        if not forms:
            print(f"  !! {cls}: forms_filter에 해당 폼 없음 — 스킵")
            return [], 0, Counter()
    saved, attempts = [], 0
    rejects = Counter()
    form_saved, form_fail = Counter(), Counter()
    max_attempts = n * 40
    FORM_FAIL_CAP = 30       # 연속 실패 시 해당 폼 제외(스탯에 명시 — 조용한 누락 금지)

    while len(saved) < n and attempts < max_attempts:
        attempts += 1
        active = [f for f in forms if form_fail[f] < FORM_FAIL_CAP]
        if not active:
            print(f"  !! {cls}: 모든 폼이 연속 실패 상한 도달, 중단")
            break
        form = min(active, key=lambda f: form_saved[f])
        cls_off = list(CLASS_FORMS).index(cls) * 100_000
        seed_k = seed * 1_000_000 + cls_off + attempts
        rng = random.Random(seed_k)
        bin_idx = len(saved) % N_BINS
        target = sample_target_binned(rng, cls, bin_idx, N_BINS)

        try:
            robot, meta = BUILDERS[form](rng, target)
        except BuildError:
            rejects["build"] += 1
            form_fail[form] += 1
            continue
        robot.finalize()

        ok, reasons, measured = V.validate(robot, meta, target,
                                           k_configs=k_configs, seed=seed_k)
        if not ok and reasons == ["self_col_conf"]:
            robot.shrink_limits(0.7)
            ok, reasons, measured = V.validate(robot, meta, target,
                                               k_configs=k_configs, seed=seed_k)
            if ok:
                meta["notes"].append("관절 리밋 30% 축소(자기충돌 회피)")
        if not ok:
            rejects.update(reasons)
            form_fail[form] += 1
            continue
        form_saved[form] += 1
        form_fail[form] = 0

        rid = f"{cls}_{form}_{len(saved):03d}"
        d = Path(out_dir) / cls / rid
        d.mkdir(parents=True, exist_ok=True)
        (d / "robot.urdf").write_text(robot.to_urdf())
        n_act = sum(1 for j in robot.joints if j.jtype != "fixed")
        meta_out = dict(
            id=rid, robot_class=cls, form=form, seed=seed_k,
            target=dict(zip("wlh", [round(t, 4) for t in target])),
            measured=measured, base_z=round(meta["base_z"], 4),
            total_mass=round(robot.total_mass(), 3),
            n_links=len(robot.links), n_joints_actuated=n_act,
            sensors=meta["sensors"], params=meta["params"],
            notes=meta["notes"], nodes=robot.node_summary(),
        )
        (d / "meta.json").write_text(
            json.dumps(meta_out, ensure_ascii=False, indent=1))
        saved.append(meta_out)

    return saved, attempts, rejects


def print_stats(cls, saved, attempts, rejects):
    print(f"\n=== {cls}: {len(saved)} 저장 / {attempts} 시도 "
          f"(통과율 {len(saved) / max(attempts, 1):.1%})")
    if rejects:
        print("  거부 사유:", dict(rejects.most_common()))
    dim, lo, hi = PRIMARY_DIM[cls]
    vals = [m["measured"][dim] for m in saved]
    hist, _ = np.histogram(vals, bins=N_BINS, range=(lo, hi))
    print(f"  주치수 {dim} 커버리지 [{lo}~{hi}m, {N_BINS}bins]: {hist.tolist()}")
    mm = [m["total_mass"] for m in saved]
    nl = [m["n_links"] for m in saved]
    if saved:
        print(f"  질량 {min(mm):.1f}~{max(mm):.1f}kg, "
              f"링크 수 {min(nl)}~{max(nl)}, "
              f"형태: {dict(Counter(m['form'] for m in saved))}")


# ---------------- 파일럿 카드 렌더 (pybullet TinyRenderer) ----------------

def render_bev(urdf_path, base_z, size=300):
    """BEV(대각선 상공에서 내려다본) 1뷰 렌더."""
    from PIL import Image
    cid = V._client()
    p.resetSimulation(physicsClientId=cid)
    body = p.loadURDF(str(urdf_path), basePosition=(0, 0, base_z),
                      flags=p.URDF_MAINTAIN_LINK_ORDER)
    lo, hi, _ = V._union_aabb(body, p.getNumJoints(body))
    center = (hi + lo) / 2
    dist = max(float(max(hi - lo)) * 1.6, 0.4)
    gnd = p.createVisualShape(p.GEOM_BOX, halfExtents=[dist, dist, 0.002],
                              rgbaColor=[0.85, 0.85, 0.85, 1])
    p.createMultiBody(baseVisualShapeIndex=gnd, basePosition=[0, 0, -0.002])
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1, nearVal=0.01, farVal=20)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=center.tolist(), distance=dist,
        yaw=45, pitch=-35, roll=0, upAxisIndex=2)
    _, _, rgb, _, _ = p.getCameraImage(size, size, view, proj,
                                       renderer=p.ER_TINY_RENDERER)
    arr = np.reshape(rgb, (size, size, 4))[:, :, :3].astype(np.uint8)
    return Image.fromarray(arr)


def make_card(cls, saved, out_dir, check_dir, cell=300, cols=5):
    from PIL import Image, ImageDraw
    tiles = []
    for m in saved:
        urdf = Path(out_dir) / cls / m["id"] / "robot.urdf"
        bev = render_bev(urdf, m["base_z"], cell)
        tile = Image.new("RGB", (cell, cell + 26), "white")
        tile.paste(bev, (0, 26))
        dr = ImageDraw.Draw(tile)
        me = m["measured"]
        dr.text((6, 5), f"{m['id']} w{me['w']} l{me['l']} h{me['h']} "
                        f"{m['total_mass']}kg", fill="black")
        tiles.append(tile)
    n_rows = (len(tiles) + cols - 1) // cols
    card = Image.new("RGB", (cell * cols, (cell + 26) * n_rows), "white")
    for i, tile in enumerate(tiles):
        card.paste(tile, ((i % cols) * cell, (i // cols) * (cell + 26)))
    Path(check_dir).mkdir(parents=True, exist_ok=True)
    path = Path(check_dir) / f"{cls}.png"
    card.save(path)
    print(f"  카드 저장: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=list(CLASS_FORMS),
                    choices=list(CLASS_FORMS))
    ap.add_argument("--n", type=int, default=20, help="클래스당 저장 개수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/urdf_pilot",
                    help="파일럿=임시 경로(기본값), 본 생성=/data/URDF/<split>")
    ap.add_argument("--check-dir", default=None, help="파일럿 카드 PNG 출력 경로")
    ap.add_argument("--k-configs", type=int, default=20)
    ap.add_argument("--forms", nargs="+", default=None,
                    help="이 폼들만 생성 (unseen-form 홀드아웃용)")
    ap.add_argument("--exclude-forms", nargs="+", default=None,
                    help="이 폼들 제외 (train 홀드아웃 제외용)")
    ap.add_argument("--card-only", action="store_true",
                    help="생성 없이 저장된 로봇들의 카드만 재렌더")
    args = ap.parse_args()

    for cls in args.classes:
        if args.card_only:
            saved = [json.loads(f.read_text()) for f in
                     sorted(Path(args.out).glob(f"{cls}/*/meta.json"))]
        else:
            ff = None
            if args.forms:
                ff = set(args.forms)
            if args.exclude_forms:
                from templates import CLASS_FORMS as _CF
                ff = (ff or set(_CF[cls])) - set(args.exclude_forms)
            saved, attempts, rejects = generate_class(
                cls, args.n, args.seed, args.out, args.k_configs,
                forms_filter=ff)
            print_stats(cls, saved, attempts, rejects)
        if args.check_dir and saved:
            make_card(cls, saved, args.out, args.check_dir)


if __name__ == "__main__":
    main()

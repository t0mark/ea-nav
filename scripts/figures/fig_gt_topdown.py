"""벤치마크 에피소드 하나의 정답 궤적 탑뷰 — 논문 figure용.

층 기하(자유 공간·벽·계단·엘베)를 탑다운으로 깔고 그 위에 GT 궤적을 얹는다.
층이 바뀌는 에피소드는 층마다 패널을 하나씩 만들어 가로로 잇는다.

사용 (airlab_hw_mansion 컨테이너):
  python /workspace/research/scripts/figure/fig_gt_topdown.py \
      --tag office_corporate_hq_6f_300_fp001 --episode ep0000 \
      --robot wheeled_diff_060 --out /workspace/research/check/figure
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "datasets", "MANSION"))
sys.path.insert(0, os.path.join(_HERE, "..", "models"))

import mansion_adapter as ma                          # noqa: E402

# 데이터 생성부 common은 모델부 common과 이름이 같아 import로는 가려진다 —
# 탑다운·경로 그리기 헬퍼가 생성부에만 있으므로 파일 경로로 직접 적재한다
_spec = importlib.util.spec_from_file_location(
    "mansion_gen_common",
    os.path.join(_HERE, "..", "datasets", "MANSION", "common.py"))
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

PATH_BGR = (60, 220, 255)      # 궤적 — 노랑
START_BGR = (90, 230, 90)      # 출발 — 초록
GOAL_BGR = (70, 70, 240)       # 목표 — 빨강
LABEL_BGR = (245, 245, 245)


def panel(geom, pts, start, goal, title):
    """한 층 패널 — 기하 + 궤적 + 출발·목표 표식."""
    img = mc.topdown(geom)
    cells = [geom.to_idx(x, z) for x, z in pts]
    if len(cells) > 1:
        mc.draw_path(img, cells, PATH_BGR)
    for (iz, ix), col, rad in ((start, START_BGR, 5), (goal, GOAL_BGR, 5)):
        if iz is not None:
            cv2.circle(img, (ix, iz), rad, col, -1)
            cv2.circle(img, (ix, iz), rad + 2, (20, 20, 20), 1)
    img = cv2.flip(img, 0)                    # z축이 위로
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                LABEL_BGR, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="office_corporate_hq_6f_300_fp001")
    ap.add_argument("--episode", default=None, help="미지정이면 첫 궤적")
    ap.add_argument("--robot", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="/workspace/research/check/figure")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = ma.DS
    ej = json.load(open(os.path.join(ds, "episodes",
                                     f"{args.tag}_episodes.json")))
    eps = {e["id"]: e for e in ej["episodes"]}

    if args.episode and args.robot:
        epid, rid = args.episode, args.robot
    else:
        pat = os.path.join(ds, "episodes", f"{args.tag}_*.npz")
        cands = sorted(glob.glob(pat))
        assert cands, f"궤적 npz 없음: {pat}"
        base = os.path.basename(cands[0])[len(args.tag) + 1:-4]
        epid, rid = base.split("_", 1)
    f = os.path.join(ds, "episodes", f"{args.tag}_{epid}_{rid}.npz")
    assert os.path.exists(f), f"없음: {f}"
    with np.load(f) as z:
        pose = z["pose"]
    e = eps[epid]
    print(f"에피소드 {epid} · 로봇 {rid} · 프레임 {len(pose)}")

    floors = ma.load_floors(args.tag)
    visited = []
    for p in pose:
        fl = int(p[0])
        if not visited or visited[-1] != fl:
            visited.append(fl)
    print(f"방문 층 {visited}")

    gl = e["goal"]
    st = e["start"]
    sx, sz = floors[st["floor"]].to_world(*st["cell"])

    panels = []
    for fl in visited:
        geom = floors[fl]
        pts = [(float(p[1]), float(p[2])) for p in pose if int(p[0]) == fl]
        s_idx = geom.to_idx(sx, sz) if int(st["floor"]) == fl else (None, None)
        g_idx = (geom.to_idx(float(gl["x"]), float(gl["z"]))
                 if int(gl["floor"]) == fl else (None, None))
        panels.append(panel(geom, pts, s_idx, g_idx, f"floor {fl}"))

    h = max(p.shape[0] for p in panels)
    row = []
    for p in panels:
        if p.shape[0] != h:
            pad = np.zeros((h - p.shape[0], p.shape[1], 3), np.uint8)
            p = np.vstack([p, pad])
        row.append(p)
        row.append(np.full((h, 6, 3), 255, np.uint8))
    img = np.hstack(row[:-1])
    if args.scale != 1.0:
        img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                         interpolation=cv2.INTER_NEAREST)
    out = os.path.join(args.out, f"gt_topdown_{args.tag[:20]}_{epid}_{rid}.png")
    cv2.imwrite(out, img)
    print(f"저장 {out} ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()

"""기존 URDF 풀에서 파일럿 카드만 재생성 (풀 자체는 건드리지 않음).

generate.py는 로봇을 새로 만들면서 카드를 그리므로, 이미 확정된 본 풀의
카드를 다시 뽑을 때는 이 스크립트를 쓴다. meta.json을 읽어 클래스별로
표본을 고르고 generate.make_card를 그대로 재사용한다.

실행:
  docker exec airlab_hw_mansion bash -c 'cd /workspace/research/scripts/datasets/URDF \
    && python cards.py --root /data/URDF --check-dir /workspace/research/check/URDF'
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "gen", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "generate.py"))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def split_cards(split_dir, check_dir, n_per_class):
    """스플릿 하나의 클래스별 카드. 표본은 id 정렬 후 균등 간격으로 고른다."""
    made = []
    for cls in sorted(os.listdir(split_dir)):
        cdir = os.path.join(split_dir, cls)
        if not os.path.isdir(cdir):
            continue
        metas = []
        for rid in sorted(os.listdir(cdir)):
            mp = os.path.join(cdir, rid, "meta.json")
            if os.path.exists(mp):
                with open(mp) as fh:
                    metas.append(json.load(fh))
        if not metas:
            continue
        step = max(1, len(metas) // n_per_class)
        picked = []
        for m in metas[::step]:
            if len(picked) >= n_per_class:
                break
            # 실기 URDF는 외부 메시를 참조해 로드가 실패할 수 있다 — 건너뛴다
            try:
                gen.render_bev(Path(split_dir) / cls / m["id"] / "robot.urdf",
                               m["base_z"], 64)
            except Exception as e:  # noqa: BLE001
                log(f"    {m['id']} 렌더 불가 — 제외 ({type(e).__name__})")
                continue
            picked.append(m)
        if not picked:
            continue
        log(f"  {cls}: 풀 {len(metas)}대 → 카드 {len(picked)}대")
        gen.make_card(cls, picked, split_dir, check_dir)
        made.append(cls)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/URDF")
    ap.add_argument("--check-dir", required=True)
    ap.add_argument("--n", type=int, default=5, help="클래스당 카드 표본 수")
    # real_robots는 외부 메시를 참조하는 실기 URDF라 기본에서 제외
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val_unseen_dims", "test_unseen_form"])
    args = ap.parse_args()

    t0 = time.time()
    for i, sp in enumerate(args.splits, 1):
        sdir = os.path.join(args.root, sp)
        if not os.path.isdir(sdir):
            log(f"[{i}/{len(args.splits)}] {sp}: 없음 — 건너뜀")
            continue
        # train은 대표 스플릿이라 check/URDF 바로 아래, 나머지는 하위 폴더
        out = (args.check_dir if sp == "train"
               else os.path.join(args.check_dir, sp))
        log(f"[{i}/{len(args.splits)}] {sp} → {out}")
        split_cards(sdir, out, args.n)
        el = time.time() - t0
        log(f"  {el:.0f}s 경과 / ETA {el / i * (len(args.splits) - i):.0f}s")

    man = os.path.join(args.root, "splits.json")
    if os.path.exists(man):
        with open(man) as fh:
            log("splits.json: " + json.dumps(json.load(fh))[:200])
    log(f"완료 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

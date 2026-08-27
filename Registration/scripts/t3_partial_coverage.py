"""R3 T3 — 参照が観測の一部しか覆わない条件を、Replica で先に作って検証する。

なぜ実データを待たないか
------------------------
実データでは reference（BIM = 411 のみ、または 410 のみ）が
source（411+410+廊下+対向室）の一部しか覆わない。**この条件は既存の評価設定に無い。**
撮ってから「スケールが発散した」と分かるのが最悪なので、正解が既知の合成データで先に通す
（指示書 [[2026-08-05_r3_ifc_loader_partial_coverage]] §1）。

やっていること
--------------
既存8シーン（`configs/sectionC/`）について、**reference 側だけ**を XY で切り、
床面積比で 100 / 75 / 50 / 30% に落とす（source は full のまま）。
各水準 × `scale_init` 2モードで `benchmark.py` をそのまま呼ぶ。
**評価経路を分岐させない**ため、config に `reference.clip` と `proposed.scale_init` を
注入した一時 config を書いて渡す形にしている。

初期スケール誤差について
------------------------
摂動 P（スケール成分 sp）を source に掛けると、`_axis_span` は範囲に対して厳密に線形なので
seed されるスケールは `s0/sp` になり、正解も `1/sp` 倍される。**比 s0 は摂動に依存しない。**
よって (scene, 被覆水準, モード) ごとに1回だけ測れば足りる。100 試行ぶん回す必要は無い。

    conda activate sni-slam
    python Registration/scripts/t3_partial_coverage.py --trials 100 --jobs 8
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))

SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2",
          "office_3", "office_4"]
LEVELS = [1.0, 0.75, 0.50, 0.30]
MODES = ["median_axes", "vertical_only"]
# 残す箱を四隅のどれに寄せるか。**中心寄せは使わない**：単室では箱が室の内側に入り、
# 壁が境界にあるので全部落ちる（room_0 の実測で 50% 被覆時に壁 0 点）。
# 隅に寄せれば、その側の壁2枚は実物のまま残る＝「BIM がより少ない完全な部屋を覆う」に近い。
ANCHORS = [(-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)]
_ANCHOR_TAG = {(-1.0, -1.0): "sw", (1.0, 1.0): "ne", (-1.0, 1.0): "nw", (1.0, -1.0): "se"}


def run_key(scene: str, level: float, anchor, mode: str) -> str:
    return "%s__cov%03d_%s__%s" % (scene, round(level * 100),
                                   _ANCHOR_TAG.get(tuple(anchor), "c"), mode)


def make_config(base: Dict, level: float, anchor, mode: str, out_dir: str) -> Dict:
    cfg = copy.deepcopy(base)
    if level < 1.0:
        cfg["reference"]["clip"] = {"keep_frac": level, "anchor": list(anchor)}
    cfg.setdefault("proposed", {})["scale_init"] = mode
    cfg["eval"]["out_dir"] = out_dir
    return cfg


def measure_init_scale(cfg: Dict) -> Optional[Dict]:
    """seed されるスケールを1回だけ測る（摂動に依存しないため）。"""
    from regbim import io_utils, preprocess
    from regbim.config import load_t_gt
    from regbim.methods.proposed import axis_spans, canonical_axes, seed_scale
    from regbim.metrics import decompose_sim3
    import numpy as np

    src = preprocess.prepare(io_utils.load_source_cloud(cfg), cfg)
    dst = preprocess.prepare(io_utils.load_reference_cloud(cfg), cfg)
    axes = canonical_axes(dst, cfg)
    dst_span = axis_spans(dst.points, dst.labels, axes)
    # source をリファレンス系へ回してから測る（register 内と同じ手順）
    from regbim import rotation
    cands = rotation.relative_rotation_candidates(src, dst, cfg)
    vals = []
    for R in cands:
        src_span = axis_spans(src.points @ R.T, src.labels, axes)
        vals.append(seed_scale(src_span, dst_span, axes, cfg["proposed"]["scale_init"]))
    try:
        T_gt = load_t_gt(cfg)
        s_true = float(decompose_sim3(T_gt)[2]) if T_gt is not None else 1.0
    except Exception:
        s_true = 1.0
    # 軸ごとの範囲比も残す。vertical_only が外れたとき「どの軸が悪いのか」を
    # 後から言えるようにするため（外れたことだけ分かっても切り分けにならない）。
    src_span0 = axis_spans(src.points @ cands[0].T, src.labels, axes)
    ratios = {k: (None if (dst_span[k] is None or src_span0[k] is None)
                  else round(dst_span[k][1] / src_span0[k][1], 4))
              for k, _, _ in axes}
    extents = {("dst_" + k): (None if dst_span[k] is None else round(dst_span[k][1], 3))
               for k, _, _ in axes}
    extents.update({("src_" + k): (None if src_span0[k] is None else round(src_span0[k][1], 3))
                    for k, _, _ in axes})
    return {"seed_scale_candidates": [round(float(v), 5) for v in vals],
            "seed_scale_median": round(float(np.median(vals)), 5),
            "true_scale": round(s_true, 5),
            "init_scale_error_ratio": round(float(np.median(vals)) / s_true, 5),
            "axis_extent_ratios": ratios, "axis_extents_m": extents,
            "coverage_achieved": (dst.meta.get("clip") or {}).get("coverage_achieved", 1.0),
            "reference_floor_area_m2": (dst.meta.get("clip") or {})
                .get("floor_area_clipped_m2"),
            "reference_class_counts": (dst.meta.get("clip") or {}).get("class_counts_after"),
            "n_reference_points": int(len(dst))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--levels", nargs="*", type=float, default=None)
    ap.add_argument("--anchors", type=int, default=3,
                    help="切る位置を何通り振るか（1 なら中心のみ）")
    ap.add_argument("--out-root", default="Registration/output/t3_coverage")
    ap.add_argument("--threads", type=int, default=4,
                    help="1 ジョブあたりの BLAS/OMP スレッド数。並列時の取り合いを防ぐ")
    ap.add_argument("--init-scale-only", action="store_true",
                    help="benchmark を回さず、初期スケールだけ測る（配管の確認用）")
    args = ap.parse_args()

    os.chdir(REPO)
    scenes = args.scenes or SCENES
    levels = args.levels or LEVELS
    anchors = ANCHORS[:max(1, args.anchors)]
    os.makedirs(args.out_root, exist_ok=True)

    jobs: List[Dict] = []
    for scene, level, mode in itertools.product(scenes, levels, MODES):
        for anchor in (anchors if level < 1.0 else [(0.0, 0.0)]):
            jobs.append({"scene": scene, "level": level, "anchor": anchor,
                         "mode": mode, "key": run_key(scene, level, anchor, mode)})
    print("%d 実行（%d シーン × %d 水準 × %d モード、100%% 以外は %d 位置）"
          % (len(jobs), len(scenes), len(levels), len(MODES), len(anchors)))

    # --- 初期スケールは摂動に依存しないので、先にまとめて測る ---
    init: Dict[str, Dict] = {}
    for j in jobs:
        base = yaml.safe_load(open("Registration/configs/sectionC/%s.yaml" % j["scene"]))
        cfg = make_config(base, j["level"], j["anchor"], j["mode"],
                          os.path.join(args.out_root, j["key"]))
        try:
            init[j["key"]] = measure_init_scale(cfg)
            print("  %-46s 被覆 %.3f  初期スケール誤差 %.4f"
                  % (j["key"], init[j["key"]]["coverage_achieved"],
                     init[j["key"]]["init_scale_error_ratio"]))
        except Exception as e:
            init[j["key"]] = {"error": "%s: %s" % (type(e).__name__, e)}
            print("  %-46s ×  %s" % (j["key"], e))
    with open(os.path.join(args.out_root, "init_scale.json"), "w") as f:
        json.dump(init, f, indent=2, ensure_ascii=False)
    if args.init_scale_only:
        return 0

    # --- benchmark を並列で回す ---
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = str(args.threads)

    tmpdir = tempfile.mkdtemp(prefix="t3cfg_")
    pending = list(jobs)
    running: List = []
    t0 = time.time()
    done = 0
    while pending or running:
        while pending and len(running) < args.jobs:
            j = pending.pop(0)
            out = os.path.join(args.out_root, j["key"])
            if os.path.exists(os.path.join(out, "summary.json")):
                done += 1
                continue                      # 再開できるようにする
            os.makedirs(out, exist_ok=True)
            base = yaml.safe_load(
                open("Registration/configs/sectionC/%s.yaml" % j["scene"]))
            cfg = make_config(base, j["level"], j["anchor"], j["mode"], out)
            cp = os.path.join(tmpdir, j["key"] + ".yaml")
            with open(cp, "w") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True)
            log = open(os.path.join(out, "run.log"), "w")
            p = subprocess.Popen(
                [sys.executable, "-W", "ignore", "Registration/scripts/benchmark.py",
                 "--config", cp, "--methods", "proposed",
                 "--trials", str(args.trials), "--out-dir", out],
                stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((p, j, log))
        time.sleep(5)
        for item in list(running):
            p, j, log = item
            if p.poll() is None:
                continue
            running.remove(item)
            log.close()
            done += 1
            print("[%5.1f 分] %3d/%3d %s rc=%d"
                  % ((time.time() - t0) / 60, done, len(jobs), j["key"], p.returncode),
                  flush=True)

    # --- 収集 ---
    rows = []
    for j in jobs:
        out = os.path.join(args.out_root, j["key"])
        sp = os.path.join(out, "summary.json")
        if not os.path.exists(sp):
            rows.append({**j, "status": "failed"})
            continue
        s = json.load(open(sp))
        m = s["methods"]["proposed"]
        agg = m["aggregate"]
        rows.append({
            "scene": j["scene"], "level_nominal": j["level"],
            "anchor": list(j["anchor"]), "mode": j["mode"],
            "coverage_achieved": init.get(j["key"], {}).get("coverage_achieved"),
            "init_scale_error_ratio": init.get(j["key"], {}).get("init_scale_error_ratio"),
            "trials": agg["robust_trials"],
            "success_rate": agg["success_rate"],
            "ci_lo": agg.get("success_ci_lo"), "ci_hi": agg.get("success_ci_hi"),
            "med_rot_deg": agg["med_rot_deg"], "med_trans": agg["med_trans"],
            "med_scale_ratio": agg["med_scale_ratio"],
            "rot_q25": agg.get("rot_deg_q25"), "rot_q75": agg.get("rot_deg_q75"),
            "trans_q25": agg.get("trans_q25"), "trans_q75": agg.get("trans_q75"),
            "scale_q25": agg.get("scale_ratio_q25"), "scale_q75": agg.get("scale_ratio_q75"),
            "failure_breakdown": m.get("failure_breakdown"),
            "status": "ok",
        })
    with open(os.path.join(args.out_root, "t3_results.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % os.path.join(args.out_root, "t3_results.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

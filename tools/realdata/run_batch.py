"""Phase 5 — マニフェスト駆動のバッチランナー（S3）。

マニフェスト `tools/realdata/manifest_realdata.yaml` を読み、シーンごとに
    変換 -> 意味ラベル -> 契約チェック -> SLAM -> ATE -> 事前チェック -> feat_cache 削除
を順に実行し、最後に `summary_realdata.csv` を書く。

設計上の注意
------------
* **feat_cache は run ごとに必ず消す**（1 run 最大 13 GB。30 run で 390 GB になる）。
  `--keep-feat-cache` を明示したときだけ残す。
* `--resume` は `eval_ate.json` が既にある run を飛ばす。長時間実行の中断に備える。
* 変換と意味ラベルは**別の conda env**（`sni-slam` / `seg2d`）で動くので、
  各ステップを `bash -lc` 経由で起動して env を切り替える。
* **シーンを事前チェック値で却下しない**（追補指示 Q7）。失敗しても記録して次へ進む。
* 乱数 seed は SNI-SLAM 側に設定箇所が無い（`run.py`/`src/` の grep でヒット 0）。
  よって run 間は独立試行になる。CSV には `seed=none` と明示的に記録する。

使い方
------
    conda activate sni-slam
    python tools/realdata/run_batch.py --order frames-asc --max-scenes 6 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CONDA_SH = "/opt/miniconda/3/etc/profile.d/conda.sh"


def sh(cmd: str, env: str, log_path: Optional[str] = None) -> int:
    """指定 conda env でコマンドを実行し、ログを tee する。"""
    full = "source %s && conda activate %s && cd %s && %s" % (CONDA_SH, env, REPO, cmd)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        full += " 2>&1 | tee -a %s" % log_path
    print("    $ [%s] %s" % (env, cmd), flush=True)
    return subprocess.call(["bash", "-lc", full])


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def read_json(path: str) -> Dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="tools/realdata/manifest_realdata.yaml")
    ap.add_argument("--scenes", nargs="*", default=None, help="scene 名で絞る")
    ap.add_argument("--order", default="frames-asc",
                    choices=["frames-asc", "manifest"],
                    help="frames-asc: フレーム数の少ない順（短い本から先に成立させる）")
    ap.add_argument("--max-scenes", type=int, default=0, help="先頭 N シーンだけ（0 で全件）")
    ap.add_argument("--runs", type=int, default=0,
                    help="run 数を上書き（0 ならマニフェストの runs_planned）")
    ap.add_argument("--mode", default="anamorphic", choices=["anamorphic", "letterbox"])
    ap.add_argument("--conf-min", type=int, default=1)
    ap.add_argument("--max-depth", type=float, default=5.0)
    ap.add_argument("--label-batch", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-feat-cache", action="store_true")
    ap.add_argument("--skip-slam", action="store_true", help="変換とラベルだけ先に流す")
    ap.add_argument("--dry-run", action="store_true",
                    help="実行計画（対象シーンと順序）を出すだけで、何も実行しない")
    ap.add_argument("--out-csv", default="output/RealData/summary_realdata.csv")
    ap.add_argument("--config-dir", default="configs/RealData",
                    help="使う config の置き場。フォールバック D なら configs/RealData/_D")
    ap.add_argument("--out-root", default="output/RealData",
                    help="出力の置き場。D なら output/RealData/_D")
    ap.add_argument("--project-labels", action="store_true",
                    help="run 後に 2D ラベルをメッシュへ投影する（追補3 §3）")
    args = ap.parse_args()

    os.chdir(REPO)
    with open(args.manifest) as f:
        man = yaml.safe_load(f)
    scenes = [s for s in man["scenes"] if s.get("status") != "rejected"]
    if args.scenes:
        scenes = [s for s in scenes if s["scene"] in args.scenes]
    if args.order == "frames-asc":
        scenes.sort(key=lambda s: s["preflight"]["n_frames"])
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    head = git_head()
    log_dir = "output/RealData/_logs"
    rows: List[Dict[str, object]] = []
    print("batch: %d scenes, order=%s, head=%s" % (len(scenes), args.order, head[:8]))
    total_frames = 0
    for s in scenes:
        n_runs_s = args.runs or int(s.get("runs_planned", 1))
        nf = s["preflight"]["n_frames"]
        total_frames += nf * n_runs_s
        print("  %-14s %-12s %6d frames x %d run  (~%.0f min/run)"
              % (s["scene"], s["scan_id"], nf, n_runs_s, nf * 29.5 / 1500.0))
    print("  ---- total %d frame-runs ~ %.1f h SLAM" % (total_frames,
                                                        total_frames * 29.5 / 1500.0 / 60))
    if args.dry_run:
        print("\n--dry-run: 何も実行しない")
        return 0

    for s in scenes:
        scene, scan = s["scene"], s["scan_id"]
        scene_dir = "data/realdata/%s" % scene
        n_runs = args.runs or int(s.get("runs_planned", 1))
        print("\n=== %s (%s) ===" % (scene, scan), flush=True)

        # ---------------- 変換 ----------------
        conv = read_json(os.path.join(scene_dir, "conversion_report.json"))
        if args.resume and conv.get("n_frames_final"):
            print("  convert: skip (%d frames already)" % conv["n_frames_final"])
        else:
            t = time.time()
            rc = sh("python tools/realdata/convert_stray.py --scan %s --scene %s "
                    "--mode %s --conf-min %d --max-depth %s --out data/realdata"
                    % (scan, scene, args.mode, args.conf_min, args.max_depth),
                    "sni-slam", "%s/convert_%s.log" % (log_dir, scene))
            print("  convert rc=%d (%.0fs)" % (rc, time.time() - t))
            conv = read_json(os.path.join(scene_dir, "conversion_report.json"))
            if rc != 0 or not conv.get("n_frames_final"):
                rows.append({"scene": scene, "run": "-", "status": "convert_failed"})
                continue

        # ---------------- 意味ラベル ----------------
        lab_path = os.path.join(scene_dir, "label_report_semantic_class.json")
        lab = read_json(lab_path)
        if args.resume and lab.get("n_frames") == conv.get("n_frames_final"):
            print("  labels: skip (%d frames already)" % lab["n_frames"])
        else:
            t = time.time()
            rc = sh("python tools/realdata/gen_labels_ade20k.py --scene-dir %s --batch %d"
                    % (scene_dir, args.label_batch),
                    "seg2d", "%s/labels_%s.log" % (log_dir, scene))
            print("  labels rc=%d (%.0fs)" % (rc, time.time() - t))
            lab = read_json(lab_path)
            if rc != 0:
                rows.append({"scene": scene, "run": "-", "status": "labels_failed"})
                continue

        # ---------------- 契約チェック ----------------
        val_path = "%s/validate_%s.json" % (log_dir, scene)
        sh("python Registration/scripts/validate_scene_data.py %s --out %s"
           % (scene_dir, val_path), "sni-slam")
        val = read_json(val_path)

        if args.skip_slam:
            rows.append({"scene": scene, "run": "-", "status": "prepared_only",
                         "n_frames": conv.get("n_frames_final")})
            continue

        # ---------------- SLAM ----------------
        for r in range(1, n_runs + 1):
            out_dir = "%s/%s/run%d" % (args.out_root, scene, r)
            ate_path = os.path.join(out_dir, "eval_ate.json")
            if args.resume and os.path.exists(ate_path):
                print("  run%d: skip (eval_ate.json exists)" % r)
            else:
                t = time.time()
                rc = sh("python -W ignore run.py %s/%s.yaml --output %s"
                        % (args.config_dir, scene, out_dir), "sni-slam",
                        "%s/slam_%s_run%d.log" % (log_dir, scene, r))
                elapsed = time.time() - t
                print("  run%d rc=%d (%.0f min)" % (r, rc, elapsed / 60))
                if rc != 0:
                    rows.append({"scene": scene, "run": r, "status": "slam_failed",
                                 "elapsed_min": round(elapsed / 60, 1),
                                 "git_head": head, "seed": "none"})
                    continue
                sh("python src/tools/eval_ate.py %s/%s.yaml --output %s"
                   % (args.config_dir, scene, out_dir), "sni-slam",
                   "%s/ate_%s_run%d.log" % (log_dir, scene, r))
                sh("python tools/realdata/precheck_scene.py --mesh %s/mesh/final_mesh_semantic.ply "
                   "--traj-gt %s/traj.txt --est-poses %s/ckpts --scene %s --out %s/precheck_%s.json"
                   % (out_dir, scene_dir, out_dir, scene, out_dir, scene), "sni-slam")
                if args.project_labels:
                    # 追補3 §3: SNI-SLAM の意味フィールドを使わず 2D ラベルを投影する
                    sh("python tools/realdata/project_labels_to_mesh.py "
                       "--mesh %s/mesh/final_mesh_semantic.ply --scene-dir %s --config %s/%s.yaml "
                       "--out %s/mesh/final_mesh_semantic_projected.ply --frame-stride 8"
                       % (out_dir, scene_dir, args.config_dir, scene, out_dir), "sni-slam",
                       "%s/project_%s.log" % (log_dir, scene))
                if not args.keep_feat_cache:
                    fc = os.path.join(out_dir, "feat_cache")
                    if os.path.isdir(fc):
                        subprocess.call(["rm", "-rf", fc])
                        print("  removed %s" % fc)

            ate = read_json(ate_path)
            pre = read_json(os.path.join(out_dir, "precheck_%s.json" % scene))
            proj = read_json(os.path.join(
                out_dir, "mesh", "final_mesh_semantic_projected_report.json"))
            rows.append({
                "scene": scene, "scan_id": scan, "run": r, "status": "ok",
                "n_frames": conv.get("n_frames_final"),
                "mode": conv.get("mode"), "conf_min": conv.get("conf_min"),
                "max_depth_m": conv.get("max_depth_m"),
                "ate_mean": ate.get("absolute_translational_error.mean"),
                "ate_median": ate.get("absolute_translational_error.median"),
                "ate_rmse": ate.get("absolute_translational_error.rmse"),
                "gravity_tilt_deg": pre.get("gravity_tilt_deg"),
                "plane_diversity": (pre.get("plane_diversity") or {}).get("n_directions"),
                "slam_start_end_m": (pre.get("drift") or {}).get("slam_start_end_m"),
                "arkit_start_end_m": (pre.get("drift") or {}).get("arkit_start_end_m"),
                "validate_pass": val.get("ok", val.get("passed")),
                "gravity_ok": pre.get("gravity_ok"),
                "wall_dirs_deg": (pre.get("plane_diversity") or {}).get("directions_deg"),
                "proj_voted_frac": proj.get("vertices_with_votes_frac"),
                "proj_class_frac": proj.get("class_fraction_voted_only"),
                "config_dir": args.config_dir,
                "git_head": head, "seed": "none",
            })

            # 途中で落ちても結果が残るよう、run ごとに CSV を書き直す
            _write_csv(args.out_csv, rows)

    _write_csv(args.out_csv, rows)
    print("\nwrote %s (%d rows)" % (args.out_csv, len(rows)))
    return 0


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    raise SystemExit(main())

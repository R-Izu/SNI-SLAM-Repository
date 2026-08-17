"""トラッキングのハイパーパラメータ探索（実データ向け）。

背景
----
SNI-SLAM のトラッキング設定は Replica（合成・ノイズ無し depth・なめらかな軌跡）で
調整されている。実データ（iPad 手持ち）では ATE 中央値 1.12 m・経路長 +28% と
収束せず、軌跡が高周波にジッタしていた（報告 E-4）。
`tracking.iters` / `lr_T` / `lr_R` / `pixels` を振って ATE が下がる設定を探す。

高速化
------
1 試行を短くするため、シーンの先頭 N フレームだけを**ハードリンク**で切り出した
試験シーンを作る（ディスクを消費せず一瞬で作れる）。
ATE の絶対値は全長のものと比較できないが、**条件間の相対比較には使える**。

使い方
------
    conda activate sni-slam
    python tools/realdata/tune_tracking.py --scene m3_room_a --frames 800
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import time
from typing import Dict, List

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CONDA_SH = "/opt/miniconda/3/etc/profile.d/conda.sh"

# label -> tracking の上書き設定
# 既定は configs/SNI-SLAM.yaml:24-38 の値
#   iters=8, lr_T=0.002, lr_R=0.001, pixels=2000, w_color=5, w_depth=1
#
# 探索の狙いは2系統：
#  (a) 最適化量を増やす（iters / lr / pixels）— 手持ちの速い動きに追従できていない仮説
#  (b) **depth を信じ色を軽くする**（w_depth / w_color）— Replica では色が完全なので
#      w_color=5 > w_depth=1 で妥当だが、実データはモーションブラーと自動露出があり
#      色の信頼度が低い。LiDAR depth の方が信頼できるので重みを逆転させる
TRIALS = {
    "t0_baseline":      {},
    "t1_iters20":       {"iters": 20},
    "t2_iters20_lr2":   {"iters": 20, "lr_T": 0.004, "lr_R": 0.002},
    "t3_iters30_lr4":   {"iters": 30, "lr_T": 0.008, "lr_R": 0.004},
    "t4_px4000":        {"iters": 20, "lr_T": 0.004, "lr_R": 0.002, "pixels": 4000},
    "t5_depth_heavy":   {"w_color": 1, "w_depth": 10},
    "t6_combo":         {"iters": 20, "lr_T": 0.004, "lr_R": 0.002,
                         "w_color": 1, "w_depth": 10},
    "t7_depth_only":    {"w_color": 0.1, "w_depth": 10},
    # --- t0/t5/t6 の結果を受けて追加した条件 -------------------------------
    # 実測: t0(既定) 0.267 m < t5(depth重視) 0.599 m < t6(iters20+lr2) 1.312 m。
    # lr を上げるほど悪化した＝最適化が「足りない」のではなく「行き過ぎている」。
    # そこで逆向き（lr を下げて反復を増やす）を試す。
    "t8_iters20_lr_half": {"iters": 20, "lr_T": 0.001, "lr_R": 0.0005},
    # 手持ちの動きは急峻なので、等速度予測での初期化がむしろ外していないかを見る。
    # 症状（フレーム単位のジッタ）とよく合う仮説。
    "t9_no_const_speed":  {"const_speed_assumption": False},
}


def sh(cmd: str, env: str = "sni-slam", log: str = None) -> int:
    full = "source %s && conda activate %s && cd %s && %s" % (CONDA_SH, env, REPO, cmd)
    if log:
        full += " > %s 2>&1" % log
    return subprocess.call(["bash", "-lc", full])


def make_subset_scene(scene: str, n: int) -> str:
    """先頭 n フレームだけの試験シーンをハードリンクで作る。"""
    src = os.path.join(REPO, "data/realdata", scene)
    dst = os.path.join(REPO, "data/realdata", "%s_t%d" % (scene, n))
    if os.path.exists(dst):
        shutil.rmtree(dst)
    for sub in ("rgb", "depth", "semantic_class"):
        os.makedirs(os.path.join(dst, sub))
        stem = {"rgb": "rgb", "depth": "depth", "semantic_class": "semantic_class"}[sub]
        for i in range(n):
            s = os.path.join(src, sub, "%s_%d.png" % (stem, i))
            d = os.path.join(dst, sub, "%s_%d.png" % (stem, i))
            os.link(s, d)          # ハードリンク: ディスクを消費しない
    with open(os.path.join(src, "traj.txt")) as f:
        lines = f.readlines()[:n]
    with open(os.path.join(dst, "traj.txt"), "w") as f:
        f.writelines(lines)
    shutil.copy(os.path.join(src, "conversion_report.json"),
                os.path.join(dst, "conversion_report.json"))
    return dst


def write_trial_config(base_cfg: str, out_cfg: str, scene_dir: str, out_dir: str,
                       overrides: Dict) -> None:
    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)
    cfg["tracking"] = dict(cfg.get("tracking", {}))
    cfg["tracking"].update(overrides)
    cfg["data"] = {"input_folder": scene_dir, "output": out_dir}
    with open(out_cfg, "w") as f:
        f.write("# 自動生成: tools/realdata/tune_tracking.py（探索用。手で編集しない）\n")
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="m3_room_a")
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--trials", nargs="*", default=None, help="ラベルで絞る")
    ap.add_argument("--out-json", default="output/RealData/_tuning/tuning_results.json")
    args = ap.parse_args()

    os.chdir(REPO)
    scene_dir = make_subset_scene(args.scene, args.frames)
    rel_scene = os.path.relpath(scene_dir, REPO)
    print("subset scene: %s (%d frames, hardlinked)" % (rel_scene, args.frames))

    base_cfg = "configs/RealData/%s.yaml" % args.scene
    os.makedirs("configs/RealData/_tuning", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    labels = args.trials or list(TRIALS)
    results: List[Dict] = []
    for label in labels:
        ov = TRIALS[label]
        cfg_path = "configs/RealData/_tuning/%s_%s.yaml" % (args.scene, label)
        out_dir = "output/RealData/_tuning/%s_%s" % (args.scene, label)
        write_trial_config(base_cfg, cfg_path, rel_scene, out_dir, ov)
        print("\n=== %s: %s ===" % (label, ov or "(既定値)"), flush=True)
        t0 = time.time()
        log = "output/RealData/_logs/tune_%s_%s.log" % (args.scene, label)
        rc = sh("python -W ignore run.py %s --output %s" % (cfg_path, out_dir), log=log)
        el = time.time() - t0
        rec = {"label": label, "overrides": ov, "rc": rc,
               "minutes": round(el / 60, 1)}
        if rc == 0:
            sh("python src/tools/eval_ate.py %s --output %s" % (cfg_path, out_dir),
               log="output/RealData/_logs/tune_ate_%s.log" % label)
            ate_path = os.path.join(out_dir, "eval_ate.json")
            if os.path.exists(ate_path):
                with open(ate_path) as f:
                    a = json.load(f)
                # eval_ate.py:240-245 は cm で出す
                rec["ate_median_m"] = round(a["absolute_translational_error.median"] / 100, 4)
                rec["ate_mean_m"] = round(a["absolute_translational_error.mean"] / 100, 4)
                rec["ate_rmse_m"] = round(a["absolute_translational_error.rmse"] / 100, 4)
        shutil.rmtree(os.path.join(out_dir, "feat_cache"), ignore_errors=True)
        results.append(rec)
        print("  -> rc=%d  ATE median=%s m  (%.1f min)" % (
            rc, rec.get("ate_median_m", "n/a"), rec["minutes"]), flush=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n%-18s %-50s %11s %7s" % ("trial", "overrides", "ATE med [m]", "min"))
    print("-" * 90)
    for r in results:
        print("%-18s %-50s %11s %7.1f" % (
            r["label"], str(r["overrides"] or "(既定値)"),
            r.get("ate_median_m", "FAILED"), r["minutes"]))
    print("\nwrote %s" % args.out_json)
    print("注意: %d フレームの部分列での値。全長の ATE とは比較しないこと（条件間の相対比較用）。"
          % args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""R26 §1 — **旧 T_gt が独立 GT からどれだけ離れているか**を測る。

なぜこれが最優先か
------------------
合成データの $T_{gt}^{\\text{旧}}$ は、**提案手法自身の無摂動解を凍結したもの**である
（`establish_gt.py` の docstring、R7 §B）。**正解かどうかは一度も確かめられていない。**
**旧 T_gt が正しければ 83.8% は生き、間違っていれば死ぬ。**

独立 GT の作り方（**提案手法を一切使わない**）
---------------------------------------------
SLAM は自分の世界座標系でメッシュを作る。Replica は `traj.txt` で GT カメラ姿勢を与える。
**この2つの姿勢列を結ぶ Sim(3) が、SLAM 座標系から Replica 座標系への変換である。**
それを 2000 対の姿勢に Umeyama を当てて求める。**登録器も提案手法も出てこない。**

    T_indep = Umeyama( estimate_c2w の位置 -> gt_c2w の位置 )

**★ 指示書 §1 は「GT-A（深度の再投影）で」と書いてあるが、GT-A ではこの量は作れない。**
  GT-A は **Replica 座標系で新しい source を作る**手順であり、既知の Sim(3) を自分で掛けるので
  正解が解析的に決まる。**しかしそれは新しいベンチマークであって、旧 T_gt の監査ではない。**
  旧 T_gt は「SLAM メッシュ → Replica 参照」の変換なので、
  **これと比べられる独立量は、姿勢列から出す変換しかない**（＝指示書の GT-B）。
  **この読み替えを勝手にやらず、報告に書いて main の判断を仰ぐ。**

**独立 GT 自身の確からしさも測る**（R26 §2-1 の警告に対応）::

    前半 1000 姿勢だけで合わせた T と、後半 1000 姿勢だけで合わせた T の d_Ω
    → これが独立 GT 自体の揺らぎである。**旧 T_gt との距離が、この揺らぎより
      小さければ「差は無い」としか言えない。**

出力は 1 シーンごとに書き出す（R23 §4-6：途中で落ちても残す）。

    conda activate sni-slam
    python Registration/scripts/r26_gt_distance.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from criterion_verdict import build_omega, d_omega      # noqa: E402
from failure_decomposition import provenance            # noqa: E402
from regbim.metrics import decompose_sim3, rotation_error_deg   # noqa: E402
from regbim.scale import umeyama                        # noqa: E402

# R26 §1-2 の事前登録された判定境界。**結果を見てから動かさない。**
BAND_SAME = 0.05
BAND_DEAD = 0.10


def sim3(R, t, s) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def fit(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    R, t, s = umeyama(est, gt, with_scaling=True)
    return sim3(R, t, s)


def latest_ckpt(run_dir: str) -> str:
    tars = sorted(glob.glob(os.path.join(run_dir, "ckpts", "*.tar")))
    if not tars:
        raise FileNotFoundError("ckpt が無い: %s" % run_dir)
    return tars[-1]


def verdict(d: float) -> str:
    if d < BAND_SAME:
        return "同一とみなせる（< %.2f m）" % BAND_SAME
    if d <= BAND_DEAD:
        return "境界（%.2f–%.2f m）" % (BAND_SAME, BAND_DEAD)
    return "旧値は無効（> %.2f m）" % BAND_DEAD


def one(scene: str, run_dir: str) -> dict:
    cfg = yaml.safe_load(open("Registration/configs/sectionC/%s.yaml" % scene))
    ck_path = latest_ckpt(run_dir)
    ck = torch.load(ck_path, map_location="cpu")
    est = np.asarray(ck["estimate_c2w_list"], dtype=np.float64)[:, :3, 3]
    gt = np.asarray(ck["gt_c2w_list"], dtype=np.float64)[:, :3, 3]
    n = len(est)

    # ckpt の gt_c2w_list が本当に traj.txt かを確かめる（読み間違いの検出）
    traj_check = None
    tp = "data/replica/%s_official/traj.txt" % scene
    if os.path.exists(tp):
        tj = np.loadtxt(tp).reshape(-1, 4, 4)[:n, :3, 3]
        traj_check = float(np.abs(tj - gt).max())

    T_ind = fit(est, gt)
    resid = float(np.sqrt(((gt - (est @ T_ind[:3, :3].T + T_ind[:3, 3])) ** 2)
                          .sum(axis=1).mean()))

    T_old = np.asarray(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"],
                       dtype=np.float64).reshape(4, 4)

    c = dict(cfg, source=dict(cfg["source"], seed=0))
    omega = build_omega(c, seed=0)

    h = n // 2
    T_a, T_b = fit(est[:h], gt[:h]), fit(est[h:], gt[h:])
    d_split = d_omega(T_a, T_b, omega)          # 独立 GT 自体の揺らぎ
    d_main = d_omega(T_old, T_ind, omega)       # **これが §1 の量**

    R_o, _, s_o = decompose_sim3(T_old)
    R_i, _, s_i = decompose_sim3(T_ind)
    return {"scene": scene, "run_dir": run_dir, "ckpt": ck_path, "n_poses": n,
            "traj_matches_ckpt_max_abs_m": traj_check,
            "t_gt_old_path": cfg["eval"]["t_gt_path"],
            "umeyama_residual_rms_m": resid,
            "d_omega_old_vs_independent_m": d_main,
            "d_omega_split_half_m": d_split,
            "rotation_error_deg": float(rotation_error_deg(R_o, R_i)),
            "scale_old": float(s_o), "scale_independent": float(s_i),
            "scale_ratio": float(abs(s_o / s_i - 1.0)),
            "omega_n": int(len(omega)),
            "verdict": verdict(d_main),
            "T_independent": T_ind.tolist(), "T_gt_old": T_old.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="Registration/configs/sectionC/manifest.yaml")
    ap.add_argument("--out", default="Registration/output/diag/r26_gt_distance.json")
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    man = yaml.safe_load(open(args.manifest))
    scenes = [s for s in man["scenes"]
              if args.scenes is None or s["name"] in args.scenes]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    print("%-10s %8s %14s %14s %12s %10s  %s"
          % ("シーン", "姿勢数", "d_Ω 旧vs独立", "半々の揺らぎ", "回転差[度]",
             "縮尺比", "判定"))
    print("-" * 104)
    for s in scenes:
        run_dir = os.path.dirname(s["ate_json"])
        try:
            r = one(s["name"], run_dir)
        except Exception as e:                       # 1 シーン落ちても続ける
            r = {"scene": s["name"], "run_dir": run_dir, "error": "%s: %s"
                 % (type(e).__name__, e)}
            print("%-10s **失敗** %s" % (s["name"], r["error"]))
        else:
            print("%-10s %8d %14.4f %14.4f %12.3f %10.5f  %s"
                  % (r["scene"], r["n_poses"], r["d_omega_old_vs_independent_m"],
                     r["d_omega_split_half_m"], r["rotation_error_deg"],
                     r["scale_ratio"], r["verdict"]))
        rows.append(r)
        # **1 件ごとに書き出す**（R23 §4-6）
        with open(args.out, "w") as f:
            json.dump({"provenance": provenance(),
                       "bands_m": {"same": BAND_SAME, "dead": BAND_DEAD},
                       "note": "独立 GT は姿勢列の Umeyama（指示書の GT-B）。"
                               "GT-A ではこの量は作れない——本文 docstring 参照",
                       "rows": rows}, f, indent=2, ensure_ascii=False)

    ok = [r for r in rows if "error" not in r]
    if ok:
        d = [r["d_omega_old_vs_independent_m"] for r in ok]
        print("\n**シーン別に出している。平均に潰さない。**")
        print("範囲 [%.4f, %.4f] m / 中央 %.4f m（n=%d）"
              % (min(d), max(d), float(np.median(d)), len(d)))
        for name, lo, hi in (("同一とみなせる", 0.0, BAND_SAME),
                             ("境界", BAND_SAME, BAND_DEAD),
                             ("旧値は無効", BAND_DEAD, np.inf)):
            k = sum(1 for x in d if lo <= x < hi)
            print("  %-16s %d / %d シーン" % (name, k, len(d)))
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

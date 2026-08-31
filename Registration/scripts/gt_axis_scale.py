"""残っている縮尺のずれが**等方か異方か**を切り分ける。

なぜ切り分けるか
----------------
手動 GT（剛体）に対して ICP が求める追加縮尺は、室内面だけに絞っても中央値 +2.17% ある。
一方、**天井高から出る独立な推定は 0.9%** である
（S0 が LiDAR 深度から実測した天井高 2.576〜2.578 m ／ BIM の室高 2.600 m）。
この食い違いは、次のどちらかを意味する。

- **等方に 2% 小さい** → 深度そのものの倍率誤差。天井高の 0.9% と矛盾するので考えにくい
- **水平だけ余計に小さい** → **ARKit の並進ドリフト**。歩いた方向に縮む/伸びるのは
  VIO の典型的な破綻であり、鉛直（重力で拘束される）には出にくい

前者なら深度の較正、後者なら軌跡の問題であり、**対処がまったく違う**。

軸は BIM の座標系（+Z 鉛直）で取る。対応は最近傍で作り、
各軸ごとに `q_axis = a * p_axis + b` を最小二乗で解く。

    conda activate sni-slam
    python Registration/scripts/gt_axis_scale.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", default="output/GT_alignment")
    ap.add_argument("--npz", default="Registration/output/ifc/m3_ifc_all.npz")
    ap.add_argument("--max-corr", type=float, default=0.30)
    ap.add_argument("--voxel", type=float, default=0.05)
    args = ap.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", ".."))
    os.chdir(repo)
    import open3d as o3d

    z = np.load(args.npz, allow_pickle=False)
    bim = z["points"][z["is_inner"].astype(bool)]      # 室内側の面のみ
    ref = o3d.geometry.PointCloud()
    ref.points = o3d.utility.Vector3dVector(bim)
    ref = ref.voxel_down_sample(args.voxel)
    ref_pts = np.asarray(ref.points)
    tree = o3d.geometry.KDTreeFlann(ref)

    rows: List[Dict] = []
    print("%-12s %9s %9s %9s   %s" % ("scene", "x 倍率", "y 倍率", "z 倍率", "対応点数"))
    print("-" * 58)
    for p in sorted(glob.glob(os.path.join(args.kit, "T_gt", "T_gt_*.json"))):
        scene = os.path.basename(p)[5:-5]
        T = np.asarray(json.load(open(p))["T_gt"], dtype=np.float64).reshape(4, 4)
        sp = os.path.join(args.kit, "source", "%s.ply" % scene)
        if not os.path.exists(sp):
            continue
        m = o3d.io.read_triangle_mesh(sp)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(m.vertices))
        pc.transform(T)
        pc = pc.voxel_down_sample(args.voxel)
        q = np.asarray(pc.points)

        # 最近傍対応（ゲート内のみ）
        idx = np.full(len(q), -1, dtype=np.int64)
        for i, x in enumerate(q):
            k, ii, d2 = tree.search_knn_vector_3d(x, 1)
            if k > 0 and d2[0] <= args.max_corr ** 2:
                idx[i] = ii[0]
        ok = idx >= 0
        if ok.sum() < 2000:
            print("%-12s 対応が %d しかない。飛ばす" % (scene, ok.sum()))
            continue
        P, Q = q[ok], ref_pts[idx[ok]]

        a = []
        for k in range(3):
            A = np.stack([P[:, k], np.ones(len(P))], axis=1)
            sol, *_ = np.linalg.lstsq(A, Q[:, k], rcond=None)
            a.append(float(sol[0]))
        rows.append({"scene": scene, "scale_x": a[0], "scale_y": a[1], "scale_z": a[2],
                     "n_corr": int(ok.sum())})
        print("%-12s %9.4f %9.4f %9.4f   %s"
              % (scene, a[0], a[1], a[2], "{:,}".format(int(ok.sum()))))

    if rows:
        A = np.array([[r["scale_x"], r["scale_y"], r["scale_z"]] for r in rows])
        print("\n【軸別の倍率】n=%d（1.0 が一致。>1 は source が小さいことを意味する）" % len(rows))
        for k, nm in enumerate(("x（水平）", "y（水平）", "z（鉛直）")):
            print("  %-10s 中央値 %.4f（%+.2f%%） / 標準偏差 %.4f"
                  % (nm, np.median(A[:, k]), 100 * (np.median(A[:, k]) - 1), A[:, k].std()))
        hor = np.median(A[:, :2], axis=1)
        ver = A[:, 2]
        print("\n  水平（x,y の中央値）%.4f（%+.2f%%） vs 鉛直 %.4f（%+.2f%%）"
              % (np.median(hor), 100 * (np.median(hor) - 1),
                 np.median(ver), 100 * (np.median(ver) - 1)))
        print("\n  対照：S0 が LiDAR 深度から実測した天井高 2.576〜2.578 m ／ BIM の室高 2.600 m")
        print("        → 鉛直方向の独立な推定は **%.2f%%**" % (100 * (2.600 / 2.577 - 1)))
        print("\n  読み方：鉛直だけが小さく水平が大きいなら **ARKit の並進ドリフト**。")
        print("          等方に大きいなら深度の倍率誤差。")
    out = os.path.join(args.kit, "gt_axis_scale.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

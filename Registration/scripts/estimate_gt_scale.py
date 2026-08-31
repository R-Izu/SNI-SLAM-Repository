"""手動 GT が剛体（縮尺 1.0 固定）だったので、**残っている縮尺のずれ**を推定する。

なぜ要るか
----------
R5 §6-4 は「**GT スケールの 1.0 からの乖離は、それ自体が入力の実寸精度の実測値になる。
必ず記録すること**」と指示している。しかし CloudCompare で `adjust scale` が OFF
だったため、10 本とも縮尺が**厳密に 1.000000000000**（det=1、異方性 0）で、
その量が測れていない。

手作業をやり直す前に、**やり直す必要があるかを数値で判断する**ために推定する。

やり方
------
手動 GT を初期値として、**Open3D の点対点 ICP（`with_scaling=True`）**を回し、
追加で掛かる縮尺を読む。提案手法の ICP は使わない（自分の手法で自分の GT を
検証すると循環するため）。対応は **BIM が幾何を持つ領域に限る**
（BIM は観測の一部しか覆わないので、廊下まで含めると被覆率を測ってしまう）。

読み方（過大な主張をしない）
----------------------------
- これは **GT を初期値にした推定**であって、手動 GT と独立ではない
- 得られるのは「剛体 GT のままで、あとどれだけ縮尺がずれているか」の**目安**
- **1% 未満なら、剛体 GT のままで実害は小さい**と判断してよい
- **数 % あるなら、`adjust scale` を ON にして取り直す価値がある**

    conda activate sni-slam
    python Registration/scripts/estimate_gt_scale.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np

CLASS_NAMES = ["background", "wall", "door", "floor", "window", "ceiling"]
PALETTE = np.array([(128, 128, 128), (255, 64, 64), (255, 200, 64),
                    (180, 220, 255), (64, 200, 255), (200, 100, 255)],
                   dtype=np.float64) / 255.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", default="output/GT_alignment")
    ap.add_argument("--max-corr", type=float, default=0.30,
                    help="対応の上限距離 [m]。手法の max_corr_dist と揃える")
    ap.add_argument("--near-bim", type=float, default=0.50,
                    help="BIM からこの距離以内の source 点だけ使う [m]。"
                         "BIM に無い廊下を対応付けから外すため")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--npz", default="Registration/output/ifc/m3_ifc_all.npz")
    ap.add_argument("--inner-only", action="store_true",
                    help="BIM の**室内側の面**だけを対応先にする（R5 Q6 の is_inner）。"
                         "IFC はソリッドなので壁は表裏2枚あり、SLAM が見るのは内側だけ。"
                         "外側の面に対応が吸われると、source を膨らませる向きに"
                         "系統的な縮尺が出る（壁厚 0.15〜0.20 m ÷ 室幅 7 m ≒ 4%）")
    args = ap.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", ".."))
    os.chdir(repo)
    import open3d as o3d

    z = np.load(args.npz, allow_pickle=False)
    bim_pts = z["points"]
    if args.inner_only:
        if "is_inner" not in z.files:
            raise SystemExit("npz に is_inner が無い。ifc_export.py を回し直すこと")
        bim_pts = bim_pts[z["is_inner"].astype(bool)]
        print("BIM: 室内側の面のみ %s / %s 点\n"
              % ("{:,}".format(len(bim_pts)), "{:,}".format(len(z["points"]))))
    ref_pc = o3d.geometry.PointCloud()
    ref_pc.points = o3d.utility.Vector3dVector(bim_pts)
    ref_pc = ref_pc.voxel_down_sample(args.voxel)
    ref_pts = np.asarray(ref_pc.points)
    ref_tree = o3d.geometry.KDTreeFlann(ref_pc)

    rows: List[Dict] = []
    print("%-12s %10s %12s %12s %10s"
          % ("scene", "追加縮尺", "1.0からの%", "対応点数", "適合度"))
    print("-" * 62)
    for p in sorted(glob.glob(os.path.join(args.kit, "T_gt", "T_gt_*.json"))):
        scene = os.path.basename(p)[5:-5]
        T = np.asarray(json.load(open(p))["T_gt"], dtype=np.float64).reshape(4, 4)
        sp = os.path.join(args.kit, "source", "%s.ply" % scene)
        if not os.path.exists(sp):
            continue
        m = o3d.io.read_triangle_mesh(sp)
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(np.asarray(m.vertices))
        src.transform(T)                       # 手動 GT を適用してから測る
        src = src.voxel_down_sample(args.voxel)

        # BIM が幾何を持つ領域に限る（未被覆を含めると被覆率を測ってしまう）
        q = np.asarray(src.points)
        keep = np.zeros(len(q), dtype=bool)
        for i, x in enumerate(q):
            k, _, d2 = ref_tree.search_knn_vector_3d(x, 1)
            keep[i] = k > 0 and d2[0] <= args.near_bim ** 2
        if keep.sum() < 1000:
            print("%-12s BIM 近傍の点が %d しかない。飛ばす" % (scene, keep.sum()))
            continue
        near = o3d.geometry.PointCloud()
        near.points = o3d.utility.Vector3dVector(q[keep])

        est = o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True)
        res = o3d.pipelines.registration.registration_icp(
            near, ref_pc, args.max_corr, np.eye(4), est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        A = np.asarray(res.transformation)[:3, :3]
        s = float(np.cbrt(abs(np.linalg.det(A))))
        rows.append({"scene": scene, "extra_scale": s,
                     "deviation_pct": 100 * (s - 1.0),
                     "n_points_near_bim": int(keep.sum()),
                     "n_points_total": int(len(q)),
                     "fitness": float(res.fitness),
                     "inlier_rmse_m": float(res.inlier_rmse)})
        print("%-12s %10.5f %11.2f%% %12s %10.3f"
              % (scene, s, 100 * (s - 1.0), "{:,}".format(int(keep.sum())),
                 res.fitness))

    if rows:
        d = np.array([r["deviation_pct"] for r in rows])
        print("\n【残っている縮尺のずれ】n=%d" % len(rows))
        print("  中央値 %+.2f%% / 平均 %+.2f%% / 標準偏差 %.2f%% / 範囲 [%+.2f%%, %+.2f%%]"
              % (np.median(d), d.mean(), d.std(), d.min(), d.max()))
        print("  |ずれ| の中央値 %.2f%%" % np.median(np.abs(d)))
        print("\n  判断の目安：")
        print("    1%% 未満 → 剛体 GT のままで実害は小さい")
        print("    数 %%    → adjust scale を ON にして取り直す価値がある")
        print("\n  ※ これは手動 GT を初期値にした推定であり、手動 GT と独立ではない。")
        print("     『入力の実寸精度』として報告するなら、本来は adjust scale を ON にした")
        print("     手動 GT から直接読むべき量である。")
    out = os.path.join(args.kit, "estimate_gt_scale.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

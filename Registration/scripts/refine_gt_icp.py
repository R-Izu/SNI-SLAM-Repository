"""手動 GT を初期値に、**幾何のみの ICP** で GT を精緻化する。

なぜ要るか
----------
撮影者の実感：「**ボコボコした SLAM 点群と綺麗な BIM の位置合わせが難しく、ズレている可能性**」。
測定もそれと整合する —— 床と天井が**ほぼ同じだけ**上に動いており（+0.090 / +0.083）、
これは「対応点の選び方で全体が上下にずれた」形そのものである
（床だけの物理的な差なら、床と天井が同じだけ動くことはない）。

R5 §6-3 は CloudCompare の手順として **`Fine registration (ICP)` での精緻化を既に挙げている**。
ここでやるのはそれを機械的に、全シーン同条件で行うことである。

循環しないこと（重要）
----------------------
- 使うのは **Open3D の点対面 ICP のみ**。**提案手法（意味 ICP・マンハッタン拘束）は使わない。**
  自分の手法で自分の GT を作ると評価が循環する（R5 §6-5）
- 初期値は**人が独立に作った手動 GT**。ICP は局所の精緻化だけを担う
- **縮尺は入れない（剛体）。** ずれが鉛直だけで異方的なので、等方の縮尺を入れると
  いま 0.2〜0.4% しかない水平が悪化する

対応先を絞る2つの条件
---------------------
1. **BIM の室内側の面のみ**（`is_inner`）。IFC はソリッドで壁が表裏2枚あり、
   外側に吸われると source を膨らませる向きに系統誤差が出る（実測で 1.4 ポイント）
2. **BIM の近傍にある source 点のみ**。BIM は 2 室だけで廊下を含まないので、
   廊下まで対応付けると被覆率に引きずられる

    conda activate sni-slam
    python Registration/scripts/refine_gt_icp.py            # 見るだけ
    python Registration/scripts/refine_gt_icp.py --write    # T_gt を更新する
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from typing import Dict, List, Optional

import numpy as np

CLASS_NAMES = ["background", "wall", "door", "floor", "window", "ceiling"]
PALETTE = np.array([(128, 128, 128), (255, 64, 64), (255, 200, 64),
                    (180, 220, 255), (64, 200, 255), (200, 100, 255)],
                   dtype=np.float64) / 255.0


def plane_z(pts: np.ndarray, nrm: np.ndarray, mask: np.ndarray,
            want_up: bool, bin_m: float = 0.01) -> Optional[float]:
    m = mask & (np.abs(nrm[:, 2]) > 0.9)
    m &= (nrm[:, 2] > 0) if want_up else (nrm[:, 2] < 0)
    z = pts[m, 2]
    if len(z) < 100:
        return None
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + bin_m, bin_m))
    if not len(hist):
        return float(np.median(z))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def rot_deg(A: np.ndarray, B: np.ndarray) -> float:
    c = (np.trace(A @ B.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", default="output/GT_alignment")
    ap.add_argument("--npz", default="Registration/output/ifc/m3_ifc_all.npz")
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--near-bim", type=float, default=0.60)
    ap.add_argument("--stages", type=float, nargs="*", default=[0.30, 0.15, 0.08],
                    help="対応の上限距離を段階的に詰める [m]")
    ap.add_argument("--write", action="store_true",
                    help="T_gt_<scene>.json を精緻化後の値で更新する"
                         "（手動版は T_gt_manual/ に退避する）")
    args = ap.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", ".."))
    os.chdir(repo)
    import open3d as o3d

    z = np.load(args.npz, allow_pickle=False)
    inner = z["is_inner"].astype(bool)
    ref = o3d.geometry.PointCloud()
    ref.points = o3d.utility.Vector3dVector(z["points"][inner])
    ref.normals = o3d.utility.Vector3dVector(z["normals"][inner])
    ref = ref.voxel_down_sample(args.voxel)
    ref.normalize_normals()
    ref_pts = np.asarray(ref.points)
    ref_nrm = np.asarray(ref.normals)
    b_fl = plane_z(z["points"], z["normals"], z["labels"] == 3, True)
    b_ce = plane_z(z["points"], z["normals"], z["labels"] == 5, False)
    ref_tree = o3d.geometry.KDTreeFlann(ref)
    print("BIM（室内側の面のみ）%s 点 / 床 %.4f / 天井 %.4f"
          % ("{:,}".format(len(ref_pts)), b_fl, b_ce))
    print()
    hdr = ("%-12s %8s %8s %9s %9s %9s %9s"
           % ("scene", "移動m", "回転°", "床前", "床後", "天井前", "天井後"))
    print(hdr); print("-" * len(hdr))

    rows: List[Dict] = []
    for p in sorted(glob.glob(os.path.join(args.kit, "T_gt", "T_gt_*.json"))):
        scene = os.path.basename(p)[5:-5]
        T0 = np.asarray(json.load(open(p))["T_gt"], dtype=np.float64).reshape(4, 4)
        sp = os.path.join(args.kit, "source", "%s.ply" % scene)
        pj = ("output/RealData/_TSDF/%s/run1/mesh/final_mesh_semantic_projected.ply"
              % scene)
        if not (os.path.exists(sp) and os.path.exists(pj)):
            continue
        m = o3d.io.read_triangle_mesh(sp)
        m.compute_vertex_normals()
        v = np.asarray(m.vertices)
        n = np.asarray(m.vertex_normals)
        mp = o3d.io.read_triangle_mesh(pj)
        cp = np.asarray(mp.vertex_colors)
        lab = (np.argmin(((cp[:, None, :] - PALETTE[None]) ** 2).sum(2), axis=1)
               if len(cp) == len(v) else None)

        # 手動 GT を適用した状態から始める
        v0 = v @ T0[:3, :3].T + T0[:3, 3]
        n0 = n @ T0[:3, :3].T
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(v0)
        src.normals = o3d.utility.Vector3dVector(n0)
        src = src.voxel_down_sample(args.voxel)

        # BIM が幾何を持つ領域だけ残す（廊下を対応付けに入れない）
        q = np.asarray(src.points)
        keep = np.zeros(len(q), dtype=bool)
        for i, x in enumerate(q):
            k, _, d2 = ref_tree.search_knn_vector_3d(x, 1)
            keep[i] = k > 0 and d2[0] <= args.near_bim ** 2
        if keep.sum() < 2000:
            print("%-12s BIM 近傍が %d 点。飛ばす" % (scene, keep.sum()))
            continue
        near = src.select_by_index(np.flatnonzero(keep))

        # 点対面 ICP を段階的に。剛体（縮尺は入れない）
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        T_ref = np.eye(4)
        fitness = rmse = float("nan")
        for d in args.stages:
            r = o3d.pipelines.registration.registration_icp(
                near, ref, d, T_ref, est,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
            T_ref = np.asarray(r.transformation)
            fitness, rmse = float(r.fitness), float(r.inlier_rmse)

        T1 = T_ref @ T0
        v1 = v @ T1[:3, :3].T + T1[:3, 3]
        n1 = n @ T1[:3, :3].T
        f0 = f1 = c0 = c1 = None
        if lab is not None:
            f0 = plane_z(v0, n0, lab == 3, True)
            f1 = plane_z(v1, n1, lab == 3, True)
            c0 = plane_z(v0, n0, lab == 5, False)
            c1 = plane_z(v1, n1, lab == 5, False)
        dt = float(np.linalg.norm(T_ref[:3, 3]))
        dr = rot_deg(T_ref[:3, :3], np.eye(3))
        print("%-12s %8.3f %8.3f %9s %9s %9s %9s"
              % (scene, dt, dr,
                 "%+.3f" % (f0 - b_fl) if f0 else "—",
                 "%+.3f" % (f1 - b_fl) if f1 else "—",
                 "%+.3f" % (c0 - b_ce) if c0 else "—",
                 "%+.3f" % (c1 - b_ce) if c1 else "—"))
        rows.append({"scene": scene, "delta_translation_m": dt, "delta_rotation_deg": dr,
                     "floor_offset_before": (f0 - b_fl) if f0 else None,
                     "floor_offset_after": (f1 - b_fl) if f1 else None,
                     "ceiling_offset_before": (c0 - b_ce) if c0 else None,
                     "ceiling_offset_after": (c1 - b_ce) if c1 else None,
                     "icp_fitness": fitness, "icp_inlier_rmse_m": rmse,
                     "n_points_used": int(keep.sum()),
                     "T_manual": T0.tolist(), "T_refined": T1.tolist()})

    if not rows:
        return 0
    def med(k):
        v = [r[k] for r in rows if r[k] is not None]
        return float(np.median(v)) if v else float("nan")
    print("\n【床・天井のずれ】精緻化の前後（中央値）")
    print("  床    %+.4f m -> %+.4f m" % (med("floor_offset_before"), med("floor_offset_after")))
    print("  天井  %+.4f m -> %+.4f m" % (med("ceiling_offset_before"), med("ceiling_offset_after")))
    print("  ICP が動かした量  並進 中央値 %.3f m / 回転 中央値 %.3f°"
          % (med("delta_translation_m"), med("delta_rotation_deg")))
    print("  ICP の inlier RMSE 中央値 %.4f m / fitness 中央値 %.3f"
          % (med("icp_inlier_rmse_m"), med("icp_fitness")))
    print("\n  ★ ずれが 0 に近づいたなら、**対応点の偏りという説明が裏づけられる**。")
    print("     近づかないなら、建物と BIM の実際の差である可能性が上がる。")

    with open(os.path.join(args.kit, "refine_gt_icp.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    if args.write:
        man_dir = os.path.join(args.kit, "T_gt_manual")
        os.makedirs(man_dir, exist_ok=True)
        for r in rows:
            s = r["scene"]
            src_json = os.path.join(args.kit, "T_gt", "T_gt_%s.json" % s)
            if not os.path.exists(os.path.join(man_dir, "T_gt_%s.json" % s)):
                shutil.copyfile(src_json, os.path.join(man_dir, "T_gt_%s.json" % s))
            with open(src_json, "w") as f:
                json.dump({"T_gt": r["T_refined"],
                           "provenance": ("撮影者の手動位置合わせ（CloudCompare, point pairs）を"
                                          "初期値に、Open3D の点対面 ICP（剛体・幾何のみ）で精緻化。"
                                          "対応先は BIM の室内側の面のみ、BIM 近傍の source 点のみ。"
                                          "提案手法は使っていない。refine_gt_icp.py"),
                           "delta_from_manual": {
                               "translation_m": r["delta_translation_m"],
                               "rotation_deg": r["delta_rotation_deg"]},
                           "icp_inlier_rmse_m": r["icp_inlier_rmse_m"],
                           "icp_fitness": r["icp_fitness"],
                           "T_manual": r["T_manual"]}, f, indent=2, ensure_ascii=False)
        print("\n手動版を %s に退避し、T_gt を精緻化後で更新しました。" % man_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

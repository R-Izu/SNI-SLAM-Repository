"""R10 §3-2 — 基準変換そのものの不確かさを測る。

問い
----
GT の精緻化（点対面・段階ゲート）と提案手法（点対点・ゲート 0.3・Tukey）は、
**別の最適化問題を解いている**。オラクル実験では、正しい初期値から始めても
収束先が基準変換から 0.139〜0.162 m 離れた。

**どちらが正しいかを決める外部の真値は無い。**
そこで **「もっともらしい幾何基準」を複数走らせ、互いにどれだけ食い違うか**を測る。
食い違いの大きさは、**「どの基準を選ぶか」に由来する不確かさ**であって、残差ではない。

**★ 6つの定式化は R10 §3-2 の指定どおりであり、結果を見る前に固定した。**
**走らせてから都合の良いものを本命に選ばない。1 を基準にするのは現行 GT だからである。**

すべて **提案手法を使わない幾何 ICP**、すべて **剛体**（縮尺は推定しない）、
すべて **手動 GT を初期値**、すべて **BIM の室内面のみ**を参照とする。

    conda activate sni-slam
    python Registration/scripts/criterion_spread.py
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
from typing import Dict, List

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)

# ★ 結果を見る前に固定した6つ（R10 §3-2 の表そのまま）
FORMULATIONS = [
    {"id": 1, "name": "点対面 ゲート0.30/0.15/0.08 voxel0.03", "loss": "plane",
     "stages": [0.30, 0.15, 0.08], "voxel": 0.03, "tukey": None},
    {"id": 2, "name": "点対面 最終ゲート0.15", "loss": "plane",
     "stages": [0.30, 0.15], "voxel": 0.03, "tukey": None},
    {"id": 3, "name": "点対面 最終ゲート0.05", "loss": "plane",
     "stages": [0.30, 0.15, 0.05], "voxel": 0.03, "tukey": None},
    {"id": 4, "name": "点対点 ゲート0.30", "loss": "point",
     "stages": [0.30], "voxel": 0.03, "tukey": None},
    {"id": 5, "name": "点対面 voxel0.05", "loss": "plane",
     "stages": [0.30, 0.15, 0.08], "voxel": 0.05, "tukey": None},
    {"id": 6, "name": "点対面 Tukey c=0.3", "loss": "plane",
     "stages": [0.30, 0.15, 0.08], "voxel": 0.03, "tukey": 0.3},
]


def rot_deg(A: np.ndarray, B: np.ndarray) -> float:
    c = (np.trace(A @ B.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", default="output/GT_alignment")
    ap.add_argument("--npz", default="Registration/output/ifc/m3_ifc_all.npz")
    ap.add_argument("--near-bim", type=float, default=0.60)
    ap.add_argument("--out", default="Registration/output/diag/criterion_spread.json")
    args = ap.parse_args()
    import open3d as o3d

    z = np.load(args.npz, allow_pickle=False)
    inner = z["is_inner"].astype(bool)
    ref_full = o3d.geometry.PointCloud()
    ref_full.points = o3d.utility.Vector3dVector(z["points"][inner])
    ref_full.normals = o3d.utility.Vector3dVector(z["normals"][inner])

    scenes = sorted(os.path.basename(p)[5:-5]
                    for p in glob.glob(os.path.join(args.kit, "T_gt", "T_gt_*.json")))
    out: Dict[str, object] = {"formulations": FORMULATIONS, "scenes": {}}

    for scene in scenes:
        sp = os.path.join(args.kit, "source", "%s.ply" % scene)
        # 手動 GT（ICP 前）を初期値にする。精緻化後を使うと定式化1に有利になる
        man = os.path.join(args.kit, "T_gt_manual", "T_gt_%s.json" % scene)
        if not (os.path.exists(sp) and os.path.exists(man)):
            continue
        T0 = np.asarray(json.load(open(man))["T_gt"], dtype=np.float64).reshape(4, 4)
        mesh = o3d.io.read_triangle_mesh(sp)
        verts = np.asarray(mesh.vertices)
        src_all = o3d.geometry.PointCloud()
        src_all.points = o3d.utility.Vector3dVector(verts)

        results = []
        for f in FORMULATIONS:
            ref = ref_full.voxel_down_sample(f["voxel"])
            ref.normalize_normals()
            tree = o3d.geometry.KDTreeFlann(ref)
            src = o3d.geometry.PointCloud(src_all)
            src.transform(T0)
            src = src.voxel_down_sample(f["voxel"])
            q = np.asarray(src.points)
            keep = np.zeros(len(q), dtype=bool)
            for i, x in enumerate(q):
                k, _, d2 = tree.search_knn_vector_3d(x, 1)
                keep[i] = k > 0 and d2[0] <= args.near_bim ** 2
            if keep.sum() < 2000:
                results.append({"id": f["id"], "status": "too_few_points"})
                continue
            near = src.select_by_index(np.flatnonzero(keep))

            if f["loss"] == "plane":
                if f["tukey"] is not None:
                    est = o3d.pipelines.registration.TransformationEstimationPointToPlane(
                        o3d.pipelines.registration.TukeyLoss(k=f["tukey"]))
                else:
                    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
            else:
                est = o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    with_scaling=False)
            T = np.eye(4)
            for g in f["stages"]:
                r = o3d.pipelines.registration.registration_icp(
                    near, ref, g, T, est,
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
                T = np.asarray(r.transformation)
            results.append({"id": f["id"], "status": "ok",
                            "T": (T @ T0).tolist(),
                            "fitness": float(r.fitness),
                            "inlier_rmse": float(r.inlier_rmse)})

        ok = [r for r in results if r.get("status") == "ok"]
        if len(ok) < 2:
            continue
        # 基準点 p0 = source の重心（source 座標系）。並進ベクトルの差は
        # source 原点の取り方に依存するので、基準点での変位も出す
        p0 = verts.mean(axis=0)
        Ts = {r["id"]: np.array(r["T"]) for r in ok}
        base = Ts[1] if 1 in Ts else Ts[min(Ts)]
        disp = {i: float(np.linalg.norm((T[:3, :3] @ p0 + T[:3, 3])
                                        - (base[:3, :3] @ p0 + base[:3, 3])))
                for i, T in Ts.items()}
        pair = {}
        for a, b in itertools.combinations(sorted(Ts), 2):
            Ta, Tb = Ts[a], Ts[b]
            pa = Ta[:3, :3] @ p0 + Ta[:3, 3]
            pb = Tb[:3, :3] @ p0 + Tb[:3, 3]
            pair["%d-%d" % (a, b)] = {
                "trans_diff_m": float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])),
                "p0_disp_m": float(np.linalg.norm(pa - pb)),
                "rot_diff_deg": rot_deg(Ta[:3, :3] / np.cbrt(abs(np.linalg.det(Ta[:3, :3]))),
                                        Tb[:3, :3] / np.cbrt(abs(np.linalg.det(Tb[:3, :3])))),
            }
        out["scenes"][scene] = {"results": results, "p0_disp_vs_1": disp,
                                "pairwise": pair}
        d = [v["p0_disp_m"] for v in pair.values()]
        print("%-12s 基準点での相互変位: 中央値 %.3f m / 最大 %.3f m  "
              "回転差 最大 %.3f度"
              % (scene, float(np.median(d)), float(np.max(d)),
                 max(v["rot_diff_deg"] for v in pair.values())))

    alld = [v["p0_disp_m"] for s in out["scenes"].values()
            for v in s["pairwise"].values()]
    if alld:
        a = np.array(alld)
        print("\n【R10 §3-3】6つの基準の相互変位（基準点 p0、全 %d シーン × 15 組）"
              % len(out["scenes"]))
        print("  中央値 %.3f m / 平均 %.3f m / 90%%点 %.3f m / 最大 %.3f m"
              % (np.median(a), a.mean(), np.percentile(a, 90), a.max()))
        print("  **閾値 0.1 m と比べる**：0.1 m を超える組 %d / %d (%.0f%%)"
              % (int((a > 0.1).sum()), len(a), 100 * (a > 0.1).mean()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

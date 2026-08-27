"""追補6 §3 — カメラ軌跡を BIM 座標へ載せ、§3-3 の3つの検証を数値で行う。

何をするか
----------
位置合わせが出した Sim(3) を、点群だけでなく **`traj.txt` の ARKit VIO 姿勢にも適用する**。
軌跡と点群は同じ座標系にあるので追加の推定は要らない。

出力
----
1. BIM 平面図に重ねた軌跡（上面図・PNG）
2. 3D の軌跡＋BIM（PNG）
3. 軌跡の BIM 座標（CSV）
4. **§3-3 の3検証**（壁の貫通・扉の通過・カメラ高さ）の数値

位置づけ（誇張しないこと。追補6 §3-4）
--------------------------------------
図（1・2）は**提示**であって証拠ではない。**証拠になるのは 3 の数値**である。
論文には「軌跡（ARKit VIO）を、提案手法が推定した Sim(3) で BIM 座標へ写した」と書く。
**SNI-SLAM がトラッキングしたとは書かない。**

    conda activate sni-slam
    python Registration/scripts/trajectory_to_bim.py \
        --traj data/realdata/m3_block_a/traj.txt \
        --sim3 Registration/output/eval_m3/T_est.json \
        --bim Registration/output/ifc/m3_ifc_all.npz \
        --out Registration/output/traj_bim/m3_block_a
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))

CLASS_NAMES = ["background", "wall", "door", "floor", "window", "ceiling"]


def load_sim3(path: str) -> np.ndarray:
    with open(path) as f:
        d = json.load(f)
    for k in ("T", "T_sim3", "matrix", "T_gt", "T_est"):
        if k in d:
            return np.asarray(d[k], dtype=np.float64).reshape(4, 4)
    raise KeyError("Sim(3) の 4x4 が見つからない: %s（キー %s）" % (path, list(d)))


def load_traj(path: str) -> np.ndarray:
    rows = [ln.split() for ln in open(path) if ln.strip()]
    return np.asarray([[float(v) for v in r] for r in rows]).reshape(-1, 4, 4)


def nn_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a の各点から b への最近傍距離。scipy を使わずに済ませる（KDTree は open3d 側）。"""
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(b)
    tree = o3d.geometry.KDTreeFlann(pc)
    out = np.empty(len(a))
    for i, p in enumerate(a):
        _, _, d2 = tree.search_knn_vector_3d(p, 1)
        out[i] = np.sqrt(d2[0])
    return out


def verify(cam: np.ndarray, bim_pts: np.ndarray, bim_lab: np.ndarray,
           bim_nrm: Optional[np.ndarray]) -> Dict:
    """§3-3 の3検証。**図ではなく数値**で出す。"""
    rep: Dict[str, object] = {}
    wall = bim_pts[bim_lab == CLASS_NAMES.index("wall")]
    door = bim_pts[bim_lab == CLASS_NAMES.index("door")]
    floor = bim_pts[bim_lab == CLASS_NAMES.index("floor")]

    # --- 壁の貫通 ---------------------------------------------------------
    # 壁面までの最短距離の分布。位置合わせが誤っていれば軌跡が壁を突き抜け、
    # 「壁のごく近く」を通る姿勢の割合が増える。壁の内部かどうかを厳密に判定するには
    # ソリッドが要るので、ここでは **距離の分布**を報告する（過大な主張をしない）。
    if len(wall):
        d = nn_dist(cam, wall)
        rep["wall"] = {
            "min_m": round(float(d.min()), 4),
            "q01_m": round(float(np.percentile(d, 1)), 4),
            "median_m": round(float(np.median(d)), 4),
            "frac_within_0.10m": round(float((d < 0.10).mean()), 4),
            "frac_within_0.05m": round(float((d < 0.05).mean()), 4),
            "n_poses": int(len(cam)),
            "note": "壁の内部かは判定していない。近接の割合で代用している",
        }

    # --- 扉の通過 ---------------------------------------------------------
    # 部屋を移る区間は扉の近くを通るはず。区間の切り出しには室の情報が要るので、
    # ここでは「扉から一定距離内を通った回数」と最接近距離を出す。
    if len(door):
        d = nn_dist(cam, door)
        near = d < 1.0
        # 連続した近接をひとかたまりの「通過」と数える
        passes = int(np.sum(np.diff(near.astype(int)) == 1)) + int(near[0])
        rep["door"] = {
            "min_m": round(float(d.min()), 4),
            "n_poses_within_1m": int(near.sum()),
            "n_passes_within_1m": passes,
        }

    # --- カメラ高さ -------------------------------------------------------
    # BIM の床面からの高さ。**平均と分散**を見る。人の身長程度でほぼ一定のはず。
    # 縮尺が数%ずれれば高さが浮くか沈むので、**縮尺誤差の別経路の検算**になる。
    if len(floor):
        # 床の上面（法線が上向き）だけを使う。ソリッドなので下面も点になっている
        if bim_nrm is not None:
            fm = (bim_lab == CLASS_NAMES.index("floor"))
            up = fm & (np.abs(bim_nrm[:, 2]) > 0.9) & (bim_nrm[:, 2] > 0)
            floor_top = bim_pts[up] if up.sum() > 100 else floor
        else:
            floor_top = floor
        z0 = float(np.median(floor_top[:, 2]))
        h = cam[:, 2] - z0
        rep["camera_height"] = {
            "floor_z_m": round(z0, 4),
            "mean_m": round(float(h.mean()), 4),
            "std_m": round(float(h.std()), 4),
            "median_m": round(float(np.median(h)), 4),
            "q25_m": round(float(np.percentile(h, 25)), 4),
            "q75_m": round(float(np.percentile(h, 75)), 4),
            "min_m": round(float(h.min()), 4),
            "max_m": round(float(h.max()), 4),
            "plausible_1.2_1.6m": bool(1.2 <= float(np.median(h)) <= 1.6),
            "note": "縮尺が数%ずれれば高さが系統的に浮くか沈む。GT 位置合わせとは別経路の検算",
        }
    return rep


def figures(cam: np.ndarray, bim_pts: np.ndarray, bim_lab: np.ndarray,
            out_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made: List[str] = []
    # 平面図：BIM の壁を薄く、軌跡を時間で色付け
    fig, ax = plt.subplots(figsize=(10, 8))
    w = bim_lab == CLASS_NAMES.index("wall")
    ax.scatter(bim_pts[w, 0], bim_pts[w, 1], s=0.4, c="#999999", linewidths=0,
               label="BIM wall")
    d = bim_lab == CLASS_NAMES.index("door")
    if d.any():
        ax.scatter(bim_pts[d, 0], bim_pts[d, 1], s=2.0, c="#e08a00", linewidths=0,
                   label="BIM door")
    ax.scatter(cam[:, 0], cam[:, 1], s=1.2,
               c=np.arange(len(cam)), cmap="viridis", linewidths=0, label="camera")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("camera trajectory in BIM coordinates (top view)")
    ax.legend(loc="best", markerscale=8, fontsize=8)
    p = os.path.join(out_dir, "traj_plan.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)

    # 高さの時系列：縮尺誤差が見えるのはここ
    fig, ax = plt.subplots(figsize=(10, 3))
    fl = bim_lab == CLASS_NAMES.index("floor")
    z0 = float(np.median(bim_pts[fl, 2])) if fl.any() else 0.0
    ax.plot(cam[:, 2] - z0, lw=0.8)
    ax.axhspan(1.2, 1.6, color="#8ec7ff", alpha=0.4, label="人の身長程度 1.2-1.6 m")
    ax.set_xlabel("frame"); ax.set_ylabel("height above BIM floor [m]")
    ax.legend(fontsize=8)
    p = os.path.join(out_dir, "traj_height.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True, help="data/realdata/<scene>/traj.txt")
    ap.add_argument("--sim3", required=True,
                    help="位置合わせが出した Sim(3) の 4x4 を持つ JSON")
    ap.add_argument("--bim", required=True, help="Registration/output/ifc/*.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    os.chdir(REPO)
    os.makedirs(args.out, exist_ok=True)

    T = load_sim3(args.sim3)
    c2w = load_traj(args.traj)[::max(args.stride, 1)]
    # カメラ中心を BIM 座標へ。Sim(3) は source(SLAM) -> reference(BIM)
    cam = (c2w[:, :3, 3] @ T[:3, :3].T) + T[:3, 3]

    z = np.load(args.bim, allow_pickle=False)
    bim_pts, bim_lab = z["points"], z["labels"]
    bim_nrm = z["normals"] if "normals" in z.files else None

    rep = verify(cam, bim_pts, bim_lab, bim_nrm)
    rep["_meta"] = {
        "traj": os.path.abspath(args.traj), "sim3": os.path.abspath(args.sim3),
        "bim": os.path.abspath(args.bim), "n_poses": int(len(cam)),
        "stride": args.stride,
        "provenance": "姿勢は ARKit VIO（traj.txt）。SNI-SLAM のトラッキングではない",
    }

    np.savetxt(os.path.join(args.out, "traj_bim.csv"),
               np.column_stack([np.arange(len(cam)) * args.stride, cam]),
               delimiter=",", header="frame,x_m,y_m,z_m", comments="", fmt="%.6f")
    with open(os.path.join(args.out, "verify.json"), "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    for p in figures(cam, bim_pts, bim_lab, args.out):
        print("wrote %s" % p)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

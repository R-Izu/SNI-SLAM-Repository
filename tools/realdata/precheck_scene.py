"""Phase 6 — SLAM 出力メッシュのシーン事前チェック（R2 事前チェック値）。

`mesh/final_mesh_semantic.ply` を入力に、`Registration/regbim` を再利用して
**位置合わせの効きやすさを左右するシーンの性質**を数値化する。

出力する指標
------------
gravity_tilt_deg      推定重力軸と世界 +Y のなす角。大きいと回転アトラクタのリスク
plane_diversity       可視壁面の水平方向の種類数。**3 未満だと並進・縮尺が不定になりうる**
wall_direction_deg    検出した壁の主方向（ヨー角）
class_counts          6クラス別の点数と割合
drift_start_end_m     SLAM 軌跡の始終点距離（T-A2。ARKit 側の値と比較する）

★ 重要：**これらの値でシーンを却下してはならない**（追補指示 Q7）。
難しいシーンを捨てると成功率を水増しすることになる。
値は「この性質が成功率を予測するか」を後で見るための**共変量**として記録する。

使い方
------
    conda activate sni-slam
    python tools/realdata/precheck_scene.py \
        --mesh output/RealData/m3_cor_d/run1/mesh/final_mesh_semantic.ply \
        --traj-gt data/realdata/m3_cor_d/traj.txt \
        --est-poses output/RealData/m3_cor_d/run1/ckpts \
        --out output/RealData/m3_cor_d/run1/precheck_m3_cor_d.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "Registration"))

from regbim import io_utils, labels as L, rotation  # noqa: E402

# regbim の rotation 設定は Registration/configs/replica_room0.yaml と同値を使う
ROT_CFG = {"rotation": {"ransac_iters": 1000, "ransac_normal_thresh_deg": 5.0,
                        "yaw_bins": 360, "yaw_use_kmeans": True}}
WORLD_UP = np.array([0.0, 1.0, 0.0])     # 変換器が書く traj は ARKit 系（+Y が上）


def plane_diversity(cloud, up: np.ndarray, min_frac: float = 0.05,
                    sep_deg: float = 20.0, bin_deg: float = 5.0) -> Dict[str, object]:
    """可視壁面の水平方向（ヨー）の種類数を数える。

    壁法線を重力軸に直交する平面へ射影し、ヨー角のヒストグラム（180° 周期。
    法線の符号は不定なので mod 180）を作る。全体の ``min_frac`` 以上を占め、
    互いに ``sep_deg`` 以上離れたピークの数を「平面多様性」とする。
    3 未満だと、その方向の並進成分と縮尺が壁だけからは決まらない。
    """
    mask = cloud.class_mask("wall")
    w = cloud.subset(mask)
    if len(w) == 0 or w.normals is None:
        return {"n_directions": 0, "directions_deg": [], "n_wall_points": 0}
    n = w.normals / (np.linalg.norm(w.normals, axis=1, keepdims=True) + 1e-12)
    horiz = n - np.outer(n @ up, up)
    mag = np.linalg.norm(horiz, axis=1)
    keep = mag > 0.5                    # ほぼ垂直な面（＝本当の壁）だけ残す
    if not keep.any():
        return {"n_directions": 0, "directions_deg": [], "n_wall_points": int(len(w))}
    horiz = horiz[keep] / mag[keep, None]

    # 重力軸に直交する基底でヨー角へ
    e1, e2 = rotation._horizontal_basis(up)
    yaw = np.degrees(np.arctan2(horiz @ e2, horiz @ e1)) % 180.0
    # ビン幅は 5°。1° ビンだと実測の法線ノイズ（±10° 程度）で山が広がり、
    # どの単一ビンも min_frac に届かず「0 方向」と誤判定する（実際に一度そうなった）。
    nbins = int(round(180.0 / bin_deg))
    hist, edges = np.histogram(yaw, bins=nbins, range=(0.0, 180.0))
    frac = hist / max(hist.sum(), 1)

    order = np.argsort(hist)[::-1]
    picked: List[float] = []
    picked_frac: List[float] = []
    for k in order:
        if frac[k] < min_frac:
            break
        c = float((edges[k] + edges[k + 1]) / 2.0)
        if all(min(abs(c - p), 180.0 - abs(c - p)) >= sep_deg for p in picked):
            picked.append(c)
            picked_frac.append(float(frac[k]))
    return {
        "n_directions": len(picked),
        "directions_deg": [round(p, 1) for p in picked],
        "direction_fracs": [round(f, 4) for f in picked_frac],
        "n_wall_points": int(len(w)),
        "n_wall_points_vertical": int(keep.sum()),
        "bin_deg": bin_deg,
        "min_frac": min_frac,
        "top_bin_fracs": [round(float(v), 4) for v in np.sort(frac)[::-1][:6]],
    }


def load_est_traj(ckpt_dir: str) -> Optional[np.ndarray]:
    """最新の ckpt から推定カメラ軌跡 (N,3) を読む。"""
    import torch
    if not os.path.isdir(ckpt_dir):
        return None
    tars = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".tar"))
    if not tars:
        return None
    ck = torch.load(os.path.join(ckpt_dir, tars[-1]), map_location="cpu")
    for key in ("estimate_c2w_list", "estimate_c2w", "est_c2w_list"):
        if key in ck:
            m = ck[key]
            m = m.numpy() if hasattr(m, "numpy") else np.asarray(m)
            return m[:, :3, 3]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--n-points", type=int, default=200000)
    ap.add_argument("--traj-gt", default=None, help="変換器が書いた traj.txt（ARKit 側）")
    ap.add_argument("--est-poses", default=None, help="ckpts ディレクトリ（SLAM 推定側）")
    args = ap.parse_args()

    res: Dict[str, object] = {
        "scene": args.scene or os.path.basename(os.path.dirname(os.path.dirname(args.mesh))),
        "mesh": args.mesh,
        "note": "これらの値でシーンを却下しないこと（追補指示 Q7）。共変量として記録する。",
    }

    cloud = io_utils._load_slam_mesh({"mesh_path": args.mesh, "n_points": args.n_points})
    res["n_points_sampled"] = int(len(cloud))

    counts = io_utils.class_counts(cloud)
    total = max(sum(counts.values()), 1)
    res["class_counts"] = counts
    res["class_fracs"] = {k: round(v / total, 5) for k, v in counts.items()}

    # --- 重力軸 ---
    try:
        up = rotation.estimate_gravity_axis(cloud, ROT_CFG)
        tilt = float(np.degrees(np.arccos(np.clip(abs(up @ WORLD_UP), -1.0, 1.0))))
        res["gravity_axis"] = [round(float(v), 5) for v in up]
        res["gravity_tilt_deg"] = round(tilt, 3)
        res["gravity_ok"] = bool(tilt <= 5.0)
    except Exception as e:
        up = WORLD_UP
        res["gravity_axis"] = None
        res["gravity_tilt_deg"] = None
        res["gravity_error"] = "%s: %s" % (type(e).__name__, e)
        res["gravity_ok"] = False

    # --- 平面多様性 ---
    pd = plane_diversity(cloud, up if res.get("gravity_axis") else WORLD_UP)
    res["plane_diversity"] = pd
    res["plane_diversity_ok"] = bool(pd["n_directions"] >= 3)

    # --- T-A2: ドリフト（始終点距離） ---
    drift: Dict[str, object] = {}
    if args.traj_gt and os.path.exists(args.traj_gt):
        rows = [ln.split() for ln in open(args.traj_gt).read().splitlines() if ln.strip()]
        m = np.asarray([[float(v) for v in r] for r in rows]).reshape(-1, 4, 4)
        p = m[:, :3, 3]
        drift["arkit_start_end_m"] = round(float(np.linalg.norm(p[-1] - p[0])), 4)
        drift["arkit_path_length_m"] = round(float(
            np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), 2)
        drift["n_frames"] = int(len(p))
    if args.est_poses:
        est = load_est_traj(args.est_poses)
        if est is not None and len(est) > 1:
            drift["slam_start_end_m"] = round(float(np.linalg.norm(est[-1] - est[0])), 4)
            drift["slam_path_length_m"] = round(float(
                np.linalg.norm(np.diff(est, axis=0), axis=1).sum()), 2)
            if "arkit_start_end_m" in drift:
                drift["start_end_deviation_m"] = round(
                    drift["slam_start_end_m"] - drift["arkit_start_end_m"], 4)
        else:
            drift["slam_start_end_m"] = None
            drift["note"] = "ckpts から推定軌跡を読めなかった"
    res["drift"] = drift

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print("scene            %s" % res["scene"])
    print("class fracs      %s" % res["class_fracs"])
    print("gravity tilt     %s deg (ok=%s)" % (res["gravity_tilt_deg"], res["gravity_ok"]))
    print("plane diversity  %s directions %s (ok=%s)" % (
        pd["n_directions"], pd["directions_deg"], res["plane_diversity_ok"]))
    print("drift            %s" % drift)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

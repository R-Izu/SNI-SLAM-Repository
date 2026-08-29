"""R5 §6-4 — 撮影者が CloudCompare で作った GT を、機械的に検算する。

なぜ要るか
----------
**手動位置合わせにも間違いは起こる。** 対応点を1点取り違えるだけで、
もっともらしいが誤った Sim(3) が出る。そして GT が誤っていると、
**その後の評価が全部その誤りを継承する**（提案手法の誤差に見えてしまう）。

検査（R5 §6-4 の5項目）
------------------------
1. **スケールが 1.0 付近か** — ARKit + LiDAR なので source は実寸のはず
2. **重力軸の一致** — source の鉛直と BIM の +Z が数度以内か
3. **床面の高さ** — 変換後の床点が BIM の床レベル付近に来るか
4. **カメラ高さ** — 軌跡を同じ変換で写し、BIM 床面から 1.2〜1.6 m でほぼ一定か
5. **壁への近接** — 軌跡が壁を突き抜けていないか

★ 3〜5 は「別経路で測った量の一致」による検算である（追補2 §1-4）。
   スケールと重力は変換行列そのものから、床高とカメラ高さは**軌跡**から出る。

★ **スケールの 1.0 からの乖離は、それ自体が入力の実寸精度の実測値になる**（R5 §6-4）。
   ARKit のドリフトと LiDAR のバイアスの合成として報告するので、必ず記録する。

    conda activate sni-slam
    python Registration/scripts/verify_gt.py                 # 全シーン
    python Registration/scripts/verify_gt.py --scenes m3_cor_c
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CLASS_NAMES = ["background", "wall", "door", "floor", "window", "ceiling"]

# 判定の基準。**結果を見てから動かさない**ため、ここに固定して書く。
SCALE_WARN = 0.03          # |s - 1| がこれを超えたら注意（3%）
SCALE_FAIL = 0.10          # これを超えたら対応点の取り違えを疑う
GRAVITY_WARN_DEG = 3.0
FLOOR_WARN_M = 0.10
CAM_H_RANGE = (1.2, 1.6)
CAM_H_STD_WARN = 0.15      # 手持ちなので多少は揺れるが、これ以上は縮尺か姿勢を疑う


def load_T(path: str) -> np.ndarray:
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, list):
        return np.asarray(d, dtype=np.float64).reshape(4, 4)
    for k in ("T_gt", "T", "T_sim3", "matrix", "transform"):
        if k in d:
            return np.asarray(d[k], dtype=np.float64).reshape(4, 4)
    raise KeyError("4x4 が見つからない: %s（キー %s）" % (path, list(d)))


def decompose(T: np.ndarray):
    """Sim(3) を (R, t, s) に分解。R は正規直交、s は等方スケール。"""
    A = T[:3, :3]
    s = float(np.cbrt(max(np.linalg.det(A), 1e-30)))
    R = A / s
    # 数値誤差を落として最も近い回転行列へ
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, T[:3, 3].copy(), s


def gravity_axis(points: np.ndarray, normals: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """点群の鉛直軸。法線があれば水平面（床・天井）の法線の主方向から取る。"""
    if normals is None or len(normals) < 100:
        return None
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    # 法線の外積行列の最小固有ベクトル…ではなく、|n・u| が大きい方向を探す。
    # 床・天井は同じ軸に対して符号が逆なので、n n^T の最大固有ベクトルでよい
    M = n.T @ n
    w, V = np.linalg.eigh(M)
    return V[:, int(np.argmax(w))]


def check(scene: str, T: np.ndarray, ref: Dict, traj: Optional[np.ndarray],
          src_pts: Optional[np.ndarray], src_nrm: Optional[np.ndarray]) -> Dict:
    R, t, s = decompose(T)
    out: Dict[str, object] = {"scene": scene}
    flags: List[str] = []

    # --- 1. スケール --------------------------------------------------------
    dev = abs(s - 1.0)
    out["scale"] = {"s": round(s, 5), "deviation_from_1": round(dev, 5),
                    "verdict": ("ok" if dev <= SCALE_WARN else
                                "warn" if dev <= SCALE_FAIL else "fail")}
    if dev > SCALE_FAIL:
        flags.append("スケールが 1.0 から %.1f%% ずれている。対応点の取り違えを疑う"
                     % (100 * dev))
    out["scale"]["note"] = ("この乖離は入力の実寸精度そのもの（ARKit ドリフト＋LiDAR バイアス）"
                            "として報告する量でもある")

    # --- 2. 重力軸 ----------------------------------------------------------
    g_src = gravity_axis(src_pts, src_nrm) if src_pts is not None else None
    if g_src is not None:
        g_in_bim = R @ g_src
        ang = np.degrees(np.arccos(np.clip(abs(g_in_bim @ np.array([0, 0, 1.0])), 0, 1)))
        out["gravity"] = {"tilt_vs_bim_up_deg": round(float(ang), 3),
                          "verdict": "ok" if ang <= GRAVITY_WARN_DEG else "warn"}
        if ang > GRAVITY_WARN_DEG:
            flags.append("重力軸が BIM の鉛直から %.1f° ずれている" % ang)
    else:
        out["gravity"] = {"verdict": "skipped", "why": "source の法線が無い"}

    # --- 3. 床面の高さ ------------------------------------------------------
    z_bim_floor = ref["floor_top_z"]
    if src_pts is not None:
        p = (src_pts @ R.T) * s + t
        # source 側の床は「下から数えて厚みのある水平帯」。ここでは下位 5% の中央値で代用する
        z_low = float(np.median(np.sort(p[:, 2])[:max(len(p) // 20, 1)]))
        d = abs(z_low - z_bim_floor)
        out["floor"] = {"src_low_z_m": round(z_low, 4),
                        "bim_floor_top_z_m": round(z_bim_floor, 4),
                        "diff_m": round(d, 4),
                        "verdict": "ok" if d <= FLOOR_WARN_M else "warn"}
        if d > FLOOR_WARN_M:
            flags.append("変換後の床が BIM の床から %.2f m ずれている" % d)

    # --- 4/5. 軌跡 ----------------------------------------------------------
    if traj is not None and len(traj):
        cam = (traj @ R.T) * s + t
        h = cam[:, 2] - z_bim_floor
        med, sd = float(np.median(h)), float(h.std())
        ok_h = CAM_H_RANGE[0] <= med <= CAM_H_RANGE[1]
        out["camera_height"] = {
            "median_m": round(med, 4), "std_m": round(sd, 4),
            "q25_m": round(float(np.percentile(h, 25)), 4),
            "q75_m": round(float(np.percentile(h, 75)), 4),
            "in_1.2_1.6m": bool(ok_h),
            "verdict": "ok" if (ok_h and sd <= CAM_H_STD_WARN) else "warn"}
        if not ok_h:
            flags.append("カメラ高さの中央値が %.2f m（人の身長程度から外れる）" % med)
        if sd > CAM_H_STD_WARN:
            flags.append("カメラ高さのばらつきが %.2f m と大きい（縮尺か姿勢を疑う）" % sd)

        wall = ref["wall_points"]
        if len(wall):
            import open3d as o3d
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(wall)
            tree = o3d.geometry.KDTreeFlann(pc)
            dmin = np.empty(len(cam))
            for i, q in enumerate(cam):
                _, _, d2 = tree.search_knn_vector_3d(q, 1)
                dmin[i] = np.sqrt(d2[0])
            out["wall_proximity"] = {
                "min_m": round(float(dmin.min()), 4),
                "q01_m": round(float(np.percentile(dmin, 1)), 4),
                "median_m": round(float(np.median(dmin)), 4),
                "frac_within_0.10m": round(float((dmin < 0.10).mean()), 4),
                "verdict": "ok" if (dmin < 0.05).mean() < 0.01 else "warn",
                "note": "壁の内部かは判定していない。近接の割合で代用している"}
            if (dmin < 0.05).mean() >= 0.01:
                flags.append("軌跡の %.1f%% が壁面 5 cm 以内。貫通を疑う"
                             % (100 * (dmin < 0.05).mean()))
    else:
        out["camera_height"] = {"verdict": "skipped", "why": "traj.txt が無い"}

    out["flags"] = flags
    out["overall"] = "ok" if not flags else "要確認"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", default="output/GT_alignment")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.chdir(REPO)
    import open3d as o3d

    man = json.load(open(os.path.join(args.kit, "manifest.json")))
    ref_pc = o3d.io.read_point_cloud(os.path.join(args.kit, man["reference"]["file"]))
    ref_pts = np.asarray(ref_pc.points)
    ref_cols = np.asarray(ref_pc.colors)
    palette = np.array([(128, 128, 128), (255, 64, 64), (255, 200, 64),
                        (180, 220, 255), (64, 200, 255), (200, 100, 255)],
                       dtype=np.float64) / 255.0
    lab = np.argmin(((ref_cols[:, None, :] - palette[None]) ** 2).sum(2), axis=1)
    fl = ref_pts[lab == CLASS_NAMES.index("floor")]
    ref = {
        # 床は上面と下面の両方が点になる（IFC のスラブはソリッド）。**上面**を取る
        "floor_top_z": float(np.percentile(fl[:, 2], 90)) if len(fl) else 0.0,
        "wall_points": ref_pts[lab == CLASS_NAMES.index("wall")],
    }

    found = sorted(glob.glob(os.path.join(args.kit, "T_gt", "T_gt_*.json")))
    if not found:
        print("T_gt がまだありません: %s/T_gt/T_gt_<scene>.json"
              % args.kit)
        print("撮影者の作業待ちです。README.md の手順を参照。")
        return 0

    results = []
    for p in found:
        scene = os.path.basename(p)[5:-5]
        if args.scenes and scene not in args.scenes:
            continue
        T = load_T(p)
        sp = os.path.join(args.kit, "source", "%s.ply" % scene)
        src_pts = src_nrm = None
        if os.path.exists(sp):
            m = o3d.io.read_triangle_mesh(sp)
            m.compute_vertex_normals()
            src_pts = np.asarray(m.vertices)
            src_nrm = np.asarray(m.vertex_normals)
        tp = "data/realdata/%s/traj.txt" % scene
        traj = None
        if os.path.exists(tp):
            rows = [ln.split() for ln in open(tp) if ln.strip()]
            traj = np.asarray([[float(v) for v in r]
                               for r in rows]).reshape(-1, 4, 4)[:, :3, 3]
        r = check(scene, T, ref, traj, src_pts, src_nrm)
        results.append(r)
        print("=== %s : %s ===" % (scene, r["overall"]))
        print("  スケール %.5f（1.0 から %.2f%%）  %s"
              % (r["scale"]["s"], 100 * r["scale"]["deviation_from_1"],
                 r["scale"]["verdict"]))
        for k in ("gravity", "floor", "camera_height", "wall_proximity"):
            if k in r:
                print("  %-15s %s" % (k, json.dumps(r[k], ensure_ascii=False)))
        for f in r["flags"]:
            print("  ⚠ %s" % f)

    if results:
        ss = [r["scale"]["s"] for r in results]
        print("\n【入力の実寸精度】GT スケールの分布（n=%d）" % len(ss))
        print("  中央値 %.5f / 平均 %.5f / 標準偏差 %.5f / 範囲 [%.5f, %.5f]"
              % (np.median(ss), np.mean(ss), np.std(ss), min(ss), max(ss)))
        print("  → ARKit のドリフトと LiDAR のバイアスの合成として報告する量")

    out = args.out or os.path.join(args.kit, "verify_gt.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

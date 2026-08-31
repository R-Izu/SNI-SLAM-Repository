"""位置合わせ後の点群から、CloudCompare が適用した Sim(3) を厳密に復元する。

なぜ行列を手写ししないか
------------------------
撮影者は **位置合わせ後の点群**を `T_gt/<scene>.ply` に保存した（行列は別途控えてある）。
行列を人が転記すると桁落ちや取り違えが入り、しかも**入ったことに気づけない**。
一方、変換前後の点群があれば行列は**一意に解ける**うえ、
**残差がその場で検証になる**（正しく解けていれば残差はゼロ近傍になる）。

やり方
------
CloudCompare の `Align (point pairs picking)` は点の順序を変えないので、
変換前 `source/<scene>.ply` の頂点 p_i と 変換後 `T_gt/<scene>.ply` の頂点 q_i が対応する。
`q = s R p + t` を最小二乗で解き（同次形の線形最小二乗）、

  1. **残差**（これが大きければ順序の仮定が崩れている＝この方法は使えない）
  2. 線形部が **相似変換になっているか**（s R の直交性）

を必ず確認してから JSON に書く。

    conda activate sni-slam
    python Registration/scripts/recover_gt_from_ply.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np


def read_points(path: str):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(path)
    v = np.asarray(m.vertices)
    if len(v):
        return v
    pc = o3d.io.read_point_cloud(path)
    return np.asarray(pc.points)


def solve_sim3(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """q = A p + t を線形最小二乗で解き、4x4 にして返す（A は制約なし）。

    相似変換かどうかは**解いた後に検査する**。最初から相似変換を仮定して
    Umeyama で解くと、実は相似でなかった場合にそれが見えなくなる。
    """
    X = np.hstack([P, np.ones((len(P), 1))])          # (N, 4)
    M, *_ = np.linalg.lstsq(X, Q, rcond=None)          # (4, 3)
    T = np.eye(4)
    T[:3, :3] = M[:3].T
    T[:3, 3] = M[3]
    return T


def check(T: np.ndarray) -> Dict[str, object]:
    A = T[:3, :3]
    s = float(np.cbrt(abs(np.linalg.det(A))))
    R = A / s if s > 0 else A
    orth = float(np.abs(R @ R.T - np.eye(3)).max())
    # 各軸のスケールが揃っているか（等方か）
    sv = np.linalg.svd(A, compute_uv=False)
    return {"scale": round(s, 6), "det": round(float(np.linalg.det(A)), 6),
            "orthogonality_error": round(orth, 8),
            "singular_values": [round(float(x), 6) for x in sv],
            "anisotropy": round(float(sv.max() / sv.min() - 1.0), 6)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", default="output/GT_alignment")
    ap.add_argument("--max-points", type=int, default=200000,
                    help="解くのに使う点数の上限（間引いても厳密解は変わらない）")
    ap.add_argument("--write", action="store_true",
                    help="検査に通ったものを T_gt_<scene>.json に書く")
    args = ap.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", ".."))
    os.chdir(repo)

    out: List[Dict] = []
    print("%-12s %9s %10s %12s %12s %10s"
          % ("scene", "点数", "縮尺 s", "残差中央値m", "残差最大m", "判定"))
    print("-" * 72)
    for p in sorted(glob.glob(os.path.join(args.kit, "T_gt", "*.ply"))):
        scene = os.path.splitext(os.path.basename(p))[0]
        src_p = os.path.join(args.kit, "source", "%s.ply" % scene)
        if not os.path.exists(src_p):
            print("%-12s 変換前が見つからない" % scene)
            continue
        P, Q = read_points(src_p), read_points(p)
        if len(P) != len(Q):
            print("%-12s 点数が違う（前 %d / 後 %d）。順序の対応が取れないので解かない"
                  % (scene, len(P), len(Q)))
            out.append({"scene": scene, "status": "point_count_mismatch",
                        "n_before": int(len(P)), "n_after": int(len(Q))})
            continue

        step = max(len(P) // args.max_points, 1)
        T = solve_sim3(P[::step], Q[::step])
        # ★残差は**全点**で測る（間引いた点だけで測ると自分に都合よく見える）
        res = np.linalg.norm((P @ T[:3, :3].T + T[:3, 3]) - Q, axis=1)
        c = check(T)
        ok = (float(np.median(res)) < 1e-4 and c["orthogonality_error"] < 1e-5
              and c["anisotropy"] < 1e-4)
        print("%-12s %9s %10.5f %12.2e %12.2e %10s"
              % (scene, "{:,}".format(len(P)), c["scale"],
                 float(np.median(res)), float(res.max()),
                 "○" if ok else "★要確認"))
        rec = {"scene": scene, "status": "ok" if ok else "check",
               "n_points": int(len(P)),
               "residual_median_m": float(np.median(res)),
               "residual_max_m": float(res.max()),
               "residual_p99_m": float(np.percentile(res, 99)),
               "T": T.tolist(), **c}
        out.append(rec)
        if args.write and ok:
            with open(os.path.join(args.kit, "T_gt", "T_gt_%s.json" % scene), "w") as f:
                json.dump({"T_gt": T.tolist(),
                           "provenance": ("CloudCompare の Align (point pairs picking) で "
                                          "撮影者が手動位置合わせした後の点群から復元。"
                                          "recover_gt_from_ply.py"),
                           "residual_median_m": float(np.median(res)),
                           "residual_max_m": float(res.max()),
                           "scale": c["scale"]}, f, indent=2, ensure_ascii=False)

    ok_rows = [r for r in out if r.get("status") == "ok"]
    if ok_rows:
        ss = [r["scale"] for r in ok_rows]
        print("\n【入力の実寸精度】GT 縮尺の分布（n=%d）" % len(ss))
        print("  中央値 %.5f / 平均 %.5f / 標準偏差 %.5f / 範囲 [%.5f, %.5f]"
              % (np.median(ss), np.mean(ss), np.std(ss), min(ss), max(ss)))
        print("  → ARKit のドリフトと LiDAR のバイアスの合成。**それ自体が結果になる量**")
    with open(os.path.join(args.kit, "recover_gt.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % os.path.join(args.kit, "recover_gt.json"))
    if args.write:
        print("T_gt_<scene>.json を書きました。次は verify_gt.py で検算します。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

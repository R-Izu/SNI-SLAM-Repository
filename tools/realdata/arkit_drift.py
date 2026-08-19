"""T-A2（追補4 §4）— ARKit VIO の相対ドリフトを実測する。

フォールバック D では姿勢が ARKit そのものなので ATE は構成上 0 になり、
評価指標として使えない。**入力点群の誤差を特徴づける唯一の実測値**が、
ARKit 軌跡が同一地点へ戻ったときの位置のずれである。

    相対ドリフト = 始終点距離 / 経路長

これは**下界**である：
  - 始終点が偶然一致した可能性を排除できない
  - 途中で相殺されたドリフトは見えない
  - ARKit 自身が内部で補正している可能性がある

複数点にするため、**始終点だけでなく「軌跡が同一地点を再訪している区間」**も探す。
軌跡上の 2 時刻 (i, j) で |t_i - t_j| が小さく、かつ間の経路長が長いものを
「再訪ペア」とみなし、同じ比を計算する。

使い方
------
    conda activate sni-slam
    python tools/realdata/arkit_drift.py --root Real-data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stray_io  # noqa: E402


def revisit_pairs(pos: np.ndarray, close_m: float, min_path_m: float,
                  max_pairs: int = 5, stride: int = 5) -> List[Dict[str, float]]:
    """軌跡が同一地点へ戻っている (i, j) を探す。

    条件: |p_i - p_j| <= close_m かつ i..j の経路長 >= min_path_m。
    見つかったペアのうち、経路長の長い順に返す（長いほどドリフトが乗るはず）。
    """
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    idx = np.arange(0, len(pos), stride)
    P = pos[idx]
    out: List[Dict[str, float]] = []
    # 総当たりは O(n^2) だが間引き後は数千点なので許容範囲
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    for a, b in tree.query_pairs(close_m):
        i, j = idx[a], idx[b]
        if i > j:
            i, j = j, i
        path = float(cum[j] - cum[i])
        if path < min_path_m:
            continue
        out.append({"i": int(i), "j": int(j),
                    "closure_m": float(np.linalg.norm(pos[j] - pos[i])),
                    "path_m": round(path, 2),
                    "relative_drift": round(float(np.linalg.norm(pos[j] - pos[i]) / path), 5)})
    out.sort(key=lambda r: -r["path_m"])
    # 区間が大きく重なるものは省いて代表だけ残す
    kept: List[Dict[str, float]] = []
    for r in out:
        if all(abs(r["i"] - k["i"]) > 200 or abs(r["j"] - k["j"]) > 200 for k in kept):
            kept.append(r)
        if len(kept) >= max_pairs:
            break
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="Real-data")
    ap.add_argument("--scans", nargs="*", default=None)
    ap.add_argument("--close-m", type=float, default=0.5,
                    help="同一地点とみなす距離。S0 のループ判定と同じ 0.5 m")
    ap.add_argument("--min-path-m", type=float, default=10.0,
                    help="再訪ペアとして採用する最小の区間経路長")
    ap.add_argument("--out", default="output/RealData/_preflight/arkit_drift.json")
    args = ap.parse_args()

    scans = args.scans or sorted(
        d for d in os.listdir(args.root)
        if os.path.exists(os.path.join(args.root, d, "odometry.csv")))

    results: List[Dict] = []
    for s in scans:
        odo = stray_io.read_odometry(os.path.join(args.root, s))
        pos = odo["pos"]
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        path = float(seg.sum())
        se = float(np.linalg.norm(pos[-1] - pos[0]))
        rec: Dict = {
            "scan_id": s, "n_frames": int(len(pos)),
            "path_length_m": round(path, 2),
            "start_end_m": round(se, 4),
            "start_end_relative_drift": round(se / path, 5) if path > 0 else None,
            "start_end_is_closure": bool(se <= args.close_m),
            "revisit_pairs": revisit_pairs(pos, args.close_m, args.min_path_m),
        }
        results.append(rec)
        print("%-12s path %7.1f m  始終点 %7.3f m (%.2f%%)  %s  再訪ペア %d 件"
              % (s, path, se, 100 * se / path,
                 "ループ" if rec["start_end_is_closure"] else "開",
                 len(rec["revisit_pairs"])))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== 閉ループから得た相対ドリフト（これが論文に書ける実測値） ===")
    vals = []
    for r in results:
        if r["start_end_is_closure"]:
            print("  %-12s 始終点 %.3f m / 経路長 %.1f m = **%.2f%%**"
                  % (r["scan_id"], r["start_end_m"], r["path_length_m"],
                     100 * r["start_end_relative_drift"]))
            vals.append(r["start_end_relative_drift"])
        for p in r["revisit_pairs"]:
            print("  %-12s 再訪 frame %5d-%5d  ずれ %.3f m / 区間 %.1f m = %.2f%%"
                  % (r["scan_id"], p["i"], p["j"], p["closure_m"], p["path_m"],
                     100 * p["relative_drift"]))
            vals.append(p["relative_drift"])
    print("\n" + "=" * 70)
    print("★★ 再訪ペアの値は独立した実測として使ってはならない ★★")
    print("=" * 70)
    print("再訪ペアは「%.1f m 以内に近づいた箇所」を探しているので、" % args.close_m)
    print("**分子が構成上 %.1f m を超えられない**。したがって" % args.close_m)
    print("  相対ドリフト <= %.1f / 区間長" % args.close_m)
    print("が常に成立し、区間が短いほど見かけの割合が大きくなる。これは選択の artifact であって")
    print("実際のドリフトの傾向ではない（区間 %.0f m のペアは上限 %.2f%% に張り付く）。"
          % (args.min_path_m, 100 * args.close_m / args.min_path_m))
    print("さらに『近くを通った』と『同じ場所に戻った』を区別できていない。")
    print()
    print("使ってよいのは、**撮影者が意図して始点へ戻したループ**の始終点距離だけである：")
    closures = [r for r in results if r["start_end_is_closure"]]
    for r in closures:
        print("  %-12s %.3f m / %.1f m = **%.2f%%**"
              % (r["scan_id"], r["start_end_m"], r["path_length_m"],
                 100 * r["start_end_relative_drift"]))
    if not closures:
        print("  （閉ループのスキャンが無い）")
    print()
    print("これも**下界**である。始終点が偶然一致した可能性、途中で相殺された分、")
    print("ARKit 自身の内部補正、いずれも排除できていない。")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

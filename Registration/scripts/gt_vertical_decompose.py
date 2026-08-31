"""床と天井を**別々に**測り、鉛直方向のずれを分解する。

なぜ要るか
----------
撮影者から「研究室の引渡し後、**床コンセントを出すために工事した**」との情報を得た。
OA フロア（置床）を後から入れたなら、**実際の歩行面は BIM の構造スラブ上面より高い**。
床のずれが10本とも正（中央値 +0.09 m）だったことと向きは合う。

しかし**それだけでは説明が閉じない**：
OA フロアが 0.09 m なら実際の室内高は 2.600 − 0.09 = **2.51 m** になるはずだが、
S0 が LiDAR 深度から実測した天井高は **2.577 m** である（0.067 m 合わない）。

そこで**床と天井を別々に**測って、
  (a) 床だけ上がっているのか（＝ OA フロア）
  (b) 天井も動いているのか（＝ BIM 自体が実物と違う）
  (c) 剛体で合わせた妥協が両方に散っているだけなのか
を切り分ける。**「工事したらしい」で説明を確定させない。**

★ 注意：GT は剛体で人が合わせたものなので、床のずれと天井のずれは
  **独立ではない**（人が中間で妥協している）。独立なのは
  **両者の差＝室内高**であり、そちらは GT に依存しない。

    conda activate sni-slam
    python Registration/scripts/gt_vertical_decompose.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", default="output/GT_alignment")
    ap.add_argument("--npz", default="Registration/output/ifc/m3_ifc_all.npz")
    args = ap.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", ".."))
    os.chdir(repo)
    import open3d as o3d

    z = np.load(args.npz, allow_pickle=False)
    bp, bl, bn = z["points"], z["labels"], z["normals"]
    b_fl = plane_z(bp, bn, bl == CLASS_NAMES.index("floor"), True)
    b_ce = plane_z(bp, bn, bl == CLASS_NAMES.index("ceiling"), False)
    print("BIM: 床上面 %.4f m / 天井下面 %.4f m / 室内高 **%.4f m**"
          % (b_fl, b_ce, b_ce - b_fl))
    print()
    print("%-12s %10s %10s %11s %10s %10s"
          % ("scene", "床 z", "天井 z", "室内高", "床のずれ", "天井のずれ"))
    print("-" * 68)

    rows: List[Dict] = []
    for p in sorted(glob.glob(os.path.join(args.kit, "T_gt", "T_gt_*.json"))):
        scene = os.path.basename(p)[5:-5]
        T = np.asarray(json.load(open(p))["T_gt"], dtype=np.float64).reshape(4, 4)
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
        if len(cp) != len(v):
            continue
        lab = np.argmin(((cp[:, None, :] - PALETTE[None]) ** 2).sum(2), axis=1)

        R = T[:3, :3]
        vt = v @ R.T + T[:3, 3]
        nt = n @ R.T
        s_fl = plane_z(vt, nt, lab == CLASS_NAMES.index("floor"), True)
        s_ce = plane_z(vt, nt, lab == CLASS_NAMES.index("ceiling"), False)
        if s_fl is None or s_ce is None:
            print("%-12s 床または天井の点が足りない" % scene)
            continue
        rows.append({"scene": scene, "floor_z": s_fl, "ceiling_z": s_ce,
                     "room_height": s_ce - s_fl,
                     "floor_offset": s_fl - b_fl, "ceiling_offset": s_ce - b_ce})
        print("%-12s %10.4f %10.4f %11.4f %+10.4f %+10.4f"
              % (scene, s_fl, s_ce, s_ce - s_fl, s_fl - b_fl, s_ce - b_ce))

    if not rows:
        return 0
    h = np.array([r["room_height"] for r in rows])
    fo = np.array([r["floor_offset"] for r in rows])
    co = np.array([r["ceiling_offset"] for r in rows])
    print("\n【★ GT に依存しない量】実測の室内高（床上面〜天井下面）")
    print("  中央値 %.4f m / 平均 %.4f m / 標準偏差 %.4f m / 範囲 [%.4f, %.4f]"
          % (np.median(h), h.mean(), h.std(), h.min(), h.max()))
    print("  BIM の室内高 %.4f m との差 **%+.4f m（%+.2f%%）**"
          % (b_ce - b_fl, np.median(h) - (b_ce - b_fl),
             100 * (np.median(h) / (b_ce - b_fl) - 1)))
    print("  ※ 床と天井の差なので、剛体 GT の上下位置には依存しない。**独立な量である。**")

    print("\n【GT に依存する量】剛体で合わせた結果の上下位置")
    print("  床のずれ   中央値 %+.4f m（範囲 %+.4f 〜 %+.4f）"
          % (np.median(fo), fo.min(), fo.max()))
    print("  天井のずれ 中央値 %+.4f m（範囲 %+.4f 〜 %+.4f）"
          % (np.median(co), co.min(), co.max()))
    print("  ※ 人が剛体で合わせているので、この2つは独立ではない（中間で妥協している）")

    print("\n【解釈】")
    print("  OA フロア（置床）を後から入れたなら、**床だけ上がって天井は動かない**ので")
    print("    - 実測の室内高は BIM より **薄くなる**（床が上がったぶん）")
    print("    - 床のずれは正、天井のずれは 0 付近になる")
    dh = np.median(h) - (b_ce - b_fl)
    print("  実測：室内高の差 %+.4f m ／ 床 %+.4f m ／ 天井 %+.4f m"
          % (dh, np.median(fo), np.median(co)))
    if dh < -0.03 and np.median(fo) > 0.03 and abs(np.median(co)) < 0.05:
        print("  → **OA フロアの仮説と整合する。**")
    elif abs(dh) < 0.05:
        print("  → 室内高はほぼ一致している。**床だけが上がっているという説明は成り立たない。**")
        print("     床のずれは剛体合わせの妥協か、BIM 全体の高さ基準の差である可能性が高い。")
    else:
        print("  → 単純な OA フロアでは説明が閉じない。天井側も動いている。")

    with open(os.path.join(args.kit, "gt_vertical_decompose.json"), "w") as f:
        json.dump({"bim_floor_z": b_fl, "bim_ceiling_z": b_ce,
                   "bim_room_height": b_ce - b_fl, "scenes": rows},
                  f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % os.path.join(args.kit, "gt_vertical_decompose.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

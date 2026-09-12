"""軌跡が壁を**貫通している**かを、符号付き距離で直接判定する。

なぜ要るか
----------
`verify_gt.py` の「壁への近接率」は、それ自身が

    "壁の内部かは判定していない。近接の割合で代用している"

と断っている代用指標である。**廊下スキャンでは壁の 10 cm 以内まで寄るのは正常**なので、
近接率が上がったこと自体は悪化を意味しない。
ところが R20 §3 の判定は**この代用指標1本で決まってしまった**。
他の3項目（カメラ高さ・室内高・重力傾き）は鉛直方向の量で、
**GT の変化は 96.7〜99.9% が水平**だったため、原理的に動かなかったからである。

判定のしかた
------------
IFC の室内向き面（`is_inner`）の法線は**室内側を向いている**
（実測：床 n_z=+1.000、天井 n_z=-1.000、壁 n_z=0.000）。
したがって壁の室内向き面の最近傍点 q とその法線 n に対し

    s = (p - q) . n

は、**s > 0 なら室内側、s < 0 なら壁の内部**である。
**近接（|s| が小さい）と貫通（s < 0）を分けて数える。**

    conda activate sni-slam
    python Registration/scripts/wall_penetration.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from failure_decomposition import provenance          # noqa: E402
from regbim.labels import NAME_TO_ID                  # noqa: E402


def load_traj(scene: str):
    p = "data/realdata/%s/traj.txt" % scene
    if not os.path.exists(p):
        return None
    rows = [ln.split() for ln in open(p) if ln.strip()]
    return np.asarray([[float(v) for v in r] for r in rows]).reshape(-1, 4, 4)[:, :3, 3]


def signed_distance(cam: np.ndarray, pts: np.ndarray, nrm: np.ndarray, k: int):
    """最近傍 k 点の符号付き距離の中央値と、ユークリッド最近傍距離。

    1点だけ見ると、壁の角や開口の縁で法線が急に変わったときに誤判定する。
    中央値を取ることで、その1点に引きずられないようにする。

    **符号だけでは足りない。** 面から遠い点は、法線の向きから見ると
    「裏側」になるが、それは壁の内部にいるという意味ではない。
    実測で -20〜-29 m の「貫通」が出たが、これは**BIM のモデル化範囲の外**を
    歩いている軌跡点だった（廊下スキャンは BIM の範囲より長い）。
    そこでユークリッド距離も返し、**壁の近傍にある点だけを判定対象にする。**
    """
    from scipy.spatial import cKDTree

    dist, idx = cKDTree(pts).query(cam, k=k, workers=-1)
    if idx.ndim == 1:
        idx, dist = idx[:, None], dist[:, None]
    d = np.einsum("nkj,nkj->nk", cam[:, None, :] - pts[idx], nrm[idx])
    return np.median(d, axis=1), dist[:, 0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="Registration/output/ifc/m3_ifc_all.npz")
    ap.add_argument("--kits", nargs="+",
                    default=["output/GT_alignment", "output/GT_alignment_probe"])
    ap.add_argument("--labels", nargs="+", default=["wall"])
    ap.add_argument("--k", type=int, default=5, help="最近傍の数（中央値を取る）")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="この深さより深く入っていたら貫通とみなす[m]")
    ap.add_argument("--radius", type=float, default=0.60,
                    help="壁面からこの距離以内の軌跡点だけを判定対象にする[m]。"
                         "外側は BIM のモデル化範囲外でありうるので判定しない")
    ap.add_argument("--out", default="Registration/output/diag/wall_penetration.json")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    ids = [NAME_TO_ID[n] for n in args.labels]
    m = z["is_inner"].astype(bool) & np.isin(z["labels"], ids)
    pts, nrm = z["points"][m], z["normals"][m]
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    print("室内向き面（%s）%d 点。法線は室内側を向いている（床 n_z=+1 で確認済み）"
          % ("+".join(args.labels), int(m.sum())))
    print("貫通の定義: 最近傍 %d 点の符号付き距離の中央値 < -%.2f m\n" % (args.k, args.tol))

    scenes = sorted(os.path.basename(p)[5:-5]
                    for p in glob.glob(os.path.join(args.kits[0], "T_gt", "T_gt_*.json")))
    out: List[Dict] = []
    print("%-12s" % "scene" + "".join("%28s" % os.path.basename(k) for k in args.kits))
    print("%-12s" % "" + "".join("%28s" % "貫通率 / 最深[m]" for _ in args.kits))
    for s in scenes:
        cam = load_traj(s)
        if cam is None or not len(cam):
            print("%-12s (軌跡なし)" % s)
            continue
        rec: Dict = {"scene": s, "n_traj": int(len(cam))}
        line = "%-12s" % s
        for kit in args.kits:
            f = os.path.join(kit, "T_gt", "T_gt_%s.json" % s)
            if not os.path.exists(f):
                line += "%28s" % "—"
                continue
            T = np.asarray(json.load(open(f))["T_gt"], dtype=np.float64).reshape(4, 4)
            w = cam @ T[:3, :3].T + T[:3, 3]
            d, eu = signed_distance(w, pts, nrm, args.k)
            near = eu < args.radius          # 判定できるのは壁の近傍だけ
            n_adj = int(near.sum())
            if n_adj == 0:
                rec[kit] = {"n_adjudicable": 0}
                line += "%28s" % "判定対象なし"
                continue
            dn = d[near]
            rec[kit] = {"n_adjudicable": n_adj,
                        "frac_adjudicable": float(near.mean()),
                        "penetration_frac": float((dn < -args.tol).mean()),
                        "min_signed_m": float(dn.min()),
                        "median_signed_m": float(np.median(dn))}
            line += "%28s" % ("%.4f / %+.3f" % (rec[kit]["penetration_frac"],
                                                rec[kit]["min_signed_m"]))
        out.append(rec)
        print(line, flush=True)

    print()
    for kit in args.kits:
        fr = [r[kit]["penetration_frac"] for r in out
              if kit in r and "penetration_frac" in r[kit]]
        mn = [r[kit]["min_signed_m"] for r in out
              if kit in r and "min_signed_m" in r[kit]]
        if not fr:
            continue
        print("%-24s 貫通率 中央 %.4f / 最大 %.4f   最深 %+.3f m"
              % (os.path.basename(kit), float(np.median(fr)), max(fr), min(mn)))
    adj = [r[args.kits[0]].get("frac_adjudicable") for r in out if args.kits[0] in r]
    adj = [a for a in adj if a is not None]
    if adj:
        print("（判定対象は壁面から %.2f m 以内の軌跡点。全体の %.1f〜%.1f%%）"
              % (args.radius, 100 * min(adj), 100 * max(adj)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"provenance": provenance(), "labels": args.labels, "k": args.k,
                   "tol": args.tol, "results": out}, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

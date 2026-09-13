"""R24 §1-1 — 指定した2面の**面間距離**を、スキャンと BIM の両方で出す。

なぜ面で測るのか
----------------
**本人は現地で面間を測る**（室内高なら床面と天井面）。**こちらも面で測らないと比較にならない。**

**やってはいけない代用**：室内高を「床点の95パーセンタイルと天井点の5パーセンタイル」で出すこと。
R20 の報告 §1-2 で、この代用が**回転で 1.5 mm 揺らぐ**ことを記録している。
**百分位は点の分布の端を拾うので、面の位置ではない。**

使い方
------
測定位置は「東端から 1 m、床から 1.2 m」のように来る。
**その位置の近傍だけを使って面を当てはめる**（天井が水平でない可能性があるため、
室全体で1つの値にしない）。

    conda activate sni-slam
    python Registration/scripts/face_distance.py --spec Registration/configs/measure_spec.json

`--spec` の形（1件ぶん）::

    {"id": "411_room_height_A",
     "config": "Registration/configs/realdata/m3_block_b__E2.yaml",
     "face_a": {"class": "floor",   "normal": "+z"},
     "face_b": {"class": "ceiling", "normal": "-z"},
     "at": [x, y, z],            # 測定位置（BIM 座標）
     "radius": 1.0,              # その位置からこの半径内の点だけを使う
     "measured_m": null}         # 現地実測値。**後から入れる**

出力（測定ごとに。**平均に潰さない**）::

    L_scan, L_bim, 面の当てはめの残差・点数・法線, そして measured_m があれば
    s_sensor = L_scan / L_measured,  Delta_bim = L_bim - L_measured
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from failure_decomposition import provenance                  # noqa: E402
from regbim import io_utils, metrics                          # noqa: E402
from regbim.labels import NAME_TO_ID                          # noqa: E402

AXES = {"+x": (0, 1.0), "-x": (0, -1.0), "+y": (1, 1.0),
        "-y": (1, -1.0), "+z": (2, 1.0), "-z": (2, -1.0)}


def fit_plane(pts: np.ndarray, axis: int, sign: float, at_axis: float,
              tol: float = 0.05, iters: int = 20) -> Optional[Dict]:
    """測定位置から見て `sign` 側にある、**いちばん近い面**の座標を推定する。

    `normal` は「面がどちら側にあるか」を表す（床は測定位置の下＝`-z`、
    天井は上＝`+z`、壁は `±x` / `±y`）。

    ★ **最初の実装では法線方向へ射影した中央値を取っていた。**
      これだと**壁の向かい合う2面が同じ点集合から同じ値を返し、距離が 0 になった**。
      面は「測定位置のどちら側にあるか」で分ける必要がある。

    ★ **百分位で面の位置を代用しない**（R20 §1-2）。まず最近傍側の点を取り、
      その近傍だけで中央値を繰り返して収束させる。
    """
    if len(pts) < 30:
        return None
    d = pts[:, axis]
    side = d > at_axis if sign > 0 else d < at_axis
    if side.sum() < 30:
        return None
    d = d[side]
    # 測定位置に最も近い面から始める（外れ値を避けるため 5 パーセンタイル側）
    c = float(np.percentile(d, 5.0 if sign > 0 else 95.0))
    for _ in range(iters):
        m = np.abs(d - c) < tol
        if m.sum() < 20:
            return None
        c_new = float(np.median(d[m]))
        if abs(c_new - c) < 1e-9:
            c = c_new
            break
        c = c_new
    m = np.abs(d - c) < tol
    return {"coord_m": c, "axis": "xyz"[axis], "side": "+" if sign > 0 else "-",
            "n_points": int(m.sum()),
            "rms_residual_m": float(np.sqrt(((d[m] - c) ** 2).mean()))}


def select(pts: np.ndarray, labels: np.ndarray, face: Dict,
           at: np.ndarray, radius: float, axis: int,
           normals: Optional[np.ndarray] = None, cos_min: float = 0.8):
    """クラス・面の向き・測定位置で点を絞る。**位置を無視して室全体を使わない。**

    ★ 半径は**面に平行な方向だけ**に当てる。3D 距離で絞ると、床から 1.2 m の位置に
      半径 1 m を当てたとき**天井（1.9 m 上）が丸ごと落ちる**（最初の実装がこれだった）。

    ★ **面の法線で絞る。** ある軸上の距離を定めうるのは、**法線がその軸を向いた面だけ**である。
      南北に伸びる壁は y 方向に点が連続して並ぶので、法線で絞らないと
      **その壁の任意の断面を「y 座標の面」として拾ってしまう**。
      実際それが起きた：壁間 10.17 m の測定に対し、1.03 m という値を返していた。
      **この修正は実測値を見て合わせたものではない**——
      「法線がその軸を向いていない面は、その軸上の距離を定義しない」という定義から出る。
    """
    m = labels == NAME_TO_ID[face["class"]]
    if normals is not None:
        nrm = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        m &= np.abs(nrm[:, axis]) > cos_min
    if radius and radius > 0:
        other = [i for i in range(3) if i != axis]
        m &= np.linalg.norm(pts[:, other] - at[other][None, :], axis=1) < radius
    return pts[m]


def measure(spec: Dict) -> Dict:
    cfg = yaml.safe_load(open(spec["config"]))
    cfg["source"] = dict(cfg["source"], seed=0)
    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    at = np.asarray(spec["at"], dtype=np.float64)
    rad = float(spec.get("radius", 1.0))

    gt_kit = spec.get("gt_kit", "output/GT_alignment")
    G = np.asarray(json.load(open(os.path.join(
        gt_kit, "T_gt", "T_gt_%s.json" % spec["scene"])))["T_gt"],
        dtype=np.float64).reshape(4, 4)
    src_pts = metrics.apply_sim3(G, src.points)      # BIM 座標へ
    R_gt = metrics.decompose_sim3(G)[0]
    src_nrm = None if src.normals is None else (np.asarray(src.normals) @ R_gt.T)

    out: Dict = {"id": spec["id"], "config": spec["config"], "gt_kit": gt_kit,
                 "at": spec["at"], "radius_m": rad,
                 "face_a": spec["face_a"], "face_b": spec["face_b"],
                 "reference_spaces": cfg["reference"].get("spaces"),
                 "reference_inner_only": cfg["reference"].get("inner_only")}
    for side, face in (("a", spec["face_a"]), ("b", spec["face_b"])):
        ax, sgn = AXES[face["normal"]]
        for who, pts, labels, nrm in (("scan", src_pts, src.labels, src_nrm),
                                      ("bim", dst.points, dst.labels, dst.normals)):
            sel = select(pts, labels, face, at, rad, ax, nrm)
            out["%s_%s" % (who, side)] = (fit_plane(sel, ax, sgn, float(at[ax]))
                                          or {"error": "点が足りない",
                                              "n_candidates": int(len(sel))})
    for who in ("scan", "bim"):
        a, b = out.get("%s_a" % who), out.get("%s_b" % who)
        if a and b and "coord_m" in a and "coord_m" in b:
            # 同じ軸上の2つの面の座標。面間距離はその差である
            out["L_%s_m" % who] = float(abs(a["coord_m"] - b["coord_m"]))
        else:
            out["L_%s_m" % who] = None

    L = spec.get("measured_m")
    if L:
        out["measured_m"] = L
        if out["L_scan_m"]:
            out["s_sensor"] = out["L_scan_m"] / L
        if out["L_bim_m"]:
            out["delta_bim_m"] = out["L_bim_m"] - L
        out["orientation"] = ("vertical" if spec["face_a"]["normal"][1] == "z"
                              else "horizontal")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="測定指定の JSON（リスト）")
    ap.add_argument("--out", default="Registration/output/diag/face_distance.json")
    args = ap.parse_args()

    specs = json.load(open(args.spec))
    if isinstance(specs, dict):
        specs = specs.get("measurements", [specs])
    rows: List[Dict] = []
    print("%-22s %12s %12s %12s %10s" % ("id", "L_scan[m]", "L_bim[m]", "実測[m]", "s_sensor"))
    for sp in specs:
        r = measure(sp)
        rows.append(r)
        print("%-22s %12s %12s %12s %10s"
              % (r["id"],
                 "—" if r["L_scan_m"] is None else "%.4f" % r["L_scan_m"],
                 "—" if r["L_bim_m"] is None else "%.4f" % r["L_bim_m"],
                 r.get("measured_m", "（未測定）"),
                 "%.5f" % r["s_sensor"] if "s_sensor" in r else "—"))
        for who in ("scan_a", "scan_b", "bim_a", "bim_b"):
            f = r.get(who) or {}
            if "coord_m" in f:
                print("     %-8s %s=%+.4f m（%s 側）/ %5d 点 / 残差 RMS %.4f m"
                      % (who, f["axis"], f["coord_m"], f["side"],
                         f["n_points"], f["rms_residual_m"]))
            else:
                print("     %-8s %s" % (who, f.get("error")))

    if any("s_sensor" in r for r in rows):
        for orient in ("vertical", "horizontal"):
            v = [r["s_sensor"] for r in rows
                 if r.get("orientation") == orient and "s_sensor" in r]
            if v:
                print("\n%s の s_sensor: n=%d / 中央 %.5f / 範囲 [%.5f, %.5f]"
                      % (orient, len(v), float(np.median(v)), min(v), max(v)))
                print("  **平均に潰さず、測定ごとの値も上に出している**")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"provenance": provenance(), "measurements": rows},
                  f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""R3 T1 の受入条件を、出力した点群そのものから検算する。

指示書 §5 の受入条件のうち、数値で確かめられるものを全部ここで見る。
とくに **単位の検算**は「室高が 2.5〜3.0 m に入るか」で行うと決まっているが、
点群の z extent（3.402 m）は**床スラブの厚みや天井裏を含む**ので、そのままでは使えない。
**床の上面から天井の下面まで**を取り出して測る。

対照は S0 の実データ実測 **2.576〜2.578 m**（同じ部屋を別々に撮った2本が 0.3% 以内で一致）。
IFC 側と実測側は独立に得た量なので、一致すれば単位・幾何の両方の検算になる
（追補2 §1-4「別経路で測った量の一致が検算になる」）。

    conda run -n bim-ifc python Registration/scripts/verify_ifc_cloud.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

CLASS_NAMES = ["background", "wall", "door", "floor", "window", "ceiling"]
MEASURED_CEILING_H = (2.576, 2.578)      # S0 preflight の実測レンジ [m]


def plane_z(pts: np.ndarray, nrm: np.ndarray, want_up: bool, bin_m: float = 0.02):
    """水平面のうち、法線が上（下）向きの点だけを集めて最頻の z を返す。

    床スラブは**ソリッド**なので上面と下面の両方が点になる。室の高さを測るには
    「上を向いた床面」と「下を向いた天井面」だけが要る。
    """
    horiz = np.abs(nrm[:, 2]) > 0.9
    side = (nrm[:, 2] > 0) if want_up else (nrm[:, 2] < 0)
    z = pts[horiz & side, 2]
    if len(z) == 0:
        return None, 0
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + bin_m, bin_m))
    if len(hist) == 0:
        return float(np.median(z)), len(z)
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1])), len(z)


def double_sided_gap(pts: np.ndarray, nrm: np.ndarray) -> float:
    """上向き床面と下向き床面の z 差＝床スラブの厚み。

    なぜ見るか：IFC の要素はソリッドなので**表裏の両面**が点になる。SLAM が観測できるのは
    室内側の1面だけなので、reference にだけ「もう1枚の面」が存在する。この距離が
    対応付けのゲート（`semantic_icp.max_corr_dist` 0.3 m）に対して大きいか小さいかで、
    悪影響が出るかどうかが決まる。
    """
    up, _ = plane_z(pts, nrm, want_up=True)
    dn, _ = plane_z(pts, nrm, want_up=False)
    if up is None or dn is None:
        return float("nan")
    return abs(up - dn)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="Registration/output/ifc")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.npz")))
    print("%-22s %9s %9s %9s %9s %8s" % ("cloud", "床上面z", "天井下面z",
                                         "室高m", "床厚m", "判定"))
    print("-" * 74)
    summary = {}
    for p in files:
        z = np.load(p, allow_pickle=False)
        pts, lab, nrm = z["points"], z["labels"], z["normals"]
        meta = json.loads(str(z["meta"]))
        fl = lab == CLASS_NAMES.index("floor")
        ce = lab == CLASS_NAMES.index("ceiling")
        name = os.path.basename(p).replace(".npz", "")
        if fl.sum() == 0 or ce.sum() == 0:
            print("%-22s  床 %d / 天井 %d 点 — 高さを測れない" % (name, fl.sum(), ce.sum()))
            continue
        z_floor, n_f = plane_z(pts[fl], nrm[fl], want_up=True)
        z_ceil, n_c = plane_z(pts[ce], nrm[ce], want_up=False)
        h = z_ceil - z_floor
        slab = double_sided_gap(pts[fl], nrm[fl])
        ok = "○" if MEASURED_CEILING_H[0] - 0.15 <= h <= MEASURED_CEILING_H[1] + 0.15 else "×"
        print("%-22s %9.3f %9.3f %9.3f %9.3f %8s"
              % (name, z_floor, z_ceil, h, slab, ok))
        summary[name] = {
            "floor_top_z_m": round(z_floor, 4), "ceiling_bottom_z_m": round(z_ceil, 4),
            "clear_height_m": round(h, 4), "floor_slab_thickness_m": round(slab, 4),
            "class_counts": meta["class_counts"],
            "n_points": int(len(pts)),
        }

    print("\n対照: S0 の実データ実測 天井高 %.3f〜%.3f m" % MEASURED_CEILING_H)
    print("  （IFC と実測は独立に得た量。一致すれば単位換算と幾何の両方の検算になる）")

    # --- 411 の凹み ＝ 410 の床面積 か（別名割当の独立な2つ目の根拠）---
    allp = os.path.join(args.dir, "m3_ifc_all.npz")
    if os.path.exists(allp):
        meta = json.loads(str(np.load(allp, allow_pickle=False)["meta"]))
        sp = {int(r["id"]): r for r in meta["spaces"] if "footprint_area_m2" in r}
        alias = {k: int(v) for k, v in meta["space_alias"].items()}
        if "411" in alias and "410" in alias:
            a, b = sp[alias["411"]], sp[alias["410"]]
            notch = a["bbox_area_m2"] - a["footprint_area_m2"]
            print("\n[別名割当の独立検算] 411 の L 字の凹み = 410 か")
            print("  411: bbox %.1f - 床 %.1f = 凹み **%.1f m2**"
                  % (a["bbox_area_m2"], a["footprint_area_m2"], notch))
            print("  410: 床 **%.1f m2**（fill_ratio %.3f ＝ 直方体）"
                  % (b["footprint_area_m2"], b["fill_ratio"]))
            print("  差 %.1f%% — 一致するなら「L 字の欠けに直方体の室が入っている」ことになり、"
                  % (100 * abs(notch - b["footprint_area_m2"]) / b["footprint_area_m2"]))
            print("  床面積の大小だけに頼らない2つ目の根拠になる")
            summary["_alias_check"] = {
                "notch_area_m2": round(notch, 2),
                "room410_area_m2": b["footprint_area_m2"],
                "rel_diff_pct": round(100 * abs(notch - b["footprint_area_m2"])
                                      / b["footprint_area_m2"], 2)}

    out = os.path.join(args.dir, "verify_ifc_cloud.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

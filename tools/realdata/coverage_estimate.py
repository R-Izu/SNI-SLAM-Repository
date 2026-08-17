"""S0 追加2 — 被覆率の概算（BIM を使わず、スキャン間の面積比で出す）。

指示書 00_Coordination/Instructions/2026-08-17_r4_realdata_slam_execution.md §2「追加2」に対応。
IFC ローダ（R3）が未実装のため BIM 床面積は使えない。代わりに

    coverage = A_ref / A_s

（A_s = そのスキャンが実際に見た床面積、A_ref = reference として与える部屋の床面積）
をスキャン間の比で概算する。

面積の定義
----------
床平面から ±0.2 m の帯に入る逆投影点を水平面（XZ）へ落とし、
5 cm グリッドで間引いたうえで **凸包面積** と **α-shape 面積** の両方を出す。
411 のような L 字（非凸）平面では凸包が面積を過大評価するため、
本スクリプトは既定で **α-shape を主**、凸包を従として扱う。

reference 面積の決め方
---------------------
「単室スキャン」として指定された scan_id 群の α-shape 面積の**中央値**を
その部屋の床面積とみなす。どの scan_id がどの部屋かは計測から決まらないので、
``--ref-411`` / ``--ref-410`` で**外から与える**（マニフェストの covers から取ることもできる）。

使い方
------
    python tools/realdata/coverage_estimate.py \
        --preflight output/RealData/_preflight \
        --ref-411 707e94b9a5 --ref-410 a499702124 \
        --out output/RealData/_preflight/coverage_estimate.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np


def load_preflight(pf_dir: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for p in sorted(glob.glob(os.path.join(pf_dir, "preflight_*.json"))):
        with open(p) as f:
            d = json.load(f)
        if "scan_id" in d:
            out[d["scan_id"]] = d
    return out


def median_area(recs: Dict[str, dict], ids: List[str], key: str) -> Optional[float]:
    vals = [recs[i][key] for i in ids if i in recs and recs[i].get(key)]
    return float(np.median(vals)) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", default="output/RealData/_preflight")
    ap.add_argument("--out", default="output/RealData/_preflight/coverage_estimate.csv")
    ap.add_argument("--ref-411", nargs="*", default=[],
                    help="411 単体スキャンの scan_id（この α-shape 面積の中央値を 411 の床面積とする）")
    ap.add_argument("--ref-410", nargs="*", default=[],
                    help="410 単体スキャンの scan_id")
    ap.add_argument("--area-key", default="alpha_shape_area_m2",
                    choices=["alpha_shape_area_m2", "hull_area_m2"])
    args = ap.parse_args()

    recs = load_preflight(args.preflight)
    if not recs:
        raise SystemExit("no preflight_*.json found under %s" % args.preflight)

    a411 = median_area(recs, args.ref_411, args.area_key)
    a410 = median_area(recs, args.ref_410, args.area_key)
    a_both = (a411 + a410) if (a411 is not None and a410 is not None) else None

    print("reference areas [%s]" % args.area_key)
    print("  411        = %s" % ("%.1f m2 (n=%d)" % (a411, len(args.ref_411))
                                 if a411 else "UNKNOWN (--ref-411 未指定)"))
    print("  410        = %s" % ("%.1f m2 (n=%d)" % (a410, len(args.ref_410))
                                 if a410 else "UNKNOWN (--ref-410 未指定)"))
    print("  411+410    = %s" % ("%.1f m2" % a_both if a_both else "UNKNOWN"))
    print()

    cols = ["scan_id", "A_s_alpha_m2", "A_s_hull_m2", "solidity",
            "coverage_ref_411", "coverage_ref_410", "coverage_ref_411_410",
            "is_ref_411", "is_ref_410"]
    rows = []
    for sid in sorted(recs):
        r = recs[sid]
        a_s = r.get(args.area_key)
        def cov(a_ref: Optional[float]) -> str:
            if a_ref is None or not a_s:
                return ""
            return "%.3f" % (a_ref / a_s)
        rows.append({
            "scan_id": sid,
            "A_s_alpha_m2": round(r.get("alpha_shape_area_m2", 0.0), 2),
            "A_s_hull_m2": round(r.get("hull_area_m2", 0.0), 2),
            "solidity": round(r.get("solidity", 0.0), 3),
            "coverage_ref_411": cov(a411),
            "coverage_ref_410": cov(a410),
            "coverage_ref_411_410": cov(a_both),
            "is_ref_411": sid in args.ref_411,
            "is_ref_410": sid in args.ref_410,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    hdr = "%-14s %10s %10s %9s %9s %9s %9s" % (
        "scan_id", "A_s_alpha", "A_s_hull", "solidity", "cov(411)", "cov(410)", "cov(both)")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-14s %10.1f %10.1f %9.3f %9s %9s %9s" % (
            r["scan_id"], r["A_s_alpha_m2"], r["A_s_hull_m2"], r["solidity"],
            r["coverage_ref_411"] or "-", r["coverage_ref_410"] or "-",
            r["coverage_ref_411_410"] or "-"))
    print("\nwrote %s" % args.out)
    print("\n注意: solidity = α-shape 面積 / 凸包面積。L 字など非凸な床では 1 を大きく下回り、"
          "\n      その差がそのまま『凸包による過大評価の程度』になる。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

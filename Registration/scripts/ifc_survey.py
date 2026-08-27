"""R3 T1 の下調べ — IFC の中身を実測する（点群は作らない）。

指示書 [[2026-08-05_r3_ifc_loader_partial_coverage]] §2 が main 側の調査値を載せているが、
**実装機側でも実測して突き合わせる**（決め打ちを避ける）。とくに：

- 単位（`IfcSIUnit` の prefix）。mm → m の係数を**読み取る**。決め打ちしない
- `IfcSlab` / `IfcCovering` の `PredefinedType`。`.FLOOR.` `.CEILING.` 以外は background に落とす
- `IfcSpace` の数と、どちらが 411（L字＝非凸）か 410（直方体）か

`bim-ifc` env で実行する（`sni-slam` は py3.7 で ifcopenshell が入らない）。

    conda run -n bim-ifc python Registration/scripts/ifc_survey.py \
        BIM_IFC_Extraction/input/m3-411.ifc
"""

from __future__ import annotations

import argparse
import collections
import json
from typing import Dict, List

import numpy as np


def length_scale_to_m(f) -> Dict[str, object]:
    """IfcSIUnit から長さの m 換算係数を読む（決め打ちしない）。"""
    prefix_pow = {
        "EXA": 18, "PETA": 15, "TERA": 12, "GIGA": 9, "MEGA": 6, "KILO": 3,
        "HECTO": 2, "DECA": 1, None: 0, "DECI": -1, "CENTI": -2, "MILLI": -3,
        "MICRO": -6, "NANO": -9, "PICO": -12,
    }
    info: Dict[str, object] = {"found": False, "scale_to_m": 1.0}
    for ua in f.by_type("IfcUnitAssignment"):
        for u in ua.Units:
            if u.is_a("IfcSIUnit") and u.UnitType == "LENGTHUNIT":
                p = u.Prefix
                info = {"found": True, "unit": u.Name, "prefix": p,
                        "scale_to_m": 10.0 ** prefix_pow.get(p, 0)}
            elif u.is_a("IfcConversionBasedUnit") and u.UnitType == "LENGTHUNIT":
                info = {"found": True, "unit": u.Name,
                        "prefix": "conversion_based",
                        "scale_to_m": float(u.ConversionFactor.ValueComponent.wrappedValue)}
    return info


def main() -> int:
    import ifcopenshell
    import ifcopenshell.geom

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ifc")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    f = ifcopenshell.open(args.ifc)
    rep: Dict[str, object] = {"path": args.ifc, "schema": f.schema}
    print("schema: %s" % f.schema)

    unit = length_scale_to_m(f)
    rep["length_unit"] = unit
    print("length unit: %s  -> x%.6g で m" % (unit.get("unit"), unit["scale_to_m"]))

    # --- クラス別件数と PredefinedType ---
    interesting = ["IfcWall", "IfcWallStandardCase", "IfcDoor", "IfcWindow",
                   "IfcSlab", "IfcCovering", "IfcColumn", "IfcBeam", "IfcRailing",
                   "IfcFurnishingElement", "IfcBuildingElementProxy", "IfcSpace"]
    counts: Dict[str, object] = {}
    for cls in interesting:
        try:
            els = f.by_type(cls)
        except Exception:
            els = []
        if not els:
            continue
        pt = collections.Counter(getattr(e, "PredefinedType", None) for e in els)
        counts[cls] = {"n": len(els),
                       "predefined_types": {str(k): v for k, v in pt.items()}}
        print("  %-26s n=%-4d  PredefinedType=%s"
              % (cls, len(els), dict(counts[cls]["predefined_types"])))
    rep["classes"] = counts

    # --- IfcSpace の形状（411=L字/非凸 vs 410=直方体 の判別材料）---
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    spaces: List[Dict[str, object]] = []
    for sp in f.by_type("IfcSpace"):
        rec: Dict[str, object] = {
            "id": sp.id(), "Name": sp.Name, "LongName": getattr(sp, "LongName", None),
        }
        try:
            shape = ifcopenshell.geom.create_shape(settings, sp)
            v = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
            # ★ ifcopenshell.geom は **既に m へ正規化して返す**（ファイルの単位が mm でも）。
            #   ここで IfcSIUnit の係数を掛けると二重適用になる。実測で確認済み:
            #   掛けると室の extent が 0.007 m（＝7 mm）になり、掛けなければ 7 m。
            #   IfcSIUnit の読み取りは「ファイルが mm 建てである」ことの記録として残し、
            #   幾何には適用しない。検算は室高が 2.5〜3.0 m に入るかで行う（受入条件）。
            faces = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
            lo, hi = v.min(axis=0), v.max(axis=0)
            ext = hi - lo

            # フットプリント面積は、**XY グリッドの占有セル数**で測る。
            # 凸包だと L 字の凹みを埋めてしまい、まさに判別したい差が消える。
            # scipy は bim-ifc env に無いので numpy だけで済ませる。
            import open3d as o3d
            m = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(faces))
            pts = np.asarray(m.sample_points_uniformly(200000).points)
            cell = 0.1
            keys = np.floor(pts[:, :2] / cell).astype(np.int64)
            n_cells = len(np.unique(keys, axis=0))
            foot_area = n_cells * cell * cell
            bbox_area = float(ext[0] * ext[1])
            rec.update({
                "n_verts": int(len(v)),
                "extent_m": [round(float(x), 3) for x in ext],
                "height_m": round(float(ext[2]), 3),
                "footprint_area_m2": round(foot_area, 2),
                "bbox_area_m2": round(bbox_area, 2),
                # 直方体なら bbox を埋めるので 1 に近い。L 字は 1 を下回る
                "fill_ratio": round(foot_area / bbox_area, 3) if bbox_area > 0 else None,
            })
        except Exception as e:
            rec["error"] = "%s: %s" % (type(e).__name__, e)
        spaces.append(rec)
        print("  IfcSpace id=%s Name=%r LongName=%r  %s"
              % (rec["id"], rec.get("Name"), rec.get("LongName"),
                 {k: rec.get(k) for k in ("extent_m", "height_m",
                                          "footprint_area_m2", "fill_ratio")}))
    rep["spaces"] = spaces

    print("\n判別の目安（2つを突き合わせる）:")
    print("  1) 床面積: S0 の実データ実測では **411 = 119.2 m² / 410 = 35.1 m²** と 3.4 倍違う。")
    print("     同じ部屋を別々に撮った2本が 0.3%% 以内で一致しているので、この値は信頼できる。")
    print("  2) fill_ratio（フットプリント面積 / bbox面積）: 直方体(410)なら 1 に近い。")
    print("     L字(411)は 1 を下回る。凸包ではなく占有セル数で測っているので凹みが残る。")

    if args.out:
        with open(args.out, "w") as fp:
            json.dump(rep, fp, indent=2, ensure_ascii=False)
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

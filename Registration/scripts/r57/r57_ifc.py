"""R57 §3-2：ConSLAM の IFC の中身と、本研究の参照の作り方での点群（bim-ifc の環境）。**回す前に commit する。**

- 要素の種類ごとの数（`f.by_type`、サブクラスを含まない数 `is_a()` の集計）、IfcSlab・IfcCovering の PredefinedType の内訳
- 単位（`ifc_export.length_scale_to_m`）、階（IfcBuildingStorey の Name・Elevation）、IfcSpace の数と名前、IfcSite・IfcMapConversion の有無
- 参照の点群：`ifc_export.build_ifc_cloud(ifc, class_map, 200000, spaces=None, storeys=None, seed=0, keep_classes=[壁・床・天井・扉・窓])`。
  class_map は R49 の実データの config の `classes.ifc_class_map`（`out/class_map.json` に書き出したもの）
出力：`~/rizu/conslam/out/ifc_info.json`、`~/rizu/conslam/out/conslam_ifc_all.npz`
"""
import json, os, sys
from collections import Counter
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
import ifcopenshell                                            # noqa: E402
import ifc_export as X                                         # noqa: E402

IFC = "/mnt/d/rizu/SNI-SLAM/data/ConSLAM/m1743877/ConSLAM_Building_00_GroundTruth_V9_wall_segment.ifc"
OUT = os.path.expanduser("~/rizu/conslam/out")


def main():
    f = ifcopenshell.open(IFC)
    info = {"schema": f.schema, "ifcopenshell": ifcopenshell.version}
    prods = Counter(e.is_a() for e in f.by_type("IfcProduct"))
    info["products_by_type"] = dict(prods.most_common())
    info["slab_predefined"] = dict(Counter(str(e.PredefinedType) for e in f.by_type("IfcSlab")))
    info["covering_predefined"] = dict(Counter(str(e.PredefinedType) for e in f.by_type("IfcCovering")))
    info["unit"] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in X.length_scale_to_m(f).items()}
    info["storeys"] = [{"name": s.Name, "elevation": s.Elevation} for s in f.by_type("IfcBuildingStorey")]
    info["spaces"] = [{"name": s.Name, "long_name": s.LongName} for s in f.by_type("IfcSpace")]
    info["n_spaces"] = len(info["spaces"])
    info["sites"] = [{"name": s.Name, "lat": list(s.RefLatitude) if s.RefLatitude else None, "lon": list(s.RefLongitude) if s.RefLongitude else None,
                      "elev": s.RefElevation} for s in f.by_type("IfcSite")]
    try:
        info["map_conversion"] = [str(m) for m in f.by_type("IfcMapConversion")]
    except Exception as e:                                      # IFC2X3 には無い
        info["map_conversion"] = "type not in schema (%s)" % type(e).__name__
    info["storey_of_elements"] = dict(Counter(X.storey_of(f).values()))
    class_map = json.load(open(os.path.join(OUT, "class_map.json")))
    res = X.build_ifc_cloud(IFC, class_map, 200000, spaces=None, storeys=None, seed=0,
                            keep_classes=["wall", "floor", "ceiling", "door", "window"])
    pts, lab = np.asarray(res["points"]), np.asarray(res["labels"])
    names = {1: "wall", 2: "door", 3: "floor", 4: "window", 5: "ceiling", 0: "background"}
    info["reference_cloud"] = {"n_points": int(len(pts)), "by_class": {names.get(int(k), str(k)): int(v) for k, v in Counter(lab.tolist()).items()},
                               "bbox_min": pts.min(0).tolist(), "bbox_max": pts.max(0).tolist(),
                               "is_inner_available": res.get("is_inner") is not None,
                               "meta": {k: v for k, v in res["meta"].items() if isinstance(v, (int, float, str, list, dict, type(None)))}}
    np.savez_compressed(os.path.join(OUT, "conslam_ifc_all.npz"), points=pts, labels=lab, normals=np.asarray(res["normals"]),
                        is_inner=(np.asarray(res["is_inner"]) if res.get("is_inner") is not None else np.zeros(0)))
    json.dump(info, open(os.path.join(OUT, "ifc_info.json"), "w"), indent=1, default=str, ensure_ascii=False)
    print(json.dumps(info, indent=1, default=str, ensure_ascii=False)[:5000])


if __name__ == "__main__":
    main()

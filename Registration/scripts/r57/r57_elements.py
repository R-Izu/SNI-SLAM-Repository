"""R57 §3-5 の材料：IFC の要素ごとの範囲（世界座標の軸にそろえた箱、m）（bim-ifc の環境）。**回す前に commit する。**
種類：IfcWall・IfcWallStandardCase・IfcColumn・IfcDoor・IfcWindow・IfcSlab。`ifcopenshell.geom.iterator`（USE_WORLD_COORDS）で形を作り、頂点の最小・最大。
単位は `ifc_export.length_scale_to_m` の倍率をかけて m にする（`build_ifc_cloud` の点と同じ座標）。出力 `~/rizu/conslam/out/elements.json`
"""
import json, os, sys
import multiprocessing
import numpy as np
import ifcopenshell, ifcopenshell.geom

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
import ifc_export as X                                         # noqa: E402

IFC = "/mnt/d/rizu/SNI-SLAM/data/ConSLAM/m1743877/ConSLAM_Building_00_GroundTruth_V9_wall_segment.ifc"
OUT = os.path.expanduser("~/rizu/conslam/out")
TYPES = ["IfcWall", "IfcWallStandardCase", "IfcColumn", "IfcDoor", "IfcWindow", "IfcSlab"]


def main():
    f = ifcopenshell.open(IFC)
    unit = X.length_scale_to_m(f)
    s = ifcopenshell.geom.settings(); s.set(s.USE_WORLD_COORDS, True)
    els = [e for t in TYPES for e in f.by_type(t) if e.is_a() == t]
    it = ifcopenshell.geom.iterator(s, f, multiprocessing.cpu_count(), include=els)
    out = []
    if it.initialize():
        while True:
            sh = it.get(); e = f.by_id(sh.id)
            v = np.asarray(sh.geometry.verts, dtype=np.float64).reshape(-1, 3)
            if len(v):
                out.append({"type": e.is_a(), "id": int(sh.id), "min": v.min(0).tolist(), "max": v.max(0).tolist()})
            if not it.next():
                break
    json.dump({"unit_scale_applied_by_geom": "geom settings return metres (ifcopenshell converts)", "file_unit": unit, "n": len(out), "elements": out},
              open(os.path.join(OUT, "elements.json"), "w"), default=str)
    from collections import Counter
    print(len(out), Counter(o["type"] for o in out), np.min([o["min"] for o in out], 0), np.max([o["max"] for o in out], 0))


if __name__ == "__main__":
    main()

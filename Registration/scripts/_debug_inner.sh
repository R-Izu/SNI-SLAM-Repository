#!/usr/bin/env bash
# 共有壁の両面が is_inner=0 になる理由を、判定条件ごとに分解する。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd /home/student/rizu/SNI-SLAM
conda activate bim-ifc

python - <<'PY'
import numpy as np
import sys
sys.path.insert(0, "Registration/scripts")
from ifc_export import space_records
import ifcopenshell

f = ifcopenshell.open("BIM_IFC_Extraction/input/m3-411.ifc")
recs = space_records(f)
for r in recs:
    print("IfcSpace id=%s Name=%r  z_range=%s  cells=%d  cell=%.2f"
          % (r["id"], r.get("Name"), r.get("z_range_m"), len(r["_cells"]), r["_cell_size"]))

# 共有壁の 411 側の面上の代表点で、判定を手で追う
eps = 0.05
zr = (0.5050, 3.1050)          # ifc_export が使う実効的な室高（床上面〜天井下面）
tests = [
    ("411 を向く面 (x=-6.921, n=+x)", np.array([-6.921, 2.0, 1.5]), np.array([1.0, 0, 0])),
    ("410 を向く面 (x=-7.071, n=-x)", np.array([-7.071, 2.0, 1.5]), np.array([-1.0, 0, 0])),
]
for name, p, n in tests:
    pp, pm = p + eps * n, p - eps * n
    print("\n%s" % name)
    for lbl, q in (("p+eps*n", pp), ("p-eps*n", pm)):
        hits = []
        for r in recs:
            if "_cells" not in r:
                continue
            cell = r["_cell_size"]
            k = tuple(np.floor(q[:2] / cell).astype(np.int64).tolist())
            inxy = k in set(map(tuple, r["_cells"].tolist()))
            inz = zr[0] <= q[2] <= zr[1]
            if inxy and inz:
                hits.append(r.get("Name"))
            print("   %-8s space %s: XY内=%s Z内=%s  (cell %s)"
                  % (lbl, r.get("Name"), inxy, inz, k))
        print("   -> %s は室内: %s" % (lbl, bool(hits)))
PY

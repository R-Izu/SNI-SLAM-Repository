"""R39 §5-2 手順 2 — 実データの source（seed 0）のクラスごとの面密度（点数 ÷ 面積）を測る。

面積：SLAM メッシュの三角形の面積を、三角形のラベル（3 頂点の色の多数決。同数なら先頭の頂点）ごとに合計。
点数：config どおり seed 0 で標本化した source（`io_utils.load_source_cloud`、200,000 点）のクラスごとの点数。
対象：廊下の 4 シーン（`m3_cor_a〜d`。R32 以降の 8 条件の source）。構造クラス（wall・floor・ceiling・door・window）。
"""
import copy, json, os, sys
import numpy as np, yaml
REPO = os.path.expanduser("~/rizu/SNI-SLAM"); os.chdir(REPO)
sys.path.insert(0, "Registration")
import open3d as o3d
from regbim import io_utils
from regbim.labels import NAME_TO_ID, color_to_label

CLS = ["wall", "floor", "ceiling", "door", "window"]
out = {}
for scene in ("m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d"):
    cfg = copy.deepcopy(yaml.safe_load(open("Registration/configs/realdata/%s__E2.yaml" % scene)))
    cfg["source"] = dict(cfg["source"], seed=0)
    mesh = o3d.io.read_triangle_mesh(cfg["source"]["mesh_path"])
    v = np.asarray(mesh.vertices); f = np.asarray(mesh.triangles)
    vl = color_to_label(np.asarray(mesh.vertex_colors))
    tl3 = vl[f]
    tl = np.where(tl3[:, 1] == tl3[:, 2], tl3[:, 1], tl3[:, 0])
    area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    src = io_utils.load_source_cloud(cfg)
    row = {"total_area_m2": float(area.sum()), "n_points": int(len(src.points))}
    for c in CLS:
        a = float(area[tl == NAME_TO_ID[c]].sum()); n = int((src.labels == NAME_TO_ID[c]).sum())
        row[c] = {"area_m2": a, "n": n, "density_per_m2": n / a if a > 0 else None}
    out[scene] = row
    print(scene, "総面積 %.1f m2" % row["total_area_m2"], " ".join("%s %.1f/m2" % (c, row[c]["density_per_m2"]) for c in CLS if row[c]["density_per_m2"]))
ds = [out[s][c]["density_per_m2"] for s in out for c in CLS if out[s][c]["density_per_m2"]]
out["median_density_per_m2"] = float(np.median(ds))
out["per_class_median"] = {c: float(np.median([out[s][c]["density_per_m2"] for s in out if isinstance(out[s], dict) and c in out[s] and out[s][c]["density_per_m2"]])) for c in CLS}
print("クラス×シーンの中央値 %.2f 点/m2" % out["median_density_per_m2"], out["per_class_median"])
json.dump(out, open("Registration/output/diag/r39_real_density.json", "w"), indent=1)

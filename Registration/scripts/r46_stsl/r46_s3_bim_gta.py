"""R46 段階3：GT-A（Replica）の BIM 側の線分を、STSL の公式の線分抽出で作り直す。**登録を回す前の修正。**

理由：`r46_s3_inputs.py` が IFC と同じ作り方（法線と位置で面に分ける）で作った Replica の線分は、シーンあたり 1〜6 本しか
取れず、office_1 では部屋を横切る誤った線分が出た（Replica の参照は IFC のような平らな面の集まりではないため）。
作り方（固定）：
1. 参照（`replica_gt`、構造 5 クラス）の wall の点と floor の点を、床の高さ z_floor を 0 にずらして取り出す（sni-slam の環境、`dump`）
2. STSL の公式の `PCD2d(walls, ground, cfg)` → `get_pcd2d()`（地面で水平化、高さで切り、平らにする）→
   `LineSegmentManager(cfg).submap_image_based_extract(pcd2d)`（**source 側と同じ公式の線分抽出**）（libim の環境、`build`）
3. 線分は pcd2d の座標（T_rp で水平化した座標）で出るので、T_rp^{-1} で参照の座標（z_floor を 0 にした座標）に戻し、xy を使う
4. 設定 A・B の config でそれぞれ作り、`BIM/GTA_<scene>_A`・`_B` に置く（register は既にあればそれを使う）。
   `bim.pcd` は壁の点、`bim2d.pcd` はそれを平らにして 0.05 m で間引いたもの（`r46_stsl_bim.py` と同じ）
使い方：python r46_s3_bim_gta.py dump （sni-slam）／ python r46_s3_bim_gta.py build （libim）
"""
import json, os, sys
import numpy as np

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]


def dump():
    import yaml
    REPO = os.path.expanduser("~/rizu/SNI-SLAM"); sys.path.insert(0, os.path.join(REPO, "Registration")); os.chdir(REPO)
    from regbim import io_utils
    from regbim.labels import NAME_TO_ID
    for s in SCENES:
        cfg = yaml.safe_load(open("Registration/configs/gt_a_plan_a/%s.yaml" % s))
        ref = io_utils.load_reference_cloud(cfg)
        zf = float(np.median(ref.points[ref.labels == NAME_TO_ID["floor"], 2]))
        sh = np.array([0.0, 0.0, zf])
        np.savez(os.path.join(ROOT, "BIM", "GTA_" + s, "ref_points.npz"), walls=ref.points[ref.labels == NAME_TO_ID["wall"]] - sh,
                 floor=ref.points[ref.labels == NAME_TO_ID["floor"]] - sh, z_floor=zf)
        print("dumped", s, flush=True)


def build():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import r46_stsl_pipeline as pl                                  # STSL の位置へ chdir する
    import open3d as o3d
    from preprocess.config import load_cfg
    from preprocess.geometry.point_cloud import PCD2d
    from preprocess.datasets.lineseg_manager import LineSegmentManager
    sys.path.insert(0, os.path.expanduser("~/rizu/SNI-SLAM/Registration/scripts/r46_stsl"))
    out = {}
    for st in "AB":
        cfg = load_cfg(pl.write_yaml(pl.SETTINGS[st][1], os.path.join(ROOT, "cfg", "gta_bim_%s.yaml" % st), {"seq": "gta_bim"}))
        for s in SCENES:
            z = np.load(os.path.join(ROOT, "BIM", "GTA_" + s, "ref_points.npz"))
            pc = lambda a: o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(a, dtype=np.float64)))
            p2 = PCD2d(pc(z["walls"]), pc(z["floor"]), cfg)
            ok = p2.get_pcd2d()
            segs = []
            if ok:
                lines = LineSegmentManager(cfg).submap_image_based_extract(p2)
                Ti = np.linalg.inv(np.asarray(p2.T_rp))
                for l in lines.linesegments:
                    a = Ti @ np.r_[l.point_a, 0.0, 1.0]; b = Ti @ np.r_[l.point_b, 0.0, 1.0]
                    segs.append([a[0], a[1], b[0], b[1]])
            segs = np.asarray(segs).reshape(-1, 4)
            d = os.path.join(ROOT, "BIM", "GTA_%s_%s" % (s, st)); os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
            np.savetxt(os.path.join(d, "bimLineSeg.txt"), segs, fmt="%.5f")
            W = z["walls"]
            o3d.io.write_point_cloud(os.path.join(d, "bim.pcd"), pc(W))
            flat = np.c_[W[:, :2], np.zeros(len(W))]
            key = np.unique(np.floor(flat[:, :2] / 0.05).astype(np.int64), axis=0, return_index=True)[1]
            o3d.io.write_point_cloud(os.path.join(d, "bim2d.pcd"), pc(flat[np.sort(key)]))
            meta = {"z_floor": float(z["z_floor"]), "n_segments": int(len(segs)), "line_extraction": "STSL official (submap_image_based_extract)",
                    "setting": st, "pcd2d_ok": bool(ok),
                    "segment_length_total_m": float(np.sum(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))) if len(segs) else 0.0}
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
            # register は BIM/<floor>/meta.json を読むので、元の名前にも同じ z_floor を残す
            json.dump({"z_floor": float(z["z_floor"])}, open(os.path.join(ROOT, "BIM", "GTA_" + s, "meta.json"), "w"))
            out["%s_%s" % (s, st)] = meta
            print(s, st, meta, flush=True)


if __name__ == "__main__":
    {"dump": dump, "build": build}[sys.argv[1]]()

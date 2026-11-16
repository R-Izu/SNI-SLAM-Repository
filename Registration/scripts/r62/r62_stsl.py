"""R62 §3-5 の 2：STSL の入力（b-2 の f 1.0・0.6）と、参照 4 つの BIM の壁の線分を作る。**回す前に commit する。STSL の登録は回さない（前処理だけ）。**

使い方：python r62_stsl.py inputs（sni-slam の環境）→ python r62_stsl.py prep（libim の環境）

inputs
- BIM：R62 の参照（`~/rizu/conslam/refs/<ref>.npz`、天井を付け直した切り出し）から、`r46_stsl_bim.lines_from_walls` と同じ作り方で
  `bimLineSeg.txt`・`bim.pcd`・`bim2d.pcd`・`meta.json` を `$ROOT/BIM/CS_<ref>/` に書く。z_floor は参照の床のクラスの点の z の中央値
  （R46 は室内面の版の床から決めたが、R62 の参照は室内面の印が無いので、切り出した参照の床から決める）
- 入力：b-2 の f 1.0・0.6 の点（`~/rizu/conslam/inputs/b2_f<f>.ply`、R62 で作った 200,000 点）。R55 の条件 A と同じ手順で、
  重力で z を上にする回転 Rg をかけ、摂動 P から作るヨーと水平の並進だけの A_P をかける（`r55_common.cond_A` と同じ式。RG の代わりに Rg）
  - **Rg の決め方が R55 と違う**：R55 は ARKit の重力（SLAM の y 上 → z 上の固定の回転）。ConSLAM の `normalized/` には重力の記録が手元に無い
    （IMU は bag の中）。R48 の合成と同じく、正解 A（`extra.json` の `b_normalized.A`）から求めた上向き u = A⁻¹ の e_z を e_z に移す回転を Rg とした
  - P：R37 の P 系列のうち `m3_cor_a__E2` の k 0〜9（`r49_run.perturbations("real", "m3_cor_a__E2")`）。§3-5 の 1 の config と同じ予定
  - T_true（入力 → IFC）= A · (A_P · Rg)⁻¹
  - 出力：`$ROOT/inputs/r62_b2_f<f>/NNNNNN.npy`・`spec.json`（R55 の `r55_stsl_inputs.py` と同じ形）
prep：`r46_stsl_pipeline.prep(setname, "A", k)` を k = 1〜5 で回す（R55 と同じ設定 A・K = 5）。登録（`register`）は呼ばない
出力のまとめ：`~/rizu/conslam/out/r62_stsl.json`
"""
import json, os, sys
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
ROOT = os.path.expanduser("~/rizu/r46_stsl_data")
OUT = os.path.expanduser("~/rizu/conslam/out")
REFS = ["S1_small", "S1_large", "S3_small", "S3_large"]


def inputs():
    for p in ("Registration", "Registration/scripts", "Registration/scripts/r49", "Registration/scripts/r46_stsl", "Registration/scripts/r55"):
        sys.path.insert(0, os.path.join(REPO, p))
    os.chdir(REPO)
    import open3d as o3d
    from r46_stsl_bim import lines_from_walls, write_pcd
    import r49_run as RR
    from r36_rotation_diag import tilt_yaw
    from regbim import metrics
    res = {"bim": {}, "inputs": {}}
    for ref in REFS:
        z = np.load(os.path.expanduser("~/rizu/conslam/refs/%s.npz" % ref))
        P, L, N = z["points"], z["labels"], z["normals"]
        zf = float(np.median(P[L == 3, 2]))
        w = (L == 1) & (np.abs(N[:, 2]) < 0.3)
        Q = P[w] - np.array([0.0, 0.0, zf]); segs = lines_from_walls(Q, N[w])
        d = os.path.join(ROOT, "BIM", "CS_" + ref); os.makedirs(d, exist_ok=True)
        np.savetxt(os.path.join(d, "bimLineSeg.txt"), segs, fmt="%.5f"); write_pcd(os.path.join(d, "bim.pcd"), Q)
        flat = np.c_[Q[:, :2], np.zeros(len(Q))]
        key = np.unique(np.floor(flat[:, :2] / 0.05).astype(np.int64), axis=0, return_index=True)[1]
        write_pcd(os.path.join(d, "bim2d.pcd"), flat[np.sort(key)])
        meta = {"reference": ref, "z_floor": zf, "n_wall_points": int(len(Q)), "n_segments": int(len(segs)),
                "segment_length_total_m": float(np.sum(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))) if len(segs) else 0.0}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1); res["bim"][ref] = meta
    A = np.array(json.load(open(os.path.join(OUT, "extra.json")))["b_normalized"]["A"])
    u = A[:3, :3].T @ np.array([0.0, 0.0, 1.0]); u /= np.linalg.norm(u); z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z); c = float(u @ z); V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Rg = np.eye(4); Rg[:3, :3] = np.eye(3) + V + V @ V * (1.0 / (1.0 + c))
    res["Rg"] = Rg.tolist(); res["tilt_of_normalized_deg"] = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    Ps = RR.perturbations("real", "m3_cor_a__E2")
    for f in ("1p0", "0p6"):
        pts = np.asarray(o3d.io.read_point_cloud(os.path.expanduser("~/rizu/conslam/inputs/b2_f%s.ply" % f)).points)
        out = os.path.join(ROOT, "inputs", "r62_b2_f%s" % f); os.makedirs(out, exist_ok=True)
        spec = {"input": "b2_f" + f, "n_points": int(len(pts)), "Rg_source": "A の上向き（GT 由来）", "trials": []}
        for k, P in enumerate(Ps):
            Pz = Rg @ np.asarray(P) @ Rg.T
            R, t, s = metrics.decompose_sim3(Pz); psi = tilt_yaw(R, np.eye(3))["yaw_deg"]
            cc, sn = np.cos(np.radians(psi)), np.sin(np.radians(psi))
            AP = np.eye(4); AP[:3, :3] = [[cc, -sn, 0], [sn, cc, 0], [0, 0, 1]]; AP[:2, 3] = t[:2]
            PA = AP @ Rg
            np.save(os.path.join(out, "%06d.npy" % k), (pts @ PA[:3, :3].T + PA[:3, 3]).astype(np.float32))
            spec["trials"].append({"index": k, "yaw_deg": float(psi), "P": np.asarray(P).tolist(), "A": AP.tolist(), "Rg": Rg.tolist(),
                                   "P_A": PA.tolist(), "T_true": (A @ np.linalg.inv(PA)).tolist()})
        json.dump(spec, open(os.path.join(out, "spec.json"), "w"), indent=1)
        res["inputs"]["r62_b2_f" + f] = {"n_points": int(len(pts)), "n_trials": len(Ps)}
    json.dump(res, open(os.path.join(OUT, "r62_stsl.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:2500])


def prep():
    sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r46_stsl"))
    os.environ["R46_STSL_ROOT"] = ROOT
    import time
    import r46_stsl_pipeline as pl
    res = json.load(open(os.path.join(OUT, "r62_stsl.json"))); res["prep"] = {}
    for f in ("1p0", "0p6"):
        name = "r62_b2_f" + f
        for k in range(1, 6):
            t0 = time.time()
            try:
                seq, seq_dir, T_trues, trp = pl.prep(name, "A", k)
                pt = json.load(open(os.path.join(ROOT, "results", name, "A_k%d_prep_time.json" % k)))
                res["prep"]["%s|k%d" % (name, k)] = {"ok": True, "s": time.time() - t0, "prep_failed": pt.get("prep_failed", []), "n_inputs": pt["n_inputs"]}
            except Exception as e:
                res["prep"]["%s|k%d" % (name, k)] = {"ok": False, "error": repr(e)[:800]}
            print(name, k, res["prep"]["%s|k%d" % (name, k)], flush=True)
    json.dump(res, open(os.path.join(OUT, "r62_stsl.json"), "w"), indent=1)


if __name__ == "__main__":
    {"inputs": inputs, "prep": prep}[sys.argv[1]]()

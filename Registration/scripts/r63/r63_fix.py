"""R63 §8-2：入力・参照の作り方の直しと、回す前の確認（sni-slam の環境、ブランチ `exp/r63`）。**回す前に commit する。位置合わせは回さない。**
R62 の物（`inputs/`・`refs/`・`r62_omega.npz`・`configs/conslam/`）は上書きせず、別の名前（`*_fix`・`r63fix_*`）で作る。

使い方：python r63_fix.py refs → inputs → omegas → configs → check

refs（§8-2 の 1）：R62 の `r62_build.refs` と同じ切り出し・付け直しに、1 つだけ足す
- 天井の帯 z ∈ [13.1, 14.1) で、**床のクラス（スラブ）の上を向いた点（法線の z > 0.7）を外す**
  （R62 と同じく、天井の帯の床のクラスで法線の z < −0.7 の点は天井にする。床の帯 z ∈ [9.7, 10.7) の床のクラスで法線の z < −0.7 の点は外す）
- 天井の帯の、床以外のクラスで上を向いた点（壁の上端など）は外さない（数だけ記録）
- 出力：`~/rizu/conslam/refs_fix/<参照>.npz`
inputs（§8-2 の 2）：R62 の `r62_build.build` と同じ走査・ラベル・つなぎ方・標本化（`default_rng(0).choice(全点, 200000)`）
- **標本化の前のつないだ点群（f ごと）で法線を推定する**：Open3D の `estimate_normals(KDTreeSearchParamHybrid(radius, max_nn))`、
  radius・max_nn は R62 の config の `preprocess.normal_radius`・`normal_max_nn`（本手法の前処理と同じ関数・同じ値）
- 標本化した点に法線を付けて PLY に書く（色はクラスのパレット）。点とラベルは R62 の入力と同じか確かめる
- 記録：つないだ点群で、法線がちょうど (0, 0, 1) の点の割合、半径 radius の中の点の数（自分を含む）が 3 未満の点の割合。標本化した点でも同じ
- 出力：`~/rizu/conslam/inputs_fix/<b2|b1>_f<f>.ply`、`~/rizu/conslam/out/r63fix_inputs.json`
omegas（§8-2 の 3）：R62 の `r62_build.omegas` と同じ作り方で、直した入力・参照から。出力 `~/rizu/conslam/out/r63fix_omega.npz`・`.json`。R62 の Ω との差も記録
configs：R62 の config と同じ（`r62_build.cfg_for`）で、source と reference の cache_path だけを `*_fix` に替える。出力 `Registration/configs/conslam_fix/`
check（§8-2 の確認）：
- (1) 参照の床・天井のラベルの点の高さ（`prepare` の後、0.1 m の区間の数の多い順に上位 3 区間。condA 報告 §4-2 と同じ表）
- (2) 入力 b-2 f1p0 を `prepare` した点：法線と正解の上向き（A⁻¹ の e_z）の角 min(θ, 180° − θ) のクラスごとの分布（diag 報告 §4-1 と同じ表）、
  法線がちょうど (0, 0, 1) の点の割合、法線の長さ（体素の平均の後）の分布
- (3) 本手法の上の向きの推定：`rotation.estimate_gravity_axis` を、`prepare(transform_cloud(src, P_k), cfg)`（k 0〜9）と `prepare(dst, cfg)` に当て、
  正解の上向きとの角（diag 報告 §2 と同じ。source の正解は (A P) の回転 R の R⁻¹ e_z）
出力：`~/rizu/conslam/out/r63fix_check.json`
"""
import copy, json, math, os, sys
import numpy as np
from scipy.spatial import cKDTree

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49", "Registration/scripts/r62"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
import r62_build as B                                          # noqa: E402

OUT = B.OUT
INP = os.path.expanduser("~/rizu/conslam/inputs_fix")
REF = os.path.expanduser("~/rizu/conslam/refs_fix")
CFGD = "Registration/configs/conslam_fix"
FS = B.FS; BOXES = B.BOXES; REFS = list(BOXES)
EZ = np.array([0.0, 0.0, 1.0])


def cfg_for(inp, f, ref):
    cfg = B.cfg_for(inp, f, ref)
    cfg["source"]["mesh_path"] = os.path.join(INP, "%s_f%s.ply" % (inp, f))
    cfg["reference"]["cache_path"] = os.path.join(REF, "%s.npz" % ref)
    return cfg


def refs():
    z = np.load(os.path.join(OUT, "conslam_ifc_dense5m_n.npz")); P = z["points"].astype(np.float64); L = z["labels"].copy(); N = z["normals"].astype(np.float64)
    down = (L == 3) & (N[:, 2] < -0.7); up = (L == 3) & (N[:, 2] > 0.7)
    band_c = (P[:, 2] >= 13.1) & (P[:, 2] < 14.1); band_f = (P[:, 2] >= 9.7) & (P[:, 2] < 10.7)
    os.makedirs(REF, exist_ok=True); res = {"refs": {}}
    for name, b in BOXES.items():
        box = (P[:, 0] >= b[0]) & (P[:, 0] < b[2]) & (P[:, 1] >= b[1]) & (P[:, 1] < b[3]) & (P[:, 2] >= B.ZLO) & (P[:, 2] < B.ZHI)
        lab = L.copy(); lab[down & band_c] = 5
        m = box & ~(down & band_f)
        n_r62 = int(m.sum())
        rm = m & up & band_c
        m &= ~rm
        other_up = box & band_c & (L != 3) & (N[:, 2] > 0.7)
        meta = {"source": "ifc", "path": B.IFC, "spaces_requested": [], "storeys_requested": [], "keep_classes": ["wall", "floor", "ceiling", "door", "window"],
                "r62_box": list(b), "r62_z": [B.ZLO, B.ZHI], "r63fix_note": "r62_build.refs と同じ切り出し・付け直しに、天井の帯の床のクラスの上向きの点を外した（r63_fix.refs）"}
        path = os.path.join(REF, "%s.npz" % name)
        np.savez_compressed(path, points=P[m], labels=lab[m], normals=N[m], meta=json.dumps(meta))
        res["refs"][name] = {"path": path, "n_points": int(m.sum()), "n_points_r62_rule": n_r62, "n_removed_up_floor_in_ceiling_band": int(rm.sum()),
                             "n_kept_up_other_class_in_ceiling_band": {int(k): int(v) for k, v in zip(*np.unique(L[other_up], return_counts=True))},
                             "by_class": {int(k): int(v) for k, v in zip(*np.unique(lab[m], return_counts=True))}}
        r62 = np.load(os.path.join(B.REF, "%s.npz" % name))
        res["refs"][name]["r62_n_points"] = int(len(r62["points"]))
    json.dump(res, open(os.path.join(OUT, "r63fix_refs.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def frac001(n):
    return float((np.abs(n - EZ) < 1e-12).all(1).mean())


def inputs():
    import open3d as o3d
    import yaml
    from regbim.labels import LABEL_COLORS
    pp = B.cfg_for("b2", "1p0", "S1_small")["preprocess"]; rad, mnn = float(pp["normal_radius"]), int(pp["normal_max_nn"])
    K, dist, M = B.calib(); o = json.load(open(os.path.join(OUT, "r62_orient.json")))
    T_l2c = M if o["chosen"] == "M" else np.linalg.inv(M)
    pcd, b2, b1 = B.scans(); m = json.load(open(os.path.join(B.W, "scan_rgb.json")))["scans"]
    os.makedirs(INP, exist_ok=True); res = {"orientation": o["chosen"], "normal_radius": rad, "normal_max_nn": mnn}
    for name, seq in (("b2", b2), ("b1", b1)):
        per = []
        for s, Tw in seq:
            x = B.read_pcd(pcd[s]); lab, inside = B.labels_for(s, x, T_l2c, K, dist, m)
            per.append((x @ Tw[:3, :3].T + Tw[:3, 3], lab))
        res[name] = {"n_scans": len(seq)}
        for f in FS:
            n = math.ceil(len(per) * float(f.replace("p", ".")))
            P = np.concatenate([p[0] for p in per[:n]]); L = np.concatenate([p[1] for p in per[:n]])
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=mnn)); Nn = np.asarray(pc.normals)
            cnt = cKDTree(P).query_ball_point(P, rad, return_length=True, workers=-1)
            idx = np.sort(np.random.default_rng(0).choice(len(P), min(200000, len(P)), replace=False))
            out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[idx])); out.colors = o3d.utility.Vector3dVector(LABEL_COLORS[L[idx]] / 255.0)
            out.normals = o3d.utility.Vector3dVector(Nn[idx])
            path = os.path.join(INP, "%s_f%s.ply" % (name, f)); o3d.io.write_point_cloud(path, out)
            old = o3d.io.read_point_cloud(os.path.join(B.INP, "%s_f%s.ply" % (name, f))); new = o3d.io.read_point_cloud(path)
            from regbim.labels import color_to_label
            e = {"n_scans": n, "n_points_concat": int(len(P)), "n_sampled": int(len(idx)), "path": path,
                 "concat_frac_normal_001": frac001(Nn), "concat_frac_nbr_lt3": float((cnt < 3).mean()), "concat_frac_nbr_le1": float((cnt <= 1).mean()),
                 "concat_frac_001_among_nbr_lt3": float(frac001(Nn[cnt < 3])) if (cnt < 3).any() else None,
                 "concat_frac_001_among_nbr_ge3": float(frac001(Nn[cnt >= 3])),
                 "concat_nbr_count": {"median": float(np.median(cnt)), "p5": float(np.percentile(cnt, 5))},
                 "sampled_frac_normal_001": frac001(Nn[idx]), "sampled_frac_nbr_lt3_in_concat": float((cnt[idx] < 3).mean()),
                 "same_points_as_r62": float(np.abs(np.asarray(old.points) - np.asarray(new.points)).max()),
                 "same_labels_as_r62": bool((color_to_label(np.asarray(old.colors)) == color_to_label(np.asarray(new.colors))).all()),
                 "ply_has_normals": bool(new.has_normals())}
            res[name]["f" + f] = e
            print(name, f, json.dumps(e), flush=True)
    json.dump(res, open(os.path.join(OUT, "r63fix_inputs.json"), "w"), indent=1, default=float)


def omegas():
    from regbim import io_utils
    from criterion_verdict import build_omega
    A = np.array(json.load(open(os.path.join(OUT, "extra.json")))["b_normalized"]["A"])
    om_b1 = build_omega(cfg_for("b1", "1p0", "S1_small"), seed=0)
    om_b2 = om_b1 @ np.linalg.inv(A)[:3, :3].T + np.linalg.inv(A)[:3, 3]
    res = {"A": A.tolist(), "n_omega": int(len(om_b1)), "ref_load": {}, "omega_bim": {}}
    out = {"omega_b1": om_b1, "omega_b2": om_b2}
    for ref in BOXES:
        c = cfg_for("b1", "1p0", ref); dst = io_utils.load_reference_cloud(c)
        res["ref_load"][ref] = {"n_points": int(len(dst.points)), "by_class": {int(k): int(v) for k, v in zip(*np.unique(dst.labels, return_counts=True))},
                                "has_normals": dst.normals is not None}
        d = cKDTree(dst.points[:, :2]).query(om_b1[:, :2], k=1)[0]
        out["omega_bim_" + ref] = om_b1[d <= 0.3]; res["omega_bim"][ref] = int((d <= 0.3).sum())
    for s in ("S1", "S3"):
        out["omega_small_%s" % s] = out["omega_bim_%s_small" % s]; res["omega_bim"]["omega_small_%s" % s] = int(len(out["omega_small_%s" % s]))
    old = np.load(os.path.join(OUT, "r62_omega.npz"))
    res["vs_r62"] = {k: {"n_r62": int(len(old[k])), "n_fix": int(len(out[k])),
                         "max_abs_diff": (float(np.abs(old[k] - out[k]).max()) if old[k].shape == out[k].shape else None)} for k in out}
    np.savez_compressed(os.path.join(OUT, "r63fix_omega.npz"), **out)
    json.dump(res, open(os.path.join(OUT, "r63fix_omega.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def configs():
    import yaml
    os.makedirs(CFGD, exist_ok=True); n = 0
    for inp in ("b2", "b1"):
        for f in FS:
            for ref in BOXES:
                yaml.safe_dump(cfg_for(inp, f, ref), open(os.path.join(CFGD, "%s_f%s__%s.yaml" % (inp, f, ref)), "w"), sort_keys=False, allow_unicode=True)
                n += 1
    print("configs", n)


def ang(u, v):
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def q(v):
    v = np.asarray(v, dtype=float)
    return {"n": int(len(v)), "median": float(np.median(v)), "p5": float(np.percentile(v, 5)), "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)), "p95": float(np.percentile(v, 95))}


def peaks(z):
    b = np.floor(np.asarray(z) / 0.1).astype(int); u, c = np.unique(b, return_counts=True); o = np.argsort(-c)[:3]
    return [{"z_lo": float(u[i] * 0.1), "n": int(c[i]), "frac": float(c[i] / len(z))} for i in o]


def check():
    from regbim import io_utils, metrics, preprocess, rotation
    from regbim.labels import NAME_TO_ID
    import r49_run as RR
    A = np.array(json.load(open(os.path.join(OUT, "extra.json")))["b_normalized"]["A"])
    PS = [np.asarray(P, dtype=np.float64) for P in RR.perturbations("real", "m3_cor_a__E2")]
    res = {"heights": {}, "normals": {}, "gravity": {}}
    cfg = cfg_for("b2", "1p0", "S1_small"); src = io_utils.load_source_cloud(cfg)
    sp = preprocess.prepare(io_utils.load_source_cloud(cfg), cfg)
    res["heights"]["input_b2_f1p0_under_A"] = {n: peaks((sp.points @ A[:3, :3].T + A[:3, 3])[:, 2][sp.labels == NAME_TO_ID[n]]) for n in ("floor", "ceiling")}
    up_true = A[:3, :3].T @ EZ; up_true /= np.linalg.norm(up_true)
    nn = np.linalg.norm(sp.normals, axis=1); c = {}
    for name in ("floor", "ceiling", "wall"):
        mm = sp.labels == NAME_TO_ID[name]
        t = np.degrees(np.arccos(np.clip(np.abs(sp.normals[mm] @ up_true) / np.maximum(nn[mm], 1e-12), 0, 1)))
        c[name] = dict(q(t), frac_within_10deg=float((t < 10).mean()), frac_over_80deg=float((t > 80).mean()), frac_normal_001=frac001(sp.normals[mm]))
    c["all"] = {"n": int(len(sp.points)), "frac_normal_001": frac001(sp.normals), "normal_norm": q(nn), "frac_norm_lt_0p5": float((nn < 0.5).mean())}
    c["before_prepare"] = {"n": int(len(src.points)), "frac_normal_001": frac001(src.normals)}
    res["normals"]["input_b2_f1p0"] = c
    for ref in REFS:
        cfg = cfg_for("b2", "1p0", ref); dst = io_utils.load_reference_cloud(cfg); dp = preprocess.prepare(dst, cfg)
        res["heights"]["ref_" + ref] = {n: peaks(dp.points[dp.labels == NAME_TO_ID[n]][:, 2]) for n in ("floor", "ceiling")}
        up_ref = rotation.estimate_gravity_axis(dp, cfg); g = {"ref_angle_deg": ang(up_ref, EZ), "src_angle_deg": []}
        for k in range(10):
            P = PS[k]; spk = preprocess.prepare(metrics.transform_cloud(src, P), cfg)
            R, _, _ = metrics.decompose_sim3(A @ P)
            g["src_angle_deg"].append(ang(rotation.estimate_gravity_axis(spk, cfg), R.T @ EZ))
        res["gravity"][ref] = g
        print(ref, "ref %.3f" % g["ref_angle_deg"], "src", [round(x, 1) for x in g["src_angle_deg"]], flush=True)
    json.dump(res, open(os.path.join(OUT, "r63fix_check.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("heights", "normals")}, indent=1))


if __name__ == "__main__":
    {"refs": refs, "inputs": inputs, "omegas": omegas, "configs": configs, "check": check}[sys.argv[1]]()

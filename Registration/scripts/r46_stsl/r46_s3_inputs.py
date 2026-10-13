"""R46 §4-1・§10-3 段階3（未知縮尺）の入力と BIM を作る（sni-slam の環境）。**回す前に commit する。**

対象（合成の縮尺である。実データは実寸なので、未知縮尺が実運用で要るかは別問題）
- GT-A：8 シーン × R45 と同じ摂動の先頭 20 試行（`output/Registration/r45/plan_a/<scene>/trial_matrices.json` の trial 0〜19）
- BIM 由来の対照：R39 §5 の E2（参照 411）・E3（410＋411）× R39 と同じ生成器の先頭 20 試行
  （`output/Registration/r39_bim/E{2,3}_plan_correlate/trial_matrices.json` の trial 0〜19）。予算のため 100 ではなく 20
STSL への入力（§8-4 と同じ考え方）
- 既知の重力で z を上にする回転 Rg（正解 G の回転から、source の座標での上向き u = R_G^T e_z を求め、u を e_z に移す最小の回転）。
  実データで ARKit の重力を与えたのと同じ扱い
- 摂動 P を Rg で水平化した Rg P Rg^T から、z まわりの twist ψ、水平の並進、**縮尺 s_P** を取り、A = Sim3(Rz(ψ), (t_x, t_y, 0), s_P)
  （未知縮尺を調べるので、P の縮尺は残す。GT-A・対照の source はもともと縮尺 1/1.135 を含む）
- 入力 q = A Rg p。正解 T_true（入力 → 参照）= G (A Rg)^{-1}
BIM（STSL の参照）
- GT-A：Replica の参照（`replica_gt`、構造 5 クラス）の壁の点から、`r46_stsl_bim.py` と同じ作り方で線分。床の高さは参照の floor の中央値
- 対照：B1 の両面の BIM（`M3_411_both`・`M3_410_411_both`）を使う（B1 で大きな失敗が最も少なかった変種。結果を見た後の選択で、STSL に有利）
出力：`$ROOT/inputs_s3/<set>/NNNNNN.npy`・`spec.json`、`$ROOT/BIM/GTA_<scene>/`
"""
import json, os, sys
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r46_stsl"))
os.chdir(REPO)
from regbim import io_utils, metrics                          # noqa: E402
from regbim.labels import NAME_TO_ID                          # noqa: E402
from r36_rotation_diag import tilt_yaw                        # noqa: E402
from r46_stsl_bim import lines_from_walls, write_pcd          # noqa: E402

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
N_TRIALS = 20


def rot_to_z(u):
    u = u / np.linalg.norm(u); z = np.array([0.0, 0.0, 1.0]); v = np.cross(u, z); c = float(u @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + V + V @ V * (1.0 / (1.0 + c))


def make_set(name, P0, G, trials):
    RG_ = metrics.decompose_sim3(G)[0]
    Rg = np.eye(4); Rg[:3, :3] = rot_to_z(RG_.T @ np.array([0.0, 0.0, 1.0]))
    out = os.path.join(ROOT, "inputs_s3", name); os.makedirs(out, exist_ok=True)
    spec = {"set": name, "trials": []}
    for t in trials[:N_TRIALS]:
        P = np.asarray(t["P"], dtype=np.float64)
        R, tt, s = metrics.decompose_sim3(Rg @ P @ np.linalg.inv(Rg))
        psi = tilt_yaw(R, np.eye(3))["yaw_deg"]
        c, sn = np.cos(np.radians(psi)), np.sin(np.radians(psi))
        A = metrics.make_sim3(np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]]), np.array([tt[0], tt[1], 0.0]), s)
        M = A @ Rg
        np.save(os.path.join(out, "%06d.npy" % t["trial"]), metrics.apply_sim3(M, P0).astype(np.float32))
        spec["trials"].append({"index": t["trial"], "yaw_deg": float(psi), "t_xy": [float(tt[0]), float(tt[1])], "s_P": float(s),
                               "P": P.tolist(), "A": A.tolist(), "Rg": Rg.tolist(), "T_true": (G @ np.linalg.inv(M)).tolist(),
                               "true_scale_of_input": float(metrics.decompose_sim3(G @ np.linalg.inv(M))[2])})
    json.dump(spec, open(os.path.join(out, "spec.json"), "w"), indent=1)
    sc = [x["true_scale_of_input"] for x in spec["trials"]]
    print(name, "inputs", len(spec["trials"]), "true scale of input %.3f..%.3f" % (min(sc), max(sc)), flush=True)


def main():
    for scene in SCENES:
        cfg = yaml.safe_load(open("Registration/configs/gt_a_plan_a/%s.yaml" % scene))
        tm = json.load(open("output/Registration/r45/plan_a/%s/trial_matrices.json" % scene))
        G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
        trials = sorted([t for t in tm["trials"] if t["method"] == "proposed" and t["trial"] >= 0], key=lambda t: t["trial"])
        make_set("gta_" + scene, io_utils.load_source_cloud(cfg).points, G, trials)
        ref = io_utils.load_reference_cloud(cfg)
        zf = float(np.median(ref.points[ref.labels == NAME_TO_ID["floor"], 2]))
        w = (ref.labels == NAME_TO_ID["wall"]) & (np.abs(ref.normals[:, 2]) < 0.3)
        Pw = ref.points[w] - np.array([0.0, 0.0, zf])
        segs = lines_from_walls(Pw, ref.normals[w])
        d = os.path.join(ROOT, "BIM", "GTA_" + scene); os.makedirs(d, exist_ok=True)
        np.savetxt(os.path.join(d, "bimLineSeg.txt"), segs, fmt="%.5f")
        write_pcd(os.path.join(d, "bim.pcd"), Pw)
        flat = np.c_[Pw[:, :2], np.zeros(len(Pw))]
        key = np.unique(np.floor(flat[:, :2] / 0.05).astype(np.int64), axis=0, return_index=True)[1]
        write_pcd(os.path.join(d, "bim2d.pcd"), flat[np.sort(key)])
        json.dump({"reference_config": "gt_a_plan_a/%s" % scene, "z_floor": zf, "n_segments": int(len(segs)),
                   "segment_length_total_m": float(np.sum(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])))},
                  open(os.path.join(d, "meta.json"), "w"), indent=1)
        print("  BIM GTA_%s segments %d" % (scene, len(segs)), flush=True)
    P0 = np.load("output/R39_BIM/bim_points_bimframe.npz")
    for e in ("E2", "E3"):
        cfg = yaml.safe_load(open("Registration/configs/r39_bim/bim_%s_plan_correlate.yaml" % e))
        tm = json.load(open("output/Registration/r39_bim/%s_plan_correlate/trial_matrices.json" % e))
        G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
        trials = sorted([t for t in tm["trials"] if t["method"] == "proposed" and t["trial"] >= 0], key=lambda t: t["trial"])
        make_set("bim_" + e, io_utils.load_source_cloud(cfg).points, G, trials)


if __name__ == "__main__":
    main()

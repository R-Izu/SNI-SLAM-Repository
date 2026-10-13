"""R46 §3-1・§8-4：段階1の実データの入力を作る（sni-slam の環境）。**回す前に commit する。**

条件：R41 の cor 8 条件（`m3_cor_a〜d` × E2・E3）。E2 の参照は 411、E3 は 410＋411（R41 と同じ config）。
**E2・E3 がすでに参照の 2 通りに当たる**ので、条件は 8 × 4（切り出しの段）= 32、各 10 試行。

切り出し（§3-1）：source（R37 の P 系列と同じ、slam_mesh の標本化 seed 0、200,000 点）を G1 で BIM 座標に置き、
その条件の参照の床（floor の点）からの**水平距離** ≤ r の点だけを残す。r ∈ {0.5, 3, 8, ∞}（∞ = 切らない）。
**G1（正解）は入力の条件づくりにだけ使う。**

摂動（§3-1・§8-4）：R37 の P 系列の 10 個（`r37_replicate.json`、条件ごと）。
- SNI-SLAM の方式（案A・重心）：source の座標で P をそのままかける（3D の回転と縮尺を含む。R41 と同じ）
- 案A・縮尺固定：P の縮尺を外したもの（benchmark の `gt_scale` と同じ：make_sim3(R, t/s, 1)）
- STSL：ARKit の重力で z を上にする回転 Rg（SLAM の座標は y が上：x→x、y→z、z→−y）の後に、
  Rg P Rg^T の z まわりの twist ψ と水平の並進 (t_x, t_y) だけをかける。入力 q = A Rg p、A = [Rz(ψ) | (t_x, t_y, 0)]。
  縮尺は実寸のまま。正解（STSL 自身の判定用）T_true = G1 (A Rg)^{-1}

出力（$R46_STSL_ROOT、既定 ~/rizu/r46_stsl_data）
- `inputs_sni/<target>_r<r>.npz`：切った source の点・ラベル・法線と、切った点の数
- `inputs/real_<target>_r<r>/NNNNNN.npy`・`spec.json`：STSL の入力
"""
import json, os, sys
import numpy as np
import yaml
from scipy.spatial import cKDTree

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from regbim import io_utils, metrics                          # noqa: E402
from regbim.labels import NAME_TO_ID                          # noqa: E402
from r36_rotation_diag import tilt_yaw                        # noqa: E402

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
TARGETS = ["m3_cor_%s__%s" % (s, e) for s in "abcd" for e in ("E2", "E3")]
CUTS = [0.5, 3.0, 8.0, None]
RG = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])   # y 上 → z 上


def cut_name(r):
    return "inf" if r is None else ("%g" % r).replace(".", "p")


def main():
    spread = json.load(open("Registration/output/diag/criterion_spread.json"))["scenes"]
    r37 = {}
    for row in json.load(open(os.path.expanduser("~/rizu/SNI-SLAM/Registration/output/diag/r37_replicate.json")))["rows"]:
        if row["series"] == "P":
            r37[(row["target"], row["k"])] = np.asarray(row["P"], dtype=np.float64)
    os.makedirs(os.path.join(ROOT, "inputs_sni"), exist_ok=True)
    summary = {}
    for target in TARGETS:
        scene = target.split("__")[0]
        cfg = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
        c0 = dict(cfg, source=dict(cfg["source"], seed=0))
        src = io_utils.load_source_cloud(c0)
        ref = io_utils.load_reference_cloud(cfg)
        G1 = np.asarray([r["T"] for r in spread[scene]["results"] if int(r["id"]) == 1][0], dtype=np.float64)
        fl = ref.points[ref.labels == NAME_TO_ID["floor"]][:, :2]
        d, _ = cKDTree(fl).query(metrics.apply_sim3(G1, src.points)[:, :2])
        Ps = [r37[(target, k)] for k in range(10)]
        for r in CUTS:
            m = np.ones(len(d), bool) if r is None else (d <= r)
            name = "%s_r%s" % (target, cut_name(r))
            np.savez_compressed(os.path.join(ROOT, "inputs_sni", name + ".npz"), points=src.points[m], labels=src.labels[m],
                                normals=src.normals[m] if src.normals is not None else np.zeros((m.sum(), 3)),
                                n_total=len(d), cut_r=-1.0 if r is None else r)
            out = os.path.join(ROOT, "inputs", "real_" + name); os.makedirs(out, exist_ok=True)
            spec = {"target": target, "cut_r": r, "n_points": int(m.sum()), "n_total": int(len(d)), "trials": []}
            pts = src.points[m] @ RG[:3, :3].T
            for k, P in enumerate(Ps):
                Pz = RG @ P @ RG.T
                R, t, s = metrics.decompose_sim3(Pz)
                psi = tilt_yaw(R, np.eye(3))["yaw_deg"]
                c, sn = np.cos(np.radians(psi)), np.sin(np.radians(psi))
                A = np.eye(4); A[:3, :3] = [[c, -sn, 0], [sn, c, 0], [0, 0, 1]]; A[:2, 3] = t[:2]
                np.save(os.path.join(out, "%06d.npy" % k), (pts @ A[:3, :3].T + A[:3, 3]).astype(np.float32))
                spec["trials"].append({"index": k, "yaw_deg": float(psi), "t_xy": [float(t[0]), float(t[1])], "P": P.tolist(),
                                       "A": A.tolist(), "Rg": RG.tolist(), "T_true": (G1 @ np.linalg.inv(A @ RG)).tolist()})
            json.dump(spec, open(os.path.join(out, "spec.json"), "w"), indent=1)
            summary[name] = {"n_points": int(m.sum()), "frac": float(m.mean())}
            print(name, summary[name], flush=True)
    json.dump(summary, open(os.path.join(ROOT, "inputs_sni", "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

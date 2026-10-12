"""R46 §8-1 陽性対照の入力を作る（sni-slam の環境で回す）。

入力：R39 §5 の BIM 由来の source（`output/R39_BIM/bim_points_bimframe.npz`、410＋411 の室内面、BIM 座標、誤差なし）。
摂動：R39 §5 の config（`configs/r39_bim/bim_E3_plan_correlate.yaml`、eval.seed 0）と同じ生成器
（`metrics.random_sim3(np.random.default_rng(seed), eval.perturb)`）の先頭 10 個の P から、**ヨー（z まわりの
twist）と水平の並進だけ**を取る（§8-4 の形）。入力 q = Rz(ψ) p + (t_x, t_y, 0)。
正解 T_true（入力 → BIM 座標）= その逆。点にラベルは付けない（STSL はラベルを使わない）。

出力：`$R46_STSL_ROOT/inputs/pc/NNNNNN.npy`（float32 の点）と `spec.json`
"""
import json, os, sys
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from regbim import metrics                                   # noqa: E402
from r36_rotation_diag import tilt_yaw                       # noqa: E402

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
N_TRIALS = 10


def main():
    cfg = yaml.safe_load(open("Registration/configs/r39_bim/bim_E3_plan_correlate.yaml"))
    P0 = np.load("output/R39_BIM/bim_points_bimframe.npz")["points"].astype(np.float64)
    rng = np.random.default_rng(int(cfg["eval"]["seed"]))
    out = os.path.join(ROOT, "inputs", "pc"); os.makedirs(out, exist_ok=True)
    spec = {"source": "output/R39_BIM/bim_points_bimframe.npz", "perturb_config": "r39_bim/bim_E3_plan_correlate", "trials": []}
    for i in range(N_TRIALS):
        P = metrics.random_sim3(rng, cfg["eval"]["perturb"])
        R, t, s = metrics.decompose_sim3(P)
        psi = tilt_yaw(R, np.eye(3))["yaw_deg"]
        c, sn = np.cos(np.radians(psi)), np.sin(np.radians(psi))
        A = np.eye(4); A[:3, :3] = [[c, -sn, 0], [sn, c, 0], [0, 0, 1]]; A[:2, 3] = t[:2]
        q = P0 @ A[:3, :3].T + A[:3, 3]
        np.save(os.path.join(out, "%06d.npy" % i), q.astype(np.float32))
        spec["trials"].append({"index": i, "yaw_deg": float(psi), "t_xy": [float(t[0]), float(t[1])],
                               "P_full": P.tolist(), "T_true": np.linalg.inv(A).tolist()})
        print(i, "yaw %.1f t (%.2f, %.2f)" % (psi, t[0], t[1]))
    json.dump(spec, open(os.path.join(out, "spec.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

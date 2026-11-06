"""R55：STSL の入力（条件 A＝B、R46 の実データと同じ作り方、切らない）を作る（sni-slam の環境）。**回す前に commit する。**

出力：`~/rizu/r46_stsl_data/inputs/r55_<target>_f<f>/NNNNNN.npy`（q = A RG p、float32）と `spec.json`
（各試行の P・A・Rg・T_true = G1 (A RG)⁻¹。`r46_stsl_pipeline.py` がそのまま読む形）。あわせて §4 S5 の確認を `Registration/output/diag/r55/inputs_check.json` に書く
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r55_common as C                                           # noqa: E402
from regbim import io_utils, metrics                            # noqa: E402
from r41_rescore import real_criteria                           # noqa: E402
from r36_rotation_diag import tilt_yaw                          # noqa: E402


def main():
    check = []
    for target in C.RR.REAL:
        G1 = np.asarray(real_criteria(target.split("__")[0])[1], dtype=np.float64)
        for f in C.FS:
            src = io_utils.load_source_cloud(C.cfg_of(target, f))
            out = os.path.join(C.STSL_ROOT, "inputs", "r55_%s_f%s" % (target, f)); os.makedirs(out, exist_ok=True)
            spec = {"target": target, "f": f, "n_points": int(len(src.points)), "trials": []}
            for k in range(10):
                P = C.P_of(target, k); PA, A, psi = C.cond_A(P)
                np.save(os.path.join(out, "%06d.npy" % k), metrics.apply_sim3(PA, src.points).astype(np.float32))
                spec["trials"].append({"index": k, "yaw_deg": psi, "P": P.tolist(), "A": A.tolist(), "Rg": C.RG.tolist(),
                                       "P_A": PA.tolist(), "T_true": (G1 @ np.linalg.inv(PA)).tolist()})
                R, t, s = metrics.decompose_sim3(PA)
                Rt, _, st = metrics.decompose_sim3(G1 @ np.linalg.inv(PA))
                check.append({"target": target, "f": f, "k": k, "scale_PA": s, "tz_A": float(A[2, 3]),
                              "PA_tilt_after_RG_deg": tilt_yaw(R @ C.RG[:3, :3].T, np.eye(3))["tilt_deg"],
                              "T_true_tilt_deg": tilt_yaw(Rt, np.eye(3))["tilt_deg"], "T_true_scale": st})
            json.dump(spec, open(os.path.join(out, "spec.json"), "w"), indent=1)
            print(target, f, len(src.points), flush=True)
    os.makedirs("Registration/output/diag/r55", exist_ok=True)
    json.dump(check, open("Registration/output/diag/r55/inputs_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""R49 段階0 §1-3：計測を入れる前と後で、R45 の GT-A 8 シーン × 先頭 5 試行の T_est がビット一致するかを確かめる（sni-slam の環境）。

使い方：python r49_bitcheck.py <ラベル> [--instrument] [並列数]
- 案A（`gt_a_plan_a/<scene>.yaml`）、P = R45 の plan_a の trial_matrices の trial 0〜4
- --instrument：`regbim.instrument` の記録を有効にして回す（記録が結果を変えないことを確かめる）
出力：`Registration/output/diag/r49/bitcheck_<ラベル>.json`（T_est と、R45 との一致）
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); os.chdir(REPO)
from regbim import io_utils, metrics                            # noqa: E402
from regbim.methods import get_method                           # noqa: E402

SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
INSTR = "--instrument" in sys.argv


def job(a):
    scene, k = a
    cfg = yaml.safe_load(open("Registration/configs/gt_a_plan_a/%s.yaml" % scene))
    tm = json.load(open("output/Registration/r45/plan_a/%s/trial_matrices.json" % scene))
    row = [t for t in tm["trials"] if t["trial"] == k][0]
    src = io_utils.load_source_cloud(cfg); dst = io_utils.load_reference_cloud(cfg)
    P = np.asarray(row["P"], dtype=np.float64)
    rec = None
    if INSTR:
        from regbim import instrument
        instrument.start()
    T = np.asarray(get_method("proposed").register(metrics.transform_cloud(src, P), dst, cfg), dtype=np.float64)
    if INSTR:
        rec = instrument.stop()
    return {"scene": scene, "trial": k, "T_est": T.tolist(), "bit_equal_r45": bool(np.array_equal(T, np.asarray(row["T_est"], dtype=np.float64))),
            "max_abs_diff_r45": float(np.abs(T - np.asarray(row["T_est"])).max()), "n_icp_calls": (len(rec["icp"]) if rec else None)}


def main():
    label = sys.argv[1]; npar = int([a for a in sys.argv[2:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[2:]) else 8
    with Pool(npar) as p:
        out = p.map(job, [(s, k) for s in SCENES for k in range(5)])
    os.makedirs("Registration/output/diag/r49", exist_ok=True)
    json.dump(out, open("Registration/output/diag/r49/bitcheck_%s.json" % label, "w"), indent=1)
    print(label, "bit_equal_r45", sum(o["bit_equal_r45"] for o in out), "/", len(out), "max diff", max(o["max_abs_diff_r45"] for o in out),
          "icp calls", sorted(set(o["n_icp_calls"] for o in out)) if INSTR else "")


if __name__ == "__main__":
    main()

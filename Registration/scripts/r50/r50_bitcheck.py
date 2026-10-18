"""R50 段階0 §3-3・§3-4：ビット一致の確認（sni-slam の環境、ブランチ exp/r50）。

- off：3 つのスイッチをすべて off（既定）で、GT-A 8 シーン × R45 の trial 0〜4 を回し、R45 の T_est と比べる（スレッドは既定）
- both：`ablation.icp_single_class` と `ablation.score_single_class` を両方 on にして、同じ 40 試行を回し、
  R49 の `single_class`（1 本ずつの実行、`~/rizu/r49_data/results/synth/single/`）の T_est と比べる。
  R49 と同じく、config は `r49_run.config("synth", scene, "single")` と同じもの（診断 record_yaw・record_stages を含む）で、
  スレッドは呼び出し側で R49 と同じに固定する（taskset -c 0-7、OMP・OpenBLAS・MKL = 8）
使い方：python r50_bitcheck.py off|both [並列数]
出力：`Registration/output/diag/r50/bitcheck_<mode>.json`
"""
import copy, json, os, sys
from multiprocessing import Pool
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r49"))
os.chdir(REPO)
from regbim import io_utils, metrics                            # noqa: E402
from regbim.methods import get_method                           # noqa: E402
import r49_run as RR                                            # noqa: E402

MODE = sys.argv[1]


def job(a):
    scene, k = a
    if MODE == "off":
        cfg = RR.config("synth", scene, "full"); cfg.pop("diagnostics", None)
        tm = json.load(open("output/Registration/r45/plan_a/%s/trial_matrices.json" % scene))
        row = [t for t in tm["trials"] if t["method"] == "proposed" and t["trial"] == k][0]
        ref_T, P = np.asarray(row["T_est"]), np.asarray(row["P"])
    else:
        cfg = copy.deepcopy(RR.config("synth", scene, "single"))
        cfg["ablation"] = {"icp_single_class": True, "score_single_class": True}
        r49 = json.load(open(RR.path("synth", "single", scene)))["rows"][str(k)]
        ref_T, P = np.asarray(r49["T_est"]), np.asarray(r49["P"])
    src = io_utils.load_source_cloud(cfg); dst = io_utils.load_reference_cloud(cfg)
    T = np.asarray(get_method("proposed").register(metrics.transform_cloud(src, P), dst, cfg), dtype=np.float64)
    return {"scene": scene, "trial": k, "bit_equal": bool(np.array_equal(T, ref_T)), "max_abs_diff": float(np.abs(T - ref_T).max())}


def main():
    npar = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    with Pool(npar) as p:
        out = p.map(job, [(s, k) for s in RR.SYNTH for k in range(5)])
    os.makedirs("Registration/output/diag/r50", exist_ok=True)
    env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    env["affinity"] = len(os.sched_getaffinity(0))
    json.dump({"mode": MODE, "env": env, "rows": out}, open("Registration/output/diag/r50/bitcheck_%s.json" % MODE, "w"), indent=1)
    print(MODE, "bit_equal", sum(o["bit_equal"] for o in out), "/", len(out), "max diff", max(o["max_abs_diff"] for o in out), env)


if __name__ == "__main__":
    main()

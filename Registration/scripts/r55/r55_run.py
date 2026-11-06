"""R55：大域の段を回す（sni-slam の環境、ブランチ `exp/r55`）。**回す前に commit する。** 指示書 §3-2・§3-3。

回すもの（方式 × 条件）。STSL は `r46_stsl_pipeline.py`（libim の環境）で別に回す。本手法の条件 B は R49・R52 の記録を使う（回さない）
- prop|A：`proposed_fixed_scale`、`translation_init: plan_correlate`、`plan_correlate: {scale_lo: 1, scale_hi: 1, scale_n: 1}`
  （R46・R47 の「案A 縮尺固定」と同じ。`r46_real_sni.make(…, "planA_fixed")`）。入力は P_A をかけた source
- fpfh|A・fgr|A：`baseline_ransac_p2l`・`baseline_fgr_p2l`、入力は P_A をかけた source
- fpfh|B・fgr|B：同じ方式、入力は R49 と同じく縮尺を外した P（`strip_scale`）をかけた source
既定のパラメータは変えない（config は R49 の完全と同じ。§4 S2）
出力：`~/rizu/r55_data/global/<method>_<cond>/<target>_f<f>.json`（"k" → T_est・P_in・time_s・diag）。途中から再開できる
使い方：python r55_run.py [並列数]
"""
import copy, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r55_common as C                                           # noqa: E402
from regbim import io_utils, metrics                            # noqa: E402
from regbim.methods import get_method                           # noqa: E402
from failure_decomposition import provenance                    # noqa: E402

RUNS = [("prop", "A"), ("fpfh", "A"), ("fgr", "A"), ("fpfh", "B"), ("fgr", "B")]
METH = {"prop": "proposed_fixed_scale", "fpfh": "baseline_ransac_p2l", "fgr": "baseline_fgr_p2l"}


def config(m, target, f):
    cfg = C.cfg_of(target, f)
    if m == "prop":
        p = cfg.setdefault("proposed", {})
        p["translation_init"] = "plan_correlate"; p["plan_correlate"] = {"scale_lo": 1.0, "scale_hi": 1.0, "scale_n": 1}
    return cfg


def P_in(cond, P):
    return C.cond_A(P)[0] if cond == "A" else C.strip_scale(P)


def path(m, cond, target, f):
    return os.path.join(C.OUT, "global", "%s_%s" % (m, cond), "%s_f%s.json" % (target, f))


_CACHE = {}


def job(a):
    m, cond, target, f, k = a
    if (target, f) not in _CACHE:
        cfg = C.cfg_of(target, f)
        _CACHE.clear(); _CACHE[(target, f)] = (io_utils.load_source_cloud(cfg), io_utils.load_reference_cloud(cfg))
    src, dst = _CACHE[(target, f)]
    Pi = P_in(cond, C.P_of(target, k))
    meth = get_method(METH[m]); cfg = config(m, target, f)
    t0 = time.perf_counter()
    T = meth.register(metrics.transform_cloud(src, Pi), dst, cfg)
    dt = time.perf_counter() - t0
    diag = None
    if m == "prop":
        pd, yd = meth.last_plan_diag, meth.last_yaw_diag
        diag = {"gen": [c["T_gen"] for c in pd["stage_candidates"]], "short": [c["T_short"] for c in pd["stage_candidates"]],
                "after_icp": yd["candidate_T"], "scores": yd["candidate_scores"]}
    return a, {"T_est": np.asarray(T, dtype=np.float64).tolist(), "P_in": Pi.tolist(), "time_s": dt, "diag": diag}


def main():
    npar = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    have, todo = {}, []
    for m, cond in RUNS:
        for target, f, k in C.trials():
            key = (m, cond, target, f)
            if key not in have:
                p = path(*key)
                have[key] = json.load(open(p)) if os.path.exists(p) else {"provenance": provenance(), "rows": {}}
            if str(k) not in have[key]["rows"]:
                todo.append((m, cond, target, f, k))
    todo.sort(key=lambda a: (a[2], a[3], a[0], a[1], a[4]))
    print("todo", len(todo), flush=True)
    t0 = time.time()
    with Pool(npar) as pool:
        for i, (a, r) in enumerate(pool.imap_unordered(job, todo, chunksize=1)):
            key = a[:4]; have[key]["rows"][str(a[4])] = r
            os.makedirs(os.path.dirname(path(*key)), exist_ok=True)
            json.dump(have[key], open(path(*key), "w"))
            print("done", i + 1, len(todo), a, "%.0fs" % r["time_s"], "elapsed %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

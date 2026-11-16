"""R63：ConSLAM で本手法を回す（sni-slam の環境、ブランチ `exp/r63`）。**回す前に commit する。** 指示書 §3-1。

- config：R62 の `Registration/configs/conslam/<b2|b1>_f<f>__<参照>.yaml`（R49 の完全な方式の写しで、source と reference だけを替えたもの）。パラメータは変えない（S2）
- 方式：full（そのまま）、score_only（`ablation: {score_single_class: true}`）、icp_only（`ablation: {icp_single_class: true}`）。R50 と同じスイッチ
- 摂動：`r49_run.perturbations("real", "m3_cor_a__E2")` の k 0〜9（全条件で共通、R62 の読み）。source の座標で `transform_cloud(src, P)`
- 記録：R49 と同じ `diag`（gen・short・after_icp・scores）と T_est・P・time_s。実行の時点で `per_trial(T_est, P, G, Ω)` の e_Ω も書く（§3-2 の確認用）。
  G は b-2 で A（`extra.json` の `b_normalized.A`）、b-1 で単位行列。Ω は `r62_omega.npz` の `omega_b2`・`omega_b1`
- 順番：段階 s5（full・b-2・f 1p0 の 40 試行）→ 段階 rest（ほか 520 試行）。s5 の後に、基準内（e_R < 45°・e_Ω < 1.0 m）が 0/40 なら止める（§4 S5）
出力：`~/rizu/r63_data/results/<方式>/<入力>_f<f>__<参照>.json`（"k" → 記録）。途中から再開できる
使い方：python r63_run.py s5 [並列数] ／ python r63_run.py rest [並列数]
"""
import copy, json, os, sys, time
from multiprocessing import Pool
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from regbim import io_utils, metrics                           # noqa: E402
from regbim.methods import get_method                          # noqa: E402
from failure_decomposition import provenance                   # noqa: E402
from r41_rescore import per_trial                              # noqa: E402
import r49_run as RR                                           # noqa: E402

OUT = os.path.expanduser("~/rizu/r63_data/results")
CO = os.path.expanduser("~/rizu/conslam/out")
REFS = ["S1_small", "S1_large", "S3_small", "S3_large"]
FS = ["0p2", "0p4", "0p6", "0p8", "1p0"]
PS = [np.asarray(P, dtype=np.float64) for P in RR.perturbations("real", "m3_cor_a__E2")]
A = np.array(json.load(open(os.path.join(CO, "extra.json")))["b_normalized"]["A"])
OM = np.load(os.path.join(CO, "r62_omega.npz"))


def jobs(stage):
    if stage == "s5":
        return [("full", "b2", "1p0", r, k) for r in REFS for k in range(10)]
    js = [("full", i, f, r, k) for i in ("b2", "b1") for f in FS for r in REFS for k in range(10) if not (i == "b2" and f == "1p0")]
    js += [(m, "b2", f, r, k) for m in ("score_only", "icp_only") for f in ("1p0", "0p6") for r in REFS for k in range(10)]
    return js


def config(m, inp, f, ref):
    cfg = yaml.safe_load(open("Registration/configs/conslam/%s_f%s__%s.yaml" % (inp, f, ref)))
    if m == "score_only":
        cfg["ablation"] = {"score_single_class": True}
    elif m == "icp_only":
        cfg["ablation"] = {"icp_single_class": True}
    return cfg


def path(m, inp, f, ref):
    return os.path.join(OUT, m, "%s_f%s__%s.json" % (inp, f, ref))


_C = {}


def job(a):
    m, inp, f, ref, k = a
    key = (inp, f, ref)
    if key not in _C:
        cfg = config("full", inp, f, ref)
        _C.clear(); _C[key] = (io_utils.load_source_cloud(cfg), io_utils.load_reference_cloud(cfg))
    src, dst = _C[key]
    cfg = config(m, inp, f, ref); P = PS[k]
    meth = get_method("proposed")
    t0 = time.perf_counter()
    T = meth.register(metrics.transform_cloud(src, P), dst, cfg)
    dt = time.perf_counter() - t0
    pd, yd = meth.last_plan_diag, meth.last_yaw_diag
    diag = {"gen": [c["T_gen"] for c in pd["stage_candidates"]], "short": [c["T_short"] for c in pd["stage_candidates"]],
            "after_icp": yd["candidate_T"], "scores": yd["candidate_scores"]}
    G = A if inp == "b2" else np.eye(4); om = OM["omega_" + inp]
    e = per_trial(np.asarray(T), P, G, om, None)
    return a, {"T_est": np.asarray(T, dtype=np.float64).tolist(), "P": P.tolist(), "time_s": dt, "diag": diag,
               "run_eO": float(e["eO"]), "run_eR": float(e["eR"]), "run_v": bool(e["v"])}


def main():
    stage = sys.argv[1]; npar = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    have, todo = {}, []
    for a in jobs(stage):
        key = a[:4]
        if key not in have:
            p = path(*key); have[key] = json.load(open(p)) if os.path.exists(p) else {"provenance": provenance(), "rows": {}}
        if str(a[4]) not in have[key]["rows"]:
            todo.append(a)
    todo.sort(key=lambda a: (a[1], a[2], a[3], a[0], a[4]))
    print("todo", len(todo), flush=True)
    t0 = time.time()
    with Pool(npar) as pool:
        for i, (a, r) in enumerate(pool.imap_unordered(job, todo, chunksize=1)):
            key = a[:4]; have[key]["rows"][str(a[4])] = r
            os.makedirs(os.path.dirname(path(*key)), exist_ok=True); json.dump(have[key], open(path(*key), "w"))
            print("done", i + 1, len(todo), a, "%.0fs" % r["time_s"], "eO %.3f eR %.1f" % (r["run_eO"], r["run_eR"]), "elapsed %.0fs" % (time.time() - t0), flush=True)
    if stage == "s5":
        ok = 0
        for r in REFS:
            for row in json.load(open(path("full", "b2", "1p0", r)))["rows"].values():
                ok += bool(row["run_v"] and row["run_eR"] < 45 and row["run_eO"] < 1.0)
        print("S5 within-line", ok, "/ 40", "STOP" if ok == 0 else "continue")


if __name__ == "__main__":
    main()

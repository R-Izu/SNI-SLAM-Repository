"""R52：観測を短く切った入力での案A（sni-slam の環境、ブランチ `exp/r52`）。**回す前に commit する。** 指示書 §3-1・§3-3。

- 方式：案A の完全な方式（`r49_run.config(data, target, "full")` と同じ config。手法のコードは R49 と同じ）
- 記録：R49 の `diag`（gen・short・after_icp・scores）に、短い精緻化の後の 16 候補の採点 `score_short`
  （`methods/proposed.py` の `last_plan_diag["stage_candidates"][i]["score_short"]`）を足す。書き出しだけで、手法は変えない
- bitcheck：GT-A 8 シーン × trial 0〜4 を、R49 の 1 本ずつの実行と同じスレッド（呼び出し側で taskset -c 0-7、OMP・OpenBLAS・MKL = 8）で回し、
  R49 の完全（`~/rizu/r49_data/results/synth/full/`）の T_est とビット一致するかを見る
- run：実データ `m3_cor_a〜d` × f ∈ {0.2, 0.4, 0.6, 0.8} × 参照 E2・E3 × R37 の P 系列 k 0〜9 = 320 試行。
  source は `output/RealData/_TSDF/<scene>/r52_f<f>/mesh/final_mesh_semantic_projected.ply`（`r52_inputs.sh` で作る）を、
  R49 と同じく `slam_mesh` として標本化（200,000 点、seed 0）。参照・P は R49 と同じ。並列（時間は測らない）
出力：`~/rizu/r52_data/results/f<f>/<target>.json`、bitcheck は `Registration/output/diag/r52/bitcheck.json`
使い方：python r52_run.py bitcheck [並列数] ／ python r52_run.py run [並列数]
"""
import copy, json, os, sys, time
from multiprocessing import Pool
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from regbim import io_utils, metrics                           # noqa: E402
from regbim.methods import get_method                          # noqa: E402
from failure_decomposition import provenance                   # noqa: E402
import r49_run as RR                                           # noqa: E402

OUT = os.path.expanduser("~/rizu/r52_data/results")
FRACS = ["0p2", "0p4", "0p6", "0p8"]


def config(target, f):
    cfg = copy.deepcopy(RR.config("real", target, "full"))
    scene = target.split("__")[0]
    cfg["source"] = dict(cfg["source"], mesh_path="output/RealData/_TSDF/%s/r52_f%s/mesh/final_mesh_semantic_projected.ply" % (scene, f))
    return cfg


def register(meth, src, dst, cfg, P):
    T = meth.register(metrics.transform_cloud(src, P), dst, cfg)
    pd, yd = meth.last_plan_diag, meth.last_yaw_diag
    diag = {"gen": [c["T_gen"] for c in pd["stage_candidates"]], "short": [c["T_short"] for c in pd["stage_candidates"]],
            "score_short": [float(c["score_short"]) for c in pd["stage_candidates"]],
            "after_icp": yd["candidate_T"], "scores": yd["candidate_scores"]}
    return np.asarray(T, dtype=np.float64), diag


def job_bit(a):
    scene, k = a
    cfg = RR.config("synth", scene, "full")
    r49 = json.load(open(RR.path("synth", "full", scene)))["rows"][str(k)]
    src = io_utils.load_source_cloud(cfg); dst = io_utils.load_reference_cloud(cfg)
    T, d = register(get_method("proposed"), src, dst, cfg, np.asarray(r49["P"]))
    return {"scene": scene, "trial": k, "bit_equal": bool(np.array_equal(T, np.asarray(r49["T_est"]))),
            "max_abs_diff": float(np.abs(T - np.asarray(r49["T_est"])).max()), "n_score_short": len(d["score_short"])}


_CACHE = {}


def job_run(a):
    f, target, k = a
    key = (f, target)
    if key not in _CACHE:
        cfg = config(target, f)
        _CACHE.clear(); _CACHE[key] = (io_utils.load_source_cloud(cfg), io_utils.load_reference_cloud(cfg))
    src, dst = _CACHE[key]
    P = RR.perturbations("real", target)[k]
    t0 = time.perf_counter()
    T, d = register(get_method("proposed"), src, dst, config(target, f), P)
    return a, {"T_est": T.tolist(), "P": P.tolist(), "time_s": time.perf_counter() - t0, "diag": d}


def main():
    mode = sys.argv[1]; npar = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if mode == "bitcheck":
        with Pool(npar) as p:
            out = p.map(job_bit, [(s, k) for s in RR.SYNTH for k in range(5)])
        env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        env["affinity"] = len(os.sched_getaffinity(0))
        os.makedirs("Registration/output/diag/r52", exist_ok=True)
        json.dump({"env": env, "rows": out}, open("Registration/output/diag/r52/bitcheck.json", "w"), indent=1)
        print("bit_equal", sum(o["bit_equal"] for o in out), "/", len(out), "max diff", max(o["max_abs_diff"] for o in out), env)
        return
    have, todo = {}, []
    for f in FRACS:
        for t in RR.REAL:
            p = os.path.join(OUT, "f" + f, t + ".json")
            have[(f, t)] = json.load(open(p)) if os.path.exists(p) else {"provenance": provenance(), "rows": {}}
            todo += [(f, t, k) for k in range(10) if str(k) not in have[(f, t)]["rows"]]
    print("todo", len(todo), flush=True)
    t0 = time.time()
    with Pool(npar) as pool:
        for i, (a, r) in enumerate(pool.imap_unordered(job_run, todo, chunksize=1)):
            f, t, k = a; have[(f, t)]["rows"][str(k)] = r
            os.makedirs(os.path.join(OUT, "f" + f), exist_ok=True)
            json.dump(have[(f, t)], open(os.path.join(OUT, "f" + f, t + ".json"), "w"))
            print("done", i + 1, len(todo), a, "%.0fs" % r["time_s"], "elapsed %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

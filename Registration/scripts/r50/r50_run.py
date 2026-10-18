"""R50 段階1 A：位置合わせ（sni-slam の環境、ブランチ `exp/r50`）。**回す前に commit する。** 指示書 §4-1。

方式（config は R49 の完全な方式と同じ。`r49_run.config(data, target, "full")` に、下の 1 つだけを足す）
- score_only：`ablation.score_single_class: true`（最終の採点 ρ だけクラスなし）
- icp_only：`ablation.icp_single_class: true`（ICP の対応づけだけクラスなし）
- geo：`reference.geometric_classes: true`（参照のクラスを幾何から作る。source のラベルは変えない）
データ・摂動は R49 と同じ（synth 8 × 100、bim 2 × 100、real 8 × 10）。完全は R49 の 1 本ずつの実行の値を使う（ここでは回さない）
診断（record_yaw・record_stages）は R49 と同じく付け、段の分類のための変換を残す
並列で回す（時間は測らない）。出力：`~/rizu/r50_data/results/<data>/<method>/<target>.json`。途中から再開できる
使い方：python r50_run.py [並列数]
"""
import copy, json, os, sys, time
from multiprocessing import Pool
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r49"))
os.chdir(REPO)
from regbim import io_utils, metrics                           # noqa: E402
from regbim.methods import get_method                          # noqa: E402
from failure_decomposition import provenance                   # noqa: E402
import r49_run as RR                                           # noqa: E402

OUT = os.path.expanduser("~/rizu/r50_data/results")
METHODS = ("score_only", "icp_only", "geo")


def config(data, target, m):
    cfg = copy.deepcopy(RR.config(data, target, "full"))
    if m == "score_only":
        cfg["ablation"] = {"score_single_class": True}
    elif m == "icp_only":
        cfg["ablation"] = {"icp_single_class": True}
    elif m == "geo":
        cfg["reference"] = dict(cfg["reference"], geometric_classes=True)
    return cfg


_CACHE = {}


def clouds(data, target, geo):
    key = (data, target, geo)
    if key not in _CACHE:
        cfg = config(data, target, "geo" if geo else "score_only")
        _CACHE.clear(); _CACHE[key] = (io_utils.load_source_cloud(cfg), io_utils.load_reference_cloud(cfg))
    return _CACHE[key]


def job(a):
    data, target, m, k = a
    src, dst = clouds(data, target, m == "geo")
    cfg = config(data, target, m); meth = get_method("proposed")
    P = RR.perturbations(data, target)[k]
    t0 = time.perf_counter()
    T = meth.register(metrics.transform_cloud(src, P), dst, cfg)
    pd, yd = meth.last_plan_diag, meth.last_yaw_diag
    diag = {"gen": [c["T_gen"] for c in pd["stage_candidates"]], "short": [c["T_short"] for c in pd["stage_candidates"]],
            "after_icp": yd["candidate_T"], "scores": yd["candidate_scores"]}
    return a, {"T_est": np.asarray(T, dtype=np.float64).tolist(), "P": P.tolist(), "time_s": time.perf_counter() - t0, "diag": diag}


def path(data, m, target):
    return os.path.join(OUT, data, m, target + ".json")


def main():
    npar = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    jobs = [(d, t, m, k) for d in ("synth", "real", "bim") for t in RR.targets(d) for m in METHODS for k in range(RR.n_trials(d))]
    have, todo = {}, []
    for a in jobs:
        key = (a[0], a[2], a[1])
        if key not in have:
            p = path(a[0], a[2], a[1])
            have[key] = json.load(open(p)) if os.path.exists(p) else {"provenance": provenance(), "rows": {}}
        if str(a[3]) not in have[key]["rows"]:
            todo.append(a)
    todo.sort(key=lambda a: (a[0], a[1], a[2] == "geo"))       # 同じ参照を続けて、読み込みを使い回す
    print("todo", len(todo), flush=True)
    t0 = time.time()
    with Pool(npar) as pool:
        for i, (a, r) in enumerate(pool.imap_unordered(job, todo, chunksize=1)):
            key = (a[0], a[2], a[1]); have[key]["rows"][str(a[3])] = r
            os.makedirs(os.path.dirname(path(*key)), exist_ok=True)
            json.dump(have[key], open(path(*key), "w"))
            print("done", i + 1, len(todo), a, "%.0fs" % r["time_s"], "elapsed %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

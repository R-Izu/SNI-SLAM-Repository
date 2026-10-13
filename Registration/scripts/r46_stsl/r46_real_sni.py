"""R46 §3-2：SNI-SLAM 側の 3 方式を、切った入力で回す（sni-slam の環境、`fix/translation-init`、R45 と同じ手法のコード）。

方式（config は `configs/realdata/<target>.yaml` の写しで、変えるのは下だけ。差分は報告に書く）
- planA      ：`proposed`、`proposed.translation_init: plan_correlate`（論文の手法、R45 と同じ）
- planA_fixed：`proposed_fixed_scale`（ICP の縮尺を 1 に固定）、`translation_init: plan_correlate`、
                `proposed.plan_correlate: {scale_lo: 1, scale_hi: 1, scale_n: 1}`（§8-6 (ii)）。P の縮尺を外して渡す（実寸）
- centroid   ：`proposed`、`translation_init: centroid`（参考）
入力：`inputs_sni/<target>_r<r>.npz`（切った source）に、R37 の P 系列の P（条件ごと 10）をかけたもの
出力：`$ROOT/results_sni/<target>_r<r>.json`（方式 × 試行の T_est、P、時間）。途中から再開できる
使い方：python r46_real_sni.py [並列数]
"""
import copy, json, os, sys, time
from multiprocessing import Pool
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from regbim import io_utils, metrics                          # noqa: E402
from regbim.labels import LabeledCloud                        # noqa: E402
from regbim.methods import get_method                         # noqa: E402
from failure_decomposition import provenance                  # noqa: E402

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
TARGETS = ["m3_cor_%s__%s" % (s, e) for s in "abcd" for e in ("E2", "E3")]
CUTS = ["0p5", "3", "8", "inf"]
METHODS = ["planA", "planA_fixed", "centroid"]


def make(cfg0, method):
    cfg = copy.deepcopy(cfg0)
    p = cfg.setdefault("proposed", {})
    if method == "planA":
        p["translation_init"] = "plan_correlate"; return "proposed", cfg, False
    if method == "planA_fixed":
        p["translation_init"] = "plan_correlate"; p["plan_correlate"] = {"scale_lo": 1.0, "scale_hi": 1.0, "scale_n": 1}
        return "proposed_fixed_scale", cfg, True
    p["translation_init"] = "centroid"; return "proposed", cfg, False


def job(a):
    target, cut, method, k = a
    cfg0 = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
    mname, cfg, strip_scale = make(cfg0, method)
    z = np.load(os.path.join(ROOT, "inputs_sni", "%s_r%s.npz" % (target, cut)))
    src = LabeledCloud(z["points"], z["labels"], z["normals"])
    dst = io_utils.load_reference_cloud(cfg)
    spec = json.load(open(os.path.join(ROOT, "inputs", "real_%s_r%s" % (target, cut), "spec.json")))
    P = np.asarray(spec["trials"][k]["P"], dtype=np.float64)
    if strip_scale:
        R, t, s = metrics.decompose_sim3(P); P = metrics.make_sim3(R, t / s, 1.0)
    t0 = time.time()
    T = get_method(mname).register(metrics.transform_cloud(src, P), dst, cfg)
    return a, {"T_est": np.asarray(T, dtype=np.float64).tolist(), "P": P.tolist(), "time_s": time.time() - t0}


def main():
    npar = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(os.path.join(ROOT, "results_sni"), exist_ok=True)
    todo = []
    for target in TARGETS:
        for cut in CUTS:
            f = os.path.join(ROOT, "results_sni", "%s_r%s.json" % (target, cut))
            done = json.load(open(f)) if os.path.exists(f) else {"provenance": provenance(), "rows": {}}
            for m in METHODS:
                for k in range(10):
                    if "%s|%d" % (m, k) not in done["rows"]:
                        todo.append((target, cut, m, k))
    print("todo", len(todo), flush=True)
    with Pool(npar) as pool:
        for (target, cut, m, k), r in pool.imap_unordered(job, todo):
            f = os.path.join(ROOT, "results_sni", "%s_r%s.json" % (target, cut))
            done = json.load(open(f)) if os.path.exists(f) else {"provenance": provenance(), "rows": {}}
            done["rows"]["%s|%d" % (m, k)] = r
            json.dump(done, open(f, "w"))
            print("done", target, cut, m, k, "%.1fs" % r["time_s"], flush=True)


if __name__ == "__main__":
    main()

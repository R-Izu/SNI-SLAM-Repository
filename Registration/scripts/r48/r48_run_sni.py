"""R48 段階1：案A と重心（参考）を回す（sni-slam の環境、`fix/translation-init`、R45〜R47 と同じ手法のコード）。**回す前に commit する。**

- 方式の config：案A = `gt_a_plan_a/room_0.yaml`（R45 の GT-A と同じ。手法のパラメータは GT-A の 8 シーンで同じ）、
  重心 = `gt_a_centroid_seeded/room_0.yaml`（R45 と同じ）。参照と source は config からではなく、下の npz から渡す
- 参照：`refs/<ref>.npz`（部分参照、構造 5 クラス、点・ラベル・法線）。source：`inputs_sni/<ref>__<cut>.npz`
- 摂動：`r48_common.perturbations()` の先頭 20（GT-A と同じ列）。入力 = P · source
- GT-A の 8 参照は R45 の結果（`output/Registration/r45/{plan_a,centroid}/<scene>/trial_matrices.json` の trial 0〜19）を再利用する
  （同じ手法のコード・同じ P。P の一致は `r48_gta_dump.py` で確かめた）。ここでは回さない
出力：`~/rizu/r48_data/results_sni/<ref>__<cut>.json`（"方式|k" → T_est・P・時間）。途中から再開できる
使い方：python r48_run_sni.py [並列数]
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r48"))
os.chdir(REPO)
from regbim import metrics                                    # noqa: E402
from regbim.labels import LabeledCloud                        # noqa: E402
from regbim.methods import get_method                         # noqa: E402
from failure_decomposition import provenance                  # noqa: E402
import r48_common as C                                        # noqa: E402

OUT = os.path.join(C.D, "results_sni")
CFG = {"planA": "Registration/configs/gt_a_plan_a/room_0.yaml", "centroid": "Registration/configs/gt_a_centroid_seeded/room_0.yaml"}


def cloud(path):
    z = np.load(path)
    return LabeledCloud(z["points"].astype(np.float64), z["labels"].astype(np.int64), z["normals"].astype(np.float64))


def job(a):
    ref, cut, m, k = a
    cfg = yaml.safe_load(open(CFG[m]))
    src = cloud(os.path.join(C.D, "inputs_sni", "%s__%s.npz" % (ref, cut))); dst = cloud(os.path.join(C.D, "refs", ref + ".npz"))
    P = C.perturbations()[k]
    t0 = time.time()
    T = get_method("proposed").register(metrics.transform_cloud(src, P), dst, cfg)
    return a, {"T_est": np.asarray(T, dtype=np.float64).tolist(), "P": P.tolist(), "time_s": time.time() - t0}


def main():
    npar = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(OUT, exist_ok=True)
    todo = []
    for ref in C.refs_multi():
        for cut in C.CUTS:
            f = os.path.join(OUT, "%s__%s.json" % (ref, cut))
            done = json.load(open(f))["rows"] if os.path.exists(f) else {}
            todo += [(ref, cut, m, k) for k in range(C.N_TRIALS) for m in ("planA", "centroid") if "%s|%d" % (m, k) not in done]
    print("todo", len(todo), flush=True)
    with Pool(npar) as pool:
        for (ref, cut, m, k), r in pool.imap_unordered(job, todo):
            f = os.path.join(OUT, "%s__%s.json" % (ref, cut))
            d = json.load(open(f)) if os.path.exists(f) else {"provenance": provenance(), "rows": {}}
            d["rows"]["%s|%d" % (m, k)] = r
            json.dump(d, open(f, "w"))
            print("done", ref, cut, m, k, "%.0fs" % r["time_s"], flush=True)


if __name__ == "__main__":
    main()

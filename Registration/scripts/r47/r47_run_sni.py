"""R47 の SNI-SLAM 側（sni-slam の環境、`fix/translation-init`、R46 と同じ手法のコード）。**回す前に commit する。**

B-1 の交差の組（16 条件 × 10 試行 × {planA, planA_fixed, centroid}）：入力 = `inputs_sni/<input_set>.npz`、参照 = `realdata/<ref_target>.yaml`、
P = R37 の P 系列（E2・E3 で同じ）。方式の config は `r46_real_sni.make` と同じ。
B-2（8 入力 × 36 向き × planA）：入力 = `inputs_sni/<scene>__E3_r<cut>.npz`、参照 = `realdata/<scene>__E3.yaml`、
P = Rz(ψ) · G1（source → 入力。G1 で BIM 座標に置き、z まわりに ψ 回す。並進 0、R37 の P の縮尺・傾きは加えない）
出力：`$ROOT/results_r47/sni/<名前>.json`（再開できる）
使い方：python r47_run_sni.py [並列数]
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r46_stsl"))
os.chdir(REPO)
from regbim import io_utils, metrics                          # noqa: E402
from regbim.labels import LabeledCloud                        # noqa: E402
from regbim.methods import get_method                         # noqa: E402
from failure_decomposition import provenance                  # noqa: E402
from r46_real_sni import make                                 # noqa: E402
from r47_common import b1_conditions, B2_INPUTS, YAWS          # noqa: E402

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
OUT = os.path.join(ROOT, "results_r47", "sni")


def g1(scene):
    sp = json.load(open("Registration/output/diag/criterion_spread.json"))["scenes"][scene]["results"]
    return np.asarray([r["T"] for r in sp if int(r["id"]) == 1][0], dtype=np.float64)


def job(a):
    kind, name, input_set, ref_target, method, k, P = a
    cfg0 = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % ref_target))
    mname, cfg, strip = make(cfg0, method)
    z = np.load(os.path.join(ROOT, "inputs_sni", input_set + ".npz"))
    src = LabeledCloud(z["points"], z["labels"], z["normals"]); dst = io_utils.load_reference_cloud(cfg)
    P = np.asarray(P, dtype=np.float64)
    if strip:
        R, t, s = metrics.decompose_sim3(P); P = metrics.make_sim3(R, t / s, 1.0)
    t0 = time.time()
    T = get_method(mname).register(metrics.transform_cloud(src, P), dst, cfg)
    return a[:6], {"T_est": np.asarray(T, dtype=np.float64).tolist(), "P": P.tolist(), "time_s": time.time() - t0}


def main():
    npar = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(OUT, exist_ok=True)
    r37 = {(r["target"], r["k"]): r["P"] for r in json.load(open("Registration/output/diag/r37_replicate.json"))["rows"] if r["series"] == "P"}
    jobs = []
    for c in b1_conditions():
        if not c["cross"]:
            continue
        for m in ("planA", "planA_fixed", "centroid"):
            for k in range(10):
                jobs.append(("b1", c["name"], c["input_set"], c["ref_target"], m, k, r37[(c["ref_target"], k)]))
    for s, cut in B2_INPUTS:
        G1 = g1(s)
        for i, y in enumerate(YAWS):
            c, sn = np.cos(np.radians(y)), np.sin(np.radians(y))
            Rz = np.eye(4); Rz[:3, :3] = [[c, -sn, 0], [sn, c, 0], [0, 0, 1]]
            jobs.append(("b2", "%s__in410_411__ref410_411_r%s_rot" % (s, cut), "%s__E3_r%s" % (s, cut), "%s__E3" % s, "planA", i, (Rz @ G1).tolist()))
    done = {}
    for f in os.listdir(OUT):
        done[f[:-5]] = json.load(open(os.path.join(OUT, f)))
    todo = [j for j in jobs if "%s|%d" % (j[4], j[5]) not in done.get(j[1], {}).get("rows", {})]
    print("todo", len(todo), "of", len(jobs), flush=True)
    with Pool(npar) as pool:
        for (kind, name, _, _, m, k), r in pool.imap_unordered(job, todo):
            f = os.path.join(OUT, name + ".json")
            d = json.load(open(f)) if os.path.exists(f) else {"provenance": provenance(), "rows": {}}
            d["rows"]["%s|%d" % (m, k)] = r
            json.dump(d, open(f, "w"))
            print("done", name, m, k, "%.0fs" % r["time_s"], flush=True)


if __name__ == "__main__":
    main()

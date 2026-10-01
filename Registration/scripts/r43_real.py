"""R43（R42 §4-4）実データ：廊下 8 条件 × 標本化 seed 1〜10（R37 の S 系列と同じ）× label_filter on/off。記述のみ。

config は R37 と同じ作り方（`r37_replicate.make_cfg` から回転の解放の項を除いたもの：source seed、translation_init = plan_correlate）。
on は R37 S 系列の T_off（回転の解放の前の解）とビット一致するかを確かめて記録する。
"""
import copy, json, os, sys
from multiprocessing import Pool
import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from regbim import io_utils                              # noqa: E402
from regbim.methods import get_method                    # noqa: E402
from failure_decomposition import provenance             # noqa: E402

TARGETS = ["m3_cor_%s__%s" % (s, e) for s in "abcd" for e in ("E2", "E3")]
SEEDS = list(range(1, 11))
OUT = "Registration/output/diag/r43/r43_real.json"


def make_cfg(cfg0, seed, lf):
    cfg = copy.deepcopy(cfg0)
    cfg["source"] = dict(cfg["source"], seed=seed)
    cfg.setdefault("proposed", {})["translation_init"] = "plan_correlate"
    cfg["proposed"]["label_filter"] = {"enabled": bool(lf)}
    return cfg


def job(a):
    target, k, seed, lf = a
    cfg = make_cfg(yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target)), seed, lf)
    T = get_method("proposed").register(io_utils.load_source_cloud(cfg), io_utils.load_reference_cloud(cfg), cfg)
    return a, np.asarray(T, dtype=np.float64).tolist()


def main():
    r37 = {(r["target"], r["k"]): np.asarray(r["T_off"]) for r in
           json.load(open("Registration/output/diag/r37_replicate.json"))["rows"] if r["series"] == "S"}
    jobs = [(t, k, s, lf) for t in TARGETS for k, s in enumerate(SEEDS) for lf in (True, False)]
    with Pool(8) as p:
        out = p.map(job, jobs, chunksize=1)
    rows = {}
    for (t, k, s, lf), T in out:
        r = rows.setdefault((t, k), {"target": t, "k": k, "seed": s})
        r["T_on" if lf else "T_off"] = T
    res = {"provenance": provenance(), "rows": list(rows.values())}
    for r in res["rows"]:
        ref = r37[(r["target"], r["k"])]
        r["on_eq_r37_S_off"] = bool(np.array_equal(np.asarray(r["T_on"]), ref))
        r["on_vs_r37_max_abs"] = float(np.abs(np.asarray(r["T_on"]) - ref).max())
    res["n_on_eq_r37"] = sum(r["on_eq_r37_S_off"] for r in res["rows"])
    print("on == R37 S off:", res["n_on_eq_r37"], "/", len(res["rows"]))
    json.dump(res, open(OUT, "w"))


if __name__ == "__main__":
    main()

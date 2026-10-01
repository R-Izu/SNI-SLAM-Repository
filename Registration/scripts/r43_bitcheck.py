"""R43 §3（R42 §4-2）：label_filter を足した後も、既定（キー無し＝on）で R35 の 16 出力とビット一致するか。

R35 と同じ呼び出し（`r35_rescore.register`：source seed 0、translation_init = centroid / plan_correlate）で再実行し、
`r35_runs.json` の T_est と `np.array_equal` で比べる。案A の 8 条件は `label_filter: {enabled: true}` を明示しても比べる。
"""
import copy, json, os, sys
from multiprocessing import Pool
import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from r35_rescore import register                        # noqa: E402
from failure_decomposition import provenance            # noqa: E402


def job(a):
    target, mode, explicit = a
    cfg0 = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
    if explicit:
        cfg0 = copy.deepcopy(cfg0); cfg0.setdefault("proposed", {})["label_filter"] = {"enabled": True}
    return a, register(cfg0, mode)["T_est"]


def main():
    rows = json.load(open("Registration/output/diag/r35_runs.json"))["rows"]
    ref = {("%s__%s" % (r["scene"], r["cond"]), r["mode"]): np.asarray(r["T_est"]) for r in rows}
    jobs = [(t, m, False) for (t, m) in ref] + [(t, m, True) for (t, m) in ref if m == "plan_correlate"]
    with Pool(8) as p:
        out = p.map(job, jobs)
    res = {"provenance": provenance(), "rows": []}
    for (t, m, ex), T in out:
        T = np.asarray(T)
        res["rows"].append({"target": t, "mode": m, "explicit_on": ex, "bit_equal": bool(np.array_equal(T, ref[(t, m)])),
                            "max_abs": float(np.abs(T - ref[(t, m)]).max())})
        print(t, m, "explicit" if ex else "default", res["rows"][-1]["bit_equal"], flush=True)
    res["all_bit_equal"] = all(r["bit_equal"] for r in res["rows"])
    print("all_bit_equal", res["all_bit_equal"])
    os.makedirs("Registration/output/diag/r43", exist_ok=True)
    json.dump(res, open("Registration/output/diag/r43/r43_bitcheck.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""R47 §4-2：案A の大きな失敗（B-1 の交差の組と B-2）を、診断を有効にして回し直し、R32 §3 の段階別に分類する（診断のみ）。

R46 で回した組の失敗は R46 の `r46_planA_diag.json` にある。保存した T_est とビット一致することを確かめる。
E_g・判定の幅は `r46_planA_diag.py` と同じ。
"""
import copy, json, os, sys
from multiprocessing import Pool
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r47"))
os.chdir(REPO)
from criterion_verdict import d_omega                                   # noqa: E402
from regbim import io_utils, metrics                                    # noqa: E402
from regbim.labels import LabeledCloud                                  # noqa: E402
from regbim.methods import get_method                                   # noqa: E402
from r41_rescore import real_criteria, per_trial                        # noqa: E402
from r47_common import b1_conditions                                    # noqa: E402

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))


def job(a):
    name, input_set, ref_target, k, saved = a
    scene = ref_target.split("__")[0]; G = real_criteria(scene)[1]; om = np.load("Registration/output/diag/omega.npz")[scene]
    cfg = copy.deepcopy(yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % ref_target)))
    cfg.setdefault("proposed", {})["translation_init"] = "plan_correlate"; cfg["diagnostics"] = {"record_yaw": True, "record_stages": True}
    z = np.load(os.path.join(ROOT, "inputs_sni", input_set + ".npz")); src = LabeledCloud(z["points"], z["labels"], z["normals"])
    P = np.asarray(saved["P"]); m = get_method("proposed")
    T = np.asarray(m.register(metrics.transform_cloud(src, P), io_utils.load_reference_cloud(cfg), cfg), dtype=np.float64)
    dO = lambda X: float(d_omega(np.asarray(X, dtype=np.float64) @ P, G, om))
    yd, pd = m.last_yaw_diag, m.last_plan_diag
    gen = min(dO(c["T_gen"]) for c in pd["stage_candidates"]); sho = min(dO(c["T_short"]) for c in pd["stage_candidates"])
    aft = min(dO(t) for t in yd["candidate_T"]); fin = dO(T)
    v = ("良い候補が残ったが別が選ばれた" if (sho < fin - 0.15 or aft < fin - 0.15) else "精緻化で悪化" if fin > gen + 0.15
         else "精緻化がほぼ動かしていない" if abs(fin - gen) < 0.15 else "精緻化で改善")
    return {"condition": name, "trial": k, "bit_equal": bool(np.array_equal(T, np.asarray(saved["T_est"]))), "final_d_omega": fin,
            "E_gen": gen, "E_short": sho, "E_after_icp": aft, "verdict": v, "stage1_scores": yd["candidate_scores"]}


def main():
    jobs = []
    for f in sorted(os.listdir(os.path.join(ROOT, "results_r47", "sni"))):
        name = f[:-5]; rows = json.load(open(os.path.join(ROOT, "results_r47", "sni", f)))["rows"]
        if name.endswith("_rot"):
            s, cut = name.split("__")[0], name.split("_r")[-2].split("_")[-1] if False else name.rsplit("_r", 2)[1]
            input_set, ref_target = "%s__E3_r%s" % (s, cut), "%s__E3" % s
        else:
            c = [x for x in b1_conditions() if x["name"] == name][0]; input_set, ref_target = c["input_set"], c["ref_target"]
        G = real_criteria(name.split("__")[0])[1]; om = np.load("Registration/output/diag/omega.npz")[name.split("__")[0]]
        for key, r in rows.items():
            meth, k = key.split("|")
            if meth != "planA":
                continue
            p = per_trial(r["T_est"], r["P"], G, om, None)
            if (not p["v"]) or p["eR"] >= 45 or p["eO"] >= 1.0:
                jobs.append((name, input_set, ref_target, int(k), r))
    print("failures to diagnose", len(jobs), flush=True)
    with Pool(8) as pool:
        out = pool.map(job, jobs)
    for o in out:
        print(json.dumps({q: o[q] for q in ("condition", "trial", "bit_equal", "final_d_omega", "E_gen", "E_short", "E_after_icp", "verdict")}, ensure_ascii=False))
    json.dump(out, open("Registration/output/diag/r47/r47_planA_diag.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()

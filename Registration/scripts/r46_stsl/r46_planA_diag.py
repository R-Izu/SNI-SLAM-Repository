"""R46 §3-3：案A（planA）の大きな失敗の試行を、診断を有効にして回し直し、R32 §3 の段階別に分類する（診断のみ）。

`r46_score.json` から planA の大きな失敗（G1）の試行を拾い、同じ入力（切った source に P）と同じ config に
diagnostics（record_yaw・record_stages）だけを足して回す。**`results_sni` の T_est とビット一致することを確かめる。**
各段階の E_g = min d_Ω(T·P, G1, Ω)（R32 §3、診断用）、判定の幅 0.15 m は `_r32_stages.py` と同じ。
"""
import copy, json, os, sys
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from criterion_verdict import d_omega                                   # noqa: E402
from failure_decomposition import provenance                            # noqa: E402
from regbim import io_utils, metrics                                    # noqa: E402
from regbim.labels import LabeledCloud                                  # noqa: E402
from regbim.methods import get_method                                   # noqa: E402
from r36_rotation_diag import tilt_yaw                                  # noqa: E402
from r41_rescore import real_criteria                                   # noqa: E402

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))


def main():
    sc = json.load(open("Registration/output/diag/r46/r46_score.json"))
    om_all = np.load("Registration/output/diag/omega.npz")
    out = {"provenance": provenance(), "trials": []}
    for name, c in sc["conditions"].items():
        for k, tr in enumerate(c["trials"]["planA"]):
            p = tr["per"]["1"]
            if p["v"] and p["eR"] < 45 and p["eO"] < 1.0:
                continue
            target, cut = name.rsplit("_r", 1); scene = target.split("__")[0]
            G = real_criteria(scene)[1]; RG = metrics.decompose_sim3(G)[0]; om = om_all[scene]
            cfg = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
            cfg = copy.deepcopy(cfg); cfg.setdefault("proposed", {})["translation_init"] = "plan_correlate"
            cfg["diagnostics"] = {"record_yaw": True, "record_stages": True}
            z = np.load(os.path.join(ROOT, "inputs_sni", name + ".npz"))
            src = LabeledCloud(z["points"], z["labels"], z["normals"]); dst = io_utils.load_reference_cloud(cfg)
            saved = json.load(open(os.path.join(ROOT, "results_sni", name + ".json")))["rows"]["planA|%d" % k]
            P = np.asarray(saved["P"], dtype=np.float64)
            m = get_method("proposed")
            T = np.asarray(m.register(metrics.transform_cloud(src, P), dst, cfg), dtype=np.float64)
            dO = lambda X: float(d_omega(np.asarray(X, dtype=np.float64) @ P, G, om))
            yawe = lambda X: tilt_yaw(metrics.decompose_sim3(np.asarray(X, dtype=np.float64) @ P)[0], RG)["yaw_deg"]
            yd, pd = m.last_yaw_diag, m.last_plan_diag
            gen = [dO(cc["T_gen"]) for cc in pd["stage_candidates"]]; sho = [dO(cc["T_short"]) for cc in pd["stage_candidates"]]
            after = [dO(t) for t in yd["candidate_T"]]; fin = dO(T)
            E_gen, E_sho, E_aft = min(gen), min(sho), min(after)
            if E_sho < fin - 0.15 or E_aft < fin - 0.15:
                verdict = "良い候補が残ったが別が選ばれた"
            elif fin > E_gen + 0.15:
                verdict = "精緻化で悪化"
            elif abs(fin - E_gen) < 0.15:
                verdict = "精緻化がほぼ動かしていない"
            else:
                verdict = "精緻化で改善"
            row = {"condition": name, "trial": k, "bit_equal": bool(np.array_equal(T, np.asarray(saved["T_est"]))),
                   "final_d_omega": fin, "E_gen": E_gen, "E_short": E_sho, "E_after_icp": E_aft, "verdict": verdict,
                   "stage1_scores": yd["candidate_scores"], "stage1_winner": yd["winner"], "stage1_margin": yd["margin"],
                   "stage1_d_after_icp": after, "stage1_yaw_err": [yawe(t) for t in yd["candidate_T"]]}
            out["trials"].append(row)
            print(json.dumps({q: row[q] for q in ("condition", "trial", "bit_equal", "final_d_omega", "E_gen", "E_short", "E_after_icp", "verdict")}, ensure_ascii=False))
            print("   scores", [round(x, 3) for x in yd["candidate_scores"]], "winner", yd["winner"], "| d", [round(x, 2) for x in after],
                  "| yaw err", [round(x, 1) for x in row["stage1_yaw_err"]], flush=True)
    json.dump(out, open("Registration/output/diag/r46/r46_planA_diag.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()

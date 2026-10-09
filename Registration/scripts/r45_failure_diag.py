"""R45 §4-5：案A で失敗した試行を、診断を有効にして回し直し、R32 §3 の段階別に分類する（診断のみ）。

- 保存した P を source に当て（benchmark と同じ `metrics.transform_cloud`）、`gt_a_plan_a/<scene>.yaml` に
  diagnostics（record_yaw・record_stages）だけを足して `proposed` を回す。**出力がベンチマークの T_est とビット一致することを確かめる**
- 段階：生成直後（T_gen）→ 短精緻化後（T_short）→ 段階1の ICP 後（向きごとの候補、score）→ 最終
- 各段階の E_g = min d_Ω(T·P, G, Ω)（R32 §3。診断用で、最小候補を勝者にしていない）
- 判定は `_r32_stages.py` と同じ幅 0.15 m
引数：<scene> <trial> [<trial> ...]
"""
import copy, json, os, sys
import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from criterion_verdict import build_omega, d_omega                      # noqa: E402
from failure_decomposition import provenance                            # noqa: E402
from regbim import io_utils, metrics                                    # noqa: E402
from regbim.methods import get_method                                   # noqa: E402
from r36_rotation_diag import tilt_yaw                                  # noqa: E402

scene, trials = sys.argv[1], [int(x) for x in sys.argv[2:]]
cfg = yaml.safe_load(open("Registration/configs/gt_a_plan_a/%s.yaml" % scene))
cfg = copy.deepcopy(cfg); cfg["diagnostics"] = {"record_yaw": True, "record_stages": True}
tm = json.load(open("output/Registration/r45/plan_a/%s/trial_matrices.json" % scene))
G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
RG = metrics.decompose_sim3(G)[0]
om = build_omega(yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % scene)), seed=0)
src = io_utils.load_source_cloud(cfg); dst = io_utils.load_reference_cloud(cfg)
saved = {t["trial"]: t for t in tm["trials"] if t["method"] == "proposed"}
out = {"provenance": provenance(), "scene": scene, "trials": []}
for k in trials:
    P = np.asarray(saved[k]["P"], dtype=np.float64)
    m = get_method("proposed")
    T = np.asarray(m.register(metrics.transform_cloud(src, P), dst, cfg), dtype=np.float64)
    dO = lambda X: float(d_omega(np.asarray(X, dtype=np.float64) @ P, G, om))
    rot = lambda X: tilt_yaw(metrics.decompose_sim3(np.asarray(X, dtype=np.float64) @ P)[0], RG)
    st, yd, pd = m.last_stage_diag, m.last_yaw_diag, m.last_plan_diag
    cands = pd["stage_candidates"]
    gen = [dO(c["T_gen"]) for c in cands]; sho = [dO(c["T_short"]) for c in cands]
    after = [dO(t) for t in yd["candidate_T"]]
    fin = dO(T)
    E_gen, E_sho, E_aft = min(gen), min(sho), min(after)
    if E_sho < fin - 0.15 or E_aft < fin - 0.15:
        verdict = "良い候補が残ったが別が選ばれた"
    elif fin > E_gen + 0.15:
        verdict = "精緻化で悪化"
    elif abs(fin - E_gen) < 0.15:
        verdict = "精緻化がほぼ動かしていない"
    else:
        verdict = "精緻化で改善"
    row = {"trial": k, "bit_equal_benchmark": bool(np.array_equal(T, np.asarray(saved[k]["T_est"]))),
           "final_d_omega": fin, "final_rot": rot(T), "E_gen": E_gen, "E_short": E_sho, "E_after_icp": E_aft,
           "verdict": verdict,
           "per_yaw": [{"yaw_index": c["yaw_index"], "score_short": c["score_short"], "d_gen": g, "d_short": s_,
                        "yaw_err_deg": rot(c["T_short"])["yaw_deg"]} for c, g, s_ in zip(cands, gen, sho)],
           "stage1": {"scores": yd["candidate_scores"], "winner": yd["winner"], "margin": yd["margin"],
                      "d_after_icp": after, "yaw_err_after_icp": [rot(t)["yaw_deg"] for t in yd["candidate_T"]]},
           "final_is_refine": st["final_is_refine"]}
    out["trials"].append(row)
    print(json.dumps({q: row[q] for q in ("trial", "bit_equal_benchmark", "final_d_omega", "E_gen", "E_short", "E_after_icp", "verdict")}, ensure_ascii=False))
    print("  stage1 scores", [round(x, 4) for x in yd["candidate_scores"]], "winner", yd["winner"], "margin", yd["margin"],
          "| d after ICP", [round(x, 3) for x in after], "| yaw err", [round(x, 1) for x in row["stage1"]["yaw_err_after_icp"]])
    best = {}
    for c, s_ in zip(cands, sho):
        best.setdefault(c["yaw_index"], []).append((round(c["score_short"], 4), round(s_, 3)))
    print("  short-refined per yaw (score, d):", {y: sorted(v, reverse=True)[:3] for y, v in best.items()})
os.makedirs("Registration/output/diag/r45", exist_ok=True)
json.dump(out, open("Registration/output/diag/r45/r45_failure_diag_%s.json" % scene, "w"), ensure_ascii=False, indent=1, default=float)

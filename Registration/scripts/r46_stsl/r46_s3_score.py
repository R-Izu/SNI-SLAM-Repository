"""R46 段階3の採点（sni-slam の環境）。**回す前に commit する。** 判定の言葉は出さない（S6）。

STSL＋縮尺探索：各 (set, 設定, k, 試行) で、59 個の縮尺のうち **STSL 自身の検証スコア（optimal_score）が最大**のものを採る。
候補なし（または optimal_score が有限でない）の縮尺は飛ばす。全縮尺で候補なしなら無効（v = 0）。
採った縮尺 σ で、source → 参照 = Shift(+z_floor) · tf · T_rp · D(σ) · A · Rg（`r46_s3_inputs.py` の A・Rg）。
採点：形 C の d_Ω（Ω は R38・R41・R45 と同じ：GT-A は `build_omega(gt_a/<scene>.yaml, seed=0)`、対照は R39 の config の
`build_omega`）、回転誤差、縮尺の相対誤差 q。**大きな失敗**（§3-3）：無効・e_R ≥ 45°・e_Ω ≥ 1.0 m。旧基準（5°・0.1 m・0.05）も数える。
試行の失敗は 5 回の多数決。設定は set ごとに大きな失敗が少ない方（同数なら A）。
案A：GT-A は R45 の plan_a（`r45_score.json`、同じ手法のコード）、対照は `output/Registration/r46_s3_planA/E{2,3}/`
（`fix/translation-init` で R39 の config・先頭 20 試行を回し直したもの。F13）。
対の比較：大きな失敗の n10（案A だけ）・n01（STSL だけ）と正確 McNemar。
予算：前処理と登録の回数、時間。
"""
import json, os, re, sys
from math import isfinite
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from criterion_verdict import build_omega                          # noqa: E402
from failure_decomposition import provenance                       # noqa: E402
from r41_rescore import per_trial, dist                            # noqa: E402
from scipy.special import comb as _comb                            # noqa: E402

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
SIGMAS = np.exp(np.linspace(np.log(0.3), np.log(3.0), 59))
SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
FLOORS = {"gta_%s" % s: "GTA_%s" % s for s in SCENES}; FLOORS.update({"bim_E2": "M3_411_both", "bim_E3": "M3_410_411_both"})
K = 5


def jl(line):
    return json.loads(re.sub(r"(?<![A-Za-z])(-?)inf(?![A-Za-z])", lambda m: m.group(1) + "Infinity", line).replace("nan", "NaN"))


def mcnemar(n10, n01):
    n = n10 + n01
    return 1.0 if n == 0 else min(1.0, 2.0 * sum(_comb(n, i, exact=True) for i in range(min(n10, n01) + 1)) / 2.0 ** n)


def big(p):
    return (not p["v"]) or p["eR"] >= 45.0 or p["eO"] >= 1.0


def old(p):
    return bool(p["v"] and p["eR"] < 5.0 and p["eO"] < 0.1 and p["q"] < 0.05)


def setup(base):
    if base.startswith("gta_"):
        s = base[4:]
        tm = json.load(open("output/Registration/r45/plan_a/%s/trial_matrices.json" % s))
        G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
        om = build_omega(yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % s)), seed=0)
        r45 = json.load(open("Registration/output/diag/r45/r45_score.json"))["scenes"][s]["trials"]["plan_a"]
        planA = [r45[str(k)] for k in range(20)]
    else:
        e = base[4:]
        cfg = yaml.safe_load(open("Registration/configs/r39_bim/bim_%s_plan_correlate.yaml" % e))
        om = build_omega(cfg)
        tm = json.load(open("output/Registration/r46_s3_planA/%s/trial_matrices.json" % e))
        G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
        rows = {t["trial"]: t for t in tm["trials"] if t["method"] == "proposed" and t["trial"] >= 0}
        planA = [per_trial(rows[k]["T_est"], rows[k]["P"], G, om, None) for k in range(20)]
    return G, om, planA


def main():
    res = {"provenance": provenance(), "sigmas": SIGMAS.tolist(), "sets": {}}
    n_prep = n_reg = 0; t_prep = []; t_reg = []
    only = sys.argv[1:] or list(FLOORS)          # 例：bim_E2 bim_E3（GT-A は main の判断待ちで回していない）
    for base in [b for b in FLOORS if b in only]:
        G, om, planA = setup(base)
        spec = json.load(open(os.path.join(ROOT, "inputs_s3", base, "spec.json")))
        zf = json.load(open(os.path.join(ROOT, "BIM", FLOORS[base], "meta.json")))["z_floor"]
        S = np.eye(4); S[2, 3] = zf
        ent = {"settings": {}}
        for st in "AB":
            runs = {i: [] for i in range(20)}
            for k in range(1, K + 1):
                best = {i: None for i in range(20)}
                for j, sg in enumerate(SIGMAS):
                    name = "s3_%s_j%02d" % (base, j)
                    d = os.path.join(ROOT, "results", name)
                    trp = json.load(open(os.path.join(d, "%s_k%d_T_rp.json" % (st, k))))
                    pt = json.load(open(os.path.join(d, "%s_k%d_prep_time.json" % (st, k))))
                    n_prep += pt["n_inputs"]; t_prep.append((pt["plane_seg_s"] + pt["python_prep_s"]) / pt["n_inputs"])
                    for r in (jl(l) for l in open(os.path.join(d, "%s_k%d_%s.jsonl" % (st, k, FLOORS[base])))):
                        n_reg += 1; t_reg.append(r["reg_s"])
                        i = r["index"]
                        if not r["candidates"] or not isfinite(r["optimal_score"]) or i in pt.get("prep_failed", []):
                            continue
                        if best[i] is None or r["optimal_score"] > best[i]["score"]:
                            best[i] = {"score": r["optimal_score"], "j": j, "tf": r["tf"], "trp": trp[str(i)]}
                for i in range(20):
                    t = spec["trials"][i]; b = best[i]
                    if b is None:
                        p = dict(per_trial(np.eye(4), None, G, om, None), v=False); sig = None
                    else:
                        D = np.diag([SIGMAS[b["j"]]] * 3 + [1.0])
                        T = S @ np.asarray(b["tf"]).reshape(4, 4) @ np.asarray(b["trp"]) @ D @ np.asarray(t["A"]) @ np.asarray(t["Rg"])
                        p = per_trial(T, None, G, om, None); sig = float(SIGMAS[b["j"]])
                    runs[i].append({"per": p, "sigma": sig, "sigma_true": t["true_scale_of_input"]})
            f = [sum(big(x["per"]) for x in runs[i]) >= 3 for i in range(20)]
            ent["settings"][st] = {"n_big_fail": sum(f), "fails": f,
                                   "n_old_success_majority": sum(sum(old(x["per"]) for x in runs[i]) >= 3 for i in range(20)),
                                   "n_invalid_majority": sum(sum(not x["per"]["v"] for x in runs[i]) >= 3 for i in range(20)),
                                   "dist": {q: dist([x["per"] for i in range(20) for x in runs[i]], q) for q in ("eR", "eO", "q")},
                                   "sigma_ratio": [x["sigma"] / x["sigma_true"] for i in range(20) for x in runs[i] if x["sigma"]],
                                   "runs": {str(i): runs[i] for i in range(20)}}
        bs = min("AB", key=lambda s: (ent["settings"][s]["n_big_fail"], s))
        fa = [big(p) for p in planA]; fs = ent["settings"][bs]["fails"]
        n10 = sum(x and not y for x, y in zip(fa, fs)); n01 = sum(y and not x for x, y in zip(fa, fs))
        ent.update({"best_setting": bs, "planA": {"n_big_fail": sum(fa), "n_old_success": sum(old(p) for p in planA),
                                                  "dist": {q: dist(planA, q) for q in ("eR", "eO", "q")}},
                    "pair": {"n10_planA_only": n10, "n01_stsl_only": n01, "p_mcnemar": mcnemar(n10, n01)}})
        res["sets"][base] = ent
        print(base, "STSL A %d B %d (best %s) | planA %d | n10 %d n01 %d p %.3g | STSL old-success %d planA old %d"
              % (ent["settings"]["A"]["n_big_fail"], ent["settings"]["B"]["n_big_fail"], bs, sum(fa), n10, n01, mcnemar(n10, n01),
                 ent["settings"][bs]["n_old_success_majority"], ent["planA"]["n_old_success"]), flush=True)
    res["budget"] = {"n_preprocess": n_prep, "n_register": n_reg, "prep_s_median": float(np.median(t_prep)),
                     "reg_s_median": float(np.median(t_reg)), "per_trial_per_setting": "59 scales x 5 runs = 295 preprocess + 295 register"}
    os.makedirs("Registration/output/diag/r46", exist_ok=True)
    json.dump(res, open("Registration/output/diag/r46/r46_s3_score.json", "w"), ensure_ascii=False, default=float)
    print("budget", res["budget"])


if __name__ == "__main__":
    main()

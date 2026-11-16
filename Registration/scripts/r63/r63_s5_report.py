"""R63 §4 S5 で止めた後の記録の集計（sni-slam の環境）。**回す前に commit する。位置合わせは回さない。**
- 本手法・完全・b-2・f 1p0 の 40 試行：最終の e_R・e_Ω・e_ΩBIM（参照ごとの Ω_BIM を A⁻¹ で b-2 の座標に置いたもの）・縮尺の比、
  段階ごと（gen・short・after_icp）の基準内（e_R < 45°・e_Ω < 1.0 m）の候補の数と最良誤差 b（E_Ω で選んだ候補の e_R・e_Ω）、ヨーの差（最終、`r51_stage0.rel` と同じ）
- §3-2 の確認：記録の T_est から計算し直した e_Ω と、実行が書いた run_eO の差
- STSL（S5 の結果の前に並行して回したもの）：k = 1 と 5 回すべての、出力なし・基準内の候補なし・別を選んだ・成功の件数（R58 と同じ分類）
出力：`~/rizu/r63_data/s5_summary.json`
"""
import glob, json, os, sys
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49", "Registration/scripts/r51"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from r41_rescore import per_trial                              # noqa: E402
from regbim import metrics                                     # noqa: E402

CO = os.path.expanduser("~/rizu/conslam/out"); ROOT = os.path.expanduser("~/rizu/r46_stsl_data")
REFS = ["S1_small", "S1_large", "S3_small", "S3_large"]
A = np.array(json.load(open(os.path.join(CO, "extra.json")))["b_normalized"]["A"]); Ai = np.linalg.inv(A)
OM = np.load(os.path.join(CO, "r62_omega.npz"))


def ok(e):
    return bool(e["v"] and e["eR"] < 45 and e["eO"] < 1.0)


def main():
    res = {"prop": {}, "stsl": {}}
    om = OM["omega_b2"]
    for ref in REFS:
        omb = OM["omega_bim_" + ref] @ Ai[:3, :3].T + Ai[:3, 3]
        rows = json.load(open(os.path.expanduser("~/rizu/r63_data/results/full/b2_f1p0__%s.json" % ref)))["rows"]
        out = []
        for k in range(10):
            r = rows[str(k)]; P = np.asarray(r["P"]); d = r["diag"]
            fin = per_trial(r["T_est"], P, A, om, omb)
            st = {}
            for s, Ts in (("gen", d["gen"]), ("short", d["short"]), ("after", d["after_icp"])):
                es = [per_trial(T, P, A, om, omb) for T in Ts]
                i = int(np.argmin([e["eO"] if e["v"] else np.inf for e in es]))
                st[s] = {"n_ok": sum(ok(e) for e in es), "n": len(es), "b_eO": float(es[i]["eO"]), "b_eR": float(es[i]["eR"])}
            Rt, _, st_s = metrics.decompose_sim3(np.asarray(r["T_est"]) @ P); Rg, _, sg = metrics.decompose_sim3(A)
            out.append({"k": k, "eR": float(fin["eR"]), "eO": float(fin["eO"]), "eOB": float(fin["eOB"]), "yaw": float(fin["yaw"]), "tilt": float(fin["tilt"]),
                        "scale_ratio": float(st_s / sg), "stages": st, "chk": abs(float(fin["eO"]) - r["run_eO"]), "time_s": r["time_s"]})
        res["prop"][ref] = out
    # STSL
    for f in ("1p0", "0p6"):
        name = "r62_b2_f" + f; spec = json.load(open(os.path.join(ROOT, "inputs", name, "spec.json")))
        for ref in REFS:
            zf = json.load(open(os.path.join(ROOT, "BIM", "CS_" + ref, "meta.json")))["z_floor"]; S = np.eye(4); S[2, 3] = zf
            omb = OM["omega_bim_" + ref] @ Ai[:3, :3].T + Ai[:3, 3]
            for j in range(1, 6):
                trp = json.load(open(os.path.join(ROOT, "processed", "%s_A_k%d" % (name, j), "30m", "T_rp.json")))
                pt = json.load(open(os.path.join(ROOT, "results", name, "A_k%d_prep_time.json" % j)))
                for l in open(os.path.join(ROOT, "results", name, "A_k%d_CS_%s.jsonl" % (j, ref))):
                    b = json.loads(l.replace("-inf", "-Infinity").replace("inf", "Infinity")); i = b["index"]
                    PA = np.asarray(spec["trials"][i]["P_A"]); Trp = np.asarray(trp[str(i)]) if str(i) in trp else np.eye(4)
                    T3 = lambda tf: S @ np.asarray(tf, dtype=np.float64).reshape(4, 4) @ Trp
                    pf = i in pt.get("prep_failed", []); nocand = len(b["candidates"]) == 0 or not np.isfinite(b["optimal_score"])
                    if pf or nocand:
                        cl = "prep_failed" if pf else "no_candidate"; e = None
                    else:
                        es = [per_trial(T3(c["tf"]), PA, A, om, omb) for c in b["candidates"]]
                        e = per_trial(T3(b["tf"]), PA, A, om, omb)
                        cl = "success" if ok(e) else ("other_chosen" if any(ok(x) for x in es) else "no_correct")
                    res["stsl"].setdefault("%s|%s" % (f, ref), []).append({"run": j, "k": i, "class": cl, "n_cand": len(b["candidates"]),
                                                                          "eO": None if e is None else float(e["eO"]), "eR": None if e is None else float(e["eR"])})
    json.dump(res, open(os.path.expanduser("~/rizu/r63_data/s5_summary.json"), "w"), indent=1)
    from collections import Counter
    for ref, out in res["prop"].items():
        print(ref, "final eO med %.2f eR med %.1f" % (np.median([o["eO"] for o in out]), np.median([o["eR"] for o in out])),
              "ok gen/short/after", [sum(o["stages"][s]["n_ok"] > 0 for o in out) for s in ("gen", "short", "after")],
              "b_gen med %.2f" % np.median([o["stages"]["gen"]["b_eO"] for o in out]), "chk max %.1e" % max(o["chk"] for o in out))
    for k, v in res["stsl"].items():
        print("stsl", k, "k1", dict(Counter(x["class"] for x in v if x["run"] == 1)), "all5", dict(Counter(x["class"] for x in v)))


if __name__ == "__main__":
    main()

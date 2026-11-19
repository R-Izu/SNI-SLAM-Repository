"""R63 §7-2 の 3 の集計：条件 A の本手法と STSL（sni-slam の環境）。**回す前に commit する。** 定義は R60、線は R63 §2（C1〜C5）。判定の言葉は出さない（線のとおりの文だけ）。

- 本手法：`~/rizu/r63_data/resultsA/<full|score_only|icp_only>/b2_f<f>__<参照>.json`。T は q = P_eff p → IFC、正解は A。
  `per_trial(T, P_eff, A, Ω_b2, M)`。M は参照の Ω_BIM（A⁻¹ で b-2 の座標に置いたもの）と、大の参照には Ω_小（小の Ω_BIM）
- 段階ごとの最良誤差 b（E_Ω で選んだ候補、gen・short・after_icp）、l = E(T_est) − E(after_icp の最良)。基準内 = v ∧ e_R < 45° ∧ e_Ω < 1.0 m。
  段階の分類（R54 の主の線）：作る（gen に基準内の候補なし）／短い ICP／絞り込み／選ぶ／成功
- STSL：R63 の登録（`results/r62_b2_f<f>/A_k<k>_CS_<参照>.jsonl`）。b は投票の後の全候補、l = E(最終) − b。出力なしは別に数える
- 線（R63 §2）：
  C1：参照ごとに、gen の b（Ω）の中央値が f 0.2 で f 1.0 より大きい参照の数（≥ 3/4）
  C2：参照ごとに f 1.0 と 0.6 を合わせた 20 試行で、l_Ω > 0.5 m の件数が score_only > full の参照の数（≥ 3/4）
  C3：f 1.0・0.6 × 参照の 8 群で、|基準内（icp_only） − 基準内（full）| ≤ 2 の群の数（≥ 7/8）
  C4：参照ごとに、full の l_Ω > 0.5 m の件数が f 0.6 > f 1.0 の参照の数（≥ 3/4）
  C5：STSL、f 0.6、k = 1 の失敗のうち「出力なし＋基準内の候補なし」の割合（≥ 0.80）
出力：`~/rizu/r63_data/trialsA<_fix>.json`、`Registration/output/diag/r63/summaryA<_fix>.json`・`tablesA<_fix>.md`・図
環境変数 R63_TAG=fix（R63 §8、2026-11-20）：直した入力・参照の記録（`resultsA_fix/`）と `r63fix_omega.npz` を使い、出力の名前に `_fix` を付ける。STSL の記録は同じもの
"""
import json, math, os, sys
from collections import Counter
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from r41_rescore import per_trial                              # noqa: E402
from failure_decomposition import provenance                   # noqa: E402

CO = os.path.expanduser("~/rizu/conslam/out"); ROOT = os.path.expanduser("~/rizu/r46_stsl_data"); TAG = os.environ.get("R63_TAG", ""); SFX = "_" + TAG if TAG else ""
RA = os.path.expanduser("~/rizu/r63_data/resultsA" + SFX)
DIAG = "Registration/output/diag/r63"
REFS = ["S1_small", "S1_large", "S3_small", "S3_large"]; FS = ["0p2", "0p4", "0p6", "0p8", "1p0"]
A = np.array(json.load(open(os.path.join(CO, "extra.json")))["b_normalized"]["A"]); Ai = np.linalg.inv(A)
Z = np.load(os.path.join(CO, "r63fix_omega.npz" if TAG == "fix" else "r62_omega.npz")); OM = Z["omega_b2"]
to_b2 = lambda X: X @ Ai[:3, :3].T + Ai[:3, 3]
OMB = {r: to_b2(Z["omega_bim_" + r]) for r in REFS}
OMS = {"S1_small": to_b2(Z["omega_small_S1"]), "S1_large": to_b2(Z["omega_small_S1"]), "S3_small": to_b2(Z["omega_small_S3"]), "S3_large": to_b2(Z["omega_small_S3"])}


def ok(e):
    return bool(e["v"] and e["eR"] < 45 and e["eO"] < 1.0)


def ev(T, P, ref):
    e = per_trial(T, P, A, OM, OMB[ref]); e2 = per_trial(T, P, A, OM, OMS[ref])
    inf = float("inf")
    return {"v": bool(e["v"]), "eO": float(e["eO"]) if e["v"] else inf, "eR": float(e["eR"]) if e["v"] else inf,
            "eOB": float(e["eOB"]) if e["v"] else inf, "eOS": float(e2["eOB"]) if e["v"] else inf, "ok": ok(e)}


def best(es):
    i = int(np.argmin([e["eO"] for e in es])); return i, es[i]


def q(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return None if not v else {"n": len(v), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                               "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95))}


def stage_class(flags):
    g, s, a, f = flags
    if f:
        return "success"
    return ["make", "short", "narrow", "select"][next(i for i, x in enumerate((g, s, a)) if not x)] if not all((g, s, a)) else "select"


def trials():
    rows = []
    for m in ("full", "score_only", "icp_only"):
        for f in FS:
            for ref in REFS:
                p = os.path.join(RA, m, "b2_f%s__%s.json" % (f, ref))
                if not os.path.exists(p):
                    continue
                for k, r in json.load(open(p))["rows"].items():
                    P = np.asarray(r["P"]); d = r["diag"]; row = {"method": m, "f": f, "ref": ref, "k": int(k), "output": True, "time_s": r["time_s"]}
                    flags = []
                    for st, Ts in (("gen", d["gen"]), ("short", d["short"]), ("after", d["after_icp"])):
                        es = [ev(T, P, ref) for T in Ts]; i, e = best(es)
                        row["b_" + st] = dict(e, idx=i); flags.append(any(x["ok"] for x in es))
                    fin = ev(r["T_est"], P, ref); row["final"] = fin; flags.append(fin["ok"])
                    row["l"] = {"O": fin["eO"] - row["b_after"]["eO"], "OB": fin["eOB"] - row["b_after"]["eOB"], "OS": fin["eOS"] - row["b_after"]["eOS"]}
                    row["class"] = stage_class(flags); row["chk"] = abs(fin["eO"] - r["run_eO"]) if fin["v"] else None
                    rows.append(row)
    for f in ("1p0", "0p6"):
        name = "r62_b2_f" + f; spec = json.load(open(os.path.join(ROOT, "inputs", name, "spec.json")))
        for ref in REFS:
            zf = json.load(open(os.path.join(ROOT, "BIM", "CS_" + ref, "meta.json")))["z_floor"]; S = np.eye(4); S[2, 3] = zf
            for j in range(1, 6):
                trp = json.load(open(os.path.join(ROOT, "processed", "%s_A_k%d" % (name, j), "30m", "T_rp.json")))
                pt = json.load(open(os.path.join(ROOT, "results", name, "A_k%d_prep_time.json" % j)))
                for l in open(os.path.join(ROOT, "results", name, "A_k%d_CS_%s.jsonl" % (j, ref))):
                    b = json.loads(l.replace("-inf", "-Infinity").replace("inf", "Infinity")); i = b["index"]
                    PA = np.asarray(spec["trials"][i]["P_A"]); Trp = np.asarray(trp[str(i)]) if str(i) in trp else np.eye(4)
                    T3 = lambda tf: S @ np.asarray(tf, dtype=np.float64).reshape(4, 4) @ Trp
                    pf = i in pt.get("prep_failed", []); nocand = len(b["candidates"]) == 0 or not np.isfinite(b["optimal_score"])
                    row = {"method": "stsl", "f": f, "ref": ref, "k": i, "run": j, "output": not (pf or nocand)}
                    if row["output"]:
                        es = [ev(T3(c["tf"]), PA, ref) for c in b["candidates"]]; ib, eb = best(es); fin = ev(T3(b["tf"]), PA, ref)
                        row.update(b_cand=dict(eb, idx=ib), final=fin, l={"O": fin["eO"] - eb["eO"], "OB": fin["eOB"] - eb["eOB"], "OS": fin["eOS"] - eb["eOS"]},
                                   n_cand=len(es), cls=("success" if fin["ok"] else ("other_chosen" if any(e["ok"] for e in es) else "no_correct")))
                    else:
                        row["cls"] = "prep_failed" if pf else "no_candidate"
                    rows.append(row)
    return rows


def main():
    os.makedirs(DIAG, exist_ok=True)
    rows = trials()
    json.dump({"provenance": provenance(), "rows": rows}, open(os.path.expanduser("~/rizu/r63_data/trialsA%s.json" % SFX), "w"), default=float)
    S = {"provenance": provenance(), "check_max": max((r.get("chk") or 0) for r in rows if r["method"] != "stsl"),
         "check_over": sum((r.get("chk") or 0) > 1e-6 for r in rows if r["method"] != "stsl")}
    sel = lambda m, f=None, ref=None: [r for r in rows if r["method"] == m and (f is None or r["f"] == f) and (ref is None or r["ref"] == ref) and r.get("run") in (None, 1)]
    md = []
    # 表 2
    md.append("### 表 2 基準内・出力なし・段階の分類（条件 A の本手法：作る / 短い ICP / 絞り込み / 選ぶ / 成功、各 10）\n")
    md.append("| 方式 | f | S1_small | S1_large | S3_small | S3_large |\n|---|---|---|---|---|---|")
    S["table2"] = {}
    for m in ("full", "score_only", "icp_only"):
        for f in FS:
            if not sel(m, f):
                continue
            cells = []
            for ref in REFS:
                c = Counter(r["class"] for r in sel(m, f, ref)); S["table2"]["%s|%s|%s" % (m, f, ref)] = dict(c)
                cells.append("/".join(str(c.get(x, 0)) for x in ("make", "short", "narrow", "select", "success")))
            md.append("| %s | %s | %s |" % (m, f.replace("p", "."), " | ".join(cells)))
    # 表 3
    md.append("\n### 表 3 段階ごとの b と最終の l（Ω、m。中央値〔25%, 75%〕/ 90% / 95%）と、Ω_BIM の b(after)・l\n")
    md.append("| 方式 | f | 参照 | b gen | b short | b after | l_Ω | l_Ω > 0.5 m の数 | b_after（Ω_BIM） | l（Ω_BIM） |\n|---|---|---|---|---|---|---|---|---|---|")
    c3 = lambda d: "—" if not d else "%.3f〔%.3f, %.3f〕/ %.3f / %.3f" % (d["median"], d["q25"], d["q75"], d["p90"], d["p95"])
    S["table3"] = {}
    for m in ("full", "score_only", "icp_only"):
        for f in FS:
            for ref in REFS:
                x = sel(m, f, ref)
                if not x:
                    continue
                e = {st: q([r["b_" + st]["eO"] for r in x]) for st in ("gen", "short", "after")}
                e["l"] = q([r["l"]["O"] for r in x]); e["n_l_over"] = sum(r["l"]["O"] > 0.5 for r in x)
                e["b_after_OB"] = q([r["b_after"]["eOB"] for r in x]); e["l_OB"] = q([r["l"]["OB"] for r in x])
                S["table3"]["%s|%s|%s" % (m, f, ref)] = e
                md.append("| %s | %s | %s | %s | %s | %s | %s | %d | %s | %s |" % (m, f.replace("p", "."), ref, c3(e["gen"]), c3(e["short"]), c3(e["after"]), c3(e["l"]), e["n_l_over"], c3(e["b_after_OB"]), c3(e["l_OB"])))
    # 表 4
    md.append("\n### 表 4 変種と完全の対の差（同じ f・参照・k、Ω、m。中央値〔25%, 75%〕）と l_Ω > 0.5 m の数（変種 / 完全）\n")
    md.append("| 変種 | f | 参照 | Δb（after） | Δl | l > 0.5 の数 |\n|---|---|---|---|---|---|")
    full = {(r["f"], r["ref"], r["k"]): r for r in sel("full")}
    S["table4"] = {}
    for m in ("score_only", "icp_only"):
        for f in ("1p0", "0p6"):
            for ref in REFS:
                x = sel(m, f, ref)
                db = [r["b_after"]["eO"] - full[(f, ref, r["k"])]["b_after"]["eO"] for r in x]; dl = [r["l"]["O"] - full[(f, ref, r["k"])]["l"]["O"] for r in x]
                nv, nf = sum(r["l"]["O"] > 0.5 for r in x), sum(full[(f, ref, r["k"])]["l"]["O"] > 0.5 for r in x)
                S["table4"]["%s|%s|%s" % (m, f, ref)] = {"db": q(db), "dl": q(dl), "n_over": [nv, nf]}
                c = lambda d: "—" if not d else "%.3f〔%.3f, %.3f〕" % (d["median"], d["q25"], d["q75"])
                md.append("| %s | %s | %s | %s | %s | %d / %d |" % (m, f.replace("p", "."), ref, c(q(db)), c(q(dl)), nv, nf))
    # 表 5
    md.append("\n### 表 5 参照の小 → 大の対の差（大 − 小、同じ f・k、最終、m。中央値〔25%, 75%〕、正 / 負）\n")
    md.append("| 方式 | f | 組 | E_Ω の差 | 正 / 負 | E_Ω小 の差 | 正 / 負 |\n|---|---|---|---|---|---|---|")
    S["table5"] = {}
    for m in ("full", "stsl"):
        for f in (FS if m == "full" else ["1p0", "0p6"]):
            for s in ("S1", "S3"):
                a1 = {r["k"]: r for r in sel(m, f, s + "_small") if r["output"]}; a2 = {r["k"]: r for r in sel(m, f, s + "_large") if r["output"]}
                ks = sorted(set(a1) & set(a2))
                d1 = [a2[k]["final"]["eO"] - a1[k]["final"]["eO"] for k in ks]; d2 = [a2[k]["final"]["eOS"] - a1[k]["final"]["eOS"] for k in ks]
                d1 = [v for v in d1 if math.isfinite(v)]; d2 = [v for v in d2 if math.isfinite(v)]
                S["table5"]["%s|%s|%s" % (m, f, s)] = {"dO": q(d1), "dOS": q(d2), "n_pairs": len(ks)}
                c = lambda d: "—" if not d else "%.3f〔%.3f, %.3f〕" % (d["median"], d["q25"], d["q75"])
                md.append("| %s | %s | %s | %s | %d / %d | %s | %d / %d |" % (m, f.replace("p", "."), s, c(q(d1)), sum(v > 0 for v in d1), sum(v < 0 for v in d1),
                                                                         c(q(d2)), sum(v > 0 for v in d2), sum(v < 0 for v in d2)))
    # 表 6（STSL）
    md.append("\n### 表 6 STSL：l = 0 の割合と分類（基準内の候補なし / 別を選んだ / 成功 / 出力なし）\n")
    md.append("| f | 参照 | k=1 の分類 | k=1 の l=0 の割合 | 5 回すべての分類 | 5 回すべての l=0 の割合 |\n|---|---|---|---|---|---|")
    S["table6"] = {}
    for f in ("1p0", "0p6"):
        for ref in REFS:
            k1 = [r for r in rows if r["method"] == "stsl" and r["f"] == f and r["ref"] == ref and r["run"] == 1]
            a5 = [r for r in rows if r["method"] == "stsl" and r["f"] == f and r["ref"] == ref]
            def cls(x):
                c = Counter(r["cls"] for r in x); return "%d / %d / %d / %d" % (c["no_correct"], c["other_chosen"], c["success"], c["no_candidate"] + c["prep_failed"])
            def l0(x):
                o = [r for r in x if r["output"]]; return (sum(abs(r["l"]["O"]) < 1e-12 for r in o) / len(o)) if o else None
            S["table6"]["%s|%s" % (f, ref)] = {"k1": dict(Counter(r["cls"] for r in k1)), "all5": dict(Counter(r["cls"] for r in a5)), "l0_k1": l0(k1), "l0_all5": l0(a5)}
            md.append("| %s | %s | %s | %.2f | %s | %.2f |" % (f.replace("p", "."), ref, cls(k1), l0(k1), cls(a5), l0(a5)))
    # C1〜C5
    C = {}
    g = {ref: (q([r["b_gen"]["eO"] for r in sel("full", "0p2", ref)]), q([r["b_gen"]["eO"] for r in sel("full", "1p0", ref)])) for ref in REFS}
    C["C1"] = {"per_ref": {r: [a["median"] if a else None, b["median"] if b else None] for r, (a, b) in g.items()},
               "n": sum(bool(a and b and a["median"] > b["median"]) for a, b in g.values())}
    C2 = {}
    for ref in REFS:
        so = sum(r["l"]["O"] > 0.5 for f in ("1p0", "0p6") for r in sel("score_only", f, ref)); fu = sum(r["l"]["O"] > 0.5 for f in ("1p0", "0p6") for r in sel("full", f, ref))
        C2[ref] = [so, fu]
    C["C2"] = {"per_ref_score_only_vs_full": C2, "n": sum(a > b for a, b in C2.values())}
    C3 = {}
    for f in ("1p0", "0p6"):
        for ref in REFS:
            C3["%s|%s" % (f, ref)] = [sum(r["final"]["ok"] for r in sel("icp_only", f, ref)), sum(r["final"]["ok"] for r in sel("full", f, ref))]
    C["C3"] = {"per_group_icp_vs_full": C3, "n": sum(abs(a - b) <= 2 for a, b in C3.values())}
    C4 = {ref: [sum(r["l"]["O"] > 0.5 for r in sel("full", "0p6", ref)), sum(r["l"]["O"] > 0.5 for r in sel("full", "1p0", ref))] for ref in REFS}
    C["C4"] = {"per_ref_f06_vs_f10": C4, "n": sum(a > b for a, b in C4.values())}
    fails = [r for r in rows if r["method"] == "stsl" and r["f"] == "0p6" and r["run"] == 1 and r["cls"] != "success"]
    C["C5"] = {"n_fail": len(fails), "n_absent": sum(r["cls"] in ("no_correct", "no_candidate", "prep_failed") for r in fails)}
    C["C5"]["frac"] = C["C5"]["n_absent"] / max(C["C5"]["n_fail"], 1)
    S["C"] = C
    S["time_full_median"] = float(np.median([r["time_s"] for r in rows if r["method"] == "full"]))
    S["time_by_method"] = {m: float(np.median([r["time_s"] for r in rows if r["method"] == m])) for m in ("full", "score_only", "icp_only") if any(r["method"] == m for r in rows)}
    json.dump(S, open(os.path.join(DIAG, "summaryA%s.json" % SFX), "w"), ensure_ascii=False, indent=1, default=float)
    open(os.path.join(DIAG, "tablesA%s.md" % SFX), "w").write("\n".join(md) + "\n")
    # 図
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"S1_small": "#D55E00", "S1_large": "#E69F00", "S3_small": "#0072B2", "S3_large": "#56B4E9"}
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for j, st in enumerate(("gen", "short", "after")):
        for ref in REFS:
            m = [np.median([r["b_" + st]["eO"] for r in sel("full", f, ref)]) if sel("full", f, ref) else np.nan for f in FS]
            ax[j].plot([0.2, 0.4, 0.6, 0.8, 1.0], m, "o-", c=col[ref], label=ref)
        ax[j].set_yscale("log"); ax[j].set_title("condition A, full: median b at %s (E_Omega)" % st); ax[j].set_xlabel("f"); ax[j].grid(alpha=0.3)
    ax[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(DIAG, "b_by_stage_A%s.png" % SFX), dpi=100); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for j, key in enumerate(("eO", "eOS")):
        for xi, f in enumerate(FS):
            for si, s in enumerate(("S1", "S3")):
                a1 = {r["k"]: r for r in sel("full", f, s + "_small")}; a2 = {r["k"]: r for r in sel("full", f, s + "_large")}
                v = [a2[k]["final"][key] - a1[k]["final"][key] for k in sorted(set(a1) & set(a2))]
                ax[j].scatter(np.full(len(v), xi + (si - 0.5) * 0.15), v, s=14, c=["#D55E00", "#0072B2"][si], label=s if xi == 0 else None)
        ax[j].axhline(0, c="k", lw=0.5); ax[j].set_xticks(range(5)); ax[j].set_xticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"]); ax[j].set_yscale("symlog", linthresh=0.1)
        ax[j].set_title("condition A, full: large - small, final %s" % ("E_Omega" if key == "eO" else "E_Omega_small")); ax[j].set_xlabel("f")
    ax[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(DIAG, "pairs_small_large_A%s.png" % SFX), dpi=100); plt.close(fig)
    print(json.dumps(S["C"], ensure_ascii=False), S["check_max"], S["check_over"], S["time_by_method"])


if __name__ == "__main__":
    main()

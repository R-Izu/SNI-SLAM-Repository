"""R60 の集計（sni-slam の環境、ブランチ `exp/r60`）。**回す前に commit する。** 指示書 §3-2〜§3-4・§2。報告の表をこのスクリプトで書き出す。

- 分布：中央値・25%・75%・90%・95%（`np.percentile` の既定）。出力なしの試行は数に入れず、件数を別に書く。+∞（無効な候補しかない段階）は件数を別に書いて分布から外す
- スキャンごと＝そのスキャンの E2 と E3 を合わせた試行（観測・propA・STSL は各 20、STSL の 5 回すべては各 100）。合成はシーンごと
- 表 3（クラス）：同じ試行（data・target・k）の対で、Δ = その方式 − full
- 表 4：short − gen、after − short（本手法の群）
- 表 5：R54 の段階の分類（主の線）ごとの、gen・short・after の b（観測の群と cls の full）
- §3-3：同じ scan・f・k・方式（STSL は回）の E3 − E2。最終の E_Ω・E_Ω411 と、本手法の after の b（Ω）。正・負・0（|差| < 1e-12）の数
- §3-4：観測の群、scan × f × ref ごとに、10 摂動の gen の b と最終の E_Ω の中央値・25%・75%、横軸 r
- L1：観測の群、スキャンごと（E2・E3 を合わせた 20 試行）の gen の b（Ω）の中央値を f 0.2 と f 1.0 で比べ、f 0.2 が大きいスキャンの数
- L2：cls の score_only − full の Δl（Ω）の、スキャンごと（実データ、20 対）・シーンごと（合成、100 対）の中央値が正の数
- L3：§3-3 の最終の E_Ω411 の差（E3 − E2）の、f × 方式ごとの、スキャンごとの中央値の符号。そろうスキャンが 4 のうち 3 以上なら、その向き
出力：`Registration/output/diag/r60/r60_summary.json`、`tables.md`、図 `pairs_<方式>.png`・`r_vs_b.png`
"""
import json, math, os, sys
from collections import Counter
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from failure_decomposition import provenance                    # noqa: E402

DIAG = "Registration/output/diag/r60"
SCANS = ["m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d"]


def qd(v):
    v = [x for x in v if x is not None]
    fin = [x for x in v if math.isfinite(x)]
    if not fin:
        return {"n": 0, "n_inf": len(v)}
    return {"n": len(fin), "n_inf": len(v) - len(fin), "median": float(np.median(fin)), "q25": float(np.percentile(fin, 25)),
            "q75": float(np.percentile(fin, 75)), "p90": float(np.percentile(fin, 90)), "p95": float(np.percentile(fin, 95))}


def fm(d, k="median"):
    return "—" if not d.get("n") else ("%.3f" % d[k])


def main():
    os.makedirs(DIAG, exist_ok=True)
    rows = json.load(open(os.path.expanduser("~/rizu/r60_data/trials.json")))["rows"]
    S = {"provenance": provenance(), "n_rows": len(rows)}
    S["check"] = {"eO_n": sum(r.get("chk_eO") is not None for r in rows), "eO_over": sum((r.get("chk_eO") or 0) > 1e-6 for r in rows),
                  "eO_max": max((r.get("chk_eO") or 0) for r in rows), "eO411_n": sum(r.get("chk_eO411") is not None for r in rows),
                  "eO411_over": sum((r.get("chk_eO411") or 0) > 1e-6 for r in rows)}
    om = np.load("Registration/output/diag/r39_omega_bim.npz")
    S["omega411_n_points"] = {s: int(len(om["%s__E2" % s])) for s in SCANS}
    md = []
    def G(sel):
        return [r for r in rows if sel(r)]
    groups = []
    for f in ("0p2", "0p4", "0p6", "0p8", "1p0"):
        groups.append(("obs f%s" % f, lambda r, f=f: r["group"] == "obs" and r["f"] == f, ["gen", "short", "after"]))
    for m in ("full", "score_only", "icp_only", "geo"):
        groups.append(("cls %s real" % m, lambda r, m=m: r["group"] == "cls" and r["method"] == m and r["data"] == "real", ["gen", "short", "after"]))
        groups.append(("cls %s synth" % m, lambda r, m=m: r["group"] == "cls" and r["method"] == m and r["data"] == "synth", ["gen", "short", "after"]))
    for f in ("1p0", "0p6"):
        groups.append(("propA f%s" % f, lambda r, f=f: r["group"] == "propA" and r["f"] == f, ["gen", "short", "after"]))
        groups.append(("stsl k1 f%s" % f, lambda r, f=f: r["group"] == "stsl" and r["f"] == f and r["run"] == 1, ["cand"]))
        groups.append(("stsl all5 f%s" % f, lambda r, f=f: r["group"] == "stsl" and r["f"] == f, ["cand"]))
    # 表 1・2
    S["b"] = {}; S["l"] = {}
    md.append("### 表 1 段階ごとの最良誤差 b（m）。全体：中央値〔25%, 75%〕/ 90% / 95%、スキャンごと（合成は全体だけ）：中央値。Ω・Ω_411\n")
    md.append("| 群 | 段階 | n（出力なし） | Ω 全体 | Ω cor_a / b / c / d | Ω_411 全体 | Ω_411 cor_a / b / c / d |\n|---|---|---|---|---|---|---|")
    for name, sel, stages in groups:
        rs = G(sel); out = [r for r in rs if r["output"]]
        for st in stages:
            e = {"all": {"O": qd([r["b_" + st]["EO"] for r in out]), "O411": qd([r["b_" + st]["EO411"] for r in out if r["data"] == "real"])}}
            for s in SCANS:
                x = [r for r in out if r["scan"] == s]
                e[s] = {"O": qd([r["b_" + st]["EO"] for r in x]), "O411": qd([r["b_" + st]["EO411"] for r in x])}
            S["b"]["%s|%s" % (name, st)] = e
            a, b2 = e["all"]["O"], e["all"]["O411"]
            cell = lambda d: "—" if not d.get("n") else "%.3f〔%.3f, %.3f〕/ %.3f / %.3f" % (d["median"], d["q25"], d["q75"], d["p90"], d["p95"])
            real = "synth" not in name
            md.append("| %s | %s | %d（%d） | %s | %s | %s | %s |" % (name, st, len(out), len(rs) - len(out), cell(a),
                      " / ".join(fm(e[s]["O"]) for s in SCANS) if real else "—", cell(b2) if real else "—",
                      " / ".join(fm(e[s]["O411"]) for s in SCANS) if real else "—"))
    md.append("\n### 表 2 最終の l（m）。全体：中央値〔25%, 75%〕/ 90% / 95%、l = 0 の数（|l| < 1e-12）、スキャンごと：中央値\n")
    md.append("| 群 | n | l_Ω 全体 | l_Ω = 0 | l_Ω cor_a / b / c / d | l_Ω411 全体 | l_Ω411 cor_a / b / c / d |\n|---|---|---|---|---|---|---|")
    for name, sel, stages in groups:
        out = [r for r in G(sel) if r["output"]]
        lo = [r["l"]["O"] for r in out]; l4 = [r["l"]["O411"] for r in out if r["data"] == "real"]
        e = {"all": {"O": qd(lo), "O411": qd(l4), "n_zero": sum(abs(x) < 1e-12 for x in lo if math.isfinite(x))}}
        for s in SCANS:
            x = [r for r in out if r["scan"] == s]
            e[s] = {"O": qd([r["l"]["O"] for r in x]), "O411": qd([r["l"]["O411"] for r in x])}
        S["l"][name] = e
        cell = lambda d: "—" if not d.get("n") else "%.3f〔%.3f, %.3f〕/ %.3f / %.3f" % (d["median"], d["q25"], d["q75"], d["p90"], d["p95"])
        real = "synth" not in name
        md.append("| %s | %d | %s | %d | %s | %s | %s |" % (name, len(out), cell(e["all"]["O"]), e["all"]["n_zero"],
                  " / ".join(fm(e[s]["O"]) for s in SCANS) if real else "—", cell(e["all"]["O411"]) if real else "—",
                  " / ".join(fm(e[s]["O411"]) for s in SCANS) if real else "—"))
    # 表 3
    md.append("\n### 表 3 クラスの群の対の差（その方式 − full、同じ試行、Ω、m）。中央値〔25%, 75%〕\n")
    md.append("| 方式 | データ | 単位 | Δb（after） | Δl |\n|---|---|---|---|---|")
    full = {(r["data"], r["target"], r["k"]): r for r in rows if r["group"] == "cls" and r["method"] == "full"}
    S["cls_pairs"] = {}
    for m in ("score_only", "icp_only", "geo"):
        for data in ("real", "synth"):
            rs = [r for r in rows if r["group"] == "cls" and r["method"] == m and r["data"] == data]
            units = SCANS if data == "real" else sorted({r["target"] for r in rs})
            for u in ["all"] + units:
                x = [r for r in rs if u == "all" or r["scan"] == u]
                db = [r["b_after"]["EO"] - full[(data, r["target"], r["k"])]["b_after"]["EO"] for r in x]
                dl = [r["l"]["O"] - full[(data, r["target"], r["k"])]["l"]["O"] for r in x]
                db = [v for v in db if math.isfinite(v)]; dl = [v for v in dl if math.isfinite(v)]
                e = {"db": qd(db), "dl": qd(dl)}
                S["cls_pairs"]["%s|%s|%s" % (m, data, u)] = e
                c = lambda d: "—" if not d.get("n") else "%.4f〔%.4f, %.4f〕（n %d）" % (d["median"], d["q25"], d["q75"], d["n"])
                md.append("| %s | %s | %s | %s | %s |" % (m, data, u, c(e["db"]), c(e["dl"])))
    # 表 4
    md.append("\n### 表 4 段階の間の b の変化（Ω、m、負＝小さくなった）。全体：中央値〔25%, 75%〕/ 90% / 95%、負の数 / n\n")
    md.append("| 群 | short − gen | 負 / n | after − short（絞り込みと通常の ICP の後の変化） | 負 / n |\n|---|---|---|---|---|")
    S["stage_change"] = {}
    for name, sel, stages in groups:
        if stages != ["gen", "short", "after"]:
            continue
        out = [r for r in G(sel) if r["output"]]
        d1 = [r["b_short"]["EO"] - r["b_gen"]["EO"] for r in out]; d2 = [r["b_after"]["EO"] - r["b_short"]["EO"] for r in out]
        d1 = [v for v in d1 if math.isfinite(v)]; d2 = [v for v in d2 if math.isfinite(v)]
        S["stage_change"][name] = {"short_minus_gen": qd(d1), "after_minus_short": qd(d2), "neg1": sum(v < 0 for v in d1), "neg2": sum(v < 0 for v in d2)}
        c = lambda d: "—" if not d.get("n") else "%.3f〔%.3f, %.3f〕/ %.3f / %.3f" % (d["median"], d["q25"], d["q75"], d["p90"], d["p95"])
        md.append("| %s | %s | %d / %d | %s | %d / %d |" % (name, c(qd(d1)), sum(v < 0 for v in d1), len(d1), c(qd(d2)), sum(v < 0 for v in d2), len(d2)))
    # 表 5
    md.append("\n### 表 5 R54 の段階の分類（主の線）ごとの b（Ω、m）。中央値〔25%, 75%〕（n）\n")
    md.append("| 群 | R54 の分類 | gen | short | after |\n|---|---|---|---|---|")
    S["by_r54_class"] = {}
    for name, sel in (("obs（全 f）", lambda r: r["group"] == "obs"), ("cls full real", lambda r: r["group"] == "cls" and r["method"] == "full" and r["data"] == "real"),
                      ("cls full synth", lambda r: r["group"] == "cls" and r["method"] == "full" and r["data"] == "synth"),
                      ("cls score_only real", lambda r: r["group"] == "cls" and r["method"] == "score_only" and r["data"] == "real"),
                      ("cls score_only synth", lambda r: r["group"] == "cls" and r["method"] == "score_only" and r["data"] == "synth")):
        rs = [r for r in rows if sel(r) and r.get("r54_class")]
        for cl in ("make", "short", "narrow", "select", "success"):
            x = [r for r in rs if r["r54_class"] == cl]
            if not x:
                continue
            e = {st: qd([r["b_" + st]["EO"] for r in x]) for st in ("gen", "short", "after")}
            S["by_r54_class"]["%s|%s" % (name, cl)] = e
            c = lambda d: "—" if not d.get("n") else "%.3f〔%.3f, %.3f〕（%d）" % (d["median"], d["q25"], d["q75"], d["n"])
            md.append("| %s | %s | %s | %s | %s |" % (name, cl, c(e["gen"]), c(e["short"]), c(e["after"])))
    # §3-3
    md.append("\n### 表 6 参照の対の差（E3 − E2、m）。中央値〔25%, 75%〕、正 / 負 / 0、スキャンごとの中央値。片方だけ出力なしの対の数\n")
    md.append("| 方式 | f | 量 | 全体 | 正 / 負 / 0 | cor_a / b / c / d | E2 だけ出力なし / E3 だけ出力なし |\n|---|---|---|---|---|---|---|")
    S["ref_pairs"] = {}
    meths = [("prop（R49/R52）", lambda r: r["group"] == "obs", ("0p2", "0p4", "0p6", "0p8", "1p0")),
             ("prop 条件 A（R55）", lambda r: r["group"] == "propA", ("1p0", "0p6")),
             ("STSL k1", lambda r: r["group"] == "stsl" and r["run"] == 1, ("1p0", "0p6"))]
    for mname, sel, fs in meths:
        idx = {(r["scan"], r["f"], r["k"], r["ref"]): r for r in rows if sel(r)}
        for f in fs:
            qs = [("final E_Ω", lambda r: r["final"]["EO"]), ("final E_Ω411", lambda r: r["final"]["EO411"])]
            if mname != "STSL k1":
                qs.append(("after の b（Ω）", lambda r: r["b_after"]["EO"]))
            for qn, fn in qs:
                d, by, only2, only3 = [], {s: [] for s in SCANS}, 0, 0
                for s in SCANS:
                    for k in range(10):
                        a2, a3 = idx.get((s, f, k, "E2")), idx.get((s, f, k, "E3"))
                        if not a2 or not a3:
                            continue
                        if not a2["output"] or not a3["output"]:
                            only2 += (not a2["output"]) and a3["output"]; only3 += (not a3["output"]) and a2["output"]; continue
                        v = fn(a3) - fn(a2)
                        if math.isfinite(v):
                            d.append(v); by[s].append(v)
                e = {"all": qd(d), "pos": sum(v > 1e-12 for v in d), "neg": sum(v < -1e-12 for v in d), "zero": sum(abs(v) <= 1e-12 for v in d),
                     "by_scan": {s: qd(v) for s, v in by.items()}, "E2_only_no_output": only2, "E3_only_no_output": only3, "values_by_scan": by}
                S["ref_pairs"]["%s|%s|%s" % (mname, f, qn)] = e
                c = "—" if not e["all"].get("n") else "%.3f〔%.3f, %.3f〕" % (e["all"]["median"], e["all"]["q25"], e["all"]["q75"])
                md.append("| %s | %s | %s | %s | %d / %d / %d | %s | %d / %d |" % (mname, f.replace("p", "."), qn, c, e["pos"], e["neg"], e["zero"],
                          " / ".join(fm(e["by_scan"][s]) for s in SCANS), only2, only3))
    # §3-4
    inp = [r for r in rows if r["group"] == "obs"]
    pts = []
    for s in SCANS:
        for f in ("0p2", "0p4", "0p6", "0p8", "1p0"):
            for ref in ("E2", "E3"):
                x = [r for r in inp if r["scan"] == s and r["f"] == f and r["ref"] == ref]
                if not x:
                    continue
                bg = [r["b_gen"]["EO"] for r in x]; fe = [r["final"]["EO"] for r in x]
                pts.append({"scan": s, "f": f, "ref": ref, "r": x[0]["r"], "b_gen": qd(bg), "final": qd(fe)})
    S["r_vs_b"] = pts
    md.append("\n### 表 7 参照の床の外形の内側の点の割合 r と、gen の b・最終の E_Ω（Ω、m、10 摂動の中央値〔25%, 75%〕）\n")
    md.append("| スキャン | f | 参照 | r | gen の b | 最終の E_Ω |\n|---|---|---|---|---|---|")
    for p in pts:
        c = lambda d: "—" if not d.get("n") else "%.3f〔%.3f, %.3f〕" % (d["median"], d["q25"], d["q75"])
        md.append("| %s | %s | %s | %.3f | %s | %s |" % (p["scan"], p["f"].replace("p", "."), p["ref"], p["r"], c(p["b_gen"]), c(p["final"])))
    # L1〜L3
    L = {}
    def med_scan(sel, key):
        return {s: float(np.median([key(r) for r in rows if sel(r) and r["scan"] == s and math.isfinite(key(r))])) for s in SCANS}
    g02 = med_scan(lambda r: r["group"] == "obs" and r["f"] == "0p2", lambda r: r["b_gen"]["EO"])
    g10 = med_scan(lambda r: r["group"] == "obs" and r["f"] == "1p0", lambda r: r["b_gen"]["EO"])
    L["L1"] = {"f0p2": g02, "f1p0": g10, "n_scans_f02_larger": sum(g02[s] > g10[s] for s in SCANS)}
    L["L2"] = {"real_scan_median": {s: S["cls_pairs"]["score_only|real|%s" % s]["dl"].get("median") for s in SCANS},
               "synth_scene_median": {k.split("|")[2]: v["dl"].get("median") for k, v in S["cls_pairs"].items() if k.startswith("score_only|synth|") and not k.endswith("|all")}}
    L["L2"]["n_real_pos"] = sum((v or 0) > 0 for v in L["L2"]["real_scan_median"].values())
    L["L2"]["n_synth_pos"] = sum((v or 0) > 0 for v in L["L2"]["synth_scene_median"].values())
    L["L3"] = {}
    for k, e in S["ref_pairs"].items():
        if k.endswith("final E_Ω411"):
            sg = {s: (np.sign(e["by_scan"][s]["median"]) if e["by_scan"][s].get("n") else 0) for s in SCANS}
            npos, nneg = sum(v > 0 for v in sg.values()), sum(v < 0 for v in sg.values())
            L["L3"][k] = {"sign_by_scan": {s: int(v) for s, v in sg.items()}, "n_pos": npos, "n_neg": nneg}
    S["L"] = L
    for e in S["ref_pairs"].values():
        e.pop("values_by_scan", None)
    json.dump(S, open(os.path.join(DIAG, "r60_summary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    open(os.path.join(DIAG, "tables.md"), "w").write("\n".join(md) + "\n")
    # 図
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"m3_cor_a": "#0072B2", "m3_cor_b": "#E69F00", "m3_cor_c": "#009E73", "m3_cor_d": "#CC79A7"}
    for mname, sel, fs in meths:
        idx = {(r["scan"], r["f"], r["k"], r["ref"]): r for r in rows if sel(r)}
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for j, (qn, fn) in enumerate((("final E_Omega", lambda r: r["final"]["EO"]), ("final E_Omega411", lambda r: r["final"]["EO411"]))):
            for xi, f in enumerate(fs):
                for si, s in enumerate(SCANS):
                    v = []
                    for k in range(10):
                        a2, a3 = idx.get((s, f, k, "E2")), idx.get((s, f, k, "E3"))
                        if a2 and a3 and a2["output"] and a3["output"]:
                            v.append(fn(a3) - fn(a2))
                    ax[j].scatter(np.full(len(v), xi + (si - 1.5) * 0.12), v, s=12, c=col[s], label=s if xi == 0 else None)
            ax[j].axhline(0, color="k", lw=0.5); ax[j].set_xticks(range(len(fs))); ax[j].set_xticklabels([f.replace("p", ".") for f in fs])
            ax[j].set_yscale("symlog", linthresh=0.1); ax[j].set_title("%s: E3 - E2, %s" % (mname.encode("ascii", "ignore").decode() or "method", qn)); ax[j].set_xlabel("f")
        ax[0].legend(); fig.tight_layout()
        tag = {"prop（R49/R52）": "prop", "prop 条件 A（R55）": "propA", "STSL k1": "stsl"}[mname]
        fig.savefig(os.path.join(DIAG, "pairs_%s.png" % tag), dpi=100); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for j, key in enumerate(("b_gen", "final")):
        for p in pts:
            d = p[key]
            if d.get("n"):
                ax[j].errorbar(p["r"], d["median"], yerr=[[d["median"] - d["q25"]], [d["q75"] - d["median"]]], fmt="o" if p["ref"] == "E2" else "s",
                               c=col[p["scan"]], ms=5, capsize=2)
        ax[j].set_yscale("log"); ax[j].set_xlabel("r (fraction of input points inside the reference floor outline)")
        ax[j].set_ylabel({"b_gen": "b at generation, E_Omega [m]", "final": "final E_Omega [m]"}[key]); ax[j].grid(alpha=0.3)
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", c=c, label=s) for s, c in col.items()] +
                 [Line2D([], [], marker="o", ls="", c="k", label="E2"), Line2D([], [], marker="s", ls="", c="k", label="E3")])
    fig.tight_layout(); fig.savefig(os.path.join(DIAG, "r_vs_b.png"), dpi=100); plt.close(fig)
    print(json.dumps(S["check"]), json.dumps(S["L"], default=float)[:3000])


if __name__ == "__main__":
    main()

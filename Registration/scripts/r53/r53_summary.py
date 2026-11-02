"""R53 の集計（sni-slam の環境、ブランチ `exp/r53`）。**回す前に commit する。** 指示書 §3-4〜§3-6。判定の言葉は出さない。

- 線（§3-4）：陰性対照のすべて（データ × f の群を合わせる）の |ΔC|・|ΔV| の 95 パーセント点を c_line・v_line、S の 5 パーセント点を s_line
  （`np.percentile` の既定の線形補間）。群ごとの分布も出す
- 型（§3-5）：III＝ΔC > c_line または ΔV > v_line、I＝III でなく S ≥ s_line、II＝III でなく S < s_line
- 陽性対照の型の件数と割合（Wilson 95%）。本体：群 × データ × f、実データのスキャンごと。型ごとの ΔC・ΔV・S・Δρ の分布。型 III の一覧
- 検算：記録の ρ（小数 5 桁）と再計算の ρ の差の最大、16 候補の score_short との差の最大（R52 の f < 1 のみ記録あり）
入力：`~/rizu/r53_data/controls.json`・`main.json`。出力：`Registration/output/diag/r53/r53_summary.json`
"""
import json, os, sys
from collections import Counter
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from failure_decomposition import provenance                    # noqa: E402

OUT = os.path.expanduser("~/rizu/r53_data")
DIAG = "Registration/output/diag/r53"


def q(v):
    v = [x for x in v if x is not None]
    return None if not v else {"n": len(v), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                               "min": float(np.min(v)), "max": float(np.max(v))}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def typ(r, L):
    if r["dC"] > L["c_line"] or (r["dV"] is not None and r["dV"] > L["v_line"]):
        return "III"
    return "I" if (r["S"] is not None and r["S"] >= L["s_line"]) else "II"


def main():
    ctl = json.load(open(os.path.join(OUT, "controls.json")))
    neg = [r for r in ctl["rows"] if r["kind"] == "neg"]; pos = [r for r in ctl["rows"] if r["kind"] == "pos"]
    L = {"c_line": float(np.percentile([abs(r["dC"]) for r in neg], 95)),
         "v_line": float(np.percentile([abs(r["dV"]) for r in neg if r["dV"] is not None], 95)),
         "s_line": float(np.percentile([r["S"] for r in neg if r["S"] is not None], 5)), "n_neg": len(neg)}
    out = {"provenance": provenance(), "controls_provenance": ctl["provenance"], "lines": L}
    key = lambda r: "%s|%s" % (r["data"], r["f"])
    out["neg_dist"] = {"all": {"absdC": q([abs(r["dC"]) for r in neg]), "absdV": q([abs(r["dV"]) for r in neg if r["dV"] is not None]),
                               "S": q([r["S"] for r in neg])}}
    for g in sorted({key(r) for r in neg}):
        rs = [r for r in neg if key(r) == g]
        out["neg_dist"][g] = {"absdC": q([abs(r["dC"]) for r in rs]), "absdV": q([abs(r["dV"]) for r in rs if r["dV"] is not None]),
                              "S": q([r["S"] for r in rs])}
    pc = Counter(typ(r, L) for r in pos)
    out["pos"] = {t: {"k": pc.get(t, 0), "frac": pc.get(t, 0) / len(pos), "wilson95": wilson(pc.get(t, 0), len(pos))} for t in ("I", "II", "III")}
    out["pos"]["n"] = len(pos)
    out["pos_dist"] = {"dC": q([r["dC"] for r in pos]), "dV": q([r["dV"] for r in pos]), "S": q([r["S"] for r in pos]),
                       "drho": q([r["drho"] for r in pos])}
    out["check"] = {"rho4_max_absdiff_vs_rec": max(r.get("rho4_max_absdiff_vs_rec", 0.0) for r in ctl["rows"])}
    if os.path.exists(os.path.join(OUT, "main.json")):
        mn = json.load(open(os.path.join(OUT, "main.json")))
        rows = mn["rows"]
        for r in rows:
            r["type"] = typ(r, L)
        out["main_provenance"] = mn["provenance"]
        out["check"]["rho4_max_absdiff_vs_rec_main"] = max(r["rho4_max_absdiff_vs_rec"] for r in rows)
        out["check"]["rho16_max_absdiff_vs_rec_main"] = max((r["rho16_max_absdiff_vs_rec"] for r in rows if "rho16_max_absdiff_vs_rec" in r), default=None)
        tab = Counter((r["kind"], r["data"], r["f"], r["type"]) for r in rows)
        out["types"] = {"%s|%s|%s" % k[:3]: {} for k in tab}
        for k, v in tab.items():
            out["types"]["%s|%s|%s" % k[:3]][k[3]] = v
        sc = Counter((r["kind"], r["scan"], r["type"]) for r in rows if r["data"] == "real")
        out["types_by_scan"] = {}
        for k, v in sc.items():
            out["types_by_scan"].setdefault("%s|%s" % k[:2], {})[k[2]] = v
        for g in ("S", "N"):
            rs = [r for r in rows if r["kind"] == g and r["data"] == "real"]
            c = Counter(r["type"] for r in rs)
            out["real_%s_totals" % g] = {"n": len(rs), **{t: c.get(t, 0) for t in ("I", "II", "III")},
                                        "n_scans_with_III": len({r["scan"] for r in rs if r["type"] == "III"})}
        out["dist_by_type"] = {}
        for g in ("S", "N"):
            for t in ("I", "II", "III"):
                rs = [r for r in rows if r["kind"] == g and r["type"] == t]
                out["dist_by_type"]["%s|%s" % (g, t)] = {"n": len(rs), "dC": q([r["dC"] for r in rs]), "dV": q([r["dV"] for r in rs]),
                                                          "S": q([r["S"] for r in rs]), "drho": q([r["drho"] for r in rs])}
        out["type3_list"] = [{k: r[k] for k in ("kind", "data", "f", "scan", "target", "trial", "dC", "dV", "S", "drho", "yaw_deg", "trans_m",
                                                "scale_ratio")} for r in rows if r["type"] == "III"]
        out["rows_compact"] = [{k: r[k] for k in ("kind", "data", "f", "target", "trial", "type", "dC", "dV", "S", "drho", "rho_c", "rho_w",
                                                  "C_c", "C_w", "V_c", "V_w", "n_O", "yaw_deg", "trans_m", "scale_ratio", "out_gt", "sel")}
                               for r in rows]
    os.makedirs(DIAG, exist_ok=True)
    json.dump(out, open(os.path.join(DIAG, "r53_summary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("lines", "pos", "check")}, indent=1))
    for k in ("real_S_totals", "real_N_totals", "types", "types_by_scan"):
        if k in out:
            print(k, json.dumps(out[k], ensure_ascii=False))


if __name__ == "__main__":
    main()

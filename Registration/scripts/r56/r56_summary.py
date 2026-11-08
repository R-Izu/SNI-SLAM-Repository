"""R56 の集計（sni-slam の環境、ブランチ `exp/r56`）。**回す前に commit する。** 指示書 §3-3・§3-5。判定の言葉は出さない。

- 段（stage）：raw（調整前）・N（クラスなし）・S（クラスあり）。方式：prop・stsl（k = 1）・fpfh・fgr。条件 A・B
- **推定なし・無効（v = 0）の試行の誤差は +∞**（§3-5）。件数は別に出す
- 2 値（補助）：主の線 v ∧ e_R < 45° ∧ e_Ω < 1.0 m
- 曲線：達成の割合 F(x) = (1/n) Σ 1[誤差 < x]（R41 と同じ「<」）。e_Ω・e_ΩBIM は x = 0〜3 m・0.01 m 刻み、e_R は 0〜10°・0.1° 刻み
- 対の優越の割合：同じ試行の対で 1[本手法 < 相手] + 0.5·1[等しい]（+∞ どうしは等しい）の平均。スキャンを単位にしたブートストラップ
  （スキャンを 2,000 回復元抽出、seed 0、抽出したスキャンの対をすべて集めた平均の 2.5%・97.5%）
- 対の誤差の差：両方が推定を出した（両方とも有限）対の e_Ω の差（本手法 − 相手）の中央値・25%・75%、中央値の同じブートストラップ区間
- 全試行の要約：中央値・75%・90%。+∞ を含む順位で、逆累積分布の規則（n 個を小さい順に並べた ⌈q n⌉ 番目、＊ 補間しない）
- 入れ替わる値：e_Ω の曲線の上で、D(x) = F_本手法(x) − F_相手(x) の符号（0 を除く）が変わる x（変わった後の最初の格子点）
出力：`Registration/output/diag/r56/r56_summary.json`、曲線 `curves.json`、図 `fig_<指標>_<段>.png`
"""
import json, math, os, sys
from collections import Counter
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from failure_decomposition import provenance                    # noqa: E402

DIAG = "Registration/output/diag/r56"
STAGES = ["raw", "N", "S"]
METHODS = ["prop", "stsl", "fpfh", "fgr"]
GROUPS = [(c, ref, f) for c in ("A", "B") for (ref, f) in (("E2", "1p0"), ("E3", "1p0"), ("E2", "0p6"), ("E3", "0p6"))]
GRID = {"eO": np.round(np.arange(0, 301) * 0.01, 2), "eOB": np.round(np.arange(0, 301) * 0.01, 2), "eR": np.round(np.arange(0, 101) * 0.1, 1)}


def err(r, st, m):
    e = r[st]
    if e is None or not e["v"] or e[m] is None:
        return math.inf
    return float(e[m])


def ok(r, st):
    e = r[st]
    return bool(e is not None and e["v"] and e["eR"] < 45.0 and e["eO"] < 1.0)


def qcdf(v, q):
    v = sorted(v); n = len(v)
    return v[max(0, math.ceil(q * n) - 1)] if n else None


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def boot(by_scan, stat):
    sc = sorted(by_scan); rng = np.random.default_rng(0); out = []
    for _ in range(2000):
        pick = [x for i in rng.integers(0, len(sc), len(sc)) for x in by_scan[sc[i]]]
        out.append(stat(pick))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def fin(x):
    return None if x is None or not math.isfinite(x) else float(x)


def main():
    rows = json.load(open(os.path.expanduser("~/rizu/r56_data/trials.json")))["rows"]
    idx = {(r["method"], r["cond"], r["target"], r["f"], r["k"]): r for r in rows}
    sel = lambda m, c, ref, f: [r for r in rows if (r["method"], r["cond"], r["ref"], r["f"]) == (m, c, ref, f)]
    S = {"provenance": provenance(), "n_rows": len(rows)}
    # 2 値
    S["binary"] = {}; S["binary_by_scan"] = {}; S["binary_pairs"] = {}
    for (c, ref, f) in GROUPS:
        for m in METHODS:
            rs = sel(m, c, ref, f); n = len(rs)
            S["binary"]["%s|%s|%s|%s" % (m, c, ref, f)] = {"n": n, "no_estimate": sum(not r["valid"] for r in rs),
                                                         **{st: {"k": sum(ok(r, st) for r in rs), "wilson95": wilson(sum(ok(r, st) for r in rs), n)} for st in STAGES}}
            for scan in sorted({r["scan"] for r in rs}):
                x = [r for r in rs if r["scan"] == scan]
                S["binary_by_scan"]["%s|%s|%s|%s|%s" % (m, c, ref, f, scan)] = {st: sum(ok(r, st) for r in x) for st in STAGES}
            if m == "prop":
                continue
            for st in STAGES:
                for g in ["all"] + sorted({r["scan"] for r in rs}):
                    cnt = Counter()
                    for r in rs:
                        if g != "all" and r["scan"] != g:
                            continue
                        p = idx[("prop", c, r["target"], f, r["k"])]
                        a, b = ok(p, st), ok(r, st)
                        cnt["both" if a and b else ("prop_only" if a else ("other_only" if b else "neither"))] += 1
                    S["binary_pairs"]["%s|%s|%s|%s|%s|%s" % (st, m, c, ref, f, g)] = {k: cnt.get(k, 0) for k in ("both", "prop_only", "other_only", "neither")}
    # 曲線
    curves = {}
    for (c, ref, f) in GROUPS:
        for m in METHODS:
            rs = sel(m, c, ref, f)
            for st in STAGES:
                for met, g in GRID.items():
                    v = np.array([err(r, st, met) for r in rs])
                    curves["%s|%s|%s|%s|%s|%s" % (met, st, m, c, ref, f)] = [float((v < x).mean()) for x in g]
    json.dump({"grid": {k: v.tolist() for k, v in GRID.items()}, "curves": curves}, open(os.path.join(DIAG, "curves.json"), "w"))
    # 対の優越・対の差・入れ替わり
    S["dominance"] = {}; S["pair_diff"] = {}; S["crossover"] = {}
    for (c, ref, f) in GROUPS:
        for m in ("stsl", "fpfh", "fgr"):
            rs = sel(m, c, ref, f)
            for st in STAGES:
                for met in ("eO", "eOB", "eR"):
                    by = {}
                    for r in rs:
                        p = idx[("prop", c, r["target"], f, r["k"])]
                        a, b = err(p, st, met), err(r, st, met)
                        by.setdefault(r["scan"], []).append(1.0 if a < b else (0.5 if a == b else 0.0))
                    allv = [x for v in by.values() for x in v]
                    S["dominance"]["%s|%s|%s|%s|%s|%s" % (met, st, m, c, ref, f)] = {
                        "n": len(allv), "value": float(np.mean(allv)), "scan_boot95": boot(by, np.mean),
                        "per_scan": {k: float(np.mean(v)) for k, v in sorted(by.items())}}
                by = {}
                for r in rs:
                    p = idx[("prop", c, r["target"], f, r["k"])]
                    a, b = err(p, st, "eO"), err(r, st, "eO")
                    if math.isfinite(a) and math.isfinite(b):
                        by.setdefault(r["scan"], []).append(a - b)
                allv = [x for v in by.values() for x in v]
                S["pair_diff"]["%s|%s|%s|%s|%s" % (st, m, c, ref, f)] = (
                    {"n_pairs": len(allv), "n_scans": len(by), "median": float(np.median(allv)), "p25": float(np.percentile(allv, 25)),
                     "p75": float(np.percentile(allv, 75)), "scan_boot95_median": boot(by, np.median)} if allv else {"n_pairs": 0})
                D = np.array(curves["eO|%s|prop|%s|%s|%s" % (st, c, ref, f)]) - np.array(curves["eO|%s|%s|%s|%s|%s" % (st, m, c, ref, f)])
                xs, last = [], 0
                for x, d in zip(GRID["eO"], D):
                    s = int(np.sign(round(d, 12)))
                    if s != 0:
                        if last != 0 and s != last:
                            xs.append({"x": float(x), "to": "prop_higher" if s > 0 else "other_higher"})
                        last = s
                first = next((int(np.sign(round(d, 12))) for d in D if round(d, 12) != 0), 0)
                S["crossover"]["%s|%s|%s|%s|%s" % (st, m, c, ref, f)] = {"first_sign": {1: "prop_higher", -1: "other_higher", 0: "equal_everywhere"}[first],
                                                                       "changes": xs}
    # 全試行の要約（+∞ を含む）
    S["summary"] = {}; S["summary_by_scan"] = {}
    for (c, ref, f) in GROUPS:
        for m in METHODS:
            rs = sel(m, c, ref, f)
            for st in STAGES:
                for met in ("eO", "eOB", "eR"):
                    v = [err(r, st, met) for r in rs]
                    S["summary"]["%s|%s|%s|%s|%s|%s" % (met, st, m, c, ref, f)] = {"n": len(v), "n_inf": sum(not math.isfinite(x) for x in v),
                                                                                 **{q: fin(qcdf(v, p)) for q, p in (("median", 0.5), ("p75", 0.75), ("p90", 0.9))}}
                    for scan in sorted({r["scan"] for r in rs}):
                        w = [err(r, st, met) for r in rs if r["scan"] == scan]
                        S["summary_by_scan"]["%s|%s|%s|%s|%s|%s|%s" % (met, st, m, c, ref, f, scan)] = {
                            "n_inf": sum(not math.isfinite(x) for x in w), **{q: fin(qcdf(w, p)) for q, p in (("median", 0.5), ("p75", 0.75), ("p90", 0.9))}}
    json.dump(S, open(os.path.join(DIAG, "r56_summary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    # 図
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"prop": "#1f77b4", "stsl": "#d62728", "fpfh": "#2ca02c", "fgr": "#9467bd"}
    name = {"prop": "proposed", "stsl": "STSL (k=1)", "fpfh": "FPFH+RANSAC", "fgr": "FGR"}
    for met in GRID:
        for st in STAGES:
            fig, ax = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
            for i, (c, ref, f) in enumerate(GROUPS):
                a = ax[i // 4][i % 4]
                for m in METHODS:
                    a.plot(GRID[met], curves["%s|%s|%s|%s|%s|%s" % (met, st, m, c, ref, f)], color=col[m], label=name[m])
                a.set_title("cond %s  ref %s  f %s" % (c, ref, f.replace("p", "."))); a.set_ylim(0, 1); a.grid(alpha=0.3)
                a.set_xlabel({"eO": "e_Omega [m]", "eOB": "e_Omega_BIM [m]", "eR": "e_R [deg]"}[met])
            ax[0][0].set_ylabel("fraction below"); ax[1][0].set_ylabel("fraction below"); ax[0][0].legend(loc="lower right")
            fig.suptitle("R56  %s  refinement=%s  (n=40 per panel, no estimate = +inf)" % (met, st)); fig.tight_layout()
            fig.savefig(os.path.join(DIAG, "fig_%s_%s.png" % (met, st)), dpi=110); plt.close(fig)
    print("done", len(rows))


if __name__ == "__main__":
    main()

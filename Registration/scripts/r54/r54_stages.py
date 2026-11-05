"""R54：段ごとの採点し直し（sni-slam の環境、ブランチ `exp/r54`）。**回す前に commit する。位置合わせは回さない。** 指示書 §3-1〜§3-3。

対象（指示書 §3-3。R52 の f 1.0 は R49 の実データと同じ試行なので 1 回だけ数える）
- R49 の完全：`~/rizu/r49_data/results/<synth|real>/full/<target>.json`（合成 8 × 100、実データ 8 × 10。実データは f = 1p0）
- R52：`~/rizu/r52_data/results/f<f>/<target>.json`（実データ f = 0p2・0p4・0p6・0p8、各 8 × 10）
- R50 の 3 方式：`~/rizu/r50_data/results/<synth|real>/<score_only|icp_only|geo>/<target>.json`（合成 800・実データ 80）
試行ごとの量
- 誤差：`r41_rescore.per_trial(T, P, G1, Ω, Ω_BIM)`。e_R（°）・e_Ω（m）・e_ΩBIM（m、実データだけ、`r39_omega_bim.npz`）・q = |s/s_G − 1|・v（有効）
- 正しい：主の線 v ∧ e_R < 45 ∧ e_Ω < 1.0、感度の線 v ∧ e_R < 10 ∧ e_Ω < 0.3（指示書 §3-2。q は使わない）
- g：diag.gen の 16 のどれかが正しい／s：diag.short の 16／a：diag.after_icp の 4／f：T_est
- 段の分類：最初に 0 になった段（作る g=0／短い精緻化 g=1,s=0／絞り込み s=1,a=0／選ぶ a=1,f=0／成功 f=1）
- 戻った：最初の 0 より後の段に 1 がある。戻った段＝その最初の段（＊ 欄 returned_at）
- 選ばれた候補の番号：diag.scores の 1 位（同じ値は記録の順、R51 と同じ）。T_est と after_icp[1 位] のビット一致も記録する（§3-1 の 2）
出す表（§3-3）：段の分類（両方の線）・スキャンごと・参照の対の 2×2・方式の対の移り・連続の誤差（全試行・成功だけ）・線の格子 12 通り
- 連続の誤差：有効な試行の有限値で、中央値・90%・95%（`np.percentile` の既定）。無効の件数は別に出す
- 線の格子：e_R ∈ {5, 10, 45}° × e_Ω ∈ {0.1, 0.3, 1, 3} m、達成＝v ∧ e_R < a ∧ e_Ω < b（R41 と同じ「<」、q は使わない）、Wilson 95%
出力：試行ごと `~/rizu/r54_data/trials.json`、まとめ `Registration/output/diag/r54/r54_summary.json`
使い方：python r54_stages.py trials [並列数] → python r54_stages.py summary
"""
import json, os, sys
from collections import Counter
from multiprocessing import Pool
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from failure_decomposition import provenance                    # noqa: E402
from r41_rescore import per_trial                               # noqa: E402
import r49_run as RR                                            # noqa: E402
from r49_score import truth                                     # noqa: E402

OUT = os.path.expanduser("~/rizu/r54_data")
DIAG = "Registration/output/diag/r54"
LINES = {"main": (45.0, 1.0), "sens": (10.0, 0.3)}
CLASSES = ["make", "short", "narrow", "select", "success"]
A_GRID, B_GRID = [5.0, 10.0, 45.0], [0.1, 0.3, 1.0, 3.0]


def sources():
    """(方式, データ, f, target, ファイル)"""
    out = []
    for data in ("synth", "real"):
        for t in RR.targets(data):
            out.append(("full", data, "1p0" if data == "real" else "-", t, RR.path(data, "full", t)))
            for m in ("score_only", "icp_only", "geo"):
                out.append((m, data, "1p0" if data == "real" else "-", t,
                            os.path.expanduser("~/rizu/r50_data/results/%s/%s/%s.json" % (data, m, t))))
    for f in ("0p2", "0p4", "0p6", "0p8"):
        for t in RR.REAL:
            out.append(("full", "real", f, t, os.path.expanduser("~/rizu/r52_data/results/f%s/%s.json" % (f, t))))
    return out


_OMB = None


def omb(target):
    global _OMB
    if _OMB is None:
        _OMB = np.load("Registration/output/diag/r39_omega_bim.npz")
    return _OMB[target] if target in _OMB.files else None


def correct(p, line):
    a, b = LINES[line]
    return bool(p["v"] and p["eR"] < a and p["eO"] < b)


def stage(flags):
    g, s, a, f = flags
    first = next((i for i, x in enumerate(flags) if not x), None)
    cls = "success" if f else CLASSES[first]
    ret = None
    if first is not None:
        later = [i for i in range(first + 1, 4) if flags[i]]
        ret = ["g", "s", "a", "f"][later[0]] if later else None
    return cls, ret


def job(src):
    m, data, f, target, path = src
    res = json.load(open(path))
    Gs, om, _ = truth(data, target); G = Gs["1"]
    ob = omb(target) if data == "real" else None
    rows, missing = [], []
    for k in range(RR.n_trials(data)):
        r = res["rows"].get(str(k))
        if r is None:
            missing.append({"k": k, "what": "row"}); continue
        d = r.get("diag") or {}
        lack = [x for x in ("gen", "short", "after_icp", "scores") if not d.get(x)] + ([] if "T_est" in r else ["T_est"])
        if lack:
            missing.append({"k": k, "what": lack}); continue
        P = np.asarray(r["P"], dtype=np.float64)
        ev = lambda T: per_trial(T, P, G, om, ob)
        pg = [ev(T) for T in d["gen"]]; ps = [ev(T) for T in d["short"]]; pa = [ev(T) for T in d["after_icp"]]; pf = ev(r["T_est"])
        top = int(np.argsort(-np.asarray(d["scores"], dtype=float), kind="stable")[0])
        row = {"method": m, "data": data, "f": f, "target": target, "scan": target.split("__")[0] if data == "real" else target,
               "ref": target.split("__")[1] if data == "real" else "-", "k": k, "n_gen": len(pg), "n_short": len(ps), "n_after": len(pa),
               "top": top, "T_est_eq_top": bool(np.array_equal(np.asarray(r["T_est"]), np.asarray(d["after_icp"][top]))),
               "v": bool(pf["v"]), "eR": float(pf["eR"]), "eO": float(pf["eO"]),
               "eOB": (None if pf["eOB"] is None else float(pf["eOB"])), "q": float(pf["q"])}
        for line in LINES:
            fl = [any(correct(p, line) for p in pg), any(correct(p, line) for p in ps), any(correct(p, line) for p in pa), correct(pf, line)]
            cls, ret = stage(fl)
            row[line] = {"g": fl[0], "s": fl[1], "a": fl[2], "f": fl[3], "class": cls, "returned_at": ret}
        rows.append(row)
    return {"src": [m, data, f, target, path], "rows": rows, "missing": missing}


def trials(npar):
    os.makedirs(OUT, exist_ok=True)
    with Pool(npar) as pool:
        outs = pool.map(job, sources(), chunksize=1)
    rows = [r for o in outs for r in o["rows"]]
    missing = [{"src": o["src"], "missing": o["missing"]} for o in outs if o["missing"]]
    json.dump({"provenance": provenance(), "rows": rows, "missing": missing}, open(os.path.join(OUT, "trials.json"), "w"), default=float)
    print("rows", len(rows), "sources with missing", len(missing), flush=True)


# --------------------------------------------------------------------------- 集計
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def q3(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return None if not v else {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95))}


def stage_table(rs, line):
    c = Counter(r[line]["class"] for r in rs)
    ret = Counter("%s→%s" % (r[line]["class"], r[line]["returned_at"]) for r in rs if r[line]["returned_at"])
    return {"n": len(rs), **{k: c.get(k, 0) for k in CLASSES}, "returned": sum(ret.values()), "returned_by": dict(ret)}


def errors(rs, line):
    inv = sum(not r["v"] for r in rs)
    succ = [r for r in rs if r[line]["f"]]
    def block(x):
        return {"n": len(x), "n_invalid": sum(not r["v"] for r in x), "eR": q3([r["eR"] for r in x if r["v"]]),
                "eO": q3([r["eO"] for r in x if r["v"]]), "eOB": q3([r["eOB"] for r in x if r["v"] and r["eOB"] is not None]),
                "q": q3([r["q"] for r in x if r["v"]])}
    return {"all": block(rs), "success_main": block(succ), "n_invalid_all": inv}


def grid(rs, key):
    out = {}
    for a in A_GRID:
        for b in B_GRID:
            k = sum(1 for r in rs if r["v"] and r[key] is not None and r["eR"] < a and r[key] < b)
            out["%g|%g" % (a, b)] = {"k": k, "n": len(rs), "frac": k / len(rs) if rs else None, "wilson95": wilson(k, len(rs))}
    return out


def summary():
    T = json.load(open(os.path.join(OUT, "trials.json")))
    rows = T["rows"]
    S = {"provenance": provenance(), "trials_provenance": T["provenance"], "missing": T["missing"], "n_rows": len(rows),
         "T_est_eq_top": dict(Counter("%s|%s" % (r["method"], r["T_est_eq_top"]) for r in rows)),
         "n_candidates": dict(Counter("%d/%d/%d" % (r["n_gen"], r["n_short"], r["n_after"]) for r in rows))}
    grp = lambda r: "%s|%s|%s|%s" % (r["method"], r["data"], r["f"], r["ref"])
    keys = sorted({grp(r) for r in rows})
    S["stage"] = {line: {g: stage_table([r for r in rows if grp(r) == g], line) for g in keys} for line in LINES}
    # データ × f をまとめた行（参照を合わせる）
    grp2 = lambda r: "%s|%s|%s" % (r["method"], r["data"], r["f"])
    S["stage_by_f"] = {line: {g: stage_table([r for r in rows if grp2(r) == g], line) for g in sorted({grp2(r) for r in rows})} for line in LINES}
    real = [r for r in rows if r["data"] == "real"]
    grp3 = lambda r: "%s|%s|%s" % (r["method"], r["f"], r["scan"])
    S["stage_by_scan"] = {line: {g: stage_table([r for r in real if grp3(r) == g], line) for g in sorted({grp3(r) for r in real})} for line in LINES}
    # 戻った試行の一覧（主の線）
    S["returned_list"] = {line: [dict({k: r[k] for k in ("method", "data", "f", "scan", "ref", "k")}, flags={x: r[line][x] for x in "gsaf"},
                                      cls=r[line]["class"], returned_at=r[line]["returned_at"])
                                 for r in rows if r[line]["returned_at"]] for line in LINES}
    # 参照の対
    pairs = {}
    by = {(r["method"], r["f"], r["scan"], r["k"], r["ref"]): r for r in real}
    for line in LINES:
        acc = {}
        for (m, f, scan, k, ref), r in by.items():
            if ref != "E2" or (m, f, scan, k, "E3") not in by:
                continue
            r3 = by[(m, f, scan, k, "E3")]
            for what, fn in (("final_success", lambda x: x[line]["f"]), ("make_lost", lambda x: x[line]["class"] == "make")):
                a2, a3 = fn(r), fn(r3)
                cell = "both" if a2 and a3 else ("E2_only" if a2 else ("E3_only" if a3 else "neither"))
                for key in ("%s|%s|%s|all" % (what, m, f), "%s|%s|%s|%s" % (what, m, f, scan)):
                    acc.setdefault(key, Counter())[cell] += 1
                if m == "full" and f in ("0p2", "0p4", "0p6"):
                    for key in ("%s|full|0p2-0p6|all" % what, "%s|full|0p2-0p6|%s" % (what, scan)):
                        acc.setdefault(key, Counter())[cell] += 1
        pairs[line] = {k: {c: v.get(c, 0) for c in ("both", "E2_only", "E3_only", "neither")} for k, v in sorted(acc.items())}
    S["ref_pairs"] = pairs
    # 方式の対（R50 の方式 対 R49 の完全、同じ試行）
    full = {(r["data"], r["target"], r["k"]): r for r in rows if r["method"] == "full" and (r["data"] == "synth" or r["f"] == "1p0")}
    trans = {}
    for line in LINES:
        for m in ("score_only", "icp_only", "geo"):
            for data in ("synth", "real"):
                c = Counter()
                for r in rows:
                    if r["method"] == m and r["data"] == data:
                        c["%s>%s" % (full[(data, r["target"], r["k"])][line]["class"], r[line]["class"])] += 1
                trans["%s|%s|%s" % (line, m, data)] = {"n": sum(c.values()), "cells": dict(c)}
    S["method_transition"] = trans
    # 連続の誤差
    S["errors"] = {g: errors([r for r in rows if grp(r) == g], "main") for g in keys}
    S["errors_by_scan"] = {g: errors([r for r in real if grp3(r) == g], "main") for g in sorted({grp3(r) for r in real})}
    # 線の格子
    S["grid"] = {g: {"eO": grid([r for r in rows if grp(r) == g], "eO"),
                     "eOB": (grid([r for r in rows if grp(r) == g], "eOB") if g.split("|")[1] == "real" else None)} for g in keys}
    os.makedirs(DIAG, exist_ok=True)
    json.dump(S, open(os.path.join(DIAG, "r54_summary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print("rows", len(rows), "missing", len(T["missing"]), "T_est_eq_top", S["T_est_eq_top"], "n_cand", S["n_candidates"])


if __name__ == "__main__":
    if sys.argv[1] == "trials":
        trials(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    else:
        summary()

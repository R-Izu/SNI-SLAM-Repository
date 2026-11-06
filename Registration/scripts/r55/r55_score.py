"""R55：共通の局所調整と採点・集計（sni-slam の環境、ブランチ `exp/r55`）。**回す前に commit する。** 指示書 §3-4〜§3-6。判定の言葉は出さない。

方式 × 条件の出どころ（T は「入力 q = P_eff p」→ BIM。誤差は per_trial(T, P_eff, G1, Ω, Ω_BIM)）
- prop|A・fpfh|A・fgr|A・fpfh|B・fgr|B：`r55_run.py` の出力（P_eff = P_in）
- prop|B：R49 の完全（f 1p0、`r49_data/results/real/full`）と R52（f 0p6、`r52_data/results/f0p6`）の記録（P_eff = P）
- stsl|A：`r46_stsl_pipeline.py <r55_<target>_f<f>> A 5 <M3_411_both|M3_410_411_both>` の結果。T = Shift(+z_floor)·tf·T_rp（R47 の `stsl_runs` と同じ）、
  P_eff = P_A。推定なし＝候補 0・最適スコアが有限でない・前処理の失敗（R47 と同じ）。STSL の条件 B は R46 の設定で、入力の作り方が条件 A と同じなので、
  同じ結果を条件 B の行にも置く（回し直さない）
- **STSL の 1 試行の値（主）は、K = 5 回のうち k = 1 の回**（ほかの方式と同じく 1 回の実行。回す前に決めた）。R47 と同じ 5 回の多数決（3 回以上の成功）の成功の数も別の行に出す
共通の調整：`r55_common.common_icp`。推定なしの試行（v = 0）は調整をかけず、件数を別に数える
主の線：v ∧ e_R < 45° ∧ e_Ω < 1.0 m。格子：e_R ∈ {5, 10, 45} × e_Ω ∈ {0.1, 0.3, 1, 3}（「<」、q は使わない）
出力：試行ごと `~/rizu/r55_data/trials.json`、まとめ `Registration/output/diag/r55/r55_summary.json`
使い方：python r55_score.py trials [並列数] → python r55_score.py summary
"""
import json, os, sys, time
from collections import Counter
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r55_common as C                                           # noqa: E402
from regbim import io_utils, metrics                            # noqa: E402
from failure_decomposition import provenance                    # noqa: E402
from r41_rescore import per_trial, real_criteria                # noqa: E402

DIAG = "Registration/output/diag/r55"
METHODS = [("prop", "A"), ("stsl", "A"), ("fpfh", "A"), ("fgr", "A"), ("prop", "B"), ("stsl", "B"), ("fpfh", "B"), ("fgr", "B")]


def jl(l):
    return json.loads(l.replace("-inf", "-Infinity").replace("inf", "Infinity"))


def stsl_runs(target, f):
    ref = target.split("__")[1]; floor = C.FLOOR[ref]; setname = "r55_%s_f%s" % (target, f)
    zf = json.load(open(os.path.join(C.STSL_ROOT, "BIM", floor, "meta.json")))["z_floor"]; S = np.eye(4); S[2, 3] = zf
    out = {}
    for j in range(1, 6):
        tp = os.path.join(C.STSL_ROOT, "processed", "%s_A_k%d" % (setname, j), "30m", "T_rp.json")
        trp = json.load(open(tp)) if os.path.exists(tp) else {}
        pt = json.load(open(os.path.join(C.STSL_ROOT, "results", setname, "A_k%d_prep_time.json" % j)))
        for r in (jl(l) for l in open(os.path.join(C.STSL_ROOT, "results", setname, "A_k%d_%s.jsonl" % (j, floor)))):
            i = r["index"]
            Trp = np.asarray(trp[str(i)]) if str(i) in trp else np.eye(4)
            T = S @ np.asarray(r["tf"], dtype=np.float64).reshape(4, 4) @ Trp
            valid = len(r["candidates"]) > 0 and np.isfinite(r["optimal_score"]) and i not in pt.get("prep_failed", [])
            out[(i, j)] = {"T": T, "valid": bool(valid), "no_candidate": len(r["candidates"]) == 0,
                           "prep_failed": i in pt.get("prep_failed", []), "reg_s": r["reg_s"],
                           "prep_s": (pt["plane_seg_s"] + pt["python_prep_s"]) / max(pt["n_inputs"], 1)}
    return out


def records(target, f):
    """(方式, 条件, k) → T・P_eff・valid・時間"""
    recs = {}
    for m, cond in (("prop", "A"), ("fpfh", "A"), ("fgr", "A"), ("fpfh", "B"), ("fgr", "B")):
        rows = json.load(open(os.path.join(C.OUT, "global", "%s_%s" % (m, cond), "%s_f%s.json" % (target, f))))["rows"]
        for k in range(10):
            r = rows[str(k)]
            recs[(m, cond, k)] = {"T": np.asarray(r["T_est"]), "P": np.asarray(r["P_in"]), "valid": True, "global_s": r["time_s"]}
    pb = C.RR.path("real", "full", target) if f == "1p0" else os.path.join(C.R52.OUT, "f" + f, target + ".json")
    rows = json.load(open(pb))["rows"]
    for k in range(10):
        r = rows[str(k)]
        recs[("prop", "B", k)] = {"T": np.asarray(r["T_est"]), "P": np.asarray(r["P"]), "valid": True, "global_s": r["time_s"]}
    st = stsl_runs(target, f)
    for k in range(10):
        PA = C.cond_A(C.P_of(target, k))[0]
        for j in range(1, 6):
            x = st[(k, j)]
            rec = {"T": x["T"], "P": PA, "valid": x["valid"], "no_candidate": x["no_candidate"], "prep_failed": x["prep_failed"],
                   "global_s": x["prep_s"] + x["reg_s"], "run": j}
            recs[("stsl_run%d" % j, "A", k)] = rec
            if j == 1:
                recs[("stsl", "A", k)] = rec; recs[("stsl", "B", k)] = rec
    return recs


def job(a):
    target, f = a
    scan, ref = target.split("__")
    G = np.asarray(real_criteria(scan)["1"], dtype=np.float64)
    om = np.load("Registration/output/diag/omega.npz")[scan]; omb = np.load("Registration/output/diag/r39_omega_bim.npz")[target]
    cfg = C.cfg_of(target, f)
    src = io_utils.load_source_cloud(cfg); dst = io_utils.load_reference_cloud(cfg)
    out, cache = [], {}
    for (m, cond, k), r in sorted(records(target, f).items(), key=lambda x: str(x[0])):
        raw = per_trial(r["T"], r["P"], G, om, omb)
        if not r["valid"]:
            raw = dict(raw, v=False)
        row = {"method": m, "cond": cond, "target": target, "scan": scan, "ref": ref, "f": f, "k": k, "valid": bool(raw["v"]),
               "no_candidate": r.get("no_candidate"), "prep_failed": r.get("prep_failed"), "global_s": r["global_s"],
               "raw": {x: (None if raw[x] is None else float(raw[x])) for x in ("v", "eR", "eO", "eOB", "q")}}
        if raw["v"]:
            key = (r["T"].tobytes(), r["P"].tobytes())          # STSL の条件 A・B は同じ記録なので 1 回だけ調整する
            if key not in cache:
                q = metrics.apply_sim3(r["P"], src.points)
                t0 = time.perf_counter(); Tr, fit, rmse = C.common_icp(q, dst.points, r["T"]); dt = time.perf_counter() - t0
                cache[key] = (Tr, fit, rmse, dt)
            Tr, fit, rmse, dt = cache[key]
            ref_ = per_trial(Tr, r["P"], G, om, omb)
            row["refined"] = {x: (None if ref_[x] is None else float(ref_[x])) for x in ("v", "eR", "eO", "eOB", "q")}
            row.update(icp_s=dt, icp_fitness=fit, icp_rmse=rmse, T_refined=Tr.tolist())
        else:
            row["refined"] = None
        row["T"] = np.asarray(r["T"]).tolist(); row["P_eff"] = np.asarray(r["P"]).tolist()
        out.append(row)
    return out


def trials(npar):
    os.makedirs(C.OUT, exist_ok=True)
    with Pool(npar) as pool:
        outs = pool.map(job, [(t, f) for t in C.RR.REAL for f in C.FS], chunksize=1)
    rows = [r for o in outs for r in o]
    json.dump({"provenance": provenance(), "icp": C.ICP, "rows": rows}, open(os.path.join(C.OUT, "trials.json"), "w"), default=float)
    print("rows", len(rows), Counter((r["method"], r["cond"]) for r in rows))


# --------------------------------------------------------------------------- 集計
def ok(e):
    return bool(e is not None and e["v"] and e["eR"] < 45.0 and e["eO"] < 1.0)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def q3(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return None if not v else {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95))}


def summary():
    T = json.load(open(os.path.join(C.OUT, "trials.json"))); rows = T["rows"]
    S = {"provenance": provenance(), "trials_provenance": T["provenance"], "icp": T["icp"]}
    main = [r for r in rows if not r["method"].startswith("stsl_run")]
    key = lambda r: (r["method"], r["cond"], r["ref"], r["f"])
    by = {}
    for r in main:
        by.setdefault(key(r), []).append(r)
    S["success"] = {}
    for kk, rs in sorted(by.items()):
        n = len(rs); kr = sum(ok(r["raw"]) for r in rs); kf = sum(ok(r["refined"]) for r in rs)
        S["success"]["|".join(kk)] = {"n": n, "no_estimate": sum(not r["valid"] for r in rs),
                                     "raw": {"k": kr, "wilson95": wilson(kr, n)}, "refined": {"k": kf, "wilson95": wilson(kf, n)}}
    # STSL の 5 回の多数決（R47 の規則）
    runs = {}
    for r in rows:
        if r["method"].startswith("stsl_run"):
            runs.setdefault((r["ref"], r["f"], r["target"], r["k"]), []).append(r)
    maj = {}
    for (ref, f, t, k), rs in runs.items():
        e = maj.setdefault("%s|%s" % (ref, f), Counter())
        e["n"] += 1; e["raw"] += sum(ok(x["raw"]) for x in rs) >= 3; e["refined"] += sum(ok(x["refined"]) for x in rs) >= 3
        e["no_candidate"] += sum(bool(x["no_candidate"]) for x in rs) >= 3
    S["stsl_majority"] = {k: dict(v) for k, v in sorted(maj.items())}
    # スキャンごと
    S["success_by_scan"] = {}
    for r in main:
        e = S["success_by_scan"].setdefault("%s|%s|%s|%s|%s" % (r["method"], r["cond"], r["ref"], r["f"], r["scan"]), Counter())
        e["n"] += 1; e["raw"] += ok(r["raw"]); e["refined"] += ok(r["refined"]); e["no_estimate"] += not r["valid"]
    S["success_by_scan"] = {k: dict(v) for k, v in sorted(S["success_by_scan"].items())}
    # 方式の対（本手法 対 各方式、同じ条件・試行）
    idx = {(r["method"], r["cond"], r["target"], r["f"], r["k"]): r for r in main}
    S["pairs"] = {}
    for stage in ("raw", "refined"):
        for r in main:
            if r["method"] == "prop":
                continue
            p = idx[("prop", r["cond"], r["target"], r["f"], r["k"])]
            a, b = ok(p[stage]), ok(r[stage])
            cell = "both" if a and b else ("prop_only" if a else ("other_only" if b else "neither"))
            for g in ("all", r["scan"]):
                e = S["pairs"].setdefault("%s|%s|%s|%s|%s|%s" % (stage, r["method"], r["cond"], r["ref"], r["f"], g), Counter())
                e[cell] += 1
    S["pairs"] = {k: {c: v.get(c, 0) for c in ("both", "prop_only", "other_only", "neither")} for k, v in sorted(S["pairs"].items())}
    # 連続の誤差
    S["errors"] = {}
    for kk, rs in sorted(by.items()):
        ent = {"n": len(rs), "no_estimate": sum(not r["valid"] for r in rs)}
        for stage in ("raw", "refined"):
            vs = [r[stage] for r in rs if r[stage] is not None and r[stage]["v"]]
            sc = [e for e in vs if ok(e)]
            ent[stage] = {"all": {x: q3([e[x] for e in vs]) for x in ("eR", "eO", "eOB", "q")},
                          "success": {x: q3([e[x] for e in sc]) for x in ("eR", "eO", "eOB", "q")}, "n_success": len(sc)}
        S["errors"]["|".join(kk)] = ent
    # 両方成功した対の e_Ω の差（本手法 − 相手）、スキャンを単位にしたブートストラップ
    S["pair_eO_diff"] = {}
    for stage in ("raw", "refined"):
        for m in ("stsl", "fpfh", "fgr"):
            for cond in ("A", "B"):
                for ref in ("E2", "E3"):
                    for f in C.FS:
                        d = {}
                        for r in main:
                            if (r["method"], r["cond"], r["ref"], r["f"]) != (m, cond, ref, f):
                                continue
                            p = idx[("prop", cond, r["target"], f, r["k"])]
                            if ok(p[stage]) and ok(r[stage]):
                                d.setdefault(r["scan"], []).append(p[stage]["eO"] - r[stage]["eO"])
                        allv = [x for v in d.values() for x in v]
                        ent = {"n_pairs": len(allv), "n_scans": len(d), "median": float(np.median(allv)) if allv else None}
                        if allv:
                            rng = np.random.default_rng(0); sc = sorted(d); bs = []
                            for _ in range(2000):
                                pick = [x for i in rng.integers(0, len(sc), len(sc)) for x in d[sc[i]]]
                                bs.append(float(np.median(pick)))
                            ent["scan_boot95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                        S["pair_eO_diff"]["%s|%s|%s|%s|%s" % (stage, m, cond, ref, f)] = ent
    # 線の格子
    S["grid"] = {}
    for kk, rs in sorted(by.items()):
        for stage in ("raw", "refined"):
            for metric in ("eO", "eOB"):
                g = {}
                for a in (5.0, 10.0, 45.0):
                    for b in (0.1, 0.3, 1.0, 3.0):
                        k = sum(1 for r in rs if r[stage] is not None and r[stage]["v"] and r[stage]["eR"] < a and r[stage][metric] < b)
                        g["%g|%g" % (a, b)] = {"k": k, "n": len(rs), "wilson95": wilson(k, len(rs))}
                S["grid"]["%s|%s|%s" % ("|".join(kk), stage, metric)] = g
    # STSL の候補なし（k = 1 の回と、5 回の多数決）
    S["stsl_no_candidate"] = {}
    for r in main:
        if r["method"] == "stsl" and r["cond"] == "A":
            e = S["stsl_no_candidate"].setdefault("%s|%s|%s" % (r["ref"], r["f"], r["scan"]), Counter())
            e["n"] += 1; e["k1_no_candidate"] += bool(r["no_candidate"]); e["k1_prep_failed"] += bool(r["prep_failed"])
    for (ref, f, t, k), rs in runs.items():
        e = S["stsl_no_candidate"]["%s|%s|%s" % (ref, f, t.split("__")[0])]
        e["majority_no_candidate"] += sum(bool(x["no_candidate"]) for x in rs) >= 3
    S["stsl_no_candidate"] = {k: dict(v) for k, v in sorted(S["stsl_no_candidate"].items())}
    # 時間
    S["time"] = {}
    for kk, rs in sorted(by.items()):
        S["time"]["|".join(kk)] = {"global_s": q3([r["global_s"] for r in rs]), "icp_s": q3([r.get("icp_s") for r in rs])}
    # R54 の追加（§3-6）：作る段で失う＝g = 0 かつ s = 0
    R = json.load(open(os.path.expanduser("~/rizu/r54_data/trials.json")))["rows"]
    real = {(r["f"], r["scan"], r["k"], r["ref"]): r for r in R if r["method"] == "full" and r["data"] == "real"}
    acc = {}
    for (f, scan, k, ref), r in real.items():
        if ref != "E2" or f not in ("0p2", "0p4", "0p6"):
            continue
        r3 = real[(f, scan, k, "E3")]
        a2 = (not r["main"]["g"]) and (not r["main"]["s"]); a3 = (not r3["main"]["g"]) and (not r3["main"]["s"])
        cell = "both" if a2 and a3 else ("E2_only" if a2 else ("E3_only" if a3 else "neither"))
        for g in ("%s|all" % f, "%s|%s" % (f, scan), "0p2-0p6|all", "0p2-0p6|%s" % scan):
            acc.setdefault(g, Counter())[cell] += 1
    S["r54_q2_strict"] = {k: {c: v.get(c, 0) for c in ("both", "E2_only", "E3_only", "neither")} for k, v in sorted(acc.items())}
    os.makedirs(DIAG, exist_ok=True)
    json.dump(S, open(os.path.join(DIAG, "r55_summary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    for k, v in S["success"].items():
        print(k, v["n"], "noest", v["no_estimate"], "raw", v["raw"]["k"], "ref", v["refined"]["k"])


if __name__ == "__main__":
    if sys.argv[1] == "trials":
        trials(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    else:
        summary()

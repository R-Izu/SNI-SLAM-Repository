"""R61 の集計（sni-slam の環境、ブランチ `exp/r61`）。**回す前に commit する。位置合わせは回さない。**

共通：正解 G1（`r49_score.truth("real", target)` の "1"）、Ω は `omega.npz` のスキャン。誤差 `per_trial(T, P, G1, Ω, None)`。
基準内（R54 の主の線）= v ∧ e_R < 45° ∧ e_Ω < 1.0 m。E_Ω は d_Ω(T P, G1, Ω)

s31（§3-1、記録だけ）：R49 の完全・実データ・f 1.0（`r49_data/results/real/full`）と R52 の f 0.6（`r52_data/results/f0p6`）の 80 試行ずつ
- Δρ = `diag.scores` の 1 位 − 2 位。R54 の段階の分類（`r54_data/trials.json` の `main.class`）ごとの分布
- 選択で失った試行（class = select）：after_icp の 4 候補のうち基準内の候補の、採点の順位（1 が最上位）と 1 位との差
rerun（§3-2）：`r61_data/rerun/<設定>_r<n>/` の各試行と R49 の記録
- `T_est` がビット一致（`np.array_equal`）する試行の数。一致しない試行の、段階ごとの候補（gen・short・after_icp）と T_est の要素の差の最大、最終の基準内が変わった試行の数
comp（§3-3）：`r61_data/comp/<成分>/` と摂動なし（`comp/none`）。P_k（全部の成分）は R49 の記録
- 候補の照合（指示書 §3-3 の固定の線）：各段階の候補を元の座標に戻し（T P）、摂動なしの候補（T）と、回転の差 ≤ 2° かつ並進の差 ≤ 0.3 m で対応を取る
- 1：最終の基準内が摂動なしと違う試行の数。2：その試行で (i) gen の候補に対応の取れないものがあるか（数）、(ii) gen はすべて対応が取れ、
  after_icp の 4 候補もすべて対応が取れ、採点の 1 位の候補が摂動なしと別の候補か。3：段階ごとの b（E_Ω 最小）の摂動なしとの差。4：前処理の後の点の数の差
出力：`~/rizu/r61_data/score_<s31|rerun|comp>.json`
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
import r49_run as RR                                            # noqa: E402
from r49_score import truth                                     # noqa: E402
from r41_rescore import per_trial                               # noqa: E402
from regbim import metrics                                      # noqa: E402

H = os.path.expanduser("~/rizu"); OUT = H + "/r61_data"
_T = {}


def tr(target):
    if target not in _T:
        Gs, om, _ = truth("real", target); _T[target] = (Gs["1"], om)
    return _T[target]


def ev(T, P, target):
    G, om = tr(target); e = per_trial(np.asarray(T), np.asarray(P), G, om, None)
    return {"v": bool(e["v"]), "eO": float(e["eO"]) if e["v"] else float("inf"), "eR": float(e["eR"]) if e["v"] else float("inf"),
            "ok": bool(e["v"] and e["eR"] < 45 and e["eO"] < 1.0)}


def q(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return None if not v else {"n": len(v), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                               "min": float(min(v)), "max": float(max(v))}


def s31():
    r54 = {(r["f"], r["target"], r["k"]): r["main"]["class"] for r in json.load(open(H + "/r54_data/trials.json"))["rows"]
           if r["method"] == "full" and r["data"] == "real"}
    res = {"rows": [], "by_class": {}}
    for f, base in (("1p0", H + "/r49_data/results/real/full/%s.json"), ("0p6", H + "/r52_data/results/f0p6/%s.json")):
        for t in RR.REAL:
            rows = json.load(open(base % t))["rows"]
            for k in range(10):
                r = rows[str(k)]; sc = np.asarray(r["diag"]["scores"]); o = np.argsort(-sc); P = r["P"]
                row = {"f": f, "target": t, "k": k, "class": r54[(f, t, k)], "scores": sc.tolist(), "drho": float(sc[o[0]] - sc[o[1]])}
                if row["class"] == "select":
                    es = [ev(T, P, t) for T in r["diag"]["after_icp"]]; rank = {int(i): j + 1 for j, i in enumerate(o)}
                    row["correct"] = [{"idx": i, "rank": rank[i], "gap_to_top": float(sc[o[0]] - sc[i]), "eO": e["eO"], "eR": e["eR"]}
                                      for i, e in enumerate(es) if e["ok"]]
                res["rows"].append(row)
    for f in ("1p0", "0p6"):
        for c in ("success", "select", "narrow", "make", "short"):
            v = [r["drho"] for r in res["rows"] if r["f"] == f and r["class"] == c]
            res["by_class"]["%s|%s" % (f, c)] = q(v)
    json.dump(res, open(OUT + "/score_s31.json", "w"), indent=1)
    for k, v in res["by_class"].items():
        print(k, v)
    for r in res["rows"]:
        if r["class"] == "select":
            print("select", r["f"], r["target"], r["k"], "drho %.4f" % r["drho"], [(c["idx"], c["rank"], round(c["gap_to_top"], 4)) for c in r["correct"]])


def rerun():
    res = {}
    for name in sorted(os.listdir(OUT + "/rerun")):
        out = {"n": 0, "bit_equal": 0, "mismatch": [], "env": None}
        for t in RR.REAL:
            p = "%s/rerun/%s/%s.json" % (OUT, name, t)
            if not os.path.exists(p):
                continue
            d = json.load(open(p)); out["env"] = d.get("env"); out["provenance"] = d.get("provenance")
            ref = json.load(open(H + "/r49_data/results/real/full/%s.json" % t))
            for k, r in d["rows"].items():
                o = ref["rows"][k]; out["n"] += 1
                if np.array_equal(np.asarray(r["T_est"]), np.asarray(o["T_est"])):
                    out["bit_equal"] += 1; continue
                m = {"target": t, "k": int(k), "T_est": float(np.abs(np.asarray(r["T_est"]) - np.asarray(o["T_est"])).max())}
                for st in ("gen", "short", "after_icp"):
                    a, b = np.asarray(r["diag"][st]), np.asarray(o["diag"][st])
                    m[st] = float(np.abs(a - b).max()) if a.shape == b.shape else "shape %s vs %s" % (a.shape, b.shape)
                e1, e0 = ev(r["T_est"], r["P"], t), ev(o["T_est"], o["P"], t)
                m["ok_new"], m["ok_r49"] = e1["ok"], e0["ok"]; out["mismatch"].append(m)
        out["ok_changed"] = sum(m["ok_new"] != m["ok_r49"] for m in out["mismatch"])
        res[name] = out
        print(name, "n", out["n"], "bit_equal", out["bit_equal"], "ok_changed", out["ok_changed"],
              "max diff T_est %.3e" % max([m["T_est"] for m in out["mismatch"]] or [0]))
    r49p = json.load(open(H + "/r49_data/results/real/full/%s.json" % RR.REAL[0]))["provenance"]
    res["_r49_provenance"] = r49p
    json.dump(res, open(OUT + "/score_rerun.json", "w"), indent=1)


def rot_trans_diff(A, B):
    Ra, ta, _ = metrics.decompose_sim3(A); Rb, tb, _ = metrics.decompose_sim3(B)
    return metrics.rotation_error_deg(Ra, Rb), float(np.linalg.norm(ta - tb))


def match(As, Bs):
    """As の各候補に、Bs のうち回転 ≤ 2°・並進 ≤ 0.3 m で対応するものの番号（無ければ None）"""
    out = []
    for A in As:
        j = None
        for i, B in enumerate(Bs):
            dr, dt = rot_trans_diff(A, B)
            if dr <= 2.0 and dt <= 0.3:
                j = i; break
        out.append(j)
    return out


def stage_eval(r, P, t):
    d = r["diag"]; res = {}
    for st in ("gen", "short", "after_icp"):
        es = [ev(T, P, t) for T in d[st]]; res[st] = min(e["eO"] for e in es)
    return res


def comp():
    base = {}
    for t in RR.REAL:
        base[t] = json.load(open("%s/comp/none/%s.json" % (OUT, t)))["rows"]
    res = {}
    names = ["full_P"] + [c for c in ("yaw", "xy", "tilt", "scale") if os.path.isdir("%s/comp/%s" % (OUT, c))]
    for name in names:
        rows = []
        for t in RR.REAL:
            d = json.load(open(H + "/r49_data/results/real/full/%s.json" % t))["rows"] if name == "full_P" else json.load(open("%s/comp/%s/%s.json" % (OUT, name, t)))["rows"]
            for k in range(10):
                r = d[str(k)]; b = base[t][str(k)]; P = np.asarray(r["P"]); I = np.eye(4)
                e, e0 = ev(r["T_est"], P, t), ev(b["T_est"], I, t)
                g = [np.asarray(T) @ P for T in r["diag"]["gen"]]; g0 = [np.asarray(T) for T in b["diag"]["gen"]]
                a = [np.asarray(T) @ P for T in r["diag"]["after_icp"]]; a0 = [np.asarray(T) for T in b["diag"]["after_icp"]]
                mg, mg0 = match(g, g0), match(g0, g)
                ma = match(a, a0)
                top, top0 = int(np.argmax(r["diag"]["scores"])), int(np.argmax(b["diag"]["scores"]))
                sb, sb0 = stage_eval(r, P, t), stage_eval(b, I, t)
                row = {"target": t, "scan": t.split("__")[0], "k": k, "ok": e["ok"], "ok_none": e0["ok"], "eO": e["eO"], "eO_none": e0["eO"],
                       "gen_unmatched": sum(x is None for x in mg), "gen_unmatched_none": sum(x is None for x in mg0),
                       "after_unmatched": sum(x is None for x in ma), "top_same": (ma[top] == top0) if ma[top] is not None else False,
                       "db": {st: sb[st] - sb0[st] for st in sb}, "b": sb, "b_none": sb0}
                if "n_prep" in r and "n_prep" in b:
                    row["dn_prep"] = r["n_prep"]["total"] - b["n_prep"]["total"]; row["n_prep"] = r["n_prep"]["total"]; row["n_prep_none"] = b["n_prep"]["total"]
                if row["ok"] != row["ok_none"]:
                    row["why"] = ("gen_changed" if (row["gen_unmatched"] or row["gen_unmatched_none"]) else
                                  ("rank_swapped" if (row["after_unmatched"] == 0 and not row["top_same"]) else "other"))
                rows.append(row)
        res[name] = rows
        for scan in sorted(set(r["scan"] for r in rows)) + ["all"]:
            x = [r for r in rows if scan == "all" or r["scan"] == scan]
            ch = [r for r in x if r["ok"] != r["ok_none"]]
            print(name, scan, "ok %d none %d changed %d" % (sum(r["ok"] for r in x), sum(r["ok_none"] for r in x), len(ch)),
                  "why", dict(Counter(r["why"] for r in ch)), "gen_unmatched>0 %d" % sum((r["gen_unmatched"] > 0) for r in x),
                  "db gen med %.3f after med %.3f" % (np.median([r["db"]["gen"] for r in x]), np.median([r["db"]["after_icp"] for r in x])),
                  "dn_prep", (q([r.get("dn_prep") for r in x]) or {}).get("median"))
    json.dump(res, open(OUT + "/score_comp.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    {"s31": s31, "rerun": rerun, "comp": comp}[sys.argv[1]]()

"""R58 段階1：STSL の候補を段ごとに採点し直す（sni-slam の環境、ブランチ `exp/r58`）。**回す前に commit する。** 指示書 §3-1〜§3-3・§7-3。

候補の採点（§3-1）：R55 の `r55_score.stsl_runs` と同じ手順。候補 c の 3D の変換 T_c = Shift(+z_floor)·tf_c·T_rp（T_rp は R55 の
`processed/<seq>/30m/T_rp.json`、無ければ単位行列）、P = P_A（`r55_common.cond_A`）、`r41_rescore.per_trial(T_c, P_A, G1, Ω, Ω_BIM)`。
主の線 v ∧ e_R < 45° ∧ e_Ω < 1.0 m、感度の線 v ∧ e_R < 10° ∧ e_Ω < 0.3 m
段の分類（§3-2、最終は調整前、**回し直しの記録で**）：前処理の失敗（R55 の prep_time の prep_failed）／候補を出せない（候補 0 または
最適スコアが有限でない）／正しい候補が無い／別を選んだ／成功。正しい候補が無いのに最終が正しい試行は別の欄（no_correct_but_final_correct）
R46 式（STSL 自身の線、`r46_score.py:94-95`）：chosen_correct（success_stsl）／correct_existed_other_chosen／no_correct_candidate、候補なし・前処理の失敗
候補の順位：`score_lamda`（Score − PunishScore·lamda）の降順（同じ値は番号の順）。正しい候補の順位＝正しい候補のうち最上位
出力：試行ごと `~/rizu/r58_data/trials.json`、まとめ `Registration/output/diag/r58/r58_summary.json`
使い方：python r58_score.py
"""
import json, os, sys
from collections import Counter
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49", "Registration/scripts/r52", "Registration/scripts/r55"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from failure_decomposition import provenance                    # noqa: E402
from r41_rescore import per_trial, real_criteria                # noqa: E402
import r55_common as C                                          # noqa: E402

OUT = os.path.expanduser("~/rizu/r58_data")
DIAG = "Registration/output/diag/r58"
LINES = {"main": (45.0, 1.0), "sens": (10.0, 0.3)}
CLS = ["prep_failed", "no_candidate", "no_correct", "other_chosen", "success", "no_correct_but_final_correct"]
R46 = ["prep_failed", "no_candidate", "no_correct_candidate", "correct_existed_other_chosen", "chosen_correct"]


def jl(l):
    return json.loads(l.replace("-inf", "-Infinity").replace("inf", "Infinity"))


def ok(p, line):
    a, b = LINES[line]
    return bool(p["v"] and p["eR"] < a and p["eO"] < b)


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(DIAG, exist_ok=True)
    rows = []
    for target in C.RR.REAL:
        scan, ref = target.split("__"); floor = C.FLOOR[ref]
        G = np.asarray(real_criteria(scan)[1]); om = np.load("Registration/output/diag/omega.npz")[scan]
        omb = np.load("Registration/output/diag/r39_omega_bim.npz")[target]
        zf = json.load(open(os.path.join(C.STSL_ROOT, "BIM", floor, "meta.json")))["z_floor"]; S = np.eye(4); S[2, 3] = zf
        for f in C.FS:
            setname = "r55_%s_f%s" % (target, f)
            for j in range(1, 6):
                tp = os.path.join(C.STSL_ROOT, "processed", "%s_A_k%d" % (setname, j), "30m", "T_rp.json")
                trp = json.load(open(tp)) if os.path.exists(tp) else {}
                pt = json.load(open(os.path.join(C.STSL_ROOT, "results", setname, "A_k%d_prep_time.json" % j)))
                old = [jl(l) for l in open(os.path.join(C.STSL_ROOT, "results", setname, "A_k%d_%s.jsonl" % (j, floor)))]
                new = [jl(l) for l in open(os.path.join(C.STSL_ROOT, "results", "r58", setname, "A_k%d_%s.jsonl" % (j, floor)))]
                for a, b in zip(old, new):
                    i = b["index"]; PA = C.cond_A(C.P_of(target, i))[0]
                    Trp = np.asarray(trp[str(i)]) if str(i) in trp else np.eye(4)
                    T3 = lambda tf: S @ np.asarray(tf, dtype=np.float64).reshape(4, 4) @ Trp
                    pf = i in pt.get("prep_failed", []); cands = b["candidates"]
                    nocand = len(cands) == 0 or not np.isfinite(b["optimal_score"])
                    fin_new = per_trial(T3(b["tf"]), PA, G, om, omb); fin_old = per_trial(T3(a["tf"]), PA, G, om, omb)
                    if pf or nocand:
                        fin_new = dict(fin_new, v=False); fin_old = dict(fin_old, v=False)
                    ce = [per_trial(T3(c["tf"]), PA, G, om, omb) for c in cands]
                    order = sorted(range(len(cands)), key=lambda q: (-cands[q]["score_lamda"], q))
                    row = {"target": target, "scan": scan, "ref": ref, "f": f, "k": i, "run": j, "prep_failed": pf, "no_candidate": nocand,
                           "n_cand": len(cands), "chosen_index": b["chosen_index"],
                           "tf_equal_r55": bool(np.array_equal(np.asarray(a["tf"]), np.asarray(b["tf"]))),
                           "chosen_tf_equal_final": (bool(np.array_equal(np.asarray(cands[b["chosen_index"]]["tf"]), np.asarray(b["tf"])))
                                                     if b["chosen_index"] >= 0 else None),
                           "final_new": {"v": bool(fin_new["v"]), "eR": float(fin_new["eR"]), "eO": float(fin_new["eO"])},
                           "final_r55": {"v": bool(fin_old["v"]), "eR": float(fin_old["eR"]), "eO": float(fin_old["eO"])},
                           "cand": [{"eR": float(e["eR"]), "eO": float(e["eO"]), "v": bool(e["v"]), "score_lamda": float(c["score_lamda"]),
                                     "stsl_correct": bool(c["correct"])} for e, c in zip(ce, cands)],
                           "stsl_success": bool(b["success_stsl"])}
                    for line in LINES:
                        good = [ok(e, line) for e in ce]; fo = ok(fin_new, line)
                        if pf:
                            cl = "prep_failed"
                        elif nocand:
                            cl = "no_candidate"
                        elif not any(good):
                            cl = "no_correct_but_final_correct" if fo else "no_correct"
                        else:
                            cl = "success" if fo else "other_chosen"
                        ranks = [order.index(q) + 1 for q in range(len(cands)) if good[q]]
                        row[line] = {"class": cl, "final_ok": fo, "final_ok_r55": ok(fin_old, line), "correct_idx": [q for q in range(len(cands)) if good[q]],
                                     "rank_best_correct": min(ranks) if ranks else None}
                    row["r46"] = ("prep_failed" if pf else "no_candidate" if not cands else
                                  "chosen_correct" if b["success_stsl"] else
                                  "correct_existed_other_chosen" if any(c["correct"] for c in cands) else "no_correct_candidate")
                    rows.append(row)
    json.dump({"provenance": provenance(), "rows": rows}, open(os.path.join(OUT, "trials.json"), "w"), default=float)
    # まとめ
    Sm = {"provenance": provenance(), "n": len(rows)}
    Sm["bit"] = {"tf_equal_r55": sum(r["tf_equal_r55"] for r in rows), "chosen_tf_equal_final": sum(bool(r["chosen_tf_equal_final"]) for r in rows),
                 "chosen_none": sum(r["chosen_tf_equal_final"] is None for r in rows)}
    Sm["mismatch"] = [dict({k: r[k] for k in ("target", "f", "k", "run", "final_r55", "final_new")}, ok_r55=r["main"]["final_ok_r55"], ok_new=r["main"]["final_ok"])
                      for r in rows if not r["tf_equal_r55"]]
    Sm["ok_differs_from_r55"] = {line: sum(r[line]["final_ok"] != r[line]["final_ok_r55"] for r in rows) for line in LINES}
    def tab(rs, line):
        c = Counter(r[line]["class"] for r in rs); return {"n": len(rs), **{k: c.get(k, 0) for k in CLS}}
    def tab46(rs):
        c = Counter(r["r46"] for r in rs); return {"n": len(rs), **{k: c.get(k, 0) for k in R46}}
    groups = [("E2", "1p0"), ("E3", "1p0"), ("E2", "0p6"), ("E3", "0p6")]
    for name, sel in (("k1", lambda r: r["run"] == 1), ("all5", lambda r: True)):
        rs = [r for r in rows if sel(r)]
        for line in LINES:
            Sm["stage_%s_%s" % (name, line)] = {**{"%s|%s" % g: tab([r for r in rs if (r["ref"], r["f"]) == g], line) for g in groups},
                                                **{"E2+E3|%s" % f: tab([r for r in rs if r["f"] == f], line) for f in ("1p0", "0p6")}}
        Sm["r46_%s" % name] = {**{"%s|%s" % g: tab46([r for r in rs if (r["ref"], r["f"]) == g]) for g in groups},
                               **{"E2+E3|%s" % f: tab46([r for r in rs if r["f"] == f]) for f in ("1p0", "0p6")}}
        mp = {"success": "chosen_correct", "other_chosen": "correct_existed_other_chosen", "no_correct": "no_correct_candidate",
              "no_candidate": "no_candidate", "prep_failed": "prep_failed", "no_correct_but_final_correct": "-"}
        Sm["line_change_%s" % name] = {"n_changed": sum(mp[r["main"]["class"]] != r["r46"] for r in rs),
                                       "pairs": dict(Counter("%s>%s" % (r["r46"], r["main"]["class"]) for r in rs if mp[r["main"]["class"]] != r["r46"]))}
    k1 = [r for r in rows if r["run"] == 1]
    Sm["stage_by_scan_k1"] = {"%s|%s|%s" % (s, ref, f): tab([r for r in k1 if (r["scan"], r["ref"], r["f"]) == (s, ref, f)], "main")
                              for s in sorted({r["scan"] for r in k1}) for ref, f in groups}
    Sm["n_cand"] = {}
    for g in groups:
        rs = [r for r in rows if (r["ref"], r["f"]) == g and r["n_cand"] > 0]
        rk = [r["main"]["rank_best_correct"] for r in rs if r["main"]["rank_best_correct"]]
        Sm["n_cand"]["%s|%s" % g] = {"n_with_cand": len(rs), "median": float(np.median([r["n_cand"] for r in rs])) if rs else None,
                                      "min": min((r["n_cand"] for r in rs), default=None), "max": max((r["n_cand"] for r in rs), default=None),
                                      "rank_best_correct": dict(Counter(str(x) for x in rk))}
    # 共通の調整の後（R55 の trials の stsl・条件 A の refined、k = 1）
    T55 = {(r["target"], r["f"], r["k"]): r for r in json.load(open(os.path.join(C.OUT, "trials.json")))["rows"] if r["method"] == "stsl" and r["cond"] == "A"}
    ref_ok = lambda e: bool(e is not None and e["v"] and e["eR"] < 45 and e["eO"] < 1.0)
    Sm["after_common_adjust_k1"] = {}
    for g in groups + [("E2+E3", "1p0"), ("E2+E3", "0p6")]:
        rs = [r for r in k1 if r["f"] == g[1] and (g[0] == "E2+E3" or r["ref"] == g[0])]
        e = {}
        for cl in ("other_chosen", "no_correct"):
            x = [r for r in rs if r["main"]["class"] == cl]
            e[cl] = {"n": len(x), "ok_after_N": sum(ref_ok(T55[(r["target"], r["f"], r["k"])]["refined"]) for r in x)}
        Sm["after_common_adjust_k1"]["%s|%s" % g] = e
    Sm["stop_rule_760"] = Sm["bit"]["tf_equal_r55"] < 760
    json.dump(Sm, open(os.path.join(DIAG, "r58_summary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print("bit", Sm["bit"], "ok_differs", Sm["ok_differs_from_r55"], "stop", Sm["stop_rule_760"])


if __name__ == "__main__":
    main()

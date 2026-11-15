"""R60：保存した候補と出力を連続の誤差で採点し直す（sni-slam の環境、ブランチ `exp/r60`）。**回す前に commit する。位置合わせは回さない。** 指示書 §3-1〜§3-4。

量（§3-1、固定）
- 評価点：Ω（`omega.npz`、R54 の e_Ω と同じ）。Ω_411 ＝ `r39_omega_bim.npz` の `<scan>__E2` を、E2 と E3 の両方に同じ点として使う（実データ）。合成は Ω だけ
- 誤差 E_M(T) = d_omega(T P, G1, M)（`r41_rescore.per_trial` の eO と eOB に M を渡して得る）。e_R・q も同じ変換から
- b(C)：候補の集合 C で E_Ω が最小の候補の E_Ω。その同じ候補の E_Ω411・e_R・q を添える（指標ごとに別の候補を選ばない）。無効な候補（v = 0）は +∞
- l_M = E_M(T_selected) − E_M(候補 b)（C_final の、E_Ω で選んだ最良の候補）。本手法の C_final は after_icp、T_selected は T_est。
  STSL の C_final は投票の後の全候補、T_selected は同じ実行の最終の tf（R58 の回し直しの記録）。出力なし（候補なし・前処理の失敗）は誤差に入れず件数を数える
- 段階：本手法は gen（16）・short（16）・after_icp（4）。STSL は投票の後（1 段階、`cand`）
群（§3-2）
- obs：本手法の完全、実データ、f 0p2〜0p8（R52）・1p0（R49）× E2・E3 × k 0〜9
- cls：full（R49）・score_only・icp_only・geo（R50）、f 1p0、合成 8 × 100・実データ 8 × 10
- stsl：R58 の回し直しの記録（`results/r58/`）、条件 A、f 1p0・0p6 × E2・E3 × k 0〜9 × 回 1〜5。T_c = Shift(+z_floor)·tf_c·T_rp、P = P_A（R55・R58 と同じ）
- propA：R55 の `~/rizu/r55_data/global/prop_A/`（diag あり）、f 1p0・0p6、P = P_in
確認（§4 S5）：T_est（STSL は最終の tf）の E_Ω を、R54（obs・cls）・R55 の trials（propA の raw）・R58 の trials（stsl の final_new）の e_Ω と、
E2 の試行の E_Ω411 を R54・R55 の e_ΩBIM と比べ、差 1e-6 以内の件数を数える
§3-4 の r：R52 の `r52_inputs.json` の `inside_411`（E2）、`inside_411 + inside_410`（E3）を `n_sampled` で割った値
出力：試行ごと `~/rizu/r60_data/trials.json`
"""
import glob, json, os, sys
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49", "Registration/scripts/r52", "Registration/scripts/r55"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from failure_decomposition import provenance                    # noqa: E402
from r41_rescore import per_trial                               # noqa: E402
from r49_score import truth                                     # noqa: E402
import r49_run as RR                                            # noqa: E402
import r55_common as C                                          # noqa: E402

OUT = os.path.expanduser("~/rizu/r60_data")
OMB = None


def om411(scan):
    global OMB
    if OMB is None:
        OMB = np.load("Registration/output/diag/r39_omega_bim.npz")
    return OMB["%s__E2" % scan]


def ev(T, P, G, om, o411):
    e = per_trial(T, P, G, om, o411)
    if not e["v"]:
        return {"v": False, "EO": float("inf"), "EO411": float("inf") if o411 is not None else None, "eR": float("inf"), "q": float("inf")}
    return {"v": True, "EO": float(e["eO"]), "EO411": (float(e["eOB"]) if o411 is not None else None), "eR": float(e["eR"]), "q": float(e["q"])}


def best(es):
    i = int(np.argmin([e["EO"] for e in es]))
    return i, es[i]


def stages_row(Ts_by_stage, T_sel, P, G, om, o411):
    row = {}
    for name, Ts in Ts_by_stage.items():
        es = [ev(T, P, G, om, o411) for T in Ts]
        i, e = best(es)
        row["b_" + name] = dict(e, idx=i)
        row["cands_" + name] = [e2["EO"] for e2 in es]
    fin = ev(T_sel, P, G, om, o411)
    row["final"] = fin
    last = list(Ts_by_stage)[-1]
    bf = row["b_" + last]
    row["l"] = {"O": fin["EO"] - bf["EO"], "O411": (fin["EO411"] - bf["EO411"]) if o411 is not None else None}
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    R54 = {(r["method"], r["data"], r["f"], r["target"], r["k"]): r for r in json.load(open(os.path.expanduser("~/rizu/r54_data/trials.json")))["rows"]}
    R55 = {(r["target"], r["f"], r["k"]): r for r in json.load(open(os.path.join(C.OUT, "trials.json")))["rows"] if r["method"] == "prop" and r["cond"] == "A"}
    R58 = {(r["target"], r["f"], r["k"], r["run"]): r for r in json.load(open(os.path.expanduser("~/rizu/r58_data/trials.json")))["rows"]}
    inp = json.load(open("Registration/output/diag/r52/r52_inputs.json"))
    rows = []
    # obs と cls（R49・R50・R52 の記録）
    srcs = [("obs", "full", "real", f, t, (RR.path("real", "full", t) if f == "1p0" else os.path.join(C.R52.OUT, "f" + f, t + ".json")))
            for f in ("0p2", "0p4", "0p6", "0p8", "1p0") for t in RR.REAL]
    for data in ("synth", "real"):
        for t in RR.targets(data):
            srcs.append(("cls", "full", data, "1p0" if data == "real" else "-", t, RR.path(data, "full", t)))
            for m in ("score_only", "icp_only", "geo"):
                srcs.append(("cls", m, data, "1p0" if data == "real" else "-", t, os.path.expanduser("~/rizu/r50_data/results/%s/%s/%s.json" % (data, m, t))))
    for grp, m, data, f, t, path in srcs:
        res = json.load(open(path))["rows"]
        Gs, om, _ = truth(data, t); G = Gs["1"]
        scan = t.split("__")[0] if data == "real" else t; ref = t.split("__")[1] if data == "real" else "-"
        o411 = om411(scan) if data == "real" else None
        for k in range(RR.n_trials(data)):
            r = res[str(k)]; d = r["diag"]; P = np.asarray(r["P"], dtype=np.float64)
            row = {"group": grp, "method": m, "data": data, "f": f, "target": t, "scan": scan, "ref": ref, "k": k, "run": None, "output": True}
            row.update(stages_row({"gen": d["gen"], "short": d["short"], "after": d["after_icp"]}, r["T_est"], P, G, om, o411))
            r54 = R54.get((m, data, f, t, k))
            if r54:
                row["chk_eO"] = abs(row["final"]["EO"] - r54["eO"]) if r54["v"] else None
                row["chk_eO411"] = (abs(row["final"]["EO411"] - r54["eOB"]) if (ref == "E2" and r54["eOB"] is not None and r54["v"]) else None)
                row["r54_class"] = r54["main"]["class"]
            if data == "real":
                c = inp["%s|%s" % (scan, f)]
                row["r"] = (c["inside_411"] if ref == "E2" else c["inside_411"] + c["inside_410"]) / c["n_sampled"]
            rows.append(row)
    # propA（R55）
    for t in RR.REAL:
        scan, ref = t.split("__"); Gs, om, _ = truth("real", t); G = Gs["1"]; o411 = om411(scan)
        for f in C.FS:
            res = json.load(open(os.path.join(C.OUT, "global", "prop_A", "%s_f%s.json" % (t, f))))["rows"]
            for k in range(10):
                r = res[str(k)]; d = r["diag"]; P = np.asarray(r["P_in"], dtype=np.float64)
                row = {"group": "propA", "method": "prop_A", "data": "real", "f": f, "target": t, "scan": scan, "ref": ref, "k": k, "run": None, "output": True}
                row.update(stages_row({"gen": d["gen"], "short": d["short"], "after": d["after_icp"]}, r["T_est"], P, G, om, o411))
                x = R55[(t, f, k)]["raw"]
                row["chk_eO"] = abs(row["final"]["EO"] - x["eO"]) if x["v"] else None
                row["chk_eO411"] = abs(row["final"]["EO411"] - x["eOB"]) if (ref == "E2" and x["v"]) else None
                rows.append(row)
    # stsl（R58 の回し直し）
    for t in RR.REAL:
        scan, ref = t.split("__"); Gs, om, _ = truth("real", t); G = Gs["1"]; o411 = om411(scan); floor = C.FLOOR[ref]
        zf = json.load(open(os.path.join(C.STSL_ROOT, "BIM", floor, "meta.json")))["z_floor"]; S = np.eye(4); S[2, 3] = zf
        for f in C.FS:
            setname = "r55_%s_f%s" % (t, f)
            for j in range(1, 6):
                tp = os.path.join(C.STSL_ROOT, "processed", "%s_A_k%d" % (setname, j), "30m", "T_rp.json")
                trp = json.load(open(tp)) if os.path.exists(tp) else {}
                pt = json.load(open(os.path.join(C.STSL_ROOT, "results", setname, "A_k%d_prep_time.json" % j)))
                for l in open(os.path.join(C.STSL_ROOT, "results", "r58", setname, "A_k%d_%s.jsonl" % (j, floor))):
                    b = json.loads(l.replace("-inf", "-Infinity").replace("inf", "Infinity")); i = b["index"]
                    PA = C.cond_A(C.P_of(t, i))[0]; Trp = np.asarray(trp[str(i)]) if str(i) in trp else np.eye(4)
                    T3 = lambda tf: S @ np.asarray(tf, dtype=np.float64).reshape(4, 4) @ Trp
                    pf = i in pt.get("prep_failed", []); nocand = len(b["candidates"]) == 0 or not np.isfinite(b["optimal_score"])
                    row = {"group": "stsl", "method": "stsl", "data": "real", "f": f, "target": t, "scan": scan, "ref": ref, "k": i, "run": j,
                           "output": not (pf or nocand), "prep_failed": pf, "no_candidate": nocand}
                    if row["output"]:
                        row.update(stages_row({"cand": [T3(c["tf"]) for c in b["candidates"]]}, T3(b["tf"]), PA, G, om, o411))
                        x = R58[(t, f, i, j)]["final_new"]
                        row["chk_eO"] = abs(row["final"]["EO"] - x["eO"]) if x["v"] else None
                        row["r58_class"] = R58[(t, f, i, j)]["main"]["class"]
                    rows.append(row)
    json.dump({"provenance": provenance(), "rows": rows}, open(os.path.join(OUT, "trials.json"), "w"), default=float)
    chk = [r.get("chk_eO") for r in rows if r.get("chk_eO") is not None]; chk2 = [r.get("chk_eO411") for r in rows if r.get("chk_eO411") is not None]
    print("rows", len(rows), "chk eO n", len(chk), "max", max(chk), "over1e-6", sum(x > 1e-6 for x in chk),
          "chk eO411 n", len(chk2), "max", max(chk2), "over1e-6", sum(x > 1e-6 for x in chk2))


if __name__ == "__main__":
    main()

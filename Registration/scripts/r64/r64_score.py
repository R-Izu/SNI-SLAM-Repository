"""R64 の集計（sni-slam の環境、ブランチ `exp/r61`）。**回す前に commit する。位置合わせは回さない。**

- 記録：`~/rizu/r64_data/<fpfh|fgr>/m3_cor_<s>__E2.json`。誤差は実行の時点で `per_trial(T, P_A, G1, Ω, Ω_BIM)` で書いたもの
- 基準内 = v ∧ e_R < 45° ∧ e_Ω < 1.0 m
- 窓なし：R55 の記録 `~/rizu/r55_data/trials.json` の method fpfh・fgr、cond A、f 1p0、ref E2 の `raw`（局所調整の前）
- 出すもの（指示書 §3 の 2）：L と窓の数（試行ごとの中央値・最小・最大）、1 試行の時間（窓の数ぶんの合計）、基準内の数（窓あり・窓なし、全体とスキャンごと）、
  選んだ窓の e_Ω・e_R の分布、診断（e_Ω が最小の窓）の e_Ω の分布と基準内の数、選んだ窓と最小の窓が同じ試行の数
出力：`~/rizu/r64_data/score.json`・`trials.json`
"""
import json, os
import numpy as np

H = os.path.expanduser("~/rizu"); OUT = H + "/r64_data"
TARGETS = ["m3_cor_%s__E2" % s for s in "abcd"]


def ok(e):
    return bool(e is not None and e["v"] and e["eR"] < 45.0 and e["eO"] < 1.0)


def q(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return None if not v else {"n": len(v), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                               "min": float(min(v)), "max": float(max(v))}


def main():
    r55 = {(r["method"], r["target"], r["k"]): r for r in json.load(open(H + "/r55_data/trials.json"))["rows"]
           if r["cond"] == "A" and r["f"] == "1p0" and r["ref"] == "E2"}
    rows, S = [], {}
    for m in ("fpfh", "fgr"):
        for t in TARGETS:
            d = json.load(open("%s/%s/%s.json" % (OUT, m, t)))["rows"]
            for k in range(10):
                r = d[str(k)]; w = r["windows"]; s = w[r["selected"]]; c = w[r["closest"]]; base = r55.get((m, t, k))
                rows.append({"method": m, "target": t, "scan": t.split("__")[0], "k": k, "n_windows": len(w), "time_s": r["time_s"], "L": r["window"]["L"],
                             "sel": {x: s[x] for x in ("ij", "coverage", "v", "eR", "eO", "eOB", "n_points")}, "sel_ok": ok(s),
                             "closest": {x: c[x] for x in ("ij", "coverage", "v", "eR", "eO", "eOB", "n_points")}, "closest_ok": ok(c),
                             "same": r["selected"] == r["closest"], "no_window_ok": ok(base["raw"]) if base else None,
                             "no_window": base["raw"] if base else None, "n_windows_ok": sum(ok(x) for x in w)})
    for m in ("fpfh", "fgr"):
        x = [r for r in rows if r["method"] == m]
        S[m] = {"L": x[0]["L"], "n_windows": q([r["n_windows"] for r in x]), "time_s": q([r["time_s"] for r in x]),
                "ok_window": sum(r["sel_ok"] for r in x), "ok_no_window": sum(bool(r["no_window_ok"]) for r in x), "n": len(x),
                "by_scan": {sc: {"ok_window": sum(r["sel_ok"] for r in x if r["scan"] == sc), "ok_no_window": sum(bool(r["no_window_ok"]) for r in x if r["scan"] == sc),
                                 "ok_closest": sum(r["closest_ok"] for r in x if r["scan"] == sc)} for sc in sorted(set(r["scan"] for r in x))},
                "sel_eO": q([r["sel"]["eO"] for r in x]), "sel_eR": q([r["sel"]["eR"] for r in x]),
                "closest_eO": q([r["closest"]["eO"] for r in x]), "ok_closest": sum(r["closest_ok"] for r in x),
                "same": sum(r["same"] for r in x), "trials_with_any_ok_window": sum(r["n_windows_ok"] > 0 for r in x),
                "no_window_eO": q([r["no_window"]["eO"] for r in x if r["no_window"]])}
        print(m, json.dumps(S[m], indent=1))
    json.dump(S, open(OUT + "/score.json", "w"), indent=1)
    json.dump(rows, open(OUT + "/trials.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""R37 §4-3 — `r37_replicate.json` を集計する（判定の表まで。判定基準は指示書のまま）。

f ＝ 対になった試行のうち、on で回転誤差と d_Ω（基準1）の**両方が off より小さくなった**割合。
Wilson 95% CI を添える。S・P 系列を分けて出し、合わせた値だけを書かない。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if not os.path.isdir("Registration"):
    os.chdir(REPO)

C_SET = ["m3_cor_b__E2", "m3_cor_b__E3", "m3_cor_c__E2", "m3_cor_d__E2", "m3_cor_d__E3"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
                       "Registration/output/diag/r37_replicate.json"))
    rows = d["rows"]
    out = {"provenance": d["provenance"], "precheck": d["precheck"]}
    g = defaultdict(list)
    for r in rows:
        g[(r["target"], r["series"])].append(r)

    def both(r):
        a, b = r["off"]["primary"], r["on"]["primary"]
        return (b["rot_deg"] < a["rot_deg"]
                and r["on"]["decomposition_primary"]["d_omega_m"]
                < r["off"]["decomposition_primary"]["d_omega_m"])

    print("## 条件 × 系列の f（回転と dΩ の両方が下がった割合）")
    table = {}
    for t in sorted({r["target"] for r in rows}):
        for s in ("S", "P"):
            rs = g[(t, s)]
            k = sum(both(r) for r in rs)
            lo, hi = wilson(k, len(rs))
            med = lambda key, side: float(np.median([r[side]["decomposition_primary"]["d_omega_m"]
                                                     if key == "d" else r[side]["primary"]["rot_deg"]
                                                     for r in rs]))
            table[(t, s)] = {"k": k, "n": len(rs), "f": k / len(rs) if rs else None,
                             "ci": [lo, hi],
                             "d_off_med": med("d", "off"), "d_on_med": med("d", "on"),
                             "rot_off_med": med("r", "off"), "rot_on_med": med("r", "on")}
            x = table[(t, s)]
            print("%-14s %s %2d/%2d f=%.2f [%.2f, %.2f]  dΩ中央 %.4f→%.4f  回転中央 %.3f→%.3f %s"
                  % (t, s, k, len(rs), x["f"], lo, hi, x["d_off_med"], x["d_on_med"],
                     x["rot_off_med"], x["rot_on_med"], "(c)" if t in C_SET else ""))
    out["per_condition"] = {"%s|%s" % k: v for k, v in table.items()}

    print("\n## シーン単位（E2＋E3 を合わせる。E2/E3 は source が同じなので独立ではない）")
    sc = {}
    for scene in sorted({r["target"].split("__")[0] for r in rows}):
        for s in ("S", "P"):
            rs = [r for r in rows if r["target"].startswith(scene) and r["series"] == s]
            k = sum(both(r) for r in rs)
            lo, hi = wilson(k, len(rs))
            sc["%s|%s" % (scene, s)] = {"k": k, "n": len(rs), "ci": [lo, hi]}
            print("%-9s %s %2d/%2d [%.2f, %.2f]" % (scene, s, k, len(rs), lo, hi))
    out["per_scene"] = sc

    print("\n## 判定の材料（(c) 5 条件で f ≥ 0.8 の数）")
    verdict = {}
    for s in ("S", "P"):
        n80 = sum(1 for t in C_SET if table[(t, s)]["f"] >= 0.8)
        verdict[s] = n80
        print("  %s 系列: %d / 5" % (s, n80))
    fails = {k: sum(1 for r in rows if r["failures"][k])
             for k in ("scale_collapse", "flip", "positive_feedback")}
    print("  失敗（%d 対）: %s" % (len(rows), fails))
    out["verdict_inputs"] = {"n_f_ge_0.8": verdict, "failures": fails, "n_pairs": len(rows)}

    print("\n## 主基準の成功数（試行単位）と3区分")
    ps = defaultdict(lambda: [0, 0, 0])
    tw = defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = r["series"]
        ps[key][0] += r["off"]["primary"]["success"]
        ps[key][1] += r["on"]["primary"]["success"]
        ps[key][2] += 1
        for side in ("off", "on"):
            tw[(key, side)][r[side]["verdict"]] += 1
    for s in ("S", "P"):
        print("  %s: 主基準 off %d/%d・on %d/%d | 3区分 off %s・on %s"
              % (s, ps[s][0], ps[s][2], ps[s][1], ps[s][2], dict(tw[(s, "off")]), dict(tw[(s, "on")])))
    out["primary_success"] = {s: {"off": ps[s][0], "on": ps[s][1], "n": ps[s][2]} for s in ps}
    out["three_way"] = {"%s|%s" % k: dict(v) for k, v in tw.items()}
    succ_rows = [(r["target"], r["series"], r["k"]) for r in rows
                 if r["on"]["primary"]["success"] or r["off"]["primary"]["success"]]
    print("  主基準で合格した試行:", succ_rows)

    print("\n## 反復・段階1・off の抜き取り")
    hit = sum(r["hit_max_iter"] for r in rows)
    drops = [r["last_iter_residual_drop"] for r in rows if r["last_iter_residual_drop"] is not None]
    print("  反復上限に達した: %d / %d。最後の反復の残差の下がり幅 中央 %.2e / 最大 %.2e / 負（上がった）%d"
          % (hit, len(rows), np.median(drops), max(drops), sum(1 for x in drops if x < 0)))
    oc = [r["off_check"] for r in rows if r["off_check"] is not None]
    print("  off 抜き取り: %d 件中 ビット一致 %d" % (len(oc), sum(o["bit_equal"] for o in oc)))
    wc = sum(r["stage1"]["winner_is_correct_rot"] for r in rows)
    print("  段階1 で正しい向きの候補が選ばれた: %d / %d" % (wc, len(rows)))
    s1 = [r["stage1"]["seed_d_omega_m"] for r in rows]
    print("  段階1 の並進誤差（正しい向きの種）: 中央 %.3f / 最大 %.3f m" % (np.median(s1), max(s1)))
    out["iter"] = {"hit_max": hit, "n": len(rows), "drop_median": float(np.median(drops)),
                   "drop_max": float(max(drops)), "drop_negative": sum(1 for x in drops if x < 0)}
    out["off_check"] = {"n": len(oc), "bit_equal": sum(o["bit_equal"] for o in oc)}
    out["stage1"] = {"winner_correct": wc, "n": len(rows),
                     "seed_d_median": float(np.median(s1)), "seed_d_max": float(max(s1))}

    leg = [(r["legacy_d_omega_primary"]["off"], r["off"]["decomposition_primary"]["d_omega_m"])
           for r in rows if r["legacy_d_omega_primary"]]
    if leg:
        ratio = [a / b for a, b in leg if b > 0]
        print("\n## 旧い採点（摂動前の Ω に T と G P^-1）/ 正しい採点 の比：中央 %.3f / 範囲 %.3f〜%.3f"
              % (np.median(ratio), min(ratio), max(ratio)))
        out["legacy_ratio"] = {"median": float(np.median(ratio)), "min": float(min(ratio)),
                               "max": float(max(ratio))}
    json.dump(out, open("Registration/output/diag/r37_summary.json", "w"), indent=1,
              ensure_ascii=False)


if __name__ == "__main__":
    main()

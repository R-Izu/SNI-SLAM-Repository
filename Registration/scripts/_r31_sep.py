"""R31 §3 — 探索半径ごとの分離境界を出し、予測 M1-a / M1-b に当てる。

**予測は R31 §3-3 に登録済み。ここで登録し直さない。**

    M1-a : max_corr_dist = 0.15 m のとき、境界は 0.17〜0.26 m に入る
    M1-b : max_corr_dist = 0.60 m のとき、境界は 0.72〜1.02 m に入る
"""

import json
import os

os.chdir("/home/student/rizu/SNI-SLAM")

PRED = {0.15: (0.17, 0.26), 0.60: (0.72, 1.02)}
FILES = [("Registration/output/diag/coverage_sweep_r0.30.json", 0.30),
         ("Registration/output/diag/coverage_sweep_r0.15.json", 0.15),
         ("Registration/output/diag/coverage_sweep_r0.60.json", 0.60)]

print("%-8s %6s %6s %12s %12s %8s %s"
      % ("半径[m]", "成功", "失敗", "成功側の端", "失敗側の端", "帯", "分離"))
print("-" * 78)
res = {}
for path, radius in FILES:
    if not os.path.exists(path):
        print("%-8.2f **未完**（%s が無い）" % (radius, os.path.basename(path)))
        continue
    d = json.load(open(path))
    rows = [r for r in d["rows"] if "error" not in r]
    s = [r for r in rows if r["success"]]
    f = [r for r in rows if not r["success"]]
    if not s or not f:
        print("%-8.2f %6d %6d   **片側しか無い。境界を出せない**"
              % (radius, len(s), len(f)))
        continue
    key = "stage1_seed_d_omega_m"
    s_edge = max(r[key] for r in s)
    f_edge = min(r[key] for r in f)
    band = sum(1 for r in rows if min(s_edge, f_edge) <= r[key] <= max(s_edge, f_edge))
    clean = s_edge < f_edge
    res[radius] = (s_edge, f_edge, clean)
    print("%-8.2f %6d %6d %12.4f %12.4f %8d %s"
          % (radius, len(s), len(f), s_edge, f_edge, 0 if clean else band,
             "完全分離" if clean else "**重なる**"))

print("\n## 予測 M1-a / M1-b の当たり外れ\n")
base = res.get(0.30)
if base:
    print("既定 0.30 m の境界: %.4f / %.4f m（比 %.2f〜%.2f 倍）"
          % (base[0], base[1], base[0] / 0.30, base[1] / 0.30))
for radius, (lo, hi) in sorted(PRED.items()):
    r = res.get(radius)
    if not r:
        print("  半径 %.2f m : **未測**" % radius)
        continue
    s_edge, f_edge, clean = r
    inside = (lo <= s_edge <= hi) and (lo <= f_edge <= hi)
    print("  半径 %.2f m : 境界 %.4f / %.4f m   予測範囲 [%.2f, %.2f] → **%s**"
          % (radius, s_edge, f_edge, lo, hi, "当たり" if inside else "外れ"))
    if base:
        print("      既定からの比: 成功側 %.2f 倍 / 失敗側 %.2f 倍   （半径の比 %.2f 倍）"
              % (s_edge / base[0], f_edge / base[1], radius / 0.30))
    if not clean:
        print("      **ただし分離していない（帯がある）。R31 §3-3 の外れ方3に当たる**")

print("\n## 対応の量（R31 §3-4：広げた代償を見る）\n")
print("%-8s %14s %14s %12s" % ("半径[m]", "半径内の対応数", "重みが正の対応", "source に占める割合"))
for path, radius in FILES:
    if not os.path.exists(path):
        continue
    rows = [r for r in json.load(open(path))["rows"]
            if "error" not in r and "correspondence" in r]
    if not rows:
        print("%-8.2f （対応数を記録していない実行）" % radius)
        continue
    import statistics as st
    print("%-8.2f %14d %14d %12.4f"
          % (radius,
             int(st.median(r["correspondence"]["n_corr_within_radius"] for r in rows)),
             int(st.median(r["correspondence"]["n_corr_positive_weight"] for r in rows)),
             st.median(r["correspondence"]["frac_src_matched"] for r in rows)))

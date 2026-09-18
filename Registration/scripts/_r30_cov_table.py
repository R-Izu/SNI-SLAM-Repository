"""被覆掃引の結果を、R30 §3-3 の予測 Q1〜Q3 に当てて読む。

**Q3（案A の効果）はまだ測れない**（案A が未実装）。Q1・Q2 のみ。
"""

import json
import os

import numpy as np

os.chdir("/home/student/rizu/SNI-SLAM")
d = json.load(open("Registration/output/diag/coverage_sweep.json"))
rows = [r for r in d["rows"] if "error" not in r]
bad = [r for r in d["rows"] if "error" in r]
print("有効 %d 件 / 失敗 %d 件\n" % (len(rows), len(bad)))
for b in bad:
    print("  失敗: keep=%s anchor=%s %s"
          % (b["keep_frac_nominal"], b["anchor"], b["error"][:70]))

print("## 隅ごとに並べる（**位置依存があるかを見る**）\n")
anchors = sorted({tuple(r["anchor"]) for r in rows})
for a in anchors:
    sub = sorted([r for r in rows if tuple(r["anchor"]) == a],
                 key=lambda x: -x["overlap_ratio"])
    print("### anchor %s" % (list(a),))
    print("  %9s %10s %10s %10s %9s %5s"
          % ("重なり率", "重心差[m]", "段階1[m]", "最終[m]", "最終回転", "成功"))
    for r in sub:
        print("  %9.4f %10.3f %10.3f %10.3f %9.2f %5s"
              % (r["overlap_ratio"], r["centroid_diff_m"],
                 r["stage1_seed_d_omega_m"], r["final_d_omega_m"],
                 r["final_rot_deg"], "○" if r["success"] else "×"))
    s = [r for r in sub if r["success"]]
    f = [r for r in sub if not r["success"]]
    if s and f:
        print("  → 境界: 成功の最小 %.4f / 失敗の最大 %.4f"
              % (min(r["overlap_ratio"] for r in s),
                 max(r["overlap_ratio"] for r in f)))
    print()

print("## Q1：重なり率を下げると段階1の並進誤差が単調に増えるか\n")
for a in anchors:
    sub = sorted([r for r in rows if tuple(r["adjust"] if False else r["anchor"]) == a],
                 key=lambda x: -x["overlap_ratio"])
    v = [r["stage1_seed_d_omega_m"] for r in sub]
    mono = all(v[i] <= v[i + 1] + 1e-9 for i in range(len(v) - 1))
    print("  anchor %-9s 段階1 誤差 %.3f → %.3f m   単調増加: %s"
          % (str(list(a)), v[0], v[-1], "**はい**" if mono else "**いいえ**"))

print("\n## Q2：成功率は境界で急に落ちるか（全隅をまとめて）\n")
allr = sorted(rows, key=lambda x: -x["overlap_ratio"])
s = [r for r in allr if r["success"]]
f = [r for r in allr if not r["success"]]
if s and f:
    lo_ok = min(r["overlap_ratio"] for r in s)
    hi_ng = max(r["overlap_ratio"] for r in f)
    print("  成功した最小の重なり率 : %.4f" % lo_ok)
    print("  失敗した最大の重なり率 : %.4f" % hi_ng)
    print("  **境界の幅 : %.4f（%.1f ポイント）**" % (abs(lo_ok - hi_ng),
                                                100 * abs(lo_ok - hi_ng)))
    overlap_region = [r for r in allr if min(lo_ok, hi_ng) <= r["overlap_ratio"]
                      <= max(lo_ok, hi_ng)]
    print("  境界帯に入る条件数 : %d" % len(overlap_region))
    if lo_ok > hi_ng:
        print("  **成功と失敗が重なっていない。単一の境界で分離できる**")
    else:
        print("  **成功と失敗が重なる帯がある。単一の境界では分離できない**")

print("\n## 最終 d_Ω と重なり率（閾値 0.1 m との関係）\n")
print("  %9s %10s %6s" % ("重なり率", "最終[m]", "成功"))
for r in allr:
    print("  %9.4f %10.3f %6s"
          % (r["overlap_ratio"], r["final_d_omega_m"], "○" if r["success"] else "×"))

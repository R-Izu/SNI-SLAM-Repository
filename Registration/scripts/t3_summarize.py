"""T3 の本結果（成功率 vs 被覆率）を表と図にする。

見たいこと（指示書 §3 T3）:
  1. 被覆率が下がると成功率がどう落ちるか
  2. median_axes と vertical_only で、どの被覆率から差がつくか
  3. 破綻の様式：初期スケールが原因か、対応付けが原因か
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "Registration/output/t3_coverage"
rows = json.load(open(os.path.join(root, "t3_results.json")))
ok = [r for r in rows if r.get("status") == "ok"]
LEVELS = [1.0, 0.75, 0.50, 0.30]
MODES = ["median_axes", "vertical_only"]


def pick(scene, level, mode):
    for r in ok:
        if r["scene"] == scene and abs(r["level_nominal"] - level) < 1e-9 \
                and r["mode"] == mode:
            return r
    return None


scenes = sorted({r["scene"] for r in ok})
print("成功率（Wilson 95%%CI）。N=%d 試行/セル、seed 固定" % ok[0]["trials"])
print("%-10s %-14s %-18s %-18s %-18s %-18s"
      % ("scene", "mode", "被覆100%", "75%", "50%", "30%"))
print("-" * 100)
for s in scenes:
    for m in MODES:
        cells = []
        for lv in LEVELS:
            r = pick(s, lv, m)
            cells.append("—" if r is None else
                         "%.2f [%.2f,%.2f]" % (r["success_rate"], r["ci_lo"], r["ci_hi"]))
        print("%-10s %-14s %-18s %-18s %-18s %-18s" % (s, m, *cells))
    print()

print("=" * 100)
print("\n【集計】被覆率ごとの成功率")
print("%-14s %8s %8s %8s %8s" % ("mode", "100%", "75%", "50%", "30%"))
agg = {}
for m in MODES:
    line = []
    for lv in LEVELS:
        rs = [pick(s, lv, m) for s in scenes]
        rs = [r for r in rs if r]
        k = sum(round(r["success_rate"] * r["trials"]) for r in rs)
        n = sum(r["trials"] for r in rs)
        p = k / n if n else float("nan")
        z = 1.96
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        agg[(m, lv)] = (p, c - h, c + h, k, n)
        line.append("%.3f" % p)
    print("%-14s %8s %8s %8s %8s" % (m, *line))

print()
for m in MODES:
    for lv in LEVELS:
        p, lo, hi, k, n = agg[(m, lv)]
        print("  %-14s 被覆 %3d%%: %.3f [%.3f, %.3f]  (%d/%d)"
              % (m, round(lv * 100), p, lo, hi, k, n))

print("\n【差】vertical_only − median_axes（正なら vertical_only が良い）")
for lv in LEVELS:
    a = agg[("median_axes", lv)]
    b = agg[("vertical_only", lv)]
    # 2標本の差の 95%CI（正規近似。N が大きいので十分）
    d = b[0] - a[0]
    se = np.sqrt(a[0] * (1 - a[0]) / a[4] + b[0] * (1 - b[0]) / b[4])
    print("  被覆 %3d%%: %+.3f [%+.3f, %+.3f]  %s"
          % (round(lv * 100), d, d - 1.96 * se, d + 1.96 * se,
             "有意" if abs(d) > 1.96 * se else "有意でない"))

print("\n【切り分け】初期スケール誤差と成功率の関係")
xs, ys, ms = [], [], []
for r in ok:
    if r.get("init_scale_error_ratio") is None:
        continue
    xs.append(abs(r["init_scale_error_ratio"] - 1.0))
    ys.append(r["success_rate"])
    ms.append(r["mode"])
xs, ys = np.array(xs), np.array(ys)


def spearman(a, b):
    def rank(x):
        o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x))
        for v in np.unique(x):
            k = x == v
            if k.sum() > 1:
                r[k] = r[k].mean()
        return r
    ra, rb = rank(a) - rank(a).mean(), rank(b) - rank(b).mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


print("  |初期スケール誤差| vs 成功率  Spearman rho = %+.3f (n=%d)"
      % (spearman(xs, ys), len(xs)))
for m in MODES:
    k = np.array([x == m for x in ms])
    if k.sum() > 3:
        print("    %-14s rho = %+.3f (n=%d)" % (m, spearman(xs[k], ys[k]), k.sum()))
print("  → 強い負なら「初期スケールが外れるほど失敗する」＝初期スケール由来。")
print("     弱ければ ICP が吸収しており、破綻の原因は対応付け側にある。")

print("\n【失敗の内訳】どの指標が閾値を超えたか")
# 「どの組み合わせで落ちたか」が切り分けの本体。回転が絡むならヨー候補の取り違え、
# 縮尺だけなら初期スケール由来、という読み方をする。
combos, crit, n_fail_tot = {}, {}, 0
for r in ok:
    fb = r.get("failure_breakdown") or {}
    n_fail_tot += fb.get("n_failed", 0)
    for k, v in (fb.get("combinations") or {}).items():
        combos[k] = combos.get(k, 0) + v
    for k, v in (fb.get("by_criterion") or {}).items():
        c = crit.setdefault(k, {"n": 0, "max_excess": 0.0})
        c["n"] += v.get("n", 0)
        if v.get("max_excess") is not None:
            c["max_excess"] = max(c["max_excess"], v["max_excess"])
print("  失敗 %d 試行 / 全 %d 試行" % (n_fail_tot, sum(r["trials"] for r in ok)))
print("  違反した指標の組み合わせ:")
for k, v in sorted(combos.items(), key=lambda kv: -kv[1]):
    print("    %-24s %5d  (%.1f%% of failures)" % (k, v, 100 * v / max(n_fail_tot, 1)))
print("  指標ごとの違反回数と最大超過:")
for k, v in crit.items():
    print("    %-14s n=%5d  max超過 %.4f" % (k, v["n"], v["max_excess"]))

# モード別・被覆別にも出す（どこでヨーが飛んでいるか）
print("\n  モード × 被覆ごとの『回転が絡む失敗』の割合:")
for m in MODES:
    line = []
    for lv in LEVELS:
        nf = rot = 0
        for s in scenes:
            r = pick(s, lv, m)
            fb = (r or {}).get("failure_breakdown") or {}
            nf += fb.get("n_failed", 0)
            for k, v in (fb.get("combinations") or {}).items():
                if "rot_deg" in k:
                    rot += v
        line.append("%.2f (%d/%d)" % (rot / nf, rot, nf) if nf else "—")
    print("    %-14s %-16s %-16s %-16s %-16s" % (m, *line))

# --- 図：成功率 vs 被覆率 ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 5))
for m, c in zip(MODES, ["#4C78A8", "#E45756"]):
    ys_ = [agg[(m, lv)][0] for lv in LEVELS]
    lo = [agg[(m, lv)][1] for lv in LEVELS]
    hi = [agg[(m, lv)][2] for lv in LEVELS]
    x = [lv * 100 for lv in LEVELS]
    ax.plot(x, ys_, "-o", color=c, label=m)
    ax.fill_between(x, lo, hi, color=c, alpha=0.18)
    for s in scenes:
        v = [pick(s, lv, m) for lv in LEVELS]
        ax.plot([lv * 100 for lv, r in zip(LEVELS, v) if r],
                [r["success_rate"] for r in v if r], color=c, alpha=0.18, lw=0.8)
ax.set_xlabel("reference coverage of the source floor area [%]")
ax.set_ylabel("recovery success rate")
ax.set_title("Success rate vs reference coverage (8 Replica scenes, 100 trials each)")
ax.invert_xaxis(); ax.grid(alpha=0.3); ax.legend()
p = os.path.join(root, "success_vs_coverage.png")
fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
print("\nwrote %s" % p)

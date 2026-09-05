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


def cells(scene, level, mode):
    """該当する**全 anchor** のセルを返す。

    ★ 以前の `pick()` は最初の1行だけ返しており、**anchor を集約していなかった**
      （R7 §4-1）。4隅で均したつもりの表が、実際には南西隅だけの値になっていた。
      傍証：報告した `0.870 [0.845, 0.892]` は k=696, n=800 の Wilson 区間と
      丸めまで一致する（n=3200 なら [0.858, 0.881]）。
    """
    return [r for r in ok
            if r["scene"] == scene and abs(r["level_nominal"] - level) < 1e-9
            and r["mode"] == mode]


def pool(rs):
    """セル群を**生の成功数と試行数**で合算する（率の平均ではない）。"""
    k = sum(round(r["success_rate"] * r["trials"]) for r in rs)
    n = sum(r["trials"] for r in rs)
    return k, n


scenes = sorted({r["scene"] for r in ok})
anchors = sorted({tuple(r["anchor"]) for r in ok})
print("★ この成功率は **自己一貫性**（手法自身の無摂動解へ戻れた率）であって、")
print("  **正解に合った率ではない**（R7 §0）。合成の T_gt は提案手法自身の出力なので、")
print("  この集計から精度を読んではならない。")
print()
print("集計の母数（R7 §4-2）: anchor %d 通り %s" % (len(anchors), anchors))
print("%-10s %-14s %-20s %-20s %-20s %-20s"
      % ("scene", "mode", "被覆100%", "75%", "50%", "30%"))
print("-" * 110)
for s in scenes:
    for m in MODES:
        cols = []
        for lv in LEVELS:
            rs = cells(s, lv, m)
            if not rs:
                cols.append("—")
                continue
            k, n = pool(rs)
            cols.append("%.2f (%d/%d, a=%d)" % (k / n, k, n, len(rs)))
        print("%-10s %-14s %-20s %-20s %-20s %-20s" % (s, m, *cols))
    print()
print("※ a= は集計に使った anchor 数。**100% 被覆は anchor を1つにまとめている**")
print("  （t3_partial_coverage.py:188-194）ので、a=1 になるのが正しい。")

print("=" * 100)
print("\n【集計】被覆率ごとの成功率")
print("%-14s %8s %8s %8s %8s" % ("mode", "100%", "75%", "50%", "30%"))
agg = {}
for m in MODES:
    line = []
    for lv in LEVELS:
        rs = [r for s in scenes for r in cells(s, lv, m)]
        k, n = pool(rs)
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
            for r in cells(s, lv, m):
                fb = r.get("failure_breakdown") or {}
                nf += fb.get("n_failed", 0)
                for k, v in (fb.get("combinations") or {}).items():
                    if "rot_deg" in k:
                        rot += v
        line.append("%.2f (%d/%d)" % (rot / nf, rot, nf) if nf else "—")
    print("    %-14s %-16s %-16s %-16s %-16s" % (m, *line))

# --- R5 §2-5: 主軸は「壁方向数」。被覆率は併記に留める --------------------
have_walls = [r for r in ok if (r.get("wall_directions") or {}).get("n_directions") is not None]
if have_walls:
    print("\n" + "=" * 100)
    print("【R5 §2 主軸】成功率 vs 残った壁方向数")
    print("  仮説: 成功率を決めているのは『どれだけ残ったか』ではなく")
    print("        『残った参照に壁が何方向あるか』（ヨー4候補を見分けられるか）")
    print("\n%-14s %-10s %8s %8s %18s %10s"
          % ("mode", "壁方向数", "セル数", "試行数", "成功率[95%CI]", "ヨー正解率"))
    print("-" * 78)
    for m in MODES:
        for nd in sorted({(r["wall_directions"]["n_directions"]) for r in have_walls}):
            rs = [r for r in have_walls
                  if r["mode"] == m and r["wall_directions"]["n_directions"] == nd]
            if not rs:
                continue
            k = sum(round(r["success_rate"] * r["trials"]) for r in rs)
            n = sum(r["trials"] for r in rs)
            p = k / n
            z = 1.96
            den = 1 + z * z / n
            c = (p + z * z / (2 * n)) / den
            h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
            yc = [r["yaw"]["frac_picked_correct"] for r in rs if r.get("yaw")]
            print("%-14s %-10d %8d %8d  %.3f [%.3f,%.3f] %10s"
                  % (m, nd, len(rs), n, p, c - h, c + h,
                     "%.3f" % np.mean(yc) if yc else "—"))

    # ヨーを正しく選べたときと選び損ねたときの成功率（機構の直接の証拠）
    yc_ok, yc_ng, mg_ok, mg_ng = [], [], [], []
    for r in have_walls:
        y = r.get("yaw")
        if not y:
            continue
        if y.get("frac_success_given_correct_yaw") is not None:
            yc_ok.append(y["frac_success_given_correct_yaw"])
        if y.get("frac_success_given_wrong_yaw") is not None:
            yc_ng.append(y["frac_success_given_wrong_yaw"])
        if y.get("median_margin_when_right") is not None:
            mg_ok.append(y["median_margin_when_right"])
        if y.get("median_margin_when_wrong") is not None:
            mg_ng.append(y["median_margin_when_wrong"])
    # R7 §4-4: セル別条件付き率の**中央値**と、全試行を**プールした率**を分けて出す。
    # 前者を P(success | correct yaw) と書いてはいけない（セルの重みが消えている）。
    n_ok = n_ok_s = n_ng = n_ng_s = 0
    for r in have_walls:
        y = r.get("yaw")
        if not y or not y.get("n"):
            continue
        c = round(y["frac_picked_correct"] * y["n"])
        w = y["n"] - c
        if y.get("frac_success_given_correct_yaw") is not None:
            n_ok += c
            n_ok_s += round(y["frac_success_given_correct_yaw"] * c)
        if y.get("frac_success_given_wrong_yaw") is not None:
            n_ng += w
            n_ng_s += round(y["frac_success_given_wrong_yaw"] * w)
    print("\n【機構の直接の証拠】※ここでの「成功」も**自己一貫性**である")
    print("  (a) セル別の条件付き率の中央値")
    if yc_ok:
        print("      ヨー正解時 %.3f (n=%d セル) / 四分位 [%.3f, %.3f]"
              % (np.median(yc_ok), len(yc_ok),
                 np.percentile(yc_ok, 25), np.percentile(yc_ok, 75)))
    if yc_ng:
        print("      ヨー誤り時 %.3f (n=%d セル) / 四分位 [%.3f, %.3f]"
              % (np.median(yc_ng), len(yc_ng),
                 np.percentile(yc_ng, 25), np.percentile(yc_ng, 75)))
    print("  (b) 全試行をプールした率（**こちらが P(成功|ヨー正解) にあたる**）")
    if n_ok:
        print("      ヨー正解時 %.3f (%d/%d 試行)" % (n_ok_s / n_ok, n_ok_s, n_ok))
    if n_ng:
        print("      ヨー誤り時 %.3f (%d/%d 試行)" % (n_ng_s / n_ng, n_ng_s, n_ng))
    print("  → 後者が前者より大きく低いなら、ヨーの選択が結果を決めていることになる")
    if mg_ok and mg_ng:
        print("\n  候補スコアの1位-2位差（margin）")
        print("    正解したとき   中央値 %.5f (n=%d)" % (np.median(mg_ok), len(mg_ok)))
        print("    取り違えたとき 中央値 %.5f (n=%d)" % (np.median(mg_ng), len(mg_ng)))
        print("  → 取り違え時の方が小さいなら『僅差で誤って選んでいる』")

    # 図：主軸 = 壁方向数
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, c in zip(MODES, ["#4C78A8", "#E45756"]):
        rs = [r for r in have_walls if r["mode"] == m]
        x = [r["wall_directions"]["n_directions"]
             + (0.06 if m == MODES[1] else -0.06) for r in rs]
        ax.scatter(x, [r["success_rate"] for r in rs], s=18, alpha=0.45,
                   color=c, label=m, linewidths=0)
    ax.set_xlabel("number of wall directions surviving in the reference")
    ax.set_ylabel("recovery success rate")
    ax.set_title("Success rate vs wall-direction diversity (R5 §2)")
    ax.grid(alpha=0.3); ax.legend()
    p2 = os.path.join(root, "success_vs_wall_directions.png")
    fig.tight_layout(); fig.savefig(p2, dpi=150); plt.close(fig)
    print("\nwrote %s" % p2)

# --- 図：成功率 vs 被覆率（併記） ---
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
        xs, ys2 = [], []
        for lv in LEVELS:
            rs = cells(s, lv, m)
            if rs:
                k, n = pool(rs)
                xs.append(lv * 100); ys2.append(k / n)
        ax.plot(xs, ys2, color=c, alpha=0.18, lw=0.8)
ax.set_xlabel("reference coverage of the source floor area [%]")
ax.set_ylabel("SELF-CONSISTENCY rate (not accuracy)")
ax.set_title("Self-consistency vs reference coverage\n"
             "(8 Replica scenes; synthetic T_gt is the method's own output)")
ax.invert_xaxis(); ax.grid(alpha=0.3); ax.legend()
p = os.path.join(root, "success_vs_coverage.png")
fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
print("\nwrote %s" % p)

"""実データ位置合わせの結果を集計し、**事前登録した P1/P2/P3 を判定する**。

[[2026-08-30_preregistration_yaw_hypothesis]] の予測：
  P1 壁方向が1方向の廊下4本は、2方向の室6本より成功率が低い
  P2 失敗した試行ではヨー候補の取り違えが過半を占める
  P3 取り違えたときの margin は、正解したときより小さい

**当たり／外れを明記する。** 後から条件を足して救わない。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "Registration/output/realdata"
rows = [r for r in json.load(open(os.path.join(root, "realdata_results.json")))
        if r.get("status") == "ok"]

# 事前登録した壁方向数（[[2026-08-27_r4f_tsdf_and_metric_retraction]] §C-3）
WALL_DIRS = {"m3_room_a": 2, "m3_room_b": 2, "m3_block_a": 2, "m3_block_b": 2,
             "m3_block_c": 2, "m3_block_d": 2,
             "m3_cor_a": 1, "m3_cor_b": 1, "m3_cor_c": 1, "m3_cor_d": 1}
VISUAL = {"m3_cor_a": 1, "m3_cor_b": 1, "m3_block_a": 3, "m3_block_b": 3,
          "m3_room_a": 5, "m3_room_b": 5, "m3_block_c": 7, "m3_block_d": 8}
COND = {"E1": "reference=410（直方体・囮あり）", "E2": "reference=411（L字）",
        "E3": "reference=411+410", "E4": "E1 を幾何のみ（意味なし）"}


def wilson(k, n, z=1.96):
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


print("=" * 96)
print("実データ位置合わせ — 10 シーン × 4 条件 × 100 試行 = %d 試行"
      % sum(r["trials"] for r in rows))
print("**1シーンも母数から外していない**（[[PROF]] C11）")
print("=" * 96)

for cond in ("E1", "E2", "E3", "E4"):
    rs = [r for r in rows if r["condition"] == cond]
    if not rs:
        continue
    print("\n### %s : %s" % (cond, COND[cond]))
    print("%-12s %4s %18s %11s %11s %11s %9s"
          % ("scene", "壁", "成功率[95%CI]", "直接回転°", "直接並進m", "直接縮尺",
             "ヨー正解"))
    print("-" * 88)
    for r in sorted(rs, key=lambda x: (-WALL_DIRS.get(x["scene"], 0), x["scene"])):
        y = r.get("yaw") or {}
        print("%-12s %4d  %.2f [%.2f,%.2f] %11s %11s %11s %9s"
              % (r["scene"], WALL_DIRS.get(r["scene"], 0),
                 r["success_rate"], r["ci_lo"], r["ci_hi"],
                 "%.2f" % r["direct_rot_deg"] if r["direct_rot_deg"] is not None else "—",
                 "%.3f" % r["direct_trans"] if r["direct_trans"] is not None else "—",
                 "%.4f" % r["direct_scale_ratio"] if r["direct_scale_ratio"] is not None else "—",
                 "%.3f" % y["frac_picked_correct"] if y else "—"))
    k = sum(round(r["success_rate"] * r["trials"]) for r in rs)
    n = sum(r["trials"] for r in rs)
    lo, hi = wilson(k, n)
    print("  条件まとめ: 成功率 %.3f [%.3f, %.3f]  (%d/%d)" % (k / n, lo, hi, k, n))

# ---- 直接誤差（GT との絶対誤差）だけを見る。ここが「実データで動いたか」の本体 ----
print("\n" + "=" * 96)
print("【★実データで位置合わせは成立したか】GT との直接誤差（摂動なしの1回）")
print("成功の基準: 回転 < 5°、並進 < 0.1 m、縮尺比 < 0.05")
print("=" * 96)
print("%-12s %-4s %4s %10s %10s %10s %8s" %
      ("scene", "cond", "壁", "回転°", "並進m", "縮尺比", "判定"))
print("-" * 66)
direct_ok = []
for r in sorted(rows, key=lambda x: (x["condition"], -WALL_DIRS.get(x["scene"], 0))):
    rd, td, sd = r["direct_rot_deg"], r["direct_trans"], r["direct_scale_ratio"]
    if rd is None:
        continue
    ok = rd < 5.0 and td < 0.1 and sd < 0.05
    direct_ok.append({"scene": r["scene"], "cond": r["condition"], "ok": ok,
                      "rot": rd, "trans": td, "scale": sd,
                      "walls": WALL_DIRS.get(r["scene"], 0)})
    print("%-12s %-4s %4d %10.2f %10.3f %10.4f %8s"
          % (r["scene"], r["condition"], WALL_DIRS.get(r["scene"], 0),
             rd, td, sd, "○" if ok else "×"))

n_ok = sum(d["ok"] for d in direct_ok)
lo, hi = wilson(n_ok, len(direct_ok))
print("\n**直接位置合わせの成功: %d / %d = %.3f [%.3f, %.3f]**"
      % (n_ok, len(direct_ok), n_ok / len(direct_ok), lo, hi))
for cond in ("E1", "E2", "E3", "E4"):
    d = [x for x in direct_ok if x["cond"] == cond]
    if d:
        print("  %s: %d/%d" % (cond, sum(x["ok"] for x in d), len(d)))
for w in (1, 2):
    d = [x for x in direct_ok if x["walls"] == w]
    if d:
        print("  壁%d方向: %d/%d" % (w, sum(x["ok"] for x in d), len(d)))

# ---- 事前登録の判定 ----
print("\n" + "=" * 96)
print("【事前登録の判定】2026-08-30 に、実データ実行 0 回の時点で記録した予測")
print("=" * 96)

# P1
cor = [r for r in rows if WALL_DIRS[r["scene"]] == 1]
room = [r for r in rows if WALL_DIRS[r["scene"]] == 2]
kc = sum(round(r["success_rate"] * r["trials"]) for r in cor)
nc = sum(r["trials"] for r in cor)
kr = sum(round(r["success_rate"] * r["trials"]) for r in room)
nr = sum(r["trials"] for r in room)
pc, pr = kc / nc, kr / nr
se = np.sqrt(pc * (1 - pc) / nc + pr * (1 - pr) / nr)
d = pc - pr
print("\nP1『壁1方向の廊下4本は、2方向の室6本より成功率が低い』")
print("  廊下(壁1方向) %.3f [%.3f, %.3f]  (%d/%d)" % (pc, *wilson(kc, nc), kc, nc))
print("  室  (壁2方向) %.3f [%.3f, %.3f]  (%d/%d)" % (pr, *wilson(kr, nr), kr, nr))
print("  差 %+.3f [%+.3f, %+.3f]" % (d, d - 1.96 * se, d + 1.96 * se))
print("  → **%s**" % ("的中（廊下の方が有意に低い）" if d < 0 and abs(d) > 1.96 * se
                     else "外れ（廊下の方が高い）" if d > 0 and abs(d) > 1.96 * se
                     else "判定できない（差が有意でない）"))

# P2
tot_fail = sum((r["failure_breakdown"] or {}).get("n_failed", 0) for r in rows)
rot_fail = 0
for r in rows:
    for k_, v in ((r["failure_breakdown"] or {}).get("combinations") or {}).items():
        if "rot_deg" in k_:
            rot_fail += v
print("\nP2『失敗した試行ではヨー候補の取り違えが過半を占める』")
print("  失敗 %d 試行のうち、回転が絡むもの %d (%.1f%%)"
      % (tot_fail, rot_fail, 100 * rot_fail / max(tot_fail, 1)))
yc = [r["yaw"]["frac_picked_correct"] for r in rows if r.get("yaw")]
print("  ヨー正解率 中央値 %.3f（全 %d 条件）" % (np.median(yc), len(yc)))
print("  → **%s**" % ("的中" if rot_fail / max(tot_fail, 1) > 0.5 else "外れ"))

# P3
mw = [r["yaw"]["median_margin_when_wrong"] for r in rows
      if r.get("yaw") and r["yaw"].get("median_margin_when_wrong") is not None]
mr = [r["yaw"]["median_margin_when_right"] for r in rows
      if r.get("yaw") and r["yaw"].get("median_margin_when_right") is not None]
print("\nP3『取り違えたときの margin は、正解したときより小さい』")
if mw and mr:
    print("  正解時   中央値 %.5f (n=%d 条件)" % (np.median(mr), len(mr)))
    print("  取り違え時 中央値 %.5f (n=%d 条件)" % (np.median(mw), len(mw)))
    print("  → **%s**" % ("的中" if np.median(mw) < np.median(mr) else "外れ"))
else:
    print("  取り違えが起きた条件が %d 件しかなく、判定できない" % len(mw))

# 機構
so = [r["yaw"]["frac_success_given_correct_yaw"] for r in rows
      if r.get("yaw") and r["yaw"].get("frac_success_given_correct_yaw") is not None]
sw = [r["yaw"]["frac_success_given_wrong_yaw"] for r in rows
      if r.get("yaw") and r["yaw"].get("frac_success_given_wrong_yaw") is not None]
print("\n【機構】ヨーの正誤で成功率がどう変わるか（合成データでは 0.940 / 0.000 だった）")
if so:
    print("  ヨー正解時 成功率 中央値 %.3f (n=%d 条件)" % (np.median(so), len(so)))
if sw:
    print("  ヨー誤り時 成功率 中央値 %.3f (n=%d 条件)" % (np.median(sw), len(sw)))

with open(os.path.join(root, "realdata_summary.json"), "w") as f:
    json.dump({"direct": direct_ok,
               "P1": {"corridor": pc, "room": pr, "diff": d, "se": float(se)},
               "P2": {"n_failed": tot_fail, "rot_involved": rot_fail},
               "P3": {"margin_right": float(np.median(mr)) if mr else None,
                      "margin_wrong": float(np.median(mw)) if mw else None}},
              f, indent=2, ensure_ascii=False)
print("\nwrote %s" % os.path.join(root, "realdata_summary.json"))

"""どの量が成功と失敗をいちばんきれいに分けるか。

**Q2 は「重なり率を境に急に落ちる」と予測していた。**
隅ごとに見ると急だが、**隅をまたぐと境界が動く**。
そこで「重なり率・重心差・段階1の誤差」のどれが分離をよく説明するかを見る。

**これは事後の探索である。予測ではない。当たり外れを主張しない。**
"""

import json
import os

os.chdir("/home/student/rizu/SNI-SLAM")
d = json.load(open("Registration/output/diag/coverage_sweep.json"))
rows = [r for r in d["rows"] if "error" not in r]
s = [r for r in rows if r["success"]]
f = [r for r in rows if not r["success"]]
print("成功 %d 件 / 失敗 %d 件\n" % (len(s), len(f)))

print("%-22s %12s %12s %10s %s"
      % ("量", "成功側の端", "失敗側の端", "重なり", "分離"))
print("-" * 76)
for key, label, lower_is_better in (
        ("overlap_ratio", "重なり率", False),
        ("centroid_diff_m", "構造重心の差 [m]", True),
        ("stage1_seed_d_omega_m", "段階1 の並進誤差 [m]", True)):
    if lower_is_better:
        s_edge = max(r[key] for r in s)      # 成功のうち最も悪い
        f_edge = min(r[key] for r in f)      # 失敗のうち最も良い
        clean = s_edge < f_edge
        band = f_edge - s_edge
    else:
        s_edge = min(r[key] for r in s)
        f_edge = max(r[key] for r in f)
        clean = s_edge > f_edge
        band = s_edge - f_edge
    n_mix = sum(1 for r in rows
                if min(s_edge, f_edge) <= r[key] <= max(s_edge, f_edge))
    print("%-22s %12.4f %12.4f %10d %s"
          % (label, s_edge, f_edge, 0 if clean else n_mix,
             "**完全に分離**（幅 %.4f）" % band if clean else "**重なる**"))

print("\n**読み方**")
print("  成功側の端 = 成功したもののうち最も条件が悪いもの")
print("  失敗側の端 = 失敗したもののうち最も条件が良いもの")
print("  この2つが入れ替わっていなければ、その量だけで成功／失敗を切り分けられる")

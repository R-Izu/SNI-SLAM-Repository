"""proposed と proposed_no_semantic が本当に同じ解を出しているのかを確かめる。

**chamfer が小数4桁まで一致して見える。** アブレーションが効いていないのか、
効いたうえで同じ解に収束しているのかを、試行ごとの誤差で見分ける。
**丸めの一致を「同一」と報告しない。**
"""

import collections
import csv
import os

os.chdir("/home/student/rizu/SNI-SLAM")

SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1",
          "office_2", "office_3", "office_4"]

print("%-10s %12s %12s %12s" % ("シーン", "試行数", "完全一致", "最大差[m]"))
print("-" * 50)
tot_same = tot_n = 0
worst = 0.0
for s in SCENES:
    p = "output/Registration/gt_a/%s/trials.csv" % s
    if not os.path.exists(p):
        continue
    by = collections.defaultdict(dict)
    for r in csv.DictReader(open(p)):
        by[int(r["trial"])][r["method"]] = r
    same = n = 0
    mx = 0.0
    for t, d in by.items():
        a, b = d.get("proposed"), d.get("proposed_no_semantic")
        if not (a and b):
            continue
        n += 1
        va, vb = float(a["gt_trans"]), float(b["gt_trans"])
        if va == vb:
            same += 1
        mx = max(mx, abs(va - vb))
    tot_same += same
    tot_n += n
    worst = max(worst, mx)
    print("%-10s %12d %12d %12.6f" % (s, n, same, mx))

print("\n合計 %d / %d 試行がビット一致  最大差 %.6f m" % (tot_same, tot_n, worst))
if tot_same == tot_n:
    print("**全試行で同一。アブレーションが解に一切影響していない**")
elif tot_same == 0:
    print("**一致する試行は無い。アブレーションは効いており、"
          "成功率が同じでも別の解である**")
else:
    print("**一部だけ一致。効いているが、多くの試行で同じ解に収束している**")

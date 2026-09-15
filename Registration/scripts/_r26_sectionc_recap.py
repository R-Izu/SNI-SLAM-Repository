"""R26 §1 の付随確認：sectionC の 8 シーンの成功率を、**何に対する成功か**を明示して並べる。

**83.8% がどう積み上がっているかを、母数つきで出す。**
"""

import json
import os

os.chdir("/home/student/rizu/SNI-SLAM")

SC = "output/Registration/sectionC"
SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1",
          "office_2", "office_3", "office_4"]

print("%-10s %-22s %10s %10s %10s" % ("シーン", "成功の基準", "proposed",
                                      "baseline", "試行数"))
print("-" * 70)
tot = {"proposed": [0, 0], "baseline_open3d": [0, 0]}
for s in SCENES:
    log = os.path.join(SC, "%s_benchmark.log" % s)
    frozen = not ("no frozen T_gt" in open(log).read())
    basis = "旧 T_gt（提案手法の解）" if frozen else "各手法 自分の無摂動解"
    import csv
    got = {}
    with open(os.path.join(SC, s, "results.csv")) as f:
        for r in csv.DictReader(f):
            got[r["method"]] = r

    def cell(m):
        r = got.get(m)
        if not r:
            return "—", 0.0, 0
        n = int(float(r["robust_trials"]))
        v = float(r["success_rate"])
        return "%.3f" % v, v * n, n

    a, an, n1 = cell("proposed")
    b, bn, n2 = cell("baseline_open3d")
    tot["proposed"][0] += an
    tot["proposed"][1] += n1
    tot["baseline_open3d"][0] += bn
    tot["baseline_open3d"][1] += n2
    print("%-10s %-22s %10s %10s %10d" % (s, basis, a, b, n1))

for m, (k, n) in tot.items():
    if n:
        print("\n%-16s 合計 %.1f / %d = **%.1f%%**" % (m, k, n, 100.0 * k / n))

"""room_1 で番人が実際に働いたかを数える。

**「落ちなくなった」だけでは足りない。** 非有限が本当に出ていて、
それが失敗として記録されたことを確かめる。出ていなければ、
今回落ちなかったのは番人のおかげではなく偶然である。
"""

import collections
import csv
import os

os.chdir("/home/student/rizu/SNI-SLAM")

for scene in ("room_1",):
    p = "output/Registration/gt_a/%s/trials.csv" % scene
    n = collections.Counter()
    deg = collections.Counter()
    for r in csv.DictReader(open(p)):
        m = r["method"]
        n[m] += 1
        if r.get("degenerate", "").strip().lower() in ("true", "1"):
            deg[m] += 1
    print("=== %s ===" % scene)
    for m in n:
        print("  %-22s degenerate %3d / %3d 試行" % (m, deg[m], n[m]))
    print("  合計 degenerate %d / %d" % (sum(deg.values()), sum(n.values())))

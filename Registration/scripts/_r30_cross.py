"""閾値曲線の要点を抜く：どこで 0 を離れ、どこで飽和するか。

**曲線から閾値を決めるためではない**（R30 §2-3）。
**「0.1 m を動かしたら結果が変わるのか」という問いに、数で答えるためである。**
"""

import json
import os

os.chdir("/home/student/rizu/SNI-SLAM")
d = json.load(open("Registration/output/diag/threshold_curve.json"))

for name, sec in d["sections"].items():
    if "curves" not in sec:
        continue
    c = sec["curves"]["trans"]
    n = sec["n"]
    first = next((x for x in c if x["k"] > 0), None)
    full = next((x for x in c if x["k"] == n), None)
    q = sec.get("quantiles", {})
    print("=== %s（n=%d）===" % (name, n))
    if q:
        print("  d_Ω 最小 %.4f m / 中央 %.4f m / 最大 %.4f m"
              % (q["min"], q["median"], q["max"]))
    print("  並進閾値を 0.02→1.0 m で振ったとき：")
    print("    最初に成功が出る閾値 : %s"
          % ("%.2f m（%d/%d）" % (first["threshold"], first["k"], n)
             if first else "**1.0 m まで 0 のまま**"))
    print("    全件が通る閾値       : %s"
          % ("%.2f m" % full["threshold"] if full else "**1.0 m でも全件は通らない**"))
    reg = next((x for x in c if abs(x["threshold"] - 0.1) < 1e-9), None)
    if reg:
        print("    事前登録 0.1 m      : %d/%d = %.1f%%"
              % (reg["k"], n, 100 * reg["rate"]))
    print()

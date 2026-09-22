"""R28 §2 の 2 — **mesh が残っている run すべての ATE を、同じ手順で出す。**

`eval_ate.json` が無い run（`260310_test3`・`Bexp_1`〜`4` など）も計算する。

手順は R26 §3-1 で検証済みのもの：**推定軌跡を GT 軌跡へ Umeyama で合わせ、
残差の RMS を取る。** この手順は `eval_ate.json` の値を 0.4380 対 0.4385 m で
再現している（同じ量である）。

**run を選ぶためではない。run 間の広がりを測るためである**（R28 §2-2）。
"""

import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration")
os.chdir("/home/student/rizu/SNI-SLAM")

from regbim.scale import umeyama      # noqa: E402

rows = []
for d in sorted(glob.glob("output/Replica/room0_official/*/")):
    name = os.path.basename(d.rstrip("/"))
    cks = sorted(glob.glob(os.path.join(d, "ckpts", "*.tar")))
    if not cks:
        continue
    try:
        ck = torch.load(cks[-1], map_location="cpu")
        est = np.asarray(ck["estimate_c2w_list"], dtype=np.float64)[:, :3, 3]
        gt = np.asarray(ck["gt_c2w_list"], dtype=np.float64)[:, :3, 3]
    except Exception as ex:
        rows.append({"run": name, "error": "%s" % type(ex).__name__})
        continue
    n = len(est)
    R, t, s = umeyama(est, gt, with_scaling=True)
    resid = np.linalg.norm((est @ (s * R).T + t) - gt, axis=1)
    saved = None
    p = os.path.join(d, "eval_ate.json")
    if os.path.exists(p):
        try:
            saved = json.load(open(p))["absolute_translational_error.rmse"]
        except Exception:
            pass
    rows.append({"run": name, "n": n,
                 "ate_cm": 100 * float(np.sqrt((resid ** 2).mean())),
                 "median_cm": 100 * float(np.median(resid)),
                 "max_cm": 100 * float(resid.max()),
                 "saved_cm": saved,
                 "ckpt": os.path.basename(cks[-1]),
                 "has_mesh": os.path.exists(
                     os.path.join(d, "mesh", "final_mesh_semantic.ply"))})

print("%-18s %6s %10s %10s %10s %12s %6s %s"
      % ("run", "姿勢数", "ATE[cm]", "中央[cm]", "最大[cm]", "保存値[cm]", "mesh", "ckpt"))
print("-" * 92)
for r in rows:
    if "error" in r:
        print("%-18s **読めない** %s" % (r["run"], r["error"]))
        continue
    agree = ("—" if r["saved_cm"] is None
             else ("一致" if abs(r["saved_cm"] - r["ate_cm"]) < 0.5 else "**不一致**"))
    print("%-18s %6d %10.2f %10.2f %10.2f %12s %6s %s"
          % (r["run"], r["n"], r["ate_cm"], r["median_cm"], r["max_cm"],
             ("—" if r["saved_cm"] is None else "%.2f(%s)" % (r["saved_cm"], agree)),
             "あり" if r["has_mesh"] else "—", r["ckpt"]))

ok = [r for r in rows if "error" not in r]
mesh = [r for r in ok if r["has_mesh"]]
print("\n## run 間の広がり")
a = [r["ate_cm"] for r in ok]
print("  全 %d run : %.2f 〜 %.2f cm（**%.1f 倍**）" % (len(a), min(a), max(a),
                                                     max(a) / min(a)))
m = [r["ate_cm"] for r in mesh]
print("  mesh がある %d run : %.2f 〜 %.2f cm（**%.1f 倍**）"
      % (len(m), min(m), max(m), max(m) / min(m)))
print("  中央 %.2f cm" % float(np.median(m)))
used = [r for r in ok if r["run"] == "260310_test4"]
if used:
    rank = sorted(m).index(used[0]["ate_cm"]) + 1
    print("\n  **位置合わせが使っている 260310_test4 は %.2f cm。**"
          "mesh がある %d run 中、小さい方から %d 番目"
          % (used[0]["ate_cm"], len(m), rank))

out = "Registration/output/diag/r28_ate_all.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"note": "Umeyama 残差 RMS。eval_ate.json と同じ量であることを R26 §3-1 で確認",
           "rows": rows}, open(out, "w"), indent=2, ensure_ascii=False)
print("\nwrote %s" % out)

"""R32 §4 / R31 §4 — **分離境界はシーンをまたいで安定しているか。**

R31 §4 が見たいことは1つだけである：

| 結果 | 意味 |
|---|---|
| 8 シーンで境界がほぼ一致 | 破綻点は手法の性質である |
| シーンごとに大きく違う | `room_0` の 0.393 / 0.468 m は一般化できない |
| 分離しないシーンがある | 律速変数の主張を弱める必要がある |

**R32 §0-1 の但し書き**：分離は半径に依存すると分かったので、
**一致しても「既定の半径のもとでの一致」でしかない。**
"""

import glob
import json
import os

import numpy as np

os.chdir("/home/student/rizu/SNI-SLAM")

FILES = [("room_0", "Registration/output/diag/coverage_sweep_r0.30.json")]
for p in sorted(glob.glob("Registration/output/diag/coverage_sweep_room_*.json")
                + glob.glob("Registration/output/diag/coverage_sweep_office_*.json")):
    FILES.append((os.path.basename(p)[len("coverage_sweep_"):-len(".json")], p))

print("%-10s %5s %5s %12s %12s %6s %s"
      % ("シーン", "成功", "失敗", "成功側の端", "失敗側の端", "帯", "分離"))
print("-" * 74)
rows = []
for scene, p in FILES:
    if not os.path.exists(p):
        print("%-10s **未完**" % scene)
        continue
    d = json.load(open(p))
    rs = [r for r in d["rows"] if "error" not in r]
    s = [r for r in rs if r["success"]]
    f = [r for r in rs if not r["success"]]
    if not s or not f:
        print("%-10s %5d %5d   **片側しか無い。境界を出せない**"
              % (scene, len(s), len(f)))
        rows.append({"scene": scene, "n_s": len(s), "n_f": len(f),
                     "clean": None})
        continue
    k = "stage1_seed_d_omega_m"
    se, fe = max(r[k] for r in s), min(r[k] for r in f)
    band = sum(1 for r in rs if min(se, fe) <= r[k] <= max(se, fe))
    clean = se < fe
    rows.append({"scene": scene, "n_s": len(s), "n_f": len(f),
                 "s_edge": se, "f_edge": fe, "band": band, "clean": clean})
    print("%-10s %5d %5d %12.4f %12.4f %6d %s"
          % (scene, len(s), len(f), se, fe, 0 if clean else band,
             "完全分離" if clean else "**重なる**"))

ok = [r for r in rows if r.get("clean") is True]
mixed = [r for r in rows if r.get("clean") is False]
none = [r for r in rows if r.get("clean") is None]
print("\n## まとめ")
print("  完全分離したシーン : %d / %d" % (len(ok), len(rows)))
print("  重なったシーン     : %d" % len(mixed))
print("  境界が出せないシーン: %d" % len(none))
if ok:
    se = [r["s_edge"] for r in ok]
    fe = [r["f_edge"] for r in ok]
    print("\n  成功側の端: %.4f 〜 %.4f m（中央 %.4f）"
          % (min(se), max(se), float(np.median(se))))
    print("  失敗側の端: %.4f 〜 %.4f m（中央 %.4f）"
          % (min(fe), max(fe), float(np.median(fe))))
    print("  **境界はシーンをまたいで %.1f 倍の幅に散る**" % (max(se) / min(se)))
    # 全シーンを一括で見たときに分離するか
    allr = []
    for scene, p in FILES:
        if os.path.exists(p):
            allr += [r for r in json.load(open(p))["rows"] if "error" not in r]
    k = "stage1_seed_d_omega_m"
    S = [r[k] for r in allr if r["success"]]
    F = [r[k] for r in allr if not r["success"]]
    if S and F:
        print("\n  **全シーンをまとめると**: 成功側の端 %.4f / 失敗側の端 %.4f → %s"
              % (max(S), min(F),
                 "分離する" if max(S) < min(F) else "**重なる**"))
        band = sum(1 for v in S + F if min(max(S), min(F)) <= v <= max(max(S), min(F)))
        print("  帯に入る条件 %d / %d" % (band, len(S) + len(F)))

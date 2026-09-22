"""R29 §6 のどの外れ方に当たったかを、段階別の候補から判定する。

段階：生成直後 → 短い精緻化後 → 通常精緻化後 → 最終
各段階の候補集合について E_g = min d_Ω(T, T_GT) を出す（R29 §3）。

**⚠ E_g の最小候補を勝者にしてはいけない。診断用である。**
"""

import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration")
sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration/scripts")
os.chdir("/home/student/rizu/SNI-SLAM")

from criterion_verdict import build_omega, d_omega     # noqa: E402

d = json.load(open("Registration/output/diag/plan_a_compare.json"))
rows = [r for r in d["rows"] if r.get("mode") == "plan_correlate"
        and "error" not in r and r.get("stage_candidates")]

print("%-16s %10s %10s %10s %10s %s"
      % ("条件", "生成直後", "短精緻化後", "最終", "採用候補", "判定"))
print("-" * 86)
cache = {}
for r in rows:
    scene = r["target"].split("__")[0]
    if scene not in cache:
        cfg = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % r["target"]))
        c = dict(cfg, source=dict(cfg["source"], seed=0))
        G = np.asarray(json.load(open(
            "output/GT_alignment_probe/T_gt/T_gt_%s.json" % scene))["T_gt"],
            dtype=np.float64).reshape(4, 4)
        cache[scene] = (build_omega(c), G)
    omega, G = cache[scene]
    gen = [d_omega(np.asarray(c["T_gen"]), G, omega) for c in r["stage_candidates"]]
    sho = [d_omega(np.asarray(c["T_short"]), G, omega) for c in r["stage_candidates"]]
    E_gen, E_sho = min(gen), min(sho)
    fin = r["final_d_omega_m"]
    # 最終に採用された候補が、短精緻化後の最良だったか
    k_best = int(np.argmin(sho))
    adopted_is_best = abs(sho[k_best] - fin) < 0.15
    if E_sho < fin - 0.15:
        verdict = "**良い候補が残ったが別が選ばれた**"
    elif fin > E_gen + 0.15:
        verdict = "精緻化で悪化"
    elif abs(fin - E_gen) < 0.15:
        verdict = "**精緻化がほぼ動かしていない**"
    else:
        verdict = "精緻化で改善"
    print("%-16s %10.3f %10.3f %10.3f %10d %s"
          % (r["target"], E_gen, E_sho, fin, k_best, verdict))

print("\n**E_g は診断用である。最小候補を勝者にしていない。**")
print("判定の幅 0.15 m は d_Ω の実測ばらつき（参照の標本で約 1 mm）より十分大きく取った")

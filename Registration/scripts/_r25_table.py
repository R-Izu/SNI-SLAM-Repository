"""R25 追補：16 件の STAGES.json から「どの段階で外れるか」を1枚の表にする。

**正解回転候補**（GT からの回転誤差が最小の候補）について、
段階1（種）と段階2（ICP 後）の d_Ω を並べる。**ICP が動かしているかを見る。**
"""

import glob
import json
import os
import re

os.chdir("/home/student/rizu/SNI-SLAM")

print("%-14s %-14s | %8s %8s %8s | %8s %8s | %s"
      % ("条件", "scale_init", "正解候補", "S1 種", "S2 ICP後", "GT誤差の変化", "最終", "最終の回転"))
print("%-14s %-14s | %8s %8s %8s | %8s %8s | %s"
      % ("", "", "の回転", "d_Ω[m]", "d_Ω[m]", "[m]", "d_Ω[m]", "[度]"))
print("-" * 104)

for d in sorted(glob.glob("output/stages_m3_cor_*__E*__*")):
    m = json.load(open(os.path.join(d, "STAGES.json")))
    files = m["files"]
    seeds = {int(re.search(r"cand(\d+)", f["file"]).group(1)): f
             for f in files if f["file"].startswith("S1_seed")}
    icps = {int(re.search(r"cand(\d+)", f["file"]).group(1)): f
            for f in files if f["file"].startswith("S2_")}
    k = min(icps, key=lambda i: icps[i]["gt_rot_deg"])
    fin = [f for f in files if f["file"].startswith("S5_")][0]
    name = os.path.basename(d).replace("stages_", "")
    cond, si = name.rsplit("__", 1)
    print("%-14s %-14s | %7.2f° %8.3f %8.3f | %+8.3f %8.3f | %7.2f°"
          % (cond, si, icps[k]["gt_rot_deg"], seeds[k]["d_omega_m"],
             icps[k]["d_omega_m"], icps[k]["d_omega_m"] - seeds[k]["d_omega_m"],
             fin["d_omega_m"], fin["gt_rot_deg"]))

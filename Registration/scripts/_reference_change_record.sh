#!/usr/bin/env bash
# R10 §2-3 — 参照を室内面に切り替えたことで、点数とクラス構成がどう変わったかを記録する。
# 併せて §2-4（扉の is_inner が低い件）も数値で残す。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd /home/student/rizu/SNI-SLAM
conda activate sni-slam

python -W ignore - <<'PY'
import copy, json, sys
import numpy as np
import yaml
sys.path.insert(0, "Registration")
from regbim import io_utils

CLS = ["background", "wall", "door", "floor", "window", "ceiling"]
print("%-10s %-10s %9s   %s" % ("参照", "面", "点数", "クラス構成"))
print("-" * 92)
rows = {}
for tag in ("all", "411", "410", "411_410"):
    spaces = {"all": None, "411": ["411"], "410": ["410"],
              "411_410": ["411", "410"]}[tag]
    for inner in (False, True):
        cfg = yaml.safe_load(open("Registration/configs/m3_ifc.yaml"))
        cfg["reference"]["spaces"] = spaces
        cfg["reference"]["inner_only"] = inner
        cfg["reference"]["cache_path"] = "Registration/output/ifc/m3_ifc_%s.npz" % tag
        c = io_utils.load_reference_cloud(cfg)
        cc = {CLS[k]: int((c.labels == k).sum()) for k in sorted(np.unique(c.labels))}
        frac = {k: round(v / len(c), 3) for k, v in cc.items()}
        print("%-10s %-10s %9d   %s"
              % (tag, "室内面のみ" if inner else "表裏両面", len(c), frac))
        rows["%s_%s" % (tag, "inner" if inner else "both")] = {
            "n": len(c), "counts": cc, "fracs": frac}

print()
print("【R10 §2-4】扉の is_inner が低い件")
z = np.load("Registration/output/ifc/m3_ifc_all.npz", allow_pickle=False)
meta = json.loads(str(z["meta"]))
print("  クラス別 is_inner: %s" % meta["is_inner_frac_by_class"])
print("  判定の探査距離は 0.30 m。**幅がそれに満たない部位では判定が粗くなる。**")
print("  扉の見込み（枠の奥行き）は 0.15〜0.20 m 程度なので、扉は原理的に不利である。")
print("  扉は match_classes に入っているが、成功判定の主因ではない")
print("  （実データの失敗は回転 90/180 度が主で、扉の寄与は小さい）。")
print("  ただし『道具の分解能が対象より粗い』型の問題が1つ残っていることは記録する。")

with open("Registration/output/ifc/reference_change_record.json", "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)
print("\nwrote Registration/output/ifc/reference_change_record.json")
PY

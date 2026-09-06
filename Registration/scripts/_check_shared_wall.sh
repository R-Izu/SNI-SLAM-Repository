#!/usr/bin/env bash
# R9 §3-3-2 — 411 と 410 を隔てる壁で、**両面が is_inner になっていないか**。
#
# 幾何（ifc_survey / _ceil_probe より）:
#   壁 id=29630  x[-7.07, -6.92]  y[-0.63, 5.04]  厚さ 0.15 m
#   IfcSpace 166(=410) x[-13.66, -7.07]   -> x = -7.07 の面が 410 を向く
#   IfcSpace 413(=411) x[-13.66,  4.56]   -> x = -6.92 の面が 411 を向く
#
# 参照が 411 のみのとき、x = -7.07 側（410 を向く面）が残っていれば、
# **スキャンが見る面の 0.15 m 裏に、もう一枚の「室内面」がある**ことになる。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd /home/student/rizu/SNI-SLAM
conda activate sni-slam

python -W ignore - <<'PY'
import numpy as np
CLS = ["background", "wall", "door", "floor", "window", "ceiling"]

for tag in ("411", "410", "all"):
    z = np.load("Registration/output/ifc/m3_ifc_%s.npz" % tag, allow_pickle=False)
    p, lab, nrm, inner = z["points"], z["labels"], z["normals"], z["is_inner"].astype(bool)
    w = lab == CLS.index("wall")
    # 共有壁の帯（x が -7.2 〜 -6.8、y が壁の範囲内）で、法線が ±x のもの
    band = w & (p[:, 0] > -7.25) & (p[:, 0] < -6.75) \
             & (p[:, 1] > -0.63) & (p[:, 1] < 5.04) & (np.abs(nrm[:, 0]) > 0.9)
    if band.sum() == 0:
        print("=== %s ===  共有壁の帯に点なし" % tag)
        continue
    face410 = band & (nrm[:, 0] < 0)      # -x を向く = 410 側
    face411 = band & (nrm[:, 0] > 0)      # +x を向く = 411 側
    print("=== %s ===  共有壁の帯 %d 点" % (tag, band.sum()))
    for nm, m in (("410 を向く面", face410), ("411 を向く面", face411)):
        if m.sum() == 0:
            print("  %-12s 0 点" % nm)
            continue
        print("  %-12s %5d 点  x 中央値 %+.3f  うち is_inner %5d (%.3f)"
              % (nm, m.sum(), float(np.median(p[m, 0])),
                 int(inner[m].sum()), float(inner[m].mean())))
    both = inner[face410].sum() > 0 and inner[face411].sum() > 0
    if both:
        d = abs(float(np.median(p[face411 & inner, 0]))
                - float(np.median(p[face410 & inner, 0])))
        print("  ★ 両面とも is_inner。間隔 %.3f m（＝壁厚）" % d)
    else:
        print("  片面のみ is_inner")
PY

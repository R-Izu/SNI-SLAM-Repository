#!/usr/bin/env bash
# R9 §3-3-2 — `is_inner` の定義を確認する。
#
# 懸念（main）：部屋どうしが共有する壁では、**両面がそれぞれ別の IfcSpace にとって
# 室内側**になりうる。参照が 411 のみのとき、410 側を向いた面が残っていれば、
# スキャンが見ている面の 0.15〜0.20 m 裏に、もう一枚の「室内面」があることになる。
set -eo pipefail
source /opt/miniconda/3/etc/profile.d/conda.sh
cd /home/student/rizu/SNI-SLAM
conda activate sni-slam

python -W ignore - <<'PY'
import json
import numpy as np

CLS = ["background", "wall", "door", "floor", "window", "ceiling"]
for tag in ("all", "411", "410"):
    p = "Registration/output/ifc/m3_ifc_%s.npz" % tag
    z = np.load(p, allow_pickle=False)
    pts, lab, inner = z["points"], z["labels"], z["is_inner"].astype(bool)
    meta = json.loads(str(z["meta"]))
    print("=== %s ===  spaces=%s  点 %d" % (tag, meta.get("spaces_requested"), len(pts)))
    print("  is_inner 全体 %.3f / クラス別 %s"
          % (inner.mean(),
             {CLS[c]: round(float(inner[lab == c].mean()), 3)
              for c in sorted(np.unique(lab))}))

    # 壁の室内面が「何枚の平面に分かれているか」を、法線方向の投影で見る。
    # 共有壁の両面が残っていれば、**近接した2枚**が同じ向きで並ぶ。
    w = (lab == CLS.index("wall")) & inner
    if w.sum() < 100:
        print("  壁の室内面が %d 点しかない" % w.sum())
        continue
    n = z["normals"][w]
    q = pts[w]
    # 法線方向をクラスタリングせず、最頻の水平法線方向を1つ取り、その方向の
    # オフセット分布に山がいくつ立つかを見る
    hor = n.copy(); hor[:, 2] = 0
    ln = np.linalg.norm(hor, axis=1)
    k = ln > 0.9
    if k.sum() < 100:
        print("  水平法線の壁点が少ない")
        continue
    hn = hor[k] / ln[k, None]
    ang = np.degrees(np.arctan2(hn[:, 1], hn[:, 0])) % 360.0
    h, e = np.histogram(ang, bins=72, range=(0, 360))
    top = int(np.argmax(h))
    c0 = 0.5 * (e[top] + e[top + 1])
    sel = np.abs(((ang - c0 + 180) % 360) - 180) < 5.0
    d = (q[k][sel] * hn[sel]).sum(axis=1)      # その法線方向へのオフセット
    hh, ee = np.histogram(d, bins=np.arange(d.min(), d.max() + 0.02, 0.02))
    peaks = [(round(float(0.5 * (ee[i] + ee[i + 1])), 3), int(hh[i]))
             for i in range(len(hh)) if hh[i] > 0.15 * hh.max()]
    print("  最頻の壁方向 %.0f度 に %d 点。その法線方向のオフセットの山（>15%%）:"
          % (c0, sel.sum()))
    print("   ", peaks[:12])
    if len(peaks) >= 2:
        gaps = [round(peaks[i + 1][0] - peaks[i][0], 3) for i in range(len(peaks) - 1)]
        near = [g for g in gaps if 0.10 <= g <= 0.25]
        print("    山の間隔:", gaps[:10])
        if near:
            print("    ★ 壁厚（0.15〜0.20 m）と同程度の間隔が %d 個ある"
                  " -> **共有壁の両面が残っている疑い**" % len(near))
PY

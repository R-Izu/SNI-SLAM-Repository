"""並進が合わない原因を切り分ける — 部分被覆下の**重心合わせ**が効いているか。

実データで回転は 25/40 で通っているのに、並進が中央値 5.9 m ずれている。
`proposed` は stage 1 の `_plane_seed` でも stage 2 でも
**構造点の重心**を並進の基準にしている（`_struct_centroid`）。

参照が観測の一部しか覆わない場合、**2つの重心は同じ点を指さない**。
`m3_cor_c`（411+410+廊下）に対し参照が 411 だけなら、重心は数 m 離れる。
指示書 R3 T2 は**縮尺**についてこの問題を予告していたが、**並進については触れていない**。

並進誤差が重心のずれと相関するなら、原因はここである。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)

from regbim import io_utils, preprocess          # noqa: E402
from regbim.labels import NAME_TO_ID             # noqa: E402

d = {x["scene"] + "__" + x["cond"]: x
     for x in json.load(open("Registration/output/realdata/realdata_summary.json"))["direct"]}

print("%-12s %-3s %10s %14s %10s" % ("scene", "cond", "並進誤差m", "重心のずれm", "回転°"))
print("-" * 56)
rows = []
for cfg_p in sorted(glob.glob("Registration/configs/realdata/*__E[123].yaml")):
    key = os.path.basename(cfg_p)[:-5]
    if key not in d:
        continue
    cfg = yaml.safe_load(open(cfg_p))
    src = preprocess.prepare(io_utils.load_source_cloud(cfg), cfg)
    dst = preprocess.prepare(io_utils.load_reference_cloud(cfg), cfg)
    ids = [NAME_TO_ID[n] for n in cfg["classes"]["structural"]]
    T = np.asarray(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"], dtype=np.float64)
    # GT で正しい位置に置いたときの、構造点の重心どうしの距離
    sp = src.points[np.isin(src.labels, ids)] @ T[:3, :3].T + T[:3, 3]
    dp = dst.points[np.isin(dst.labels, ids)]
    off = float(np.linalg.norm(sp.mean(0) - dp.mean(0)))
    rows.append((d[key]["trans"], off, d[key]["rot"]))
    print("%-12s %-3s %10.3f %14.3f %10.2f"
          % (d[key]["scene"], d[key]["cond"], d[key]["trans"], off, d[key]["rot"]))

a = np.array(rows)
ok = a[:, 2] < 5.0                      # 回転が通ったものだけで見る
print("\n全 %d 件: 並進誤差 と 重心のずれ の相関 r = %+.3f"
      % (len(a), np.corrcoef(a[:, 0], a[:, 1])[0, 1]))
if ok.sum() > 3:
    print("回転が通った %d 件: r = %+.3f"
          % (ok.sum(), np.corrcoef(a[ok, 0], a[ok, 1])[0, 1]))
    print("  並進誤差 中央値 %.3f m / 重心のずれ 中央値 %.3f m"
          % (np.median(a[ok, 0]), np.median(a[ok, 1])))
    print("  比（並進誤差 / 重心のずれ）中央値 %.2f"
          % np.median(a[ok, 0] / np.maximum(a[ok, 1], 1e-6)))
print("\n→ 強い正の相関なら、原因は**部分被覆下で重心が対応しないこと**である。")

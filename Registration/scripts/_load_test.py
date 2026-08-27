"""受入条件の最後の1つ — `reference.type: ifc` が **位置合わせ本体の env** で読めるか。

`sni-slam` は py3.7 で ifcopenshell を持てないので、`io_utils._load_ifc_reference` は
bim-ifc が書いた .npz を読む経路になる。**その経路が実際に通ることを確かめる。**
併せて、下流が要求する条件（重力軸推定に floor/ceiling が要る）を満たすか見る。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)

from regbim.io_utils import load_reference_cloud, class_counts   # noqa: E402

cfg = yaml.safe_load(open("Registration/configs/m3_ifc.yaml"))

for name, spaces, cache in (
        ("全体(411+410)", None, "Registration/output/ifc/m3_ifc_all.npz"),
        ("411 (L字)", ["411"], "Registration/output/ifc/m3_ifc_411.npz"),
        ("410 (直方体)", ["410"], "Registration/output/ifc/m3_ifc_410.npz"),
        ("411+410", ["411", "410"], "Registration/output/ifc/m3_ifc_411_410.npz")):
    c = dict(cfg)
    c["reference"] = dict(cfg["reference"], spaces=spaces, cache_path=cache)
    cloud = load_reference_cloud(c)
    cc = class_counts(cloud)
    n = cloud.normals
    horiz_up = int(((np.abs(n[:, 2]) > 0.9) & (n[:, 2] > 0)).sum())
    print("%-14s N=%d  %s" % (name, len(cloud), cc))
    print("%-14s   法線あり=%s / 水平上向き %d 点 / 重力推定に必要な floor,ceiling = %s"
          % ("", n is not None, horiz_up,
             "○" if (cc.get("floor", 0) > 0 and cc.get("ceiling", 0) > 0) else "× 停止する"))

# キャッシュが config と食い違うときに黙って通らないこと（E1/E2 の取り違え防止）
c = dict(cfg)
c["reference"] = dict(cfg["reference"], spaces=["410"],
                      cache_path="Registration/output/ifc/m3_ifc_411.npz")
try:
    load_reference_cloud(c)
    print("\n[×] spaces が食い違うキャッシュを黙って読んでしまった")
except ValueError as e:
    print("\n[○] 食い違うキャッシュを拒否: %s" % str(e)[:100])

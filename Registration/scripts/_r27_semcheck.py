"""GT-A を作る前に：semantic_class の png が**何の id を持っているか**を確かめる。

`replica_to_six` は **Replica のクラス id**（93=wall, 40=floor, 31=ceiling, …）で書かれている。
png がクラス id を持つならそのまま使える。**インスタンス id なら `id_to_label` を挟む必要がある。**
**確かめずに作らない。**
"""

import json
import os
import sys

import cv2
import numpy as np

os.chdir("/home/student/rizu/SNI-SLAM")
scene = sys.argv[1] if len(sys.argv) > 1 else "room_0"

p = "data/replica/%s_official/semantic_class/semantic_class_0.png" % scene
im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
print("%s  shape=%s dtype=%s" % (p, im.shape, im.dtype))
u, c = np.unique(im, return_counts=True)
o = np.argsort(-c)
print("出現する id（多い順・上位 15）:")
for i in o[:15]:
    print("   id %4d : %7d 画素 (%.1f%%)" % (u[i], c[i], 100.0 * c[i] / im.size))
print("id の個数 %d / 最大 %d" % (len(u), u.max()))

info = json.load(open("data/replica/%s/habitat/info_semantic.json" % scene))
id_to_label = np.asarray(info["id_to_label"], dtype=np.int64)
print("\ninfo_semantic.json: id_to_label の長さ %d（インスタンス数）" % len(id_to_label))
print("  そこに現れるクラス id の種類: %s" % sorted(set(id_to_label.tolist()))[:25])
names = {c["id"]: c["name"] for c in info["classes"]}
for k in (93, 40, 31, 37, 97):
    print("  クラス %3d = %s" % (k, names.get(k, "（無い）")))

print("\n**判定**")
cls_ids = set(id_to_label.tolist())
as_class = len(set(u.tolist()) - cls_ids - {0})
as_inst = len(set(u.tolist()) - set(range(len(id_to_label))) - {0})
print("  png の id のうち、クラス id に無いもの: %d 種" % as_class)
print("  png の id のうち、インスタンス id の範囲外のもの: %d 種" % as_inst)
print("  → 前者が 0 ならクラス id。後者が 0 ならインスタンス id")

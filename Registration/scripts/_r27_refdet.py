"""参照点群が実行ごとに変わるかを確かめる（ビット一致を主張する前に）。

`_load_replica_reference` は `trimesh.sample.sample_surface` を seed 無しで呼ぶ。
**呼ぶたびに別の 20 万点が出るなら、どの実行もビット一致しない。**
コードを変えていなくても数値が動くので、**「変更がビット一致」という証明の立て方自体が使えない。**
"""

import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from regbim import io_utils   # noqa: E402

cfg = yaml.safe_load(open("Registration/configs/gt_a/room_0.yaml"))

a = io_utils.load_reference_cloud(cfg).points
b = io_utils.load_reference_cloud(cfg).points
print("参照点群を 2 回読んだ： %d 点 / %d 点" % (len(a), len(b)))
same = a.shape == b.shape and np.array_equal(a, b)
print("**同一か： %s**" % ("はい" if same else "いいえ（実行ごとに変わる）"))
if not same and a.shape == b.shape:
    print("  同じ添字の点どうしの距離: 中央 %.4f m / 最大 %.4f m"
          % (float(np.median(np.linalg.norm(a - b, axis=1))),
             float(np.linalg.norm(a - b, axis=1).max())))

s = cfg["source"]
c = io_utils.load_source_cloud(cfg).points
d = io_utils.load_source_cloud(cfg).points
print("\nsource（%s）を 2 回読んだ： 同一か： %s"
      % (s["type"], "はい" if np.array_equal(c, d) else "**いいえ**"))

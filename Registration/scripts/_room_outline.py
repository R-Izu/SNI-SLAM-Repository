"""室 411 の平面形を、y ごとの床の x 範囲として出す（1回限りの確認）。

18 m の東西測線が本当に室内を貫いているかを、図に頼らず数値で確かめる。
"""

import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from regbim import io_utils   # noqa: E402
from regbim.labels import NAME_TO_ID   # noqa: E402

cfg = yaml.safe_load(open("Registration/configs/realdata/m3_block_b__E2.yaml"))
dst = io_utils.load_reference_cloud(cfg)
fl = dst.points[dst.labels == NAME_TO_ID["floor"]]
print("BIM 床 %d 点   x [%.2f, %.2f]  y [%.2f, %.2f]"
      % (len(fl), fl[:, 0].min(), fl[:, 0].max(), fl[:, 1].min(), fl[:, 1].max()))
print("\n  y 帯 [m]        床の x 範囲 [m]        幅 [m]   点数")
lo, hi = fl[:, 1].min(), fl[:, 1].max()
for y0 in np.arange(np.floor(lo), np.ceil(hi), 1.0):
    m = (fl[:, 1] >= y0) & (fl[:, 1] < y0 + 1.0)
    if m.sum() < 20:
        print("  %+5.1f .. %+5.1f      （床点なし）" % (y0, y0 + 1))
        continue
    x = fl[m, 0]
    print("  %+5.1f .. %+5.1f     %+7.2f .. %+7.2f     %6.2f   %6d"
          % (y0, y0 + 1, x.min(), x.max(), x.max() - x.min(), m.sum()))

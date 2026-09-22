"""R34 §2-2 — **N=59 に戻して、式そのものを検証する。**

**予測 R1 は R34 §2-2 に登録済み。ここで登録し直さない。**

    予測 R1：セル境界の欠陥を直した小例は、N=59 でも 3 通りすべてが診断基準を満たす。

**N 以外は何も変えない。** 既定（170）はコード側に残したまま、ここでだけ上書きする。
"""

import os
import sys

import numpy as np

sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration")
sys.path.insert(0, "/home/student/rizu/SNI-SLAM/Registration/tests")
os.chdir("/home/student/rizu/SNI-SLAM")

from regbim import plan_correlate as pc          # noqa: E402
from regbim.labels import NAME_TO_ID             # noqa: E402
from test_plan_candidates import apply, box_room, sim3   # noqa: E402

OFF = 0.07          # 検査と同じずらし（セル格子から外す）
CRIT_D, CRIT_S = 1.0, 0.05          # R29 §3 の診断基準。**緩めない**


def run(nscale: int):
    dst_p, dst_l, dst_n = box_room(OFF, OFF, 8, 5)
    room = box_room(OFF, OFF, 8, 5)
    cor = box_room(16 + OFF, OFF, 22, 2.2)
    bp = np.vstack([room[0], cor[0]])
    bl = np.concatenate([room[1], cor[1]])
    bn = np.vstack([room[2], cor[2]])

    scales = np.exp(np.linspace(np.log(0.3), np.log(3.0), nscale))
    rows = []
    for s_true, t_true in ((1.0, [0.0, 0.0, 0.0]),
                           (0.85, [3.0, -2.0, 0.4]),
                           (1.30, [-5.0, 4.0, -0.3])):
        M = sim3(np.eye(3), np.asarray(t_true), s_true)
        src_p = apply(M, bp)
        T_truth = np.linalg.inv(M)
        cands, _ = pc.generate_candidates(src_p, bl, bn, dst_p, dst_l, dst_n,
                                          [np.eye(3)], NAME_TO_ID,
                                          {"scale_n": nscale})
        best, best_s = None, None
        for c in cands:
            d = np.linalg.norm(apply(c["T"], src_p) - apply(T_truth, src_p), axis=1)
            r = float(np.sqrt((d ** 2).mean()))
            if best is None or r < best:
                best, best_s = r, abs(c["scale"] * s_true - 1.0)
        # 式が予測する量：最寄り格子点の誤差 × 広がり
        k = int(np.argmin(np.abs(scales - 1.0 / s_true)))
        rel = abs(scales[k] * s_true - 1.0)
        rows.append({"s_true": s_true, "d": best, "se": best_s, "rel": rel,
                     "disp8": rel * 8.0,
                     "ok": best is not None and best < CRIT_D and best_s < CRIT_S})
    return rows


for n in (59, 170):
    print("\n## N = %d（刻み %.4f%%）"
          % (n, 100 * (np.exp(np.log(3.0 / 0.3) / (n - 1)) - 1)))
    print("  %-8s %10s %10s %12s %12s %6s"
          % ("真の縮尺", "d_Ω[m]", "縮尺誤差", "最寄り格子", "L=8m の変位", "判定"))
    for r in run(n):
        print("  %-8.2f %10.3f %9.1f%% %11.3f%% %10.0f mm %6s"
              % (r["s_true"], r["d"], 100 * r["se"], 100 * r["rel"],
                 1000 * r["disp8"], "合格" if r["ok"] else "**不合格**"))
    n_ok = sum(1 for r in run(n) if r["ok"])
    print("  → %d / 3 合格" % n_ok)

print("\n**予測 R1（R34 §2-2）：N=59 でも 3 通りすべてが診断基準を満たす**")
print("基準（R29 §3）：d_Ω < 1.0 m かつ 縮尺誤差 < 5%。**緩めていない**")
print("式の基準（R33 §3-1）：最寄り格子点の誤差 × 広がり ≤ セル幅の半分 = 125 mm")

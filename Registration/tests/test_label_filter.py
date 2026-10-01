"""R42 §4-2 `label_filter`：既定（on）は従来と同じ。off は background の点を対応に入れる。

1. 引数を省いたときと label_filter=True が同じ行列を返す（既定の動作が変わらない）
2. background の点が無ければ、on と off は同じ行列を返す（off が足すのは background の対応だけ）
3. 参照の面の近くに background の点を足すと、on の結果は足す前と同じ、off の結果は変わる
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from synthetic import make_room  # noqa: E402
from regbim.labels import LabeledCloud, NAME_TO_ID  # noqa: E402
from regbim.metrics import apply_sim3, make_sim3  # noqa: E402
from regbim.semantic_icp import semantic_icp  # noqa: E402

ICP_CFG = {
    "semantic_icp": {"max_iter": 60, "max_corr_dist": 0.5, "tukey_c": 0.5,
                     "with_scaling": True, "rotation_fixed": False,
                     "convergence_delta": 1e-7},
    "classes": {"match_classes": ["wall", "floor", "ceiling", "door", "window"]},
}
FAILS = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (("  " + detail) if detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    room = make_room(seed=5)
    T_true = make_sim3(np.eye(3), np.array([0.3, -0.2, 0.1]), 1.1)
    dst = LabeledCloud(apply_sim3(T_true, room.points), room.labels.copy(), room.normals.copy())
    init = make_sim3(np.eye(3), np.zeros(3), 1.0)

    src = LabeledCloud(room.points, room.labels.copy(), room.normals.copy())
    src.labels[src.labels == NAME_TO_ID["background"]] = NAME_TO_ID["wall"]   # background を無くす
    a = semantic_icp(src, dst, init, ICP_CFG, rotation_fixed=True)
    b = semantic_icp(src, dst, init, ICP_CFG, rotation_fixed=True, label_filter=True)
    c = semantic_icp(src, dst, init, ICP_CFG, rotation_fixed=True, label_filter=False)
    check("既定 = label_filter=True", np.array_equal(a, b))
    check("background が無ければ on = off", np.array_equal(b, c))

    # 床の上 0.1 m に、床からずれた background の板（家具の天板を模す）を足す
    rng = np.random.default_rng(0)
    fl = src.points[src.labels == NAME_TO_ID["floor"]]
    slab = fl[rng.choice(len(fl), 800, replace=False)] + np.array([0.0, 0.0, 0.1])
    src2 = LabeledCloud(np.vstack([src.points, slab]),
                        np.concatenate([src.labels, np.full(len(slab), NAME_TO_ID["background"])]))
    on2 = semantic_icp(src2, dst, init, ICP_CFG, rotation_fixed=True, label_filter=True)
    off2 = semantic_icp(src2, dst, init, ICP_CFG, rotation_fixed=True, label_filter=False)
    check("background を足しても on は変わらない", np.allclose(on2, b, atol=1e-9),
          "max|Δ| %.2e" % np.abs(on2 - b).max())
    check("background を足すと off は変わる", not np.allclose(off2, c, atol=1e-6),
          "max|Δ| %.2e" % np.abs(off2 - c).max())
    if FAILS:
        print("FAILED:", FAILS)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

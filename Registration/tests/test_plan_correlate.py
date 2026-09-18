"""R29 §5 の小例検査 —— **案A を実装する前に、これを全部通す。**

| 検査 | 内容 |
|---|---|
| 相関の移動符号 | 既知の平行移動を与えて、符号と量が合うか |
| 端の処理 | 循環相関になっていないか |
| 既知の未知縮尺 | 既知の相似変換を戻せるか |
| 余分な廊下 | 参照に無い構造を足しても、正しい候補が残るか |
| **同型の複数部屋** | **同じ形の部屋を複数置く。どちらも完全一致しうる。**
根拠なく片方を正解と断定するなら、候補検証が不足している |

**小例の既知変換は評価器だけが持つ。実データの GT を候補探索に混ぜない。**

R29 §1-3 の反例（1次元投影では区別できない配置）も、ここで数値ごと検査する。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from regbim import plan_correlate as pc  # noqa: E402

FAILS = []


def check(name, ok, extra=""):
    print("  %-4s %s%s" % ("OK" if ok else "NG", name, ("  " + extra) if extra else ""))
    if not ok:
        FAILS.append(name)


def rect_walls(cx, cy, w, h, n=40):
    """軸に沿った長方形の壁点と法線。x 法線の2面と y 法線の2面。"""
    xs = np.linspace(cx - w / 2, cx + w / 2, n)
    ys = np.linspace(cy - h / 2, cy + h / 2, n)
    pts, nrm = [], []
    for x in (cx - w / 2, cx + w / 2):          # x を定める面
        for y in ys:
            pts.append([x, y, 0.0])
            nrm.append([1.0, 0.0, 0.0])
    for y in (cy - h / 2, cy + h / 2):          # y を定める面
        for x in xs:
            pts.append([x, y, 0.0])
            nrm.append([0.0, 1.0, 0.0])
    return np.asarray(pts), np.asarray(nrm)


def best_shift(src_pts, src_nrm, dst_pts, dst_nrm, cell=0.25, n_peaks=3,
               min_sep_m=0.5):
    sx, sy = pc.wall_groups(src_pts, src_nrm)
    dx, dy = pc.wall_groups(dst_pts, dst_nrm)
    return pc.horizontal_candidates(sx, sy, dx, dy, cell, n_peaks, min_sep_m)


def main() -> int:
    cell = 0.25

    print("1. 相関の移動符号と量（**既知の平行移動を与える**）")
    p, n = rect_walls(0, 0, 4, 6)
    for truth in ([2.0, 0.0], [0.0, 3.0], [-1.5, 2.5], [5.0, -4.0]):
        moved = p + np.array([truth[0], truth[1], 0.0])
        cands = best_shift(p, n, moved, n, cell)
        got = np.asarray(cands[0]["shift_xy"]) if cands else np.array([9e9, 9e9])
        ok = np.allclose(got, truth, atol=cell)
        check("平行移動 %s を復元" % truth, ok, "得た値 %s" % np.round(got, 3).tolist())

    print("\n2. 端の処理（**循環相関になっていないか**）")
    # 大きく離した配置。循環なら「回り込んだ」偽のピークが最大になる。
    far = p + np.array([40.0, 30.0, 0.0])
    cands = best_shift(p, n, far, n, cell)
    got = np.asarray(cands[0]["shift_xy"]) if cands else np.array([9e9, 9e9])
    check("40 m 離れた平行移動でも正しく復元", np.allclose(got, [40.0, 30.0], atol=cell),
          "得た値 %s" % np.round(got, 3).tolist())
    a = np.zeros((3, 3)); a[0, 0] = 1
    b = np.zeros((3, 3)); b[2, 2] = 1
    c = pc.correlate_full(a, b)
    check("相関の形が n_a+n_b-1 になっている", c.shape == (5, 5), "形 %s" % (c.shape,))
    check("回り込みのピークが無い（一致数の総和が 1）", c.sum() == 1.0,
          "総和 %g" % c.sum())

    print("\n3. 1次元投影では区別できない配置（R29 §1-3 の反例）")
    A = np.array([[0, 0], [1, 3], [4, 1]], dtype=float)
    B = np.array([[0, 3], [1, 1], [4, 0]], dtype=float)
    gA, _ = pc.occupancy(A, 1.0, np.zeros(2), (4, 5))
    gB, _ = pc.occupancy(B, 1.0, np.zeros(2), (4, 5))
    check("x への投影が等しい", np.array_equal(gA.sum(axis=0), gB.sum(axis=0)))
    check("y への投影が等しい", np.array_equal(gA.sum(axis=1), gB.sum(axis=1)))
    check("2次元の一致数は A-A が 3", float((gA * gA).sum()) == 3.0)
    check("2次元の一致数は A-B が 0", float((gA * gB).sum()) == 0.0,
          "**投影が同じでも2次元なら分かれる**")

    print("\n4. 既知の未知縮尺（**縮尺を当ててから相関すれば戻せるか**）")
    s_true, t_true = 1.35, np.array([3.0, -2.0])
    scaled = p.copy()
    scaled[:, :2] = p[:, :2] * s_true + t_true
    cands = best_shift(scaled, n, p, n, cell)      # 正しい縮尺で割ってから
    scaled_back = scaled.copy()
    scaled_back[:, :2] = scaled[:, :2] / s_true
    cands = best_shift(scaled_back, n, p, n, cell)
    got = np.asarray(cands[0]["shift_xy"]) if cands else np.array([9e9, 9e9])
    check("縮尺を戻せば平行移動も戻る", np.allclose(got, -t_true / s_true, atol=cell),
          "得た値 %s / 期待 %s" % (np.round(got, 3).tolist(),
                                np.round(-t_true / s_true, 3).tolist()))

    print("\n5. 余分な廊下（**参照に無い構造を足しても正しい候補が残るか**）")
    room, rn = rect_walls(0, 0, 4, 6)
    corr_p, corr_n = rect_walls(14, 0, 24, 2)      # 参照に無い長い廊下
    src = np.vstack([room, corr_p])
    srcn = np.vstack([rn, corr_n])
    cands = best_shift(src, srcn, room, rn, cell, n_peaks=5)
    shifts = [np.asarray(c["shift_xy"]) for c in cands]
    hit = any(np.allclose(sh, [0.0, 0.0], atol=cell) for sh in shifts)
    check("正しい候補（移動 0）が上位に残る", hit,
          "上位 %s" % [np.round(s, 2).tolist() for s in shifts[:3]])

    print("\n6. **同型の複数部屋**（R29 §5・R31 §2。省かない）")
    r1, n1 = rect_walls(0, 0, 4, 6)
    r2, n2 = rect_walls(12, 0, 4, 6)               # まったく同じ形をもう1つ
    src2 = np.vstack([r1, r2])
    srcn2 = np.vstack([n1, n2])
    cands = best_shift(src2, srcn2, r1, n1, cell, n_peaks=5, min_sep_m=2.0)
    shifts = [np.asarray(c["shift_xy"]) for c in cands]
    scores = [c["score"] for c in cands]
    hit0 = any(np.allclose(sh, [0.0, 0.0], atol=cell) for sh in shifts)
    hit12 = any(np.allclose(sh, [-12.0, 0.0], atol=cell) for sh in shifts)
    check("部屋1に合わせる候補が出る", hit0)
    check("部屋2に合わせる候補も出る", hit12,
          "上位 %s" % [np.round(s, 2).tolist() for s in shifts[:3]])
    tied = (len(scores) >= 2 and abs(scores[0] - scores[1]) < 1e-9)
    check("**2つの候補が同点である**（どちらも完全一致しうる）", tied,
          "スコア %s" % scores[:3])
    print("     → **相関だけでは決められない。根拠なく片方を正解と断定しない。**")
    print("     → 案A はこの 2 つを**両方候補として残し**、後段の検証で選ぶ。")

    print()
    if FAILS:
        print("FAILED: %s" % ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""案A の候補生成が、**既知の相似変換を候補のなかに含む**こと。

R29 §6 の外れ方の1つめ——
「探索範囲内に真値があるのに、生成候補に GT へ近いものが無い」——
**を、実データへ行く前に小例で潰す。**

**GT は評価器だけが持つ。** 生成側には参照と source しか渡さない。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from regbim import plan_correlate as pc          # noqa: E402
from regbim.labels import NAME_TO_ID             # noqa: E402

FAILS = []


def check(name, ok, extra=""):
    print("  %-4s %s%s" % ("OK" if ok else "NG", name, ("  " + extra) if extra else ""))
    if not ok:
        FAILS.append(name)


def box_room(cx, cy, w, h, z0=0.0, z1=2.6, n=26):
    """壁・床・天井を持つ直方体の部屋。法線つき。"""
    pts, nrm, lab = [], [], []
    xs = np.linspace(cx - w / 2, cx + w / 2, n)
    ys = np.linspace(cy - h / 2, cy + h / 2, n)
    zs = np.linspace(z0, z1, 8)
    for x in (cx - w / 2, cx + w / 2):
        for y in ys:
            for z in zs:
                pts.append([x, y, z]); nrm.append([1, 0, 0]); lab.append(NAME_TO_ID["wall"])
    for y in (cy - h / 2, cy + h / 2):
        for x in xs:
            for z in zs:
                pts.append([x, y, z]); nrm.append([0, 1, 0]); lab.append(NAME_TO_ID["wall"])
    for x in xs:
        for y in ys:
            pts.append([x, y, z0]); nrm.append([0, 0, 1]); lab.append(NAME_TO_ID["floor"])
            pts.append([x, y, z1]); nrm.append([0, 0, 1]); lab.append(NAME_TO_ID["ceiling"])
    return (np.asarray(pts, float), np.asarray(lab), np.asarray(nrm, float))


def sim3(R, t, s):
    T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
    return T


def apply(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def main() -> int:
    dst_p, dst_l, dst_n = box_room(0, 0, 8, 5)

    print("1. 既知の相似変換を候補に含むか（**source は参照より広い**）")
    # source = 参照の部屋 ＋ 参照に無い廊下。実データの被覆不一致を模す。
    room_p, room_l, room_n = box_room(0, 0, 8, 5)
    cor_p, cor_l, cor_n = box_room(16, 0, 22, 2.2)
    base_p = np.vstack([room_p, cor_p])
    base_l = np.concatenate([room_l, cor_l])
    base_n = np.vstack([room_n, cor_n])

    for s_true, t_true in ((1.0, [0.0, 0.0, 0.0]),
                           (0.85, [3.0, -2.0, 0.4]),
                           (1.30, [-5.0, 4.0, -0.3])):
        M = sim3(np.eye(3), np.asarray(t_true), s_true)   # source を作る変換
        src_p = apply(M, base_p)
        src_n = base_n
        T_truth = np.linalg.inv(M)                         # 正解（評価器だけが持つ）

        cands, diag = pc.generate_candidates(
            src_p, base_l, src_n, dst_p, dst_l, dst_n,
            [np.eye(3)], NAME_TO_ID)
        # **判定は R29 §3 の「粗い候補の診断基準」を使う**（ここで作らない）：
        #   回転誤差 5° 未満 かつ 縮尺の相対誤差 5% 未満 かつ d_Ω < 1 m
        best, best_s = None, None
        for c in cands:
            d = np.linalg.norm(apply(c["T"], src_p) - apply(T_truth, src_p), axis=1)
            r = float(np.sqrt((d ** 2).mean()))
            se = abs(c["scale"] * s_true - 1.0)     # 候補の縮尺 × source の縮尺
            if best is None or r < best:
                best, best_s = r, se
        ok = best is not None and best < 1.0 and best_s < 0.05
        check("s=%.2f t=%s の正解に近い候補がある" % (s_true, t_true), ok,
              "最良候補 d_Ω %.3f m（基準 1.0）/ 縮尺誤差 %.1f%%（基準 5%%）/ 候補 %d 個"
              % (best if best else -1, 100 * (best_s if best_s else -1), len(cands)))

    print("\n2. 候補に通し番号が付き、重複していないこと（R29 §3）")
    cands, diag = pc.generate_candidates(base_p, base_l, base_n,
                                         dst_p, dst_l, dst_n,
                                         [np.eye(3)], NAME_TO_ID)
    ids = [c["cand_id"] for c in cands]
    check("cand_id が連番で重複しない", len(ids) == len(set(ids)))
    check("向きごとの枠が記録される", len(diag["per_yaw"]) == 1,
          "per_yaw=%s" % diag["per_yaw"])

    print("\n3. 4 向きぶんの枠が独立に確保されること（R29 §1-3 の違い 3）")
    rots = []
    for k in range(4):
        a = k * np.pi / 2
        rots.append(np.array([[np.cos(a), -np.sin(a), 0],
                              [np.sin(a), np.cos(a), 0], [0, 0, 1.0]]))
    cands, diag = pc.generate_candidates(base_p, base_l, base_n,
                                         dst_p, dst_l, dst_n, rots, NAME_TO_ID)
    per = {d["yaw_index"]: d["n_kept"] for d in diag["per_yaw"]}
    check("4 向きすべてに候補が確保される", len(per) == 4 and all(v > 0 for v in per.values()),
          "向きごとの数 %s" % per)
    check("合計が向きごとの上限×4 以内", len(cands) <= 4 * pc.DEFAULTS["per_yaw_coarse"],
          "合計 %d 個" % len(cands))

    print("\n4. 鉛直候補が作れないときは補わずに数えること（R29 §2-2）")
    no_fc = base_l.copy()
    no_fc[(no_fc == NAME_TO_ID["floor"]) | (no_fc == NAME_TO_ID["ceiling"])] = \
        NAME_TO_ID["wall"]
    cands2, diag2 = pc.generate_candidates(base_p, no_fc, base_n,
                                           dst_p, dst_l, dst_n,
                                           [np.eye(3)], NAME_TO_ID)
    check("床も天井も無いと候補を作らない", len(cands2) == 0, "候補 %d 個" % len(cands2))
    check("不足を数えている", diag2["vertical_shortfall"] > 0,
          "不足 %d 回" % diag2["vertical_shortfall"])

    print()
    if FAILS:
        print("FAILED: %s" % ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

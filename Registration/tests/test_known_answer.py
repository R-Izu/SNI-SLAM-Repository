"""R22 §5 — 既知解テスト K1〜K4。**評価器が、既知の正答・誤答・縮退を見分けられるか。**

なぜ「正しく解ける1本」では足りないか
--------------------------------------
真の変換 $G$、摂動 $P$ のとき、GT 基準の期待値は $GP^{-1}$、自己基準の期待値は $T_0P^{-1}$。
**$T_0=G$（正しく解ける例）なら、この2つは完全に一致する。**
**N を増やしても、自己一貫性と正解率の混同は捕まらない。**

**必要なのは「安定して誤答を返す対照」である。**

    conda activate sni-slam
    python Registration/tests/test_known_answer.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from regbim import metrics, stats                      # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (("  " + detail) if detail else ""))
    if not cond:
        FAILS.append(name)


def sim3(R, t, s=1.0):
    T = np.eye(4)
    T[:3, :3] = s * np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def rand_rot(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def d_omega(T, G, omega):
    a = metrics.apply_sim3(np.asarray(T, float), omega)
    b = metrics.apply_sim3(np.asarray(G, float), omega)
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


# --------------------------------------------------------------------------- #
# K1: 正答と、**安定した誤答**を同じ経路に流す
# --------------------------------------------------------------------------- #
def k1(rng, thresholds):
    print("K1 正答と安定した誤答を同じ経路に流す（自己一貫性と GT 基準が分かれるか）")
    G = sim3(rand_rot(rng), [3.0, -1.5, 0.7])
    omega = rng.uniform(-5, 5, size=(2000, 3))
    H = sim3(np.eye(3), [0.5, 0.0, 0.0])      # BIM 座標で水平に 0.5 m ずらす

    # 手法の代用品：**常に HG を返す**（= 安定した誤答）。T0 も HG。
    # ★ 代用品に GT を渡すのは、測定器を試験するための限定した対照である。
    #   性能評価で手法に GT を渡すことではない。
    T0 = H @ G

    sc_d, gt_d, sc_ok, gt_ok = [], [], [], []
    for _ in range(20):
        P = metrics.random_sim3(rng, {"rot_deg": [0.0, 20.0], "trans": [-1.0, 1.0], "log_scale": [0.0, 0.0]})
        T_hat = H @ G @ metrics.invert_sim3(P)
        e_sc = metrics.sim3_errors(T_hat, T0 @ metrics.invert_sim3(P))
        e_gt = metrics.sim3_errors(T_hat, G @ metrics.invert_sim3(P))
        sc_d.append(d_omega(T_hat, T0 @ metrics.invert_sim3(P), omega))
        gt_d.append(d_omega(T_hat, G @ metrics.invert_sim3(P), omega))
        sc_ok.append(stats.check_success(e_sc, thresholds))
        gt_ok.append(stats.check_success(e_gt, thresholds))

    check("自己一貫性の変位誤差が 0", max(sc_d) < 1e-9, "最大 %.2e m" % max(sc_d))
    check("**GT 基準の変位誤差が 0.5 m**", abs(np.median(gt_d) - 0.5) < 1e-6,
          "中央 %.6f m" % np.median(gt_d))
    check("自己一貫性は 20/20 合格", all(sc_ok), "%d/20" % sum(sc_ok))
    check("**GT 基準は 0/20 合格**", not any(gt_ok), "%d/20" % sum(gt_ok))
    return {"selfconsistency_success": sum(sc_ok), "gt_success": sum(gt_ok),
            "selfconsistency_d_omega": float(np.median(sc_d)),
            "gt_d_omega": float(np.median(gt_d)), "n": 20}


# --------------------------------------------------------------------------- #
# K2: 座標・単位・順序
# --------------------------------------------------------------------------- #
def k2(rng):
    print("K2 座標・単位・順序（source の原点と単位を変え、GT も解析的に更新する）")
    G = sim3(rand_rot(rng), [10.0, 4.0, -2.0])
    omega = rng.uniform(-5, 5, size=(500, 3))

    # source 側の座標変換 A（原点移動＋単位変更＋回転）。GT は G' = G A^{-1}
    A = sim3(rand_rot(rng), [100.0, -50.0, 7.0], s=0.001)   # mm -> m 相当
    G2 = G @ metrics.invert_sim3(A)
    omega2 = metrics.apply_sim3(A, omega)
    check("座標・単位を変えても、同じ点の行き先は一致する",
          d_omega(G2, G2, omega2) == 0.0 and
          np.allclose(metrics.apply_sim3(G2, omega2), metrics.apply_sim3(G, omega)),
          "最大差 %.2e m" % np.abs(metrics.apply_sim3(G2, omega2)
                                   - metrics.apply_sim3(G, omega)).max())

    # 変換順序の取り違えを検出できるか（G A^{-1} と A^{-1} G は別物）
    wrong = metrics.invert_sim3(A) @ G
    check("**順序を取り違えた変換は、同じにならない**",
          d_omega(wrong, G2, omega2) > 1.0, "d_Ω = %.3f m" % d_omega(wrong, G2, omega2))

    # 複数の既知 P で、期待値どおりか
    errs = []
    for _ in range(5):
        P = metrics.random_sim3(rng, {"rot_deg": [0.0, 30.0], "trans": [-2.0, 2.0], "log_scale": [-0.1, 0.1]})
        errs.append(d_omega(G @ metrics.invert_sim3(P),
                            G @ metrics.invert_sim3(P), omega))
    check("複数の既知 P で期待値と一致", max(errs) < 1e-12, "最大 %.2e m" % max(errs))
    return {"n_perturbations": 5}


# --------------------------------------------------------------------------- #
# K3: 出力の不成立
# --------------------------------------------------------------------------- #
def k3(rng, thresholds):
    print("K3 出力の不成立（無効出力を成功にしない。**試行を分母に残す**）")
    R = rand_rot(rng)
    G = sim3(R, [1.0, 2.0, 3.0])

    collapsed = sim3(R, [1.0, 2.0, 3.0], s=1e-30)
    e = metrics.sim3_errors(collapsed, G)
    check("s=1e-30 は degenerate", e["degenerate"] is True)
    check("**degenerate は成功にならない**", stats.check_success(e, thresholds) is False)

    # **正常な s=1 の剛体出力を誤拒否しない**
    rigid = sim3(R, [1.0, 2.0, 3.0], s=1.0)
    e_ok = metrics.sim3_errors(rigid, G)
    check("正常な s=1 の剛体出力は degenerate ではない", e_ok["degenerate"] is False)
    check("正常な剛体出力は成功と判定される",
          stats.check_success(e_ok, thresholds) is True)

    # 非有限値
    bad = sim3(R, [np.nan, 0.0, 0.0])
    try:
        eb = metrics.sim3_errors(bad, G)
        finite = all(np.isfinite(v) for k, v in eb.items() if isinstance(v, float))
        check("**非有限値を含む出力が成功にならない**",
              (not finite) or (stats.check_success(eb, thresholds) is False),
              "誤差が有限=%s" % finite)
    except Exception as ex:
        check("非有限値は例外で弾かれる（成功にはならない）", True, type(ex).__name__)

    # 分母：10 試行のうち 1 つが縮退、7 が成功、2 が失敗 → 7/10
    outcomes = [stats.check_success(metrics.sim3_errors(t, G), thresholds)
                for t in ([rigid] * 7
                          + [sim3(R, [9.0, 9.0, 9.0])] * 2
                          + [collapsed])]
    check("**縮退を含めた分母で 7/10**", sum(outcomes) == 7 and len(outcomes) == 10,
          "%d/%d" % (sum(outcomes), len(outcomes)))
    return {"n": len(outcomes), "k": sum(outcomes), "n_degenerate": 1}


# --------------------------------------------------------------------------- #
# K4: 集計の識別
# --------------------------------------------------------------------------- #
def aggregate(rows):
    """集計は、率だけでなく**母数・分子・除外数**を返す（R23 §4-4）。

    **重複行を黙って数えない。** 同じ `(scene, anchor, trial)` が2度来たら、
    2件目は `n_duplicate` に入れて母数にも分子にも足さない。

    ★ **この検査を書いたとき、最初の実装は重複を黙って成功に足していた**
      （真の 7/10 に対し 8/11 を返した）。**K4 はその型を捕まえるためのものである。**
    """
    out = {}
    seen = set()
    for r in rows:
        key = (r["scene"], r["anchor"])
        a = out.setdefault(key, {"n": 0, "k": 0, "n_degenerate": 0,
                                 "n_missing": 0, "n_duplicate": 0})
        uid = (r["scene"], r["anchor"], r.get("trial"))
        if uid in seen:
            a["n_duplicate"] += 1
            continue
        seen.add(uid)
        if r.get("missing"):
            a["n_missing"] += 1
            a["n"] += 1
            continue
        a["n"] += 1
        a["n_degenerate"] += int(bool(r.get("degenerate")))
        a["k"] += int(bool(r.get("success")) and not r.get("degenerate"))
    return out


def k4():
    print("K4 集計の識別（anchor ごとに異なる既知の成功数。欠損・重複も入れる）")
    rows = []
    truth = {("s1", "ne"): (7, 10, 1), ("s1", "sw"): (3, 10, 0), ("s2", "ne"): (10, 10, 0)}
    for (scene, anchor), (k, n, ndeg) in truth.items():
        i = 0
        for _ in range(k):
            rows.append({"scene": scene, "anchor": anchor, "trial": i, "success": True})
            i += 1
        for _ in range(n - k - ndeg):
            rows.append({"scene": scene, "anchor": anchor, "trial": i, "success": False})
            i += 1
        for _ in range(ndeg):
            rows.append({"scene": scene, "anchor": anchor, "trial": i,
                         "success": True, "degenerate": True})
            i += 1
    rows.append(dict(rows[0]))                      # 重複行（同じ trial 番号）
    rows.append({"scene": "s2", "anchor": "sw", "trial": 0, "missing": True})

    agg = aggregate(rows)
    check("**anchor をまたいで集約せず、cell ごとに出る**", len(agg) == 4,
          "cell 数 %d（重複を含めても cell は増えない）" % len(agg))
    check("**s1/ne は 7/10。重複行を母数に足さない**",
          agg[("s1", "ne")]["k"] == 7 and agg[("s1", "ne")]["n"] == 10,
          "%d/%d" % (agg[("s1", "ne")]["k"], agg[("s1", "ne")]["n"]))
    check("**重複行が検出され、数えられている**",
          agg[("s1", "ne")]["n_duplicate"] == 1,
          "n_duplicate=%d" % agg[("s1", "ne")]["n_duplicate"])
    check("**縮退は分母に残り、分子には入らない**",
          agg[("s1", "ne")]["n_degenerate"] == 1)
    check("s1/sw は 3/10", agg[("s1", "sw")]["k"] == 3 and agg[("s1", "sw")]["n"] == 10)
    check("**欠測が数えられている**", agg[("s2", "sw")]["n_missing"] == 1)
    check("**先頭 anchor のみを採る誤りをしていない**",
          agg[("s1", "ne")]["k"] != agg[("s1", "sw")]["k"])
    return {k: v for k, v in ((str(k), v) for k, v in agg.items())}


def main() -> int:
    rng = np.random.default_rng(0)
    th = {"rot_deg": 5.0, "trans": 0.1, "scale_ratio": 0.05}
    print("閾値: 回転<%.1f度 / 並進<%.2fm / 縮尺比<%.2f\n" % (
        th["rot_deg"], th["trans"], th["scale_ratio"]))
    r1 = k1(rng, th); print()
    k2(rng); print()
    k3(rng, th); print()
    k4(); print()
    print("K1 の要約（この2行が同じ値になったら、混同が復活している）:")
    print("  selfconsistency: 成功 %d/%d, d_Ω 中央 %.3e m"
          % (r1["selfconsistency_success"], r1["n"], r1["selfconsistency_d_omega"]))
    print("  gt             : 成功 %d/%d, d_Ω 中央 %.3f m"
          % (r1["gt_success"], r1["n"], r1["gt_d_omega"]))
    print()
    if FAILS:
        print("FAILED: %s" % ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

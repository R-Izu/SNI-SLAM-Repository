"""R7 §3-3 — 指標の定義が意図どおりかを、小例で確かめる。

**20,800 試行を回し直す前にここを通す。**
今回の事故（自己一貫性を正解率として報告した）は、
**「同じ誤答に安定して戻る手法が満点を取る」**という性質を検算していなかったために起きた。
その性質を、まさにそういう手法を作って確かめる。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from regbim import metrics, stats                       # noqa: E402

SUCC = {"rot_deg": 5.0, "trans": 0.1, "scale_ratio": 0.05}


def _run(T_method_lands_on, T_gt, n_trials=8, seed=0):
    """`benchmark._evaluate` と同じ2つの誤差を、最小構成で再現する。

    手法は「摂動 P で動かされた source を、毎回まったく同じ世界座標 ``T0`` へ置く」
    ものとする。**自己一貫性は構成上つねに完璧**であり、
    それが正解かどうかとは無関係である。
    """
    rng = np.random.default_rng(seed)
    perturb = {"rot_deg": [0.0, 180.0], "trans": [-2.0, 2.0], "log_scale": [-0.4, 0.4]}
    T0 = np.asarray(T_method_lands_on, dtype=np.float64)
    sc, gt = [], []
    for _ in range(n_trials):
        P = metrics.random_sim3(rng, perturb)
        Tt = T0 @ metrics.invert_sim3(P)          # 手法は毎回 T0 に戻る
        e_sc = metrics.sim3_errors(Tt, T0 @ metrics.invert_sim3(P))
        e_gt = metrics.sim3_errors(Tt, T_gt @ metrics.invert_sim3(P))
        sc.append(e_sc)
        gt.append(e_gt)
    return sc, gt


def test_identical_when_solution_equals_gt():
    """検算1：T0 = T_gt なら、自己一貫性と GT 基準は一致する。"""
    T = metrics.make_sim3(np.eye(3), np.zeros(3), 1.0)
    sc, gt = _run(T, T)
    assert all(stats.check_success(e, SUCC) for e in sc)
    assert all(stats.check_success(e, SUCC) for e in gt)


def test_selfconsistency_perfect_while_gt_is_5m_off():
    """検算2（**本命**）：GT から 5 m ずれた解に毎回戻る手法。

    自己一貫性は満点、GT 基準の並進誤差は 5 m でなければならない。
    **この2つが同じ数字になる実装だったのが、今回の事故である。**
    """
    T_gt = metrics.make_sim3(np.eye(3), np.zeros(3), 1.0)
    T_wrong = metrics.make_sim3(np.eye(3), np.array([5.0, 0.0, 0.0]), 1.0)
    sc, gt = _run(T_wrong, T_gt)

    assert all(stats.check_success(e, SUCC) for e in sc), "自己一貫性は満点のはず"
    assert all(e["trans"] == pytest.approx(5.0, abs=1e-6) for e in gt), \
        "GT 基準の並進誤差は 5 m のはず"
    assert not any(stats.check_success(e, SUCC) for e in gt), "GT 基準では全滅のはず"


def test_aggregation_averages_over_cells():
    """検算3：成功率 1.0 と 0.0 の2セルを集計したら 0.5 になる。

    `t3_summarize.pick()` が anchor を集約せず最初の1行だけ返していたため、
    4隅ぶんの平均のつもりが1隅の値になっていた（R7 §4-1）。
    """
    cells = [{"success_rate": 1.0, "trials": 100}, {"success_rate": 0.0, "trials": 100}]
    k = sum(round(c["success_rate"] * c["trials"]) for c in cells)
    n = sum(c["trials"] for c in cells)
    assert k == 100 and n == 200
    assert k / n == pytest.approx(0.5)
    lo, hi = stats.wilson_ci(k, n)
    assert lo < 0.5 < hi
    # 1隅だけ見ると 1.0 になってしまうことも明示しておく
    assert cells[0]["success_rate"] == 1.0

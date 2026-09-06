"""Unit tests for the coverage-invariant translation seeding.

The point of the module is that a *partially observed* reference still yields the
right offset. That is exactly what these tests check, by construction: the
reference keeps only a subset of the source's walls.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from regbim.plane_match import offset_candidates, plane_offset   # noqa: E402


def _walls(positions, n=400, noise=0.01, seed=0):
    """1-D coordinates of points lying on planes at ``positions``."""
    rng = np.random.default_rng(seed)
    return np.concatenate([p + rng.normal(0, noise, n) for p in positions])


def test_recovers_a_known_shift_with_full_overlap():
    dst = _walls([0.0, 3.0, 6.0])
    src = dst - 2.5                       # source sits 2.5 m short of the reference
    cands = offset_candidates(src, dst, bin_m=0.05, top_k=3)
    assert cands, "no candidate returned"
    assert cands[0][0] == pytest.approx(2.5, abs=0.05)


def test_recovers_the_shift_when_the_reference_covers_only_part():
    """★ 本命：参照が source の壁の一部しか持たない場合。

    重心や範囲の中点はこの状況で必ずずれる。相互相関は残った壁で合わせられる。
    間隔を不等にしてあるのは、等間隔だと後述のとおり解が一意に決まらないため。
    """
    src = _walls([0.0, 2.2, 5.1, 9.4, 13.0])      # 室 + 廊下まで見えている
    dst = _walls([0.0, 2.2])                       # BIM は最初の2枚だけ
    cands = offset_candidates(src, dst, bin_m=0.05, top_k=3)
    assert cands
    assert cands[0][0] == pytest.approx(0.0, abs=0.05)

    # 比較：重心を合わせると、覆われていない壁のぶんだけずれる
    centroid_offset = dst.mean() - src.mean()
    assert abs(centroid_offset) > 4.0, "この構成では重心は大きくずれるはず"


def test_periodic_structure_is_genuinely_ambiguous():
    """等間隔の壁では、正解を**一意には決められない**。

    src が 0,3,6,9,12、dst が 0,3 のとき、オフセット 0 / −3 / −6 / −9 は
    **どれも同じだけ重なる**。これは実装の欠陥ではなく構造の性質であり、
    廊下のような反復構造でまさに起きる。
    **だからこの関数は1つに決め打たず候補を返し、選択は下流のスコアに委ねる。**
    """
    src = _walls([0.0, 3.0, 6.0, 9.0, 12.0])
    dst = _walls([0.0, 3.0])
    cands = offset_candidates(src, dst, bin_m=0.05, top_k=6, min_separation_m=0.5)

    # 実測：-6.00 / -3.00 / +0.00 / -9.00 が 0.1618〜0.1581（差 2.3%）で並び、
    # 5位（-12.00）は 0.0796 と約半分に落ちる。**上位4つは 2 枚重なる解、
    # 5位以降は 1 枚しか重ならない解**という構造がそのまま出ている。
    top = cands[0][1]
    tied = [d for d, s in cands if s > 0.95 * top]
    assert len(tied) == 4, "2枚重なる解が4つ同点で並ぶはず: %s" % cands
    assert any(abs(d) < 0.05 for d in tied), "正解が同点の中に入っていること"
    assert cands[4][1] < 0.6 * top, "1枚しか重ならない解ははっきり落ちるはず"
    # **正解が1位とは限らない**（ここでは -6.00 が1位）。だから候補として返す。
    assert abs(cands[0][0]) > 0.05


def test_candidates_are_separated():
    src = _walls([0.0, 4.0])
    dst = _walls([1.0, 5.0])
    cands = offset_candidates(src, dst, bin_m=0.05, top_k=4, min_separation_m=0.5)
    ds = [d for d, _ in cands]
    for a in range(len(ds)):
        for b in range(a + 1, len(ds)):
            assert abs(ds[a] - ds[b]) >= 0.5


def test_too_few_points_returns_nothing():
    assert offset_candidates(np.zeros(3), np.zeros(3)) == []


def test_plane_offset_uses_the_dominant_plane():
    """鉛直軸は単一ピーク。少数の外れ値に引かれないこと。"""
    rng = np.random.default_rng(0)
    src = np.concatenate([rng.normal(0.0, 0.005, 1000), rng.normal(2.0, 0.005, 20)])
    dst = np.concatenate([rng.normal(0.5, 0.005, 1000), rng.normal(3.0, 0.005, 20)])
    assert plane_offset(src, dst, bin_m=0.02) == pytest.approx(0.5, abs=0.03)

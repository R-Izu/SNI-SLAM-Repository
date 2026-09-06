"""R8 §2 — 縮尺を固定したときの並進が、その縮尺のもとで最小二乗解になっていること。

**性質そのものをテストにする**（`test_metric_definitions.py` と同じ扱い）。

以前は `scale_translation` の返り値 `s` を呼び出し側で 1.0 に上書きしていた。
`t = mu_dst - s*mu_src` は `s` に依存するので、`s` だけ差し替えると
`t` は自由縮尺のままになり、解が `(s_free - 1) * mu_src` ずれる。
**source の重心が原点から数 m 離れていれば、ずれもメートル級になる。**
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from regbim.scale import scale_translation, umeyama      # noqa: E402


def _example():
    """外部レビューが出した数値例。

    p_x = 9, 10, 11 ／ q_x = 0.6 p_x + 3
      自由縮尺の解 : s = 0.6, t = 3
      s = 1 の最小二乗解 : t = mean(q) - mean(p) = 9 - 10 = -1
    """
    p = np.zeros((3, 3)); p[:, 0] = [9.0, 10.0, 11.0]
    q = np.zeros((3, 3)); q[:, 0] = 0.6 * p[:, 0] + 3.0
    return p, q


def test_free_scale_solution():
    p, q = _example()
    t, s = scale_translation(p, q)
    assert s == pytest.approx(0.6, abs=1e-9)
    assert t[0] == pytest.approx(3.0, abs=1e-9)


def test_fixed_scale_translation_is_the_least_squares_solution():
    """★本命：s=1 に固定したとき t は −1 になる（3 ではない）。"""
    p, q = _example()
    t, s = scale_translation(p, q, fixed_scale=1.0)
    assert s == 1.0
    assert t[0] == pytest.approx(-1.0, abs=1e-9)


def test_overwriting_s_afterwards_is_wrong_by_the_predicted_amount():
    """旧実装のずれが (1 - s_free) * mu_src であることを明示しておく。

    t_free - t_fixed = (mu_dst - s_f mu_src) - (mu_dst - 1 mu_src) = (1 - s_f) mu_src
    """
    p, q = _example()
    t_free, s_free = scale_translation(p, q)
    t_fixed, _ = scale_translation(p, q, fixed_scale=1.0)
    predicted = (1.0 - s_free) * p.mean(axis=0)
    assert (t_free - t_fixed)[0] == pytest.approx(predicted[0], abs=1e-9)
    # この例では 4 m ずれる。重心が原点から遠いほど大きくなる
    assert abs(t_free[0] - t_fixed[0]) == pytest.approx(4.0, abs=1e-9)


def test_agrees_with_umeyama_when_rotation_is_identity():
    """`umeyama` は元から正しい。回転が厳密に単位行列なら両者は一致すべき。

    ノイズを入れると `umeyama` は僅かな回転を推定してしまい、
    「回転固定の解」と比べる意味が無くなるので、ここでは入れない。
    """
    rng = np.random.default_rng(0)
    p = rng.random((50, 3)) * 5.0 + np.array([10.0, -4.0, 2.0])
    q = p + np.array([0.3, -0.2, 0.1])
    t_st, s_st = scale_translation(p, q, fixed_scale=1.0)
    R_um, t_um, s_um = umeyama(p, q, with_scaling=False)
    assert s_um == pytest.approx(s_st)
    assert np.allclose(R_um, np.eye(3), atol=1e-9)
    assert np.allclose(t_um, t_st, atol=1e-9)
    assert np.allclose(t_st, [0.3, -0.2, 0.1], atol=1e-9)

"""2次元の壁配置相関による並進候補の生成（R29 案A の中核）。

なぜ 1 次元の投影照合では足りないのか（R29 §1-3）
--------------------------------------------------
既存の ``plane_match`` は x と y を**独立に**照合する。**2次元にすれば x と y の関係が残る。**
レビューが数値で示した反例::

    A = {(0,0), (1,3), (4,1)},  B = {(0,3), (1,1), (4,0)}

**この2つは x への投影も y への投影も等しい。**
しかし同じ位置での2次元一致数は、A と A で 3、A と B で 0 である。

考え方
------
**参照が小さく source が広くても、両方を含む画像を作れば、画像を平行移動しながら
重なりを調べることで「source のどこが参照に対応するか」も同時に探索できる。**
**先に室の範囲を決める必要がない**——これが「切り出しの鶏と卵」を解く形である。

実装で外してはいけない点（R29 §2-2）
------------------------------------
- **各軸で少なくとも ``n_P + n_Q - 1`` へゼロ埋めする。循環相関を使わない**
- **配列添字から移動量への変換を、既知の平行移動を与えた小例で検査する**
- 数値のため source から ``a``、参照から ``b`` を引いたなら、元座標へ
  ``t = b - s R a + u`` で戻す。**中心どうしを対応点とみなしているのではない**
- **点数で重みを付けない**（セルが占有されていれば 1）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# 壁を「法線がその軸を向いている」と見なす角度（R29 §2-1：20 度以内）
WALL_AXIS_TOL_DEG = 20.0


def wall_groups(points: np.ndarray, normals: np.ndarray,
                tol_deg: float = WALL_AXIS_TOL_DEG) -> Tuple[np.ndarray, np.ndarray]:
    """壁点を、法線が x 軸寄りの群と y 軸寄りの群に分ける。

    **法線の正負は同一視する**（R29 §1-2）。壁の表裏で群が割れないようにするためである。
    どちらの軸にも十分近くない面（斜め壁）は**どちらにも入れない**——
    その面は軸方向の位置を定めないので、入れると相関が濁る。
    """
    if normals is None or len(points) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    n = np.asarray(normals, dtype=np.float64)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    c = np.cos(np.deg2rad(tol_deg))
    gx = np.abs(n[:, 0]) >= c          # 法線が x 軸を向く＝x 位置を定める面
    gy = np.abs(n[:, 1]) >= c
    return points[gx][:, :2], points[gy][:, :2]


def occupancy(xy: np.ndarray, cell: float,
              origin: Optional[np.ndarray] = None,
              shape: Optional[Tuple[int, int]] = None):
    """占有格子を作る。**点数で重みを付けない**（占有なら 1）。

    返すのは ``(grid, origin)``。``grid[i, j]`` はセル
    ``origin + (j, i) * cell`` に点があれば 1。
    """
    if len(xy) == 0:
        g = np.zeros(shape or (1, 1), dtype=np.float64)
        return g, (origin if origin is not None else np.zeros(2))
    if origin is None:
        origin = np.floor(xy.min(axis=0) / cell) * cell
    idx = np.floor((xy - origin) / cell).astype(np.int64)
    if shape is None:
        shape = (int(idx[:, 1].max()) + 1, int(idx[:, 0].max()) + 1)
    keep = ((idx[:, 0] >= 0) & (idx[:, 1] >= 0)
            & (idx[:, 1] < shape[0]) & (idx[:, 0] < shape[1]))
    g = np.zeros(shape, dtype=np.float64)
    if keep.any():
        g[idx[keep, 1], idx[keep, 0]] = 1.0
    return g, np.asarray(origin, dtype=np.float64)


def correlate_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a`` を動かして ``b`` に重ねたときの一致数（完全相関）。

    **循環相関にしない**ため、各軸を ``n_a + n_b - 1`` 以上へゼロ埋めしてから
    FFT で畳み込む。返る配列の添字 ``(i, j)`` は「``a`` を行方向へ ``i - (n_a0 - 1)``、
    列方向へ ``j - (n_a1 - 1)`` だけ動かしたときの一致数」に対応する
    （``shift_from_index`` がこの変換を持つ）。
    """
    sa, sb = np.asarray(a.shape), np.asarray(b.shape)
    full = sa + sb - 1
    fs = [int(2 ** np.ceil(np.log2(v))) if v > 0 else 1 for v in full]
    A = np.fft.rfft2(a[::-1, ::-1], s=fs)      # 相関＝反転してからの畳み込み
    B = np.fft.rfft2(b, s=fs)
    c = np.fft.irfft2(A * B, s=fs)[:full[0], :full[1]]
    return np.rint(c).astype(np.float64)       # 0/1 格子なので整数になるはず


def shift_from_index(idx: Tuple[int, int], a_shape: Tuple[int, int],
                     cell: float) -> np.ndarray:
    """相関の添字を、``a`` に与える平行移動 ``(dx, dy)`` [m] へ直す。

    **ここが間違えやすい**（R29 §2-2）ので、小例検査で符号と量を必ず確かめること。
    """
    di = idx[0] - (a_shape[0] - 1)      # 行 = y
    dj = idx[1] - (a_shape[1] - 1)      # 列 = x
    return np.array([dj * cell, di * cell], dtype=np.float64)


def top_peaks(corr: np.ndarray, n_peaks: int, min_sep_cells: int) -> List[Tuple]:
    """相関の上位ピークを、互いに ``min_sep_cells`` 以上離して拾う。

    **同点のときは添字順**で決める（R29 §2-1：再現性のため順序を固定する）。
    """
    order = np.argsort(-corr.ravel(), kind="stable")
    out: List[Tuple] = []
    for flat in order:
        if corr.ravel()[flat] <= 0:
            break
        i, j = np.unravel_index(flat, corr.shape)
        if all(max(abs(i - pi), abs(j - pj)) >= min_sep_cells for pi, pj, _ in out):
            out.append((int(i), int(j), float(corr[i, j])))
        if len(out) >= n_peaks:
            break
    return out


def horizontal_candidates(src_xy_x: np.ndarray, src_xy_y: np.ndarray,
                          dst_xy_x: np.ndarray, dst_xy_y: np.ndarray,
                          cell: float, n_peaks: int,
                          min_sep_m: float) -> List[Dict]:
    """縮尺を当てた source の壁像と参照の壁像を相関させ、水平位置の候補を返す。

    x 法線群と y 法線群の相関を**足してから**ピークを取る。
    別々に取ると x と y の関係が切れ、``plane_match`` と同じ弱点に戻る。
    """
    if (len(src_xy_x) + len(src_xy_y) == 0) or (len(dst_xy_x) + len(dst_xy_y) == 0):
        return []
    all_src = np.vstack([v for v in (src_xy_x, src_xy_y) if len(v)])
    all_dst = np.vstack([v for v in (dst_xy_x, dst_xy_y) if len(v)])
    o_s = np.floor(all_src.min(axis=0) / cell) * cell
    o_d = np.floor(all_dst.min(axis=0) / cell) * cell
    sh_s = (int(np.floor((all_src[:, 1].max() - o_s[1]) / cell)) + 1,
            int(np.floor((all_src[:, 0].max() - o_s[0]) / cell)) + 1)
    sh_d = (int(np.floor((all_dst[:, 1].max() - o_d[1]) / cell)) + 1,
            int(np.floor((all_dst[:, 0].max() - o_d[0]) / cell)) + 1)

    corr = None
    for s_xy, d_xy in ((src_xy_x, dst_xy_x), (src_xy_y, dst_xy_y)):
        gs, _ = occupancy(s_xy, cell, o_s, sh_s)
        gd, _ = occupancy(d_xy, cell, o_d, sh_d)
        c = correlate_full(gs, gd)
        corr = c if corr is None else corr + c
    if corr is None:
        return []
    sep = max(1, int(round(min_sep_m / cell)))
    out = []
    for i, j, score in top_peaks(corr, n_peaks, sep):
        # a(source) を動かす量。原点の差もここで戻す。
        shift = shift_from_index((i, j), sh_s, cell) + (o_d - o_s)
        out.append({"shift_xy": shift.tolist(), "score": score,
                    "index": [int(i), int(j)]})
    return out

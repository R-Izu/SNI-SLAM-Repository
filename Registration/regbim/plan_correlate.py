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


# --------------------------------------------------------------------------- #
# 鉛直位置：床どうし・天井どうしの高さを合わせる
# --------------------------------------------------------------------------- #
def vertical_candidates(src_z: Dict[str, np.ndarray], dst_z: Dict[str, np.ndarray],
                        cell: float) -> List[Dict]:
    """床どうし・天井どうしの一致から、鉛直位置の候補を作る。

    **床と天井を取り違える対応は作らない**（R29 §1-2）。
    同じクラスどうしだけを突き合わせるので、クラス名をキーに回す。

    多層でも破綻しないよう、平均ではなく **1 次元の占有相関**で合わせる。
    **候補が 1 つも作れない場合は空を返す**——GT や手入力で補わない（R29 §2-2）。
    """
    out: List[Dict] = []
    for name in ("floor", "ceiling"):
        s, d = src_z.get(name), dst_z.get(name)
        if s is None or d is None or len(s) < 10 or len(d) < 10:
            continue
        o_s = np.floor(s.min() / cell) * cell
        o_d = np.floor(d.min() / cell) * cell
        gs = np.zeros(int(np.floor((s.max() - o_s) / cell)) + 1)
        gd = np.zeros(int(np.floor((d.max() - o_d) / cell)) + 1)
        gs[np.floor((s - o_s) / cell).astype(np.int64)] = 1.0
        gd[np.floor((d - o_d) / cell).astype(np.int64)] = 1.0
        c = correlate_full(gs[None, :], gd[None, :])[0]
        k = int(np.argmax(c))
        dz = (k - (len(gs) - 1)) * cell + (o_d - o_s)
        out.append({"dz": float(dz), "score": float(c[k]), "from": name})
    out.sort(key=lambda r: -r["score"])
    return out


def _dedup(cands: List[Dict], xy_tol: float, z_tol: float,
           log_s_tol: float) -> List[Dict]:
    """同じ山を1つにまとめる（R29 §2-1 の同一視の条件。**3つとも満たすとき同じ**）。"""
    kept: List[Dict] = []
    for c in cands:
        dup = False
        for k in kept:
            if (np.linalg.norm(np.asarray(c["shift_xy"]) - np.asarray(k["shift_xy"])) < xy_tol
                    and abs(c["dz"] - k["dz"]) < z_tol
                    and abs(np.log(c["scale"]) - np.log(k["scale"])) < log_s_tol):
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


# R29 §2-1 の設定値。**測定前に固定したもの。結果を見てから変えない。**
DEFAULTS = {
    "cell_m": 0.25,              # 粗い画像のセル幅（参照座標）
    "scale_lo": 0.3,             # 縮尺探索の下限
    "scale_hi": 3.0,             # 上限
    "scale_n": 59,               # 対数間隔の点数
    "peak_sep_m": 0.5,           # 水平ピークの最小間隔
    "n_peaks": 3,                # 水平ピークの数
    "n_vertical": 2,             # 鉛直候補の数
    "dedup_xy_m": 0.5,           # 同一視：水平位置差
    "dedup_z_m": 0.25,           # 同一視：高さ差
    "dedup_log_s": 0.04,         # 同一視：対数縮尺差
    "max_points": 5000,          # source・参照それぞれの上限
    "subsample_seed": 0,         # **全条件共通の固定 seed**
    "per_yaw_coarse": 4,         # 各向きから短い精緻化へ渡す数
    "wall_axis_tol_deg": WALL_AXIS_TOL_DEG,
}


def _subsample_mask(n: int, n_max: int, seed: int) -> np.ndarray:
    """**全条件共通の固定 seed** で間引く添字を返す（R29 §2-1）。"""
    if n <= n_max:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, n_max, replace=False))


def _subsample_by_role(pts, labels, normals, n_max, seed, wall, floor, ceil):
    """**役割ごとに** 5,000 点の枠を当てる（R29 §2-1 の「それぞれ最大 5,000 点」）。

    ★ **最初の実装は全クラスまとめて 5,000 点に間引いていた。**
      床と天井が点数の 6 割以上を占めるので、**水平相関を動かす壁がやせ細り、
      占有格子に穴が空いた。** その結果、正解の縮尺 1.0 での一致が
      108 セルから 41 セルまで落ち、**誤った縮尺 0.888 に負けた**
      （間引きを外すと 1.0 が 108 対 68 で明確に勝つことを確認済み）。

      **枠の大きさ（5,000）は R29 §2-1 のまま変えていない。**
      変えたのは「何に対する 5,000 か」であり、R29 が定めていなかった点である。
      水平相関を動かすのは壁だけ、鉛直を動かすのは床と天井だけなので、
      **それぞれに枠を与える。**
    """
    idx = []
    for cid in (wall, floor, ceil):
        w = np.flatnonzero(labels == cid)
        if len(w):
            idx.append(w[_subsample_mask(len(w), n_max, seed)])
    if not idx:
        return pts[:0], labels[:0], (None if normals is None else normals[:0])
    k = np.sort(np.concatenate(idx))
    return pts[k], labels[k], (None if normals is None else normals[k])


def generate_candidates(src_pts, src_labels, src_normals,
                        dst_pts, dst_labels, dst_normals,
                        rotations, name_to_id, cfg=None) -> Tuple[List[Dict], Dict]:
    """案A の候補生成（R29 §1-2）。

    **向きごとに枠を確保する**（`plane_match` は全向きを一括で絞ってしまう）。
    返すのは ``(候補のリスト, 診断)``。候補には通し番号 ``cand_id`` が付く
    （R29 §3：同じ変換を段階をまたいで追跡するため）。

    **GT は一切受け取らない。** 探索は参照と source だけで閉じている。
    """
    p = dict(DEFAULTS, **(cfg or {}))
    wall, floor, ceil = (name_to_id["wall"], name_to_id["floor"],
                         name_to_id["ceiling"])

    sp, sl, sn = _subsample_by_role(src_pts, src_labels, src_normals,
                                    p["max_points"], p["subsample_seed"],
                                    wall, floor, ceil)
    dp, dl, dn = _subsample_by_role(dst_pts, dst_labels, dst_normals,
                                    p["max_points"], p["subsample_seed"],
                                    wall, floor, ceil)

    d_wall_m = dl == wall
    dx_xy, dy_xy = wall_groups(dp[d_wall_m], None if dn is None else dn[d_wall_m],
                               p["wall_axis_tol_deg"])
    dst_z = {"floor": dp[dl == floor][:, 2], "ceiling": dp[dl == ceil][:, 2]}

    scales = np.exp(np.linspace(np.log(p["scale_lo"]), np.log(p["scale_hi"]),
                                int(p["scale_n"])))
    diag = {"n_scales": len(scales), "per_yaw": [],
            "vertical_shortfall": 0, "n_raw": 0}
    out: List[Dict] = []
    cid = 0
    for ri, R in enumerate(rotations):
        rot_pts = sp @ np.asarray(R).T
        rot_nrm = None if sn is None else sn @ np.asarray(R).T
        w_m = sl == wall
        per_yaw: List[Dict] = []
        for s in scales:
            sx, sy = wall_groups((rot_pts[w_m] * s),
                                 None if rot_nrm is None else rot_nrm[w_m],
                                 p["wall_axis_tol_deg"])
            hs = horizontal_candidates(sx, sy, dx_xy, dy_xy, p["cell_m"],
                                       int(p["n_peaks"]), p["peak_sep_m"])
            if not hs:
                continue
            src_z = {"floor": rot_pts[sl == floor][:, 2] * s,
                     "ceiling": rot_pts[sl == ceil][:, 2] * s}
            vs = vertical_candidates(src_z, dst_z, p["cell_m"])[:int(p["n_vertical"])]
            if not vs:
                diag["vertical_shortfall"] += 1     # **補わない。数える**
                continue
            for h in hs:
                for v in vs:
                    per_yaw.append({"scale": float(s), "shift_xy": h["shift_xy"],
                                    "dz": v["dz"], "score_h": h["score"],
                                    "score_v": v["score"], "yaw_index": ri,
                                    "vertical_from": v["from"]})
        diag["n_raw"] += len(per_yaw)
        # R29 §2-1 の並べ方：水平相関値 → 同点なら鉛直 → 同点なら固定した添字順
        per_yaw = sorted(enumerate(per_yaw),
                         key=lambda t: (-t[1]["score_h"], -t[1]["score_v"], t[0]))
        per_yaw = [c for _, c in per_yaw]
        per_yaw = _dedup(per_yaw, p["dedup_xy_m"], p["dedup_z_m"], p["dedup_log_s"])
        kept = per_yaw[:int(p["per_yaw_coarse"])]
        diag["per_yaw"].append({"yaw_index": ri, "n_after_dedup": len(per_yaw),
                                "n_kept": len(kept)})
        for c in kept:
            s, R_ = c["scale"], np.asarray(R, dtype=np.float64)
            # t = b - s R a + u。**中心どうしを対応点とみなしているのではない**
            t = np.array([c["shift_xy"][0], c["shift_xy"][1], c["dz"]])
            T = np.eye(4)
            T[:3, :3] = s * R_
            T[:3, 3] = t
            out.append(dict(c, cand_id=cid, T=T))
            cid += 1
    diag["n_candidates"] = len(out)
    return out, diag

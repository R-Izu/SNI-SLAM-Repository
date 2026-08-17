"""S0 事前フライトチェック — SLAM を回さずにスキャン品質を判定する。

指示書 00_Coordination/Instructions/2026-08-17_r4_realdata_slam_execution.md §2「追加1」に対応。
``odometry.csv`` と ``depth/`` だけで、1 run 38 分を投じる前に赤いスキャンを弾く。

算出する指標
------------
フレーム数の整合   rgb / depth / confidence / odometry の 4 者一致
始終点距離         軌跡の始点-終点ユークリッド距離（<= 0.5 m でループ閉じとみなす）
経路長・所要時間   軌跡の積算長、フレーム数 / 30fps と timestamp 差の双方
水平被覆           床高さ ±0.2 m の逆投影点の水平凸包面積（および α-shape 面積）
天井の可視性       カメラ高さ +0.5 m より上の点の割合、床上 2.2〜3.6 m の水平面ピーク
depth 有効率       conf >= 1 の画素率、depth == 0 の割合
depth 距離分布     ヒストグラム（5 m 付近の打ち切りの有無）
姿勢規約の確認     OpenCV / OpenGL 両解釈での 2 フレーム間 NN 距離中央値・床法線

出力
----
<out>/preflight_all.csv        1 行 = 1 スキャン
<out>/preflight_<scan>.json    スキャンごとの全指標（ヒストグラム含む）
<out>/traj_<scan>.png          軌跡俯瞰図（床点の散布図つき）

使い方
------
    conda activate sni-slam
    python tools/realdata/preflight.py --root Real-data --out output/RealData/_preflight
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stray_io  # noqa: E402

# --- 判定閾値 -------------------------------------------------------------
LOOP_CLOSE_M = 0.5          # 始終点距離がこれ以下ならループ閉じ
CEILING_MIN_FRAC = 0.02     # 天井帯に入る点の割合の下限
CEILING_H_RANGE = (2.2, 3.6)  # 床から天井までの妥当な高さ [m]
NN_MEDIAN_MAX_M = 0.08      # 2 フレーム間 NN 距離中央値の受入上限（指示書 S1）
FLOOR_NORMAL_MAX_DEG = 3.0  # 床法線と重力軸のなす角の受入上限（指示書 S1）
FLOOR_BAND_M = 0.2          # 「床面付近」の帯の半幅


# ===========================================================================
# 幾何ヘルパ
# ===========================================================================

def _voxel_downsample_2d(pts2d: np.ndarray, voxel: float) -> np.ndarray:
    if len(pts2d) == 0:
        return pts2d
    keys = np.floor(pts2d / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts2d[np.sort(idx)]


def convex_hull_area(pts2d: np.ndarray) -> float:
    """2D 凸包面積 [m^2]。点が足りない場合は 0。"""
    from scipy.spatial import ConvexHull
    if len(pts2d) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts2d).volume)  # 2D では volume が面積
    except Exception:   # QhullError の import 位置が scipy 版で異なるため広めに捕捉
        return 0.0


def alpha_shape_area(pts2d: np.ndarray, alpha: float = 0.5) -> float:
    """α-shape（α-complex）の面積 [m^2]。

    Delaunay 三角形のうち外接円半径が ``alpha`` 以下のものだけを残して面積を足す。
    L 字など非凸な床形状で、凸包が面積を過大評価する分を補正するために使う。
    shapely が env に無いため scipy.spatial.Delaunay で直接実装している。
    """
    from scipy.spatial import Delaunay
    if len(pts2d) < 4:
        return 0.0
    try:
        tri = Delaunay(pts2d)
    except Exception:   # 同上
        return 0.0
    p = pts2d[tri.simplices]                      # (T,3,2)
    a = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    b = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        circum_r = np.where(area > 1e-12, a * b * c / (4.0 * area), np.inf)
    return float(area[circum_r <= alpha].sum())


def fit_plane_normal(pts: np.ndarray) -> np.ndarray:
    """点群に最小二乗平面を当て、単位法線を返す。"""
    if len(pts) < 3:
        return np.array([0.0, 0.0, 0.0])
    c = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    n = vt[-1]
    return n / (np.linalg.norm(n) + 1e-12)


def _y_histogram(y: np.ndarray, bin_w: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """世界 Y のヒストグラム。外れ値を除くため 0.2〜99.8 パーセンタイルで切る。"""
    lo, hi = np.percentile(y, [0.2, 99.8])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < bin_w:
        return np.zeros(0, dtype=int), np.zeros(0)
    edges = np.arange(lo, hi + bin_w, bin_w)
    hist, edges = np.histogram(y, bins=edges)
    return hist, (edges[:-1] + edges[1:]) / 2.0


def _peak_of_extreme_run(hist: np.ndarray, rel_thr: float, from_top: bool) -> Optional[int]:
    """閾値を超える連続 run のうち、最下（または最上）の run の頂点 index を返す。

    水平面（床・天井・机上面）は Y ヒストグラムで鋭いピークになり、垂直面（壁）は
    Y 方向に散るのでピークにならない。床は「下から数えて最初の有意なピーク」、
    天井は「上から数えて最初の有意なピーク」として取る。
    単純な argmax では、教室の机上面のように床より点数の多い水平面を拾ってしまう。
    """
    if hist.size == 0 or hist.max() <= 0:
        return None
    above = hist >= rel_thr * hist.max()
    if not above.any():
        return None
    if from_top:
        end = len(above) - 1 - int(np.argmax(above[::-1]))
        start = end
        while start - 1 >= 0 and above[start - 1]:
            start -= 1
    else:
        start = int(np.argmax(above))
        end = start
        while end + 1 < len(above) and above[end + 1]:
            end += 1
    return start + int(np.argmax(hist[start:end + 1]))


def detect_floor_y(y: np.ndarray, rel_thr: float = 0.25) -> float:
    """世界 Y のヒストグラムから床高さを推定する（+Y が上）。"""
    hist, centers = _y_histogram(y)
    k = _peak_of_extreme_run(hist, rel_thr, from_top=False)
    if k is None:
        return float(np.percentile(y, 0.5))
    # ビン幅 5 cm の量子化誤差を消すため、ピーク近傍の中央値で精密化する
    near = y[np.abs(y - centers[k]) <= 0.08]
    return float(np.median(near)) if len(near) >= 50 else float(centers[k])


def detect_ceiling(y: np.ndarray, floor_y: float,
                   rel_thr: float = 0.25) -> Tuple[Optional[float], float]:
    """床上 1.8 m より上で最上位の有意ピークを探し (天井高 [m], 帯内点の割合) を返す。"""
    sel = y[y >= floor_y + 1.8]
    if len(sel) < 50:
        return None, 0.0
    hist, centers = _y_histogram(sel)
    k = _peak_of_extreme_run(hist, rel_thr, from_top=True)
    if k is None:
        return None, 0.0
    ceil_y = float(centers[k])
    frac = float(np.count_nonzero(np.abs(y - ceil_y) <= 0.15) / max(len(y), 1))
    return ceil_y - floor_y, frac


# ===========================================================================
# 姿勢規約の確認（2 フレーム間 NN 距離・床法線）
# ===========================================================================

def check_pose_convention(
    scan_dir: str,
    odo: Dict[str, np.ndarray],
    depth_files: List[str],
    conf_files: List[str],
    pairs: int = 5,
    gap: int = 3,
    conf_min: int = 1,
) -> Dict[str, Dict[str, float]]:
    """OpenCV / OpenGL 両解釈で 2 フレーム間 NN 距離中央値と床法線角を測る。"""
    from scipy.spatial import cKDTree
    n = len(depth_files)
    if n < gap + 2:
        return {}
    idxs = np.linspace(int(n * 0.2), int(n * 0.8) - gap, pairs).astype(int)
    idxs = [i for i in idxs if 0 <= i and i + gap < n]

    out: Dict[str, Dict[str, float]] = {}
    for conv in ("opencv", "opengl"):
        c2w = stray_io.build_c2w(odo, convention=conv)
        med: List[float] = []
        acc: List[np.ndarray] = []
        for i in idxs:
            clouds = []
            for j in (i, i + gap):
                d = stray_io.read_depth_m(depth_files[j])
                cf = stray_io.read_confidence(conf_files[j]) if conf_files else None
                clouds.append(stray_io.backproject_frame(
                    d, cf, odo["intr"][j], c2w[j], conf_min=conf_min, pix_stride=2))
            if len(clouds[0]) < 200 or len(clouds[1]) < 200:
                continue
            dist, _ = cKDTree(clouds[1]).query(clouds[0], k=1)
            med.append(float(np.median(dist)))
            acc.append(clouds[0])
        if not acc:
            continue
        pts = np.concatenate(acc, axis=0)
        # 最下層 5% を床とみなして法線を測る（軸は解釈ごとに変わりうるので Y 固定で見る）
        thr = np.percentile(pts[:, 1], 5.0)
        nrm = fit_plane_normal(pts[pts[:, 1] <= thr])
        if nrm[1] < 0:
            nrm = -nrm
        ang = float(np.degrees(np.arccos(np.clip(abs(nrm[1]), -1.0, 1.0))))
        out[conv] = {
            "nn_median_m": float(np.median(med)) if med else float("nan"),
            "floor_normal": [float(v) for v in nrm],
            "floor_normal_vs_up_deg": ang,
        }
    return out


# ===========================================================================
# スキャン 1 本の解析
# ===========================================================================

def analyze_scan(
    scan_dir: str,
    frame_stride: int,
    pix_stride: int,
    conf_min: int,
    alpha: float,
    check_convention: bool = True,
    max_depth: float = 8.0,
) -> Dict:
    scan_id = os.path.basename(scan_dir.rstrip("/"))
    t0 = time.time()
    res: Dict = {"scan_id": scan_id}

    # --- フレーム数の整合 ---
    odo = stray_io.read_odometry(scan_dir)
    depth_files = stray_io.list_depth_files(scan_dir)
    conf_files = stray_io.list_confidence_files(scan_dir)
    n_rgb = stray_io.rgb_frame_count(scan_dir)
    n_odo, n_dep, n_cnf = len(odo["pos"]), len(depth_files), len(conf_files)
    res["n_rgb"], res["n_depth"], res["n_conf"], res["n_odometry"] = n_rgb, n_dep, n_cnf, n_odo
    res["frames_consistent"] = bool(n_rgb == n_dep == n_cnf == n_odo)

    # --- 軌跡 ---
    pos = odo["pos"]
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    res["start_end_dist_m"] = float(np.linalg.norm(pos[-1] - pos[0]))
    res["path_length_m"] = float(seg.sum())
    res["duration_s_by_timestamp"] = float(odo["timestamp"][-1] - odo["timestamp"][0])
    res["duration_s_by_fps"] = float(n_odo / stray_io.FPS)
    res["loop_closed"] = bool(res["start_end_dist_m"] <= LOOP_CLOSE_M)
    res["traj_extent_m"] = [float(v) for v in (pos.max(axis=0) - pos.min(axis=0))]

    # --- depth 品質と逆投影（フレーム間引き） ---
    sel = list(range(0, n_dep, frame_stride))
    c2w = stray_io.build_c2w(odo, convention="opencv")
    clouds, zero_frac, conf_ok_frac, zs = [], [], [], []
    for i in sel:
        d = stray_io.read_depth_m(depth_files[i])
        cf = stray_io.read_confidence(conf_files[i]) if conf_files else None
        zero_frac.append(float(np.count_nonzero(d <= 0.0) / d.size))
        if cf is not None:
            conf_ok_frac.append(float(np.count_nonzero(cf >= conf_min) / cf.size))
        zs.append(d[d > 0.05][::7])
        if i < len(c2w):
            clouds.append(stray_io.backproject_frame(
                d, cf, odo["intr"][i], c2w[i], conf_min=conf_min,
                pix_stride=pix_stride, max_depth=max_depth))
    res["frames_sampled"] = len(sel)
    res["conf_min"], res["max_depth_m"] = conf_min, max_depth
    res["depth_zero_frac"] = float(np.mean(zero_frac)) if zero_frac else float("nan")
    res["depth_conf_ok_frac"] = float(np.mean(conf_ok_frac)) if conf_ok_frac else float("nan")

    zall = np.concatenate(zs) if zs else np.zeros(0, dtype=np.float32)
    if len(zall):
        hist, edges = np.histogram(zall, bins=np.arange(0.0, 10.01, 0.25))
        res["depth_hist_bins"] = [float(v) for v in edges]
        res["depth_hist"] = [int(v) for v in hist]
        res["depth_p50_m"] = float(np.percentile(zall, 50))
        res["depth_p99_m"] = float(np.percentile(zall, 99))
        res["depth_max_m"] = float(zall.max())
        # 打ち切り判定: 4.5-5.5 m に不自然な突出（直前 1 m 平均の 1.5 倍以上）があるか
        band = (edges[:-1] >= 4.5) & (edges[:-1] < 5.5)
        prev = (edges[:-1] >= 3.5) & (edges[:-1] < 4.5)
        res["depth_truncation_5m"] = bool(
            hist[prev].mean() > 0 and hist[band].max() > 1.5 * hist[prev].mean())

    pts = np.concatenate(clouds, axis=0) if clouds else np.zeros((0, 3), np.float32)
    res["n_points"] = int(len(pts))
    if len(pts) < 1000:
        res["error"] = "too few back-projected points"
        res["elapsed_s"] = time.time() - t0
        return res

    # --- 床・天井 ---
    y = pts[:, 1]
    floor_y = detect_floor_y(y)
    cam_y = float(np.median(pos[:, 1]))
    res["floor_y"] = floor_y
    res["camera_height_m"] = cam_y - floor_y
    res["frac_above_cam_plus_0.5"] = float(np.count_nonzero(y > cam_y + 0.5) / len(y))
    ceil_h, ceil_frac = detect_ceiling(y, floor_y)
    res["ceiling_height_m"] = ceil_h
    res["ceiling_band_frac"] = ceil_frac
    res["ceiling_ok"] = bool(
        ceil_h is not None
        and CEILING_H_RANGE[0] <= ceil_h <= CEILING_H_RANGE[1]
        and ceil_frac >= CEILING_MIN_FRAC)

    # 床法線（OpenCV 解釈のまま。重力軸 = +Y との角度）
    fl = pts[np.abs(y - floor_y) <= 0.1]
    if len(fl) >= 100:
        nrm = fit_plane_normal(fl)
        if nrm[1] < 0:
            nrm = -nrm
        res["floor_normal"] = [float(v) for v in nrm]
        res["floor_normal_vs_up_deg"] = float(
            np.degrees(np.arccos(np.clip(abs(nrm[1]), -1.0, 1.0))))

    # --- 水平被覆（床帯の逆投影点） ---
    fb = pts[np.abs(y - floor_y) <= FLOOR_BAND_M][:, [0, 2]]   # XZ 平面
    fb_ds = _voxel_downsample_2d(fb.astype(np.float64), 0.05)
    res["n_floor_points"] = int(len(fb))
    res["n_floor_points_ds"] = int(len(fb_ds))
    res["hull_area_m2"] = convex_hull_area(fb_ds)
    res["alpha_shape_area_m2"] = alpha_shape_area(fb_ds, alpha=alpha)
    res["solidity"] = float(res["alpha_shape_area_m2"] / res["hull_area_m2"]) \
        if res["hull_area_m2"] > 0 else 0.0

    # --- 姿勢規約 ---
    if check_convention:
        res["pose_convention"] = check_pose_convention(
            scan_dir, odo, depth_files, conf_files, conf_min=conf_min)

    res["elapsed_s"] = time.time() - t0
    res["_pts_for_plot"] = fb_ds
    res["_traj_for_plot"] = pos[:, [0, 2]]
    return res


# ===========================================================================
# 描画
# ===========================================================================

def plot_scan(res: Dict, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fb = res.get("_pts_for_plot")
    tr = res.get("_traj_for_plot")
    fig, ax = plt.subplots(figsize=(7, 7))
    if fb is not None and len(fb):
        ax.scatter(fb[:, 0], fb[:, 1], s=0.4, c="0.75", linewidths=0, label="floor points")
    if tr is not None and len(tr):
        ax.plot(tr[:, 0], tr[:, 1], "-", lw=1.2, color="tab:blue", label="trajectory")
        ax.plot(tr[0, 0], tr[0, 1], "o", ms=9, color="tab:green", label="start")
        ax.plot(tr[-1, 0], tr[-1, 1], "s", ms=9, color="tab:red", label="end")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("world X [m]")
    ax.set_ylabel("world Z [m]")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(
        "%s\nstart-end %.2f m / path %.1f m / hull %.1f m2 / alpha %.1f m2\n"
        "ceiling %s (h=%s, frac=%.3f)" % (
            res["scan_id"], res.get("start_end_dist_m", float("nan")),
            res.get("path_length_m", float("nan")),
            res.get("hull_area_m2", 0.0), res.get("alpha_shape_area_m2", 0.0),
            "OK" if res.get("ceiling_ok") else "NG",
            "%.2f" % res["ceiling_height_m"] if res.get("ceiling_height_m") else "None",
            res.get("ceiling_band_frac", 0.0)),
        fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ===========================================================================
# CSV
# ===========================================================================

CSV_COLS = [
    "scan_id", "verdict",
    "n_rgb", "n_depth", "n_conf", "n_odometry", "frames_consistent",
    "duration_s_by_fps", "path_length_m", "start_end_dist_m", "loop_closed",
    "traj_extent_x", "traj_extent_y", "traj_extent_z",
    "hull_area_m2", "alpha_shape_area_m2", "solidity",
    "camera_height_m", "ceiling_height_m", "ceiling_band_frac",
    "frac_above_cam_plus_0.5", "ceiling_ok",
    "floor_normal_vs_up_deg", "nn_median_opencv_m", "nn_median_opengl_m",
    "depth_conf_ok_frac", "depth_zero_frac", "depth_p50_m", "depth_p99_m",
    "depth_max_m", "depth_truncation_5m", "n_points", "frames_sampled",
]


def verdict_of(res: Dict) -> Tuple[str, List[str]]:
    """赤 / 黄 / 緑 と理由を返す。"""
    reasons: List[str] = []
    if res.get("error"):
        return "RED", [res["error"]]
    if not res.get("frames_consistent"):
        reasons.append("frame count mismatch")
    if not res.get("ceiling_ok"):
        reasons.append("ceiling not detected")
    nn = (res.get("pose_convention", {}).get("opencv", {}) or {}).get("nn_median_m")
    if nn is not None and nn == nn and nn >= NN_MEDIAN_MAX_M:
        reasons.append("pose NN median %.3f m >= %.2f" % (nn, NN_MEDIAN_MAX_M))
    fn = res.get("floor_normal_vs_up_deg")
    if fn is not None and fn > FLOOR_NORMAL_MAX_DEG:
        reasons.append("floor normal %.1f deg > %.1f" % (fn, FLOOR_NORMAL_MAX_DEG))
    if reasons:
        return "RED", reasons
    warn: List[str] = []
    if not res.get("loop_closed"):
        warn.append("no loop closure (start-end %.2f m)" % res.get("start_end_dist_m", -1))
    if res.get("depth_conf_ok_frac", 1.0) < 0.9:
        warn.append("low confidence rate %.2f" % res.get("depth_conf_ok_frac", -1))
    return ("YELLOW", warn) if warn else ("GREEN", [])


def row_of(res: Dict) -> Dict[str, object]:
    ext = res.get("traj_extent_m", [None, None, None])
    pc = res.get("pose_convention", {})
    row = {c: "" for c in CSV_COLS}
    for k in CSV_COLS:
        if k in res:
            row[k] = res[k]
    row["traj_extent_x"], row["traj_extent_y"], row["traj_extent_z"] = ext
    row["nn_median_opencv_m"] = (pc.get("opencv") or {}).get("nn_median_m", "")
    row["nn_median_opengl_m"] = (pc.get("opengl") or {}).get("nn_median_m", "")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="Real-data", help="スキャン群のルート")
    ap.add_argument("--out", default="output/RealData/_preflight")
    ap.add_argument("--scans", nargs="*", default=None, help="対象 scan_id（既定は全件）")
    ap.add_argument("--frame-stride", type=int, default=15, help="逆投影に使うフレーム間引き")
    ap.add_argument("--pix-stride", type=int, default=2, help="逆投影の画素間引き")
    ap.add_argument("--conf-min", type=int, default=1, help="採用する confidence の下限")
    ap.add_argument("--alpha", type=float, default=0.5, help="α-shape の α [m]")
    ap.add_argument("--max-depth", type=float, default=8.0,
                    help="逆投影に採用する depth の上限 [m]。iPad LiDAR の実測レンジは "
                         "約 5 m で、それ以遠は ARKit の外挿値なので被覆面積を膨らませる")
    ap.add_argument("--no-convention-check", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scans = args.scans or sorted(
        d for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d))
        and os.path.exists(os.path.join(args.root, d, "odometry.csv")))

    rows, summary = [], []
    for k, s in enumerate(scans, 1):
        print("[%d/%d] %s ..." % (k, len(scans), s), flush=True)
        try:
            res = analyze_scan(
                os.path.join(args.root, s), args.frame_stride, args.pix_stride,
                args.conf_min, args.alpha, not args.no_convention_check,
                max_depth=args.max_depth)
        except Exception as e:  # 1 本の失敗で全体を落とさない
            import traceback
            traceback.print_exc()
            res = {"scan_id": s, "error": "%s: %s" % (type(e).__name__, e)}
        v, reasons = verdict_of(res)
        res["verdict"], res["verdict_reasons"] = v, reasons
        try:
            plot_scan(res, os.path.join(args.out, "traj_%s.png" % s))
        except Exception as e:
            print("  plot failed: %s" % e)
        dump = {k2: v2 for k2, v2 in res.items() if not k2.startswith("_")}
        with open(os.path.join(args.out, "preflight_%s.json" % s), "w") as f:
            json.dump(dump, f, indent=2, ensure_ascii=False)
        rows.append(row_of(res))
        summary.append((s, v, reasons))
        print("  -> %s %s (%.1fs)" % (v, ("; ".join(reasons)) if reasons else "",
                                      res.get("elapsed_s", 0.0)), flush=True)

    import csv
    csv_path = os.path.join(args.out, "preflight_all.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s" % csv_path)
    print("%-14s %-7s %s" % ("scan_id", "verdict", "reasons"))
    for s, v, r in summary:
        print("%-14s %-7s %s" % (s, v, "; ".join(r)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

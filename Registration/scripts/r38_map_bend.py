"""R38 §4 — SLAM 地図の曲がりを測る。**手法は再実行しない。** 保存済みの変換と点群だけを使う。

固定した手順（結果を見る前に決めた。R38 §4-1・§4-2）
------------------------------------------------------
- 点：source（seed 0）を `preprocess.prepare` した点群の wall 点（`rotation.py` と同じ前処理・同じ法線）
- 置き方：G1（`criterion_spread.json` の基準 1）で BIM 座標へ置く
- 領域：G1 で置いた wall 点のうち、その条件の BIM 参照のいずれかの点（壁・床・天井・扉・窓。参照に入っているもの全部）
  から**水平距離 0.3 m 以内**を「部屋」、それ以外を「廊下」。8 条件それぞれで分ける
- ヨー：法線を BIM 座標へ回し、水平成分が 0.3 を超えるものだけを使う（`rotation.py` と同じ条件）。
  水平の角度を 90° で畳んで [-45°, 45°) に置き（BIM の軸が 0° に来る向き）、**幅 0.25° の bin で投票**する。
  ピーク（bin の中心）と、ピーク ±5° 内の平均の両方を出す
- 差：Δψ = ψ_廊下 − ψ_部屋（符号つき）
- 不確かさ：部屋・廊下それぞれの点を復元抽出し 1,000 回推定し直した Δψ の 2.5〜97.5% 点（乱数 seed 0）
- 区間：廊下の点を BIM の X で 5 m ごとに分け、区間ごとにピーク ±5° 内の平均（点が 200 未満の区間は出さない）
- 手法の解：R35 の案A の解（`r35_runs.json`、回転固定）で同じ点を置いたときの廊下のヨーと、G1 に対するヨー誤差 ψ_err
- §4-4：G1 で置いた wall 点と、BIM（E2＝411）の長辺の壁（法線が ±Y の壁面のうち、Y の最小・最大の 2 枚）との
  **壁に垂直な方向の離れ**（室の外向きを正）を、X 方向 1 m ごとの区間の中央値で出す。
  区間ごとに、壁の高さ（BIM の床〜天井）を 10 等分した段のうち SLAM の点がある段の割合も出す

    conda activate sni-slam
    python Registration/scripts/r38_map_bend.py --png-dir <PNG の出力先>
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import yaml
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
if not os.path.isdir("Registration"):
    os.chdir(REPO)

from failure_decomposition import provenance                 # noqa: E402
from r36_rotation_diag import tilt_yaw                       # noqa: E402
from regbim import io_utils, metrics, preprocess             # noqa: E402

TARGETS = ["m3_cor_a__E2", "m3_cor_a__E3", "m3_cor_b__E2", "m3_cor_b__E3",
           "m3_cor_c__E2", "m3_cor_c__E3", "m3_cor_d__E2", "m3_cor_d__E3"]
ROOM_DIST = 0.3
BIN = 0.25
WIN = 5.0
N_BOOT = 1000
SEG = 5.0
SEG_MIN = 200


def g1(scene):
    sp = json.load(open("Registration/output/diag/criterion_spread.json"))["scenes"][scene]
    return np.asarray(next(f["T"] for f in sp["results"] if f["id"] == 1), dtype=np.float64)


def r35(target):
    rows = json.load(open("Registration/output/diag/r35_runs.json"))["rows"]
    return np.asarray(next(r["T_est"] for r in rows if "%s__%s" % (r["scene"], r["cond"]) == target
                           and r["mode"] == "plan_correlate"), dtype=np.float64)


def yaw_deg(normals_bim):
    """法線（BIM 座標）→ 水平の角度 [-45, 45)。水平成分 0.3 以下は NaN。"""
    h = normals_bim[:, :2]
    mag = np.linalg.norm(h, axis=1)
    a = np.degrees(np.arctan2(h[:, 1], h[:, 0]))
    a = (a + 45.0) % 90.0 - 45.0
    a[mag <= 0.3] = np.nan
    return a


def vote(a):
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan"), 0
    hist, edges = np.histogram(a, bins=int(90 / BIN), range=(-45.0, 45.0))
    pk = float(edges[int(np.argmax(hist))] + BIN / 2)
    d = (a - pk + 45.0) % 90.0 - 45.0
    near = d[np.abs(d) <= WIN]
    return pk, float(pk + near.mean()) if len(near) else float("nan"), int(len(a))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png-dir", required=True)
    ap.add_argument("--out", default="Registration/output/diag/r38_map_bend.json")
    args = ap.parse_args()
    os.makedirs(args.png_dir, exist_ok=True)
    res = {"provenance": provenance(), "params": {"room_dist_m": ROOM_DIST, "bin_deg": BIN, "win_deg": WIN,
                                                  "n_boot": N_BOOT, "seg_m": SEG, "seg_min": SEG_MIN},
           "conditions": {}, "segments": {}, "wall_offsets": {}}
    rng = np.random.default_rng(0)
    seg_plot = {}

    for target in TARGETS:
        scene, cond = target.split("__")
        cfg = copy.deepcopy(yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target)))
        cfg["source"] = dict(cfg["source"], seed=0)
        src = preprocess.prepare(io_utils.load_source_cloud(cfg), cfg)
        dst = io_utils.load_reference_cloud(cfg)
        wall = src.subset(src.class_mask("wall"))
        G = g1(scene)
        RG = metrics.decompose_sim3(G)[0]
        Tm = r35(target)
        Rm = metrics.decompose_sim3(Tm)[0]
        P = metrics.apply_sim3(G, wall.points)
        a_g = yaw_deg(wall.normals @ RG.T)
        a_m = yaw_deg(wall.normals @ Rm.T)
        dist, _ = cKDTree(dst.points[:, :2]).query(P[:, :2], k=1, workers=-1)
        room = dist <= ROOM_DIST
        cor = ~room

        pr = vote(a_g[room]); pc = vote(a_g[cor]); pcm = vote(a_m[cor]); prm = vote(a_m[room])
        # 復元抽出
        ar, ac = a_g[room & np.isfinite(a_g)], a_g[cor & np.isfinite(a_g)]
        bp, bm = [], []
        for _ in range(N_BOOT):
            r1 = vote(rng.choice(ar, len(ar), replace=True))
            c1 = vote(rng.choice(ac, len(ac), replace=True))
            bp.append(c1[0] - r1[0]); bm.append(c1[1] - r1[1])
        ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        yerr = tilt_yaw(Rm, RG)
        res["conditions"][target] = {
            "n_wall": int(len(wall)), "n_room_used": pr[2], "n_corr_used": pc[2],
            "room_peak_deg": pr[0], "room_mean_deg": pr[1],
            "corr_peak_deg": pc[0], "corr_mean_deg": pc[1],
            "dpsi_peak_deg": pc[0] - pr[0], "dpsi_mean_deg": pc[1] - pr[1],
            "dpsi_peak_ci95": ci(bp), "dpsi_mean_ci95": ci(bm),
            "method_psi_err_deg": yerr["yaw_deg"], "method_tilt_deg": yerr["tilt_deg"],
            "method_corr_peak_deg": pcm[0], "method_corr_mean_deg": pcm[1],
            "method_room_peak_deg": prm[0], "method_room_mean_deg": prm[1]}
        c = res["conditions"][target]
        print("%-14s 部屋 %6d 点 ψ %+.3f/%+.3f | 廊下 %6d 点 ψ %+.3f/%+.3f | Δψ %+.3f [%+.2f,%+.2f] / %+.3f [%+.2f,%+.2f] | 手法 ψerr %+.3f 廊下ψ(手法) %+.3f/%+.3f"
              % (target, pr[2], pr[0], pr[1], pc[2], pc[0], pc[1], c["dpsi_peak_deg"], *c["dpsi_peak_ci95"],
                 c["dpsi_mean_deg"], *c["dpsi_mean_ci95"], yerr["yaw_deg"], pcm[0], pcm[1]), flush=True)

        # 区間ごとの推移（廊下）
        xs = P[cor, 0]; ac_all = a_g[cor]
        edges = np.arange(np.floor(xs.min() / SEG) * SEG, xs.max() + SEG, SEG)
        segs = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (xs >= lo) & (xs < hi) & np.isfinite(ac_all)
            if m.sum() < SEG_MIN:
                continue
            d = (ac_all[m] - pc[0] + 45.0) % 90.0 - 45.0
            near = d[np.abs(d) <= WIN]
            segs.append({"x_lo": float(lo), "x_hi": float(hi), "n": int(m.sum()),
                         "mean_deg": float(pc[0] + near.mean()) if len(near) else None,
                         "peak_deg": vote(ac_all[m])[0]})
        res["segments"][target] = segs
        seg_plot.setdefault(scene, {})[cond] = (segs, pr[1])

        # §4-4：411 の長辺の壁との離れ（E2 のときだけ、b・c・d）
        if cond == "E2" and scene in ("m3_cor_b", "m3_cor_c", "m3_cor_d"):
            from regbim.labels import NAME_TO_ID
            bw = dst.points[dst.labels == NAME_TO_ID["wall"]]
            bn = dst.normals[dst.labels == NAME_TO_ID["wall"]] if dst.normals is not None else None
            if bn is None:
                bnn = preprocess.prepare(dst, cfg)
                bw = bnn.points[bnn.labels == NAME_TO_ID["wall"]]
                bn = bnn.normals[bnn.labels == NAME_TO_ID["wall"]]
            yw = bw[np.abs(bn[:, 1]) > 0.9]
            zlo = float(np.percentile(dst.points[dst.labels == NAME_TO_ID["floor"], 2], 50))
            zhi = float(np.percentile(dst.points[dst.labels == NAME_TO_ID["ceiling"], 2], 50))
            walls = {}
            for name, yv, outward in (("south_far_from_corridor", yw[:, 1].min(), -1.0),
                                      ("north_corridor_side", yw[:, 1].max(), +1.0)):
                sel = yw[np.abs(yw[:, 1] - yv) < 0.2]
                y0 = float(np.median(sel[:, 1]))
                x0, x1 = float(sel[:, 0].min()), float(sel[:, 0].max())
                # SLAM の wall 点：法線が ±Y、BIM の壁面から ±0.5 m
                ny = np.abs((wall.normals @ RG.T)[:, 1])
                m = (ny > 0.8) & (np.abs(P[:, 1] - y0) < 0.5) & (P[:, 0] >= x0) & (P[:, 0] <= x1)
                bins = []
                for lo in np.arange(np.floor(x0), x1, 1.0):
                    mm = m & (P[:, 0] >= lo) & (P[:, 0] < lo + 1.0)
                    if mm.sum() < 20:
                        bins.append({"x_lo": float(lo), "n": int(mm.sum()), "offset_m": None, "height_cov": None})
                        continue
                    off = (P[mm, 1] - y0) * outward
                    zs = P[mm, 2]
                    levels = np.linspace(zlo, zhi, 11)
                    cov = np.mean([np.any((zs >= levels[k]) & (zs < levels[k + 1])) for k in range(10)])
                    bins.append({"x_lo": float(lo), "n": int(mm.sum()), "offset_m": float(np.median(off)),
                                 "height_cov": float(cov), "z_max": float(zs.max())})
                walls[name] = {"y_m": y0, "x_range": [x0, x1], "bins": bins}
                print("  %s %s（Y=%.2f, X %.1f〜%.1f）:" % (scene, name, y0, x0, x1),
                      " ".join("%.0f:%s/%s" % (b["x_lo"], "—" if b["offset_m"] is None else "%+.3f" % b["offset_m"],
                                               "—" if b["height_cov"] is None else "%.1f" % b["height_cov"]) for b in bins))
            res["wall_offsets"][scene] = {"floor_z": zlo, "ceiling_z": zhi, "walls": walls}
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)

    # PNG：区間ごとのヨー（シーンごと）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for scene, dd in seg_plot.items():
        fig, ax = plt.subplots(figsize=(8, 3.6), dpi=110)
        for cond, colr in (("E2", "#2a78d6"), ("E3", "#eb6834")):
            if cond not in dd:
                continue
            segs, room_mean = dd[cond]
            x = [(s["x_lo"] + s["x_hi"]) / 2 for s in segs if s["mean_deg"] is not None]
            y = [s["mean_deg"] for s in segs if s["mean_deg"] is not None]
            ax.plot(x, y, marker="o", color=colr, label="corridor %s (BIM %s)" % (cond, "411" if cond == "E2" else "411+410"))
            ax.axhline(room_mean, color=colr, linestyle=(0, (4, 3)), linewidth=1,
                       label="room region %s: %.2f°" % (cond, room_mean))
        ax.axhline(0, color="#898781", linewidth=0.8)
        ax.set_xlabel("BIM X [m] (5 m segments, corridor region, placed by G1)")
        ax.set_ylabel("wall yaw [deg] (mean within ±5° of peak)")
        ax.set_title("%s: wall yaw along the corridor" % scene)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(color="#e1e0d9", linewidth=0.6)
        p = os.path.join(args.png_dir, "r38_%s_yaw_along_corridor.png" % scene)
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        print("wrote", p, os.path.getsize(p))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

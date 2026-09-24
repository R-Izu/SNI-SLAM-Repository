"""R37 §3 — 本人の目視用 3D データ（PLY・PNG）を書き出す。**再計算なし**（保存済みの変換を使う）。

すべて BIM 座標。色の規約（R37 §3-1）：
    BIM 参照 灰 / source を G1 で置いたもの 緑 / R35（回転固定）赤 / R36 §4（回転解放）青 / 選ばれなかった候補 橙

間引き：各 PLY は 50,000 点以下。点の並び順のまま `np.linspace(0, N-1, n)` の添字で取る（乱数なし）。
2 層を 1 ファイルに入れるとき（V4）は、各層 25,000 点以下。

    conda activate sni-slam
    python Registration/scripts/r37_visual.py --out /mnt/d/.../08_images/2026-09-24_r37_visual
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import open3d as o3d                                           # noqa: E402

from failure_decomposition import provenance                   # noqa: E402
from regbim import io_utils, metrics, preprocess, rotation     # noqa: E402

N_MAX = 50000
COL = {"bim": (0.55, 0.55, 0.55), "g1": (0.10, 0.65, 0.20), "r35": (0.85, 0.10, 0.10),
       "rel": (0.10, 0.30, 0.90), "cand": (1.00, 0.55, 0.00)}
NAME = {"bim": "BIM 参照（灰）", "g1": "source を G1 で置いたもの（緑）",
        "r35": "R35 の解・回転固定（赤）", "rel": "R36 §4 の解・回転解放（青）",
        "cand": "選ばれなかった候補（橙）"}
SLAB_HALF = 0.25       # 断面の半幅 [m]（幅 0.5 m）
END_PCT = (5.0, 95.0)  # 廊下の「両端付近」＝長軸方向の source 点の 5% 点と 95% 点


def decimate(p: np.ndarray, n: int = N_MAX) -> np.ndarray:
    if len(p) <= n:
        return np.arange(len(p))
    return np.linspace(0, len(p) - 1, n).astype(int)


def write_ply(path: str, pts: np.ndarray, rgb: np.ndarray) -> int:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(rgb, dtype=np.float64), 0, 1))
    o3d.io.write_point_cloud(path, pc, write_ascii=False)
    return len(pts)


def load(target: str):
    cfg = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
    cfg = copy.deepcopy(cfg)
    cfg["source"] = dict(cfg["source"], seed=0)
    return cfg, io_utils.load_source_cloud(cfg), io_utils.load_reference_cloud(cfg)


def G1(scene: str) -> np.ndarray:
    sp = json.load(open("Registration/output/diag/criterion_spread.json"))["scenes"][scene]
    return np.asarray(next(f["T"] for f in sp["results"] if f["id"] == 1), dtype=np.float64)


def T_r35(target: str) -> np.ndarray:
    rows = json.load(open("Registration/output/diag/r35_runs.json"))["rows"]
    return np.asarray(next(r["T_est"] for r in rows if "%s__%s" % (r["scene"], r["cond"]) == target
                           and r["mode"] == "plan_correlate"), dtype=np.float64)


def T_rel(target: str) -> np.ndarray:
    rows = json.load(open("Registration/output/diag/r36_release.json"))["rows"]
    return np.asarray(next(r["T_end"] for r in rows if r["target"] == target), dtype=np.float64)


def long_axis(pts: np.ndarray) -> int:
    """BIM の X(0)・Y(1) のうち、点の広がりが大きい方。側面図と断面の向きに使う。"""
    ext = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
    return int(np.argmax(ext))


def png_views(path_base: str, layers: List[Tuple[np.ndarray, Tuple]], ax_long: int,
              title: str) -> List[str]:
    """真上（BIM の X–Y）と横（長軸–Z。短軸の負側から見る）の正射影。"""
    out = []
    for view in ("top", "side"):
        fig, ax = plt.subplots(figsize=(12, 6 if view == "side" else 9), dpi=110)
        for pts, col in layers:
            if view == "top":
                x, y = pts[:, 0], pts[:, 1]
            else:
                x, y = pts[:, ax_long], pts[:, 2]
            ax.scatter(x, y, s=0.15, c=[col], linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlabel("BIM X [m]" if view == "top" else "BIM %s [m]" % "XY"[ax_long])
        ax.set_ylabel("BIM Y [m]" if view == "top" else "BIM Z [m]")
        ax.set_title("%s — %s" % (title, "top (X–Y)" if view == "top" else "side (%s–Z)" % "XY"[ax_long]))
        p = "%s_%s.png" % (path_base, view)
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        out.append(p)
    return out


def png_section(path: str, layers, ax_long: int, title: str) -> str:
    """断面：長軸方向に見る（横軸は短軸、縦軸は Z）。"""
    ax_short = 1 - ax_long
    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    for pts, col in layers:
        ax.scatter(pts[:, ax_short], pts[:, 2], s=1.0, c=[col], linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("BIM %s [m]" % "XY"[ax_short]); ax.set_ylabel("BIM Z [m]")
    ax.set_title(title)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rec: List[Dict] = []          # README 用の一覧

    def ply(name, pts, key, scene, source):
        idx = decimate(pts)
        n = write_ply(os.path.join(args.out, name), pts[idx],
                      np.tile(COL[key], (len(idx), 1)))
        rec.append({"file": name, "scene": scene, "color": NAME[key], "points": n,
                    "points_before": int(len(pts)), "source": source})
        return pts[idx]

    # ---------------- V1 / V3：G1・R35・§4 と BIM ----------------
    for tag, target in (("V1", "m3_cor_d__E2"), ("V3", "m3_cor_b__E2")):
        cfg, src, dst = load(target)
        scene = target.split("__")[0]
        P = {"g1": metrics.apply_sim3(G1(scene), src.points),
             "r35": metrics.apply_sim3(T_r35(target), src.points),
             "rel": metrics.apply_sim3(T_rel(target), src.points)}
        lay = [(ply("%s_%s_bim.ply" % (tag, target), dst.points, "bim", target,
                    "参照 config `%s.yaml`" % target), COL["bim"])]
        srcs = {"g1": "`criterion_spread.json` %s の基準 id 1" % scene,
                "r35": "`r35_runs.json` の %s・plan_correlate の `T_est`" % target,
                "rel": "`r36_release.json` の %s の `T_end`" % target}
        for k in ("g1", "r35", "rel"):
            lay.append((ply("%s_%s_%s.ply" % (tag, target, k), P[k], k, target, srcs[k]),
                        COL[k]))
        axl = long_axis(dst.points)
        for p in png_views(os.path.join(args.out, "%s_%s" % (tag, target)), lay, axl,
                           "%s %s: BIM grey / G1 green / R35 red / R36§4 blue" % (tag, target)):
            rec.append({"file": os.path.basename(p), "scene": target, "color": "上の 4 PLY を重ねた図",
                        "points": None, "points_before": None, "source": "同上"})

    # ---------------- V2：m3_cor_d の wall 点を壁法線のヨーのずれで塗る ----------------
    target = "m3_cor_d__E2"
    cfg, src, dst = load(target)
    src_p = preprocess.prepare(src, cfg)
    up = rotation.estimate_gravity_axis(src_p, cfg)
    e1, e2 = rotation._horizontal_basis(up)
    wall = src_p.subset(src_p.class_mask("wall"))
    nh = wall.normals - (wall.normals @ up)[:, None] * up
    mag = np.linalg.norm(nh, axis=1)
    keep = mag > 0.3                           # rotation.py と同じ（ほぼ水平な法線を落とす）
    nh = nh[keep] / mag[keep, None]
    ang = np.degrees(np.mod(np.arctan2(nh @ e2, nh @ e1), np.pi / 2.0))
    bins = int(cfg["rotation"]["yaw_bins"])
    hist, edges = np.histogram(ang, bins=bins, range=(0.0, 90.0))
    phi = float(edges[int(np.argmax(hist))] + 90.0 / bins / 2.0)
    dev = (ang - phi + 45.0) % 90.0 - 45.0     # 支配方向からのずれ。90° で畳む
    VMAX = 5.0
    cmap = plt.get_cmap("coolwarm")
    rgb = cmap((np.clip(dev, -VMAX, VMAX) + VMAX) / (2 * VMAX))[:, :3]
    pts = metrics.apply_sim3(G1("m3_cor_d"), wall.points[keep])
    idx = decimate(pts)
    name = "V2_m3_cor_d_wall_yawdev.ply"
    write_ply(os.path.join(args.out, name), pts[idx], rgb[idx])
    rec.append({"file": name, "scene": "m3_cor_d", "color": "壁法線のヨーのずれ（凡例 PNG）",
                "points": int(len(idx)), "points_before": int(len(pts)),
                "source": "source（seed 0、前処理後）の wall 点・G1 で配置。支配方向 %.3f°（source 座標、bin 幅 %.2f°）" % (phi, 90.0 / bins)})
    axl = long_axis(pts)
    for view in ("top", "side"):
        fig, ax = plt.subplots(figsize=(12, 6 if view == "side" else 9), dpi=110)
        bp = dst.points[decimate(dst.points)]
        x0, y0 = (bp[:, 0], bp[:, 1]) if view == "top" else (bp[:, axl], bp[:, 2])
        ax.scatter(x0, y0, s=0.1, c=[COL["bim"]], linewidths=0, rasterized=True, alpha=0.4)
        q = pts[idx]
        x, y = (q[:, 0], q[:, 1]) if view == "top" else (q[:, axl], q[:, 2])
        sc = ax.scatter(x, y, s=0.3, c=np.clip(dev[idx], -VMAX, VMAX), cmap=cmap,
                        vmin=-VMAX, vmax=VMAX, linewidths=0, rasterized=True)
        fig.colorbar(sc, ax=ax, label="wall normal yaw − dominant [deg] (folded at 90°, clipped ±5°)")
        ax.set_aspect("equal")
        ax.set_title("V2 m3_cor_d wall yaw deviation (placed by G1), BIM grey — %s" % view)
        p = os.path.join(args.out, "V2_m3_cor_d_wall_yawdev_%s.png" % view)
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        rec.append({"file": os.path.basename(p), "scene": "m3_cor_d", "color": "V2 の図（色の凡例を含む）",
                    "points": None, "points_before": None, "source": "同上"})
    fig, ax = plt.subplots(figsize=(6, 1.2), dpi=110)
    fig.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-VMAX, VMAX), cmap=cmap),
                 cax=ax, orientation="horizontal",
                 label="wall normal yaw − dominant direction [deg] (blue −5 … white 0 … red +5; beyond ±5 clipped)")
    p = os.path.join(args.out, "V2_legend.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    rec.append({"file": "V2_legend.png", "scene": "m3_cor_d", "color": "V2 の色の凡例",
                "points": None, "points_before": None, "source": "—"})
    v2_stats = {"n_wall_used": int(len(dev)), "dominant_deg_src_frame": phi,
                "frac_abs_dev_le_1": float((np.abs(dev) <= 1).mean()),
                "frac_abs_dev_gt_5": float((np.abs(dev) > 5).mean())}

    # ---------------- V4：G1 で置いた source と BIM、全体と両端の断面 ----------------
    v4_info = {}
    for scene in ("m3_cor_b", "m3_cor_c", "m3_cor_d"):
        target = "%s__E2" % scene
        cfg, src, dst = load(target)
        S = metrics.apply_sim3(G1(scene), src.points)
        axl = long_axis(S)
        half = N_MAX // 2
        si, bi = decimate(S, half), decimate(dst.points, half)
        name = "V4_%s_G1_full.ply" % scene
        write_ply(os.path.join(args.out, name), np.vstack([S[si], dst.points[bi]]),
                  np.vstack([np.tile(COL["g1"], (len(si), 1)), np.tile(COL["bim"], (len(bi), 1))]))
        rec.append({"file": name, "scene": target, "color": "緑（G1 の source）＋灰（BIM）",
                    "points": int(len(si) + len(bi)), "points_before": int(len(S) + len(dst.points)),
                    "source": "G1＝`criterion_spread.json` %s 基準 id 1" % scene})
        for p in png_views(os.path.join(args.out, "V4_%s_G1_full" % scene),
                           [(dst.points[bi], COL["bim"]), (S[si], COL["g1"])], axl,
                           "V4 %s: G1 green / BIM grey" % scene):
            rec.append({"file": os.path.basename(p), "scene": target, "color": "全体図",
                        "points": None, "points_before": None, "source": "同上"})
        pos = np.percentile(S[:, axl], END_PCT)
        v4_info[scene] = {"long_axis": "XY"[axl], "slab_centers_m": pos.tolist()}
        for end, c in zip(("endA", "endB"), pos):
            sm = np.abs(S[:, axl] - c) < SLAB_HALF
            bm = np.abs(dst.points[:, axl] - c) < SLAB_HALF
            s2, b2 = S[sm], dst.points[bm]
            si2, bi2 = decimate(s2, half), decimate(b2, half)
            name = "V4_%s_G1_section_%s.ply" % (scene, end)
            write_ply(os.path.join(args.out, name), np.vstack([s2[si2], b2[bi2]]),
                      np.vstack([np.tile(COL["g1"], (len(si2), 1)),
                                 np.tile(COL["bim"], (len(bi2), 1))]))
            rec.append({"file": name, "scene": target,
                        "color": "緑（G1 の source）＋灰（BIM）。BIM %s = %.2f m ± 0.25 m の薄切り"
                                 % ("XY"[axl], c),
                        "points": int(len(si2) + len(bi2)), "points_before": int(len(s2) + len(b2)),
                        "source": "同上"})
            p = png_section(os.path.join(args.out, "V4_%s_G1_section_%s.png" % (scene, end)),
                            [(b2[bi2], COL["bim"]), (s2[si2], COL["g1"])], axl,
                            "V4 %s section %s (BIM %s=%.2f±0.25 m), view along %s"
                            % (scene, end, "XY"[axl], c, "XY"[axl]))
            rec.append({"file": os.path.basename(p), "scene": target, "color": "断面図",
                        "points": None, "points_before": None, "source": "同上"})

    # ---------------- V5：m3_cor_c E3 の選ばれた解と、選ばれなかった 0.240 m の候補 ----------------
    target = "m3_cor_c__E3"
    cfg, src, dst = load(target)
    pa = json.load(open("Registration/output/diag/plan_a_compare.json"))
    row = next(x for x in pa["rows"] if x["target"] == target and x["mode"] == "plan_correlate")
    cand8 = next(c for c in row["stage_candidates"] if c["cand_id"] == 8)
    P = {"r35": metrics.apply_sim3(T_r35(target), src.points),
         "cand": metrics.apply_sim3(np.asarray(cand8["T_short"]), src.points)}
    lay = [(ply("V5_%s_bim.ply" % target, dst.points, "bim", target,
                "参照 config `%s.yaml`（411＋410）" % target), COL["bim"])]
    lay.append((ply("V5_%s_selected_r35.ply" % target, P["r35"], "r35", target,
                    "`r35_runs.json` の %s・plan_correlate の `T_est`（選ばれた解）" % target), COL["r35"]))
    lay.append((ply("V5_%s_unselected_cand8.ply" % target, P["cand"], "cand", target,
                    "`plan_a_compare.json`（R32）の %s・plan_correlate、`stage_candidates` の cand_id 8 の `T_short`（短い精緻化の後）" % target),
                COL["cand"]))
    for p in png_views(os.path.join(args.out, "V5_%s" % target), lay, long_axis(dst.points),
                       "V5 %s: BIM grey / selected (R35) red / unselected cand 8 orange" % target):
        rec.append({"file": os.path.basename(p), "scene": target, "color": "上の 3 PLY を重ねた図",
                    "points": None, "points_before": None, "source": "同上"})

    json.dump({"provenance": provenance(), "files": rec, "v2": v2_stats, "v4": v4_info,
               "cand8": {k: cand8[k] for k in ("cand_id", "yaw_index", "score_short")}},
              open(os.path.join(args.out, "manifest.json"), "w"), indent=2, ensure_ascii=False)
    print("wrote %d entries to %s" % (len(rec), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

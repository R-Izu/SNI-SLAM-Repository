"""R36 §3 — 案A の回転誤差がどこから来ているかを分ける（診断のみ。手法は変えない）。

出すもの
--------
§3-1  回転誤差を「傾き」と「ヨー」に分ける（基準1 と 6 基準）
      傾き  θ_tilt = ∠(R̂ᵀ e_z, R_Gᵀ e_z)
      ヨー  ψ     = R_rel = R̂ R_Gᵀ を e_z まわりの twist と swing に分けたときの twist（符号つき）
      （swing の角度は θ_tilt と恒等的に等しい）
§3-2  段階1（`rotation.py`）の中間量を、`register` と同じ前処理済みの点群で計算し直す
§3-3  オラクル回転：R̂ を R_G に置き換え、並進・縮尺だけを回転固定の ICP で解き直す
      （段階2 と同じ `semantic_icp(..., rotation_fixed=True)`）。
      **回転誤差は構成上 0。結果として書かない。** 見るのは d_Ω と2項分解だけ。

手法の再実行は、勝者の種（段階1 の出力）を得るためだけに行う。
最終 T̂ が R35 の `r35_runs.json` とビット一致することを確かめる。

    conda activate sni-slam
    python Registration/scripts/r36_rotation_diag.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from criterion_verdict import d_omega                        # noqa: E402
from failure_decomposition import provenance                 # noqa: E402
from r35_rescore import decompose                            # noqa: E402
from regbim import io_utils, metrics, preprocess, rotation   # noqa: E402
from regbim.labels import NAME_TO_ID                         # noqa: E402
from regbim.methods import get_method                        # noqa: E402
from regbim.metrics import make_sim3                         # noqa: E402
from regbim.semantic_icp import semantic_icp                 # noqa: E402

PRIMARY = 1
EZ = np.array([0.0, 0.0, 1.0])
KIND = {"m3_cor_c__E3": "a", "m3_cor_a__E2": "b", "m3_cor_a__E3": "b"}   # 他は (c)


# --------------------------------------------------------------------------- #
# §3-1 傾きとヨー
# --------------------------------------------------------------------------- #
def tilt_yaw(R_est: np.ndarray, R_gt: np.ndarray) -> Dict:
    Rrel = R_est @ R_gt.T
    a, b = R_est.T @ EZ, R_gt.T @ EZ
    tilt = float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))
    # swing-twist（twist 軸 = e_z）。四元数 (w, x, y, z) の (w, z) が twist
    tr = np.trace(Rrel)
    w = np.sqrt(max(0.0, 1.0 + tr)) / 2.0
    if w > 1e-8:
        z = (Rrel[1, 0] - Rrel[0, 1]) / (4.0 * w)
    else:                                   # 180° 近傍（案A では起きないが守る）
        z = np.sqrt(max(0.0, (1.0 + Rrel[2, 2] - Rrel[0, 0] - Rrel[1, 1]))) / 2.0
    psi = float(np.degrees(2.0 * np.arctan2(z, w)))
    psi = (psi + 180.0) % 360.0 - 180.0
    total = metrics.rotation_error_deg(R_est, R_gt)
    return {"tilt_deg": tilt, "yaw_deg": psi, "rot_deg": total,
            "recombined_deg": float(np.hypot(tilt, psi)),
            "recombined_minus_rot_deg": float(np.hypot(tilt, psi) - total)}


# --------------------------------------------------------------------------- #
# §3-2 段階1 の中間量（rotation.py と同じ計算を、途中の量を残して行う）
# --------------------------------------------------------------------------- #
def gravity_diag(cloud, cfg) -> Dict:
    rcfg = cfg["rotation"]
    floor = cloud.subset(cloud.class_mask("floor"))
    ceil = cloud.subset(cloud.class_mask("ceiling"))
    normals = np.concatenate([c.normals for c in (floor, ceil) if len(c)], axis=0)
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    # `_robust_axis` と同じ乱数・同じ手順で内点を再現する
    rng = np.random.default_rng(0)
    cos_t = np.cos(np.radians(rcfg["ransac_normal_thresh_deg"]))
    iters = int(min(rcfg["ransac_iters"], max(len(n), 1)))
    best, bc = None, -1
    for i in rng.integers(0, len(n), size=iters):
        inl = np.abs(n @ n[i]) > cos_t
        if int(inl.sum()) > bc:
            bc, best = int(inl.sum()), inl
    M = n[best].T @ n[best]
    wv, _ = np.linalg.eigh(M)
    up = rotation.estimate_gravity_axis(cloud, cfg)          # 実際に使う軸
    ang = np.degrees(np.arccos(np.clip(np.abs(n[best] @ up), -1.0, 1.0)))
    return {"n_floor": int(len(floor)), "n_ceiling": int(len(ceil)),
            "n_normals": int(len(n)), "n_inliers": bc,
            "inlier_frac": bc / len(n),
            "inlier_angle_to_axis_deg": {"median": float(np.median(ang)),
                                         "rms": float(np.sqrt((ang ** 2).mean())),
                                         "p90": float(np.percentile(ang, 90))},
            "eig_ratio_l2_l1": float(wv[-2] / wv[-1]),
            # 内点法線の平均方向の不確かさの目安（RMS / √N）
            "axis_se_deg": float(np.sqrt((ang ** 2).mean()) / np.sqrt(bc)),
            "up": up.tolist(),
            "up_tilt_from_ez_deg": float(np.degrees(np.arccos(abs(up @ EZ))))}


def yaw_diag(cloud, up, cfg) -> Dict:
    rcfg = cfg["rotation"]
    wall = cloud.subset(cloud.class_mask("wall"))
    e1, e2 = rotation._horizontal_basis(up)
    nh = wall.normals - (wall.normals @ up)[:, None] * up
    mag = np.linalg.norm(nh, axis=1)
    nh = nh[mag > 0.3]
    nh = nh / (np.linalg.norm(nh, axis=1, keepdims=True) + 1e-12)
    ang = np.mod(np.arctan2(nh @ e2, nh @ e1), np.pi / 2.0)
    bins = int(rcfg["yaw_bins"])
    hist, edges = np.histogram(ang, bins=bins, range=(0.0, np.pi / 2.0))
    width = 90.0 / bins
    k = int(np.argmax(hist))
    phi = float(np.degrees(edges[k]) + width / 2.0)
    # 2位：ピーク以外の最大 bin と、ピークから 2° 以上離れた最大 bin（循環距離）
    idx = np.arange(bins)
    cd = np.minimum(np.abs(idx - k), bins - np.abs(idx - k)) * width
    h2 = hist.copy(); h2[k] = -1
    k2 = int(np.argmax(h2))
    far = np.where(cd >= 2.0, hist, -1)
    k3 = int(np.argmax(far))
    # ピーク周りの広がり（循環差、±5° 以内の標準偏差）
    dd = (np.degrees(ang) - phi + 45.0) % 90.0 - 45.0
    near = dd[np.abs(dd) <= 5.0]
    return {"n_wall": int(len(wall)), "n_used": int(len(nh)),
            "bin_width_deg": width, "n_bins": bins,
            "peak_deg": phi, "peak_count": int(hist[k]),
            "second_bin_deg": float(np.degrees(edges[k2]) + width / 2.0),
            "second_bin_count": int(hist[k2]),
            "second_peak_ge2deg_deg": float(np.degrees(edges[k3]) + width / 2.0),
            "second_peak_ge2deg_count": int(hist[k3]),
            "frac_within_1deg": float((np.abs(dd) <= 1.0).mean()),
            "frac_within_5deg": float((np.abs(dd) <= 5.0).mean()),
            "std_within_5deg_deg": float(near.std()) if len(near) else None,
            "mean_within_5deg_minus_peak_deg": float(near.mean()) if len(near) else None,
            "e1": e1.tolist()}


# --------------------------------------------------------------------------- #
def swap_rotation(T: np.ndarray, R_new: np.ndarray, c: np.ndarray) -> np.ndarray:
    """回転だけを R_new に替える。点 c の写り先は保つ（縮尺はそのまま）。"""
    R, t, s = metrics.decompose_sim3(T)
    t2 = t + s * (R - R_new) @ c
    return make_sim3(R_new, t2, s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="Registration/output/diag/r35_runs.json")
    ap.add_argument("--spread", default="Registration/output/diag/criterion_spread.json")
    ap.add_argument("--omega-cache", default="Registration/output/diag/omega.npz")
    ap.add_argument("--out", default="Registration/output/diag/r36_rotation_diag.json")
    args = ap.parse_args()

    runs = [r for r in json.load(open(args.runs))["rows"] if r["mode"] == "plan_correlate"]
    spread = json.load(open(args.spread))["scenes"]
    z = np.load(args.omega_cache)
    struct_ids = None
    out: List[Dict] = []

    for r in runs:
        scene, cond = r["scene"], r["cond"]
        target = "%s__%s" % (scene, cond)
        cfg0 = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
        cfg = copy.deepcopy(cfg0)
        cfg["source"] = dict(cfg["source"], seed=0)
        cfg.setdefault("proposed", {})["translation_init"] = "plan_correlate"
        cfg.setdefault("diagnostics", {})
        cfg["diagnostics"]["record_yaw"] = True
        cfg["diagnostics"]["record_stages"] = True
        struct_ids = [NAME_TO_ID[n] for n in cfg["classes"]["structural"]]

        src = io_utils.load_source_cloud(cfg)
        dst = io_utils.load_reference_cloud(cfg)
        m = get_method("proposed")
        T_hat = np.asarray(m.register(src, dst, cfg), dtype=np.float64)
        T_r35 = np.asarray(r["T_est"], dtype=np.float64)
        bit_equal = bool(np.array_equal(T_hat, T_r35))
        st = m.last_stage_diag
        win = int(np.argmax(st["scores"]))
        seed_T = np.asarray(st["seeds"][win], dtype=np.float64)

        src_p = preprocess.prepare(src, cfg)
        dst_p = preprocess.prepare(dst, cfg)
        R_hat, _, s_hat = metrics.decompose_sim3(T_hat)
        cands = rotation.relative_rotation_candidates(src_p, dst_p, cfg)
        cand_diff = [float(np.abs(R_hat - C).max()) for C in cands]

        # §3-1
        per = []
        for f in spread[scene]["results"]:
            if "T" not in f:
                continue
            RG = metrics.decompose_sim3(np.asarray(f["T"], dtype=np.float64))[0]
            per.append(dict(id=f["id"], **tilt_yaw(R_hat, RG)))

        # §3-2
        g_src = gravity_diag(src_p, cfg)
        g_dst = gravity_diag(dst_p, cfg)
        y_src = yaw_diag(src_p, np.asarray(g_src["up"]), cfg)
        y_dst = yaw_diag(dst_p, np.asarray(g_dst["up"]), cfg)
        # 参照の壁の向きが座標軸からどれだけ回っているか（0°/90° からのずれ）
        y_dst["axis_offset_deg"] = float(min(y_dst["peak_deg"], 90.0 - y_dst["peak_deg"]))

        # §3-3 オラクル回転
        G = np.asarray(next(f["T"] for f in spread[scene]["results"]
                            if f["id"] == PRIMARY), dtype=np.float64)
        RG, _, sG = metrics.decompose_sim3(G)
        omega = z[scene]
        c_src = src_p.points[np.isin(src_p.labels, struct_ids)].mean(axis=0)
        oracle = {}
        for name, T0 in (("from_winner_seed", seed_T), ("from_final", T_hat)):
            init = swap_rotation(T0, RG, c_src)
            T_or = semantic_icp(src_p, dst_p, init, cfg, rotation_fixed=True)
            _, _, s_or = metrics.decompose_sim3(T_or)
            oracle[name] = {
                "init_decomposition": decompose(init, G, omega),
                "decomposition": decompose(T_or, G, omega),
                "scale_ratio": float(abs(s_or / sG - 1.0)),
                "T": np.asarray(T_or).tolist()}

        row = {"target": target, "kind": KIND.get(target, "c"),
               "bit_equal_r35": bit_equal,
               "winner_seed_index": win,
               "R_hat_vs_stage1_candidates_maxabs": cand_diff,
               "per_criterion": per,
               "gravity_src": g_src, "gravity_dst": g_dst,
               "yaw_src": y_src, "yaw_dst": y_dst,
               "r35_decomposition": decompose(T_hat, G, omega),
               "r35_scale_ratio": float(abs(s_hat / sG - 1.0)),
               "oracle_rotation": oracle}
        out.append(row)
        p1 = next(p for p in per if p["id"] == PRIMARY)
        print("%-14s (%s) bit=%s 傾き %.3f° ヨー %+.3f° 回転 %.3f° | "
              "yaw src %.3f° dst %.3f° | dΩ R35 %.4f → オラクル %.4f / %.4f"
              % (target, row["kind"], bit_equal, p1["tilt_deg"], p1["yaw_deg"],
                 p1["rot_deg"], y_src["peak_deg"], y_dst["peak_deg"],
                 row["r35_decomposition"]["d_omega_m"],
                 oracle["from_winner_seed"]["decomposition"]["d_omega_m"],
                 oracle["from_final"]["decomposition"]["d_omega_m"]), flush=True)
        with open(args.out, "w") as f:
            json.dump({"provenance": provenance(), "primary_criterion": PRIMARY,
                       "rows": out}, f, indent=2, ensure_ascii=False)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

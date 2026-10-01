"""R42 §3 — 点群の面からの離れ（雑音の大きさ）を測る。手法は再実行しない。

測り方（R42 §3-2・§6-1。結果を見る前にここで固定する）
------------------------------------------------------
1. 点群は `register` と同じ前処理（`preprocess.prepare`：法線推定 → voxel 0.05 m）をしたもの
2. クラスごと（wall が判定用。floor・ceiling は記述）に、平らな面を順に取り出す
   a. 残りの点に RANSAC で平面を当てる（Open3D `segment_plane`、距離 RANSAC_DIST、3 点、RANSAC_ITERS 回、乱数 seed 0）
   b. その平面から BAND 以内の点を候補にし、DBSCAN（eps DBSCAN_EPS、最小 DBSCAN_MIN 点）で最大の連結成分を 1 枚の面とする
   c. 面の点で平面を当て直す（重心と、共分散の最小固有ベクトル＝単位法線 n）
   d. 符号つき残差 r_i = n·(p_i − p0)、σ = 1.4826 · median|r_i − median r_j|（§6-1）
   e. 面積 = 面内の 2 次元座標を CELL 角の格子に落とし、点のある格子の数 × CELL²
   f. 面積 MIN_AREA 以上なら面として記録する。面の点は残りから外す（面積が足りなくても外す）
   g. RANSAC の当てはまりが MIN_INLIERS 点未満になるか、MAX_PLANES 回で止める
3. シーンの値 = その点群の wall の面ごとの σ の中央値

感度の確認（§3-3）：R40 §4 の入力 N 1・2・5 cm で、測った σ が足した雑音の ±30% 以内。誤差なしで 0.5 cm 未満
判定（§6-2）：R40 §4 の回転誤差の中央（0・1・2・5 cm → 0.146・0.27・0.66・1.92°）を σ について区分線形に補間した
予測 ê_R と、実測 0.62°（R41 D3 S off、G1 の中央）を比べる。[0.5 ê_R, 2 ê_R] に入るかを、
4 シーンの σ の中央値と、各シーンの σ のそれぞれで見る。5 cm を超える σ は補間の範囲外として記録する（外挿しない）

実行：python Registration/scripts/r42_noise.py check   （対照 4 入力だけ）
      python Registration/scripts/r42_noise.py real    （実データ 10 シーン。check が通った後だけ）
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import open3d as o3d
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from regbim import io_utils, preprocess                              # noqa: E402
from regbim.labels import NAME_TO_ID                                 # noqa: E402
from failure_decomposition import provenance                         # noqa: E402

RANSAC_DIST = 0.05
RANSAC_ITERS = 2000
BAND = 0.20
DBSCAN_EPS = 0.15
DBSCAN_MIN = 5
CELL = 0.10
MIN_AREA = 2.0
MIN_INLIERS = 200
MAX_PLANES = 40
CLASSES = ["wall", "floor", "ceiling"]

CONTROLS = {"zero": ("r40_bim/err_zero", 0.0), "N_1cm": ("r40_bim/err_N_1cm", 0.01),
            "N_2cm": ("r40_bim/err_N_2cm", 0.02), "N_5cm": ("r40_bim/err_N_5cm", 0.05)}
COR = ["m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d"]
OTHER = ["m3_block_a", "m3_block_b", "m3_block_c", "m3_block_d", "m3_room_a", "m3_room_b"]
NOISE_CM = [0.0, 1.0, 2.0, 5.0]
ROT_MED = [0.146, 0.27, 0.66, 1.92]          # R40 §4 の proposed（案A）の回転誤差の中央 [°]
OBSERVED = 0.62                              # R41 §4-1 D3 S off、G1 の中央 [°]
OUT = "Registration/output/diag/r42"


def planes(points: np.ndarray) -> list:
    o3d.utility.random.seed(0)
    rest = np.arange(len(points)); out = []
    for _ in range(MAX_PLANES):
        if len(rest) < MIN_INLIERS:
            break
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[rest]))
        model, inl = pc.segment_plane(RANSAC_DIST, 3, RANSAC_ITERS)
        if len(inl) < MIN_INLIERS:
            break
        a = np.asarray(model[:3]); a = a / np.linalg.norm(a)
        dist = np.abs(points[rest] @ a + model[3] / np.linalg.norm(model[:3]))
        cand = np.where(dist < BAND)[0]
        lab = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[rest[cand]])).cluster_dbscan(DBSCAN_EPS, DBSCAN_MIN))
        if (lab >= 0).sum() == 0:
            rest = np.delete(rest, inl); continue
        big = np.bincount(lab[lab >= 0]).argmax()
        sel = rest[cand[lab == big]]
        P = points[sel]; p0 = P.mean(0)
        w, V = np.linalg.eigh(np.cov((P - p0).T)); n = V[:, 0]
        r = (P - p0) @ n
        sigma = float(1.4826 * np.median(np.abs(r - np.median(r))))
        uv = (P - p0) @ V[:, 1:]
        area = float(len({tuple(c) for c in np.floor(uv / CELL).astype(np.int64)}) * CELL * CELL)
        if area >= MIN_AREA:
            out.append({"n_points": int(len(sel)), "area_m2": area, "sigma_m": sigma, "normal": n.tolist()})
        rest = np.setdiff1d(rest, sel, assume_unique=True)
    return out


def measure(cfg_path: str, seed=None) -> dict:
    cfg = yaml.safe_load(open("Registration/configs/%s.yaml" % cfg_path))
    spec = dict(cfg["source"])
    if spec.get("type") == "slam_mesh":
        spec["seed"] = 0 if seed is None else seed
    cloud = preprocess.prepare(io_utils.load_source_cloud(dict(cfg, source=spec)), cfg)
    res = {"config": cfg_path, "n_points_prepared": int(len(cloud.points))}
    for c in CLASSES:
        pl = planes(cloud.points[cloud.labels == NAME_TO_ID[c]])
        s = [p["sigma_m"] for p in pl]
        res[c] = {"planes": pl, "n_planes": len(pl), "area_total_m2": float(sum(p["area_m2"] for p in pl)),
                  "sigma_median_m": (float(np.median(s)) if s else None),
                  "sigma_q": ({q: float(np.percentile(s, q)) for q in (5, 25, 50, 75, 95)} if s else None)}
    return res


def predict(sigma_m: float):
    cm = sigma_m * 100.0
    if cm > NOISE_CM[-1]:
        return None
    return float(np.interp(cm, NOISE_CM, ROT_MED))


def verdict_of(pred):
    if pred is None:
        return "範囲外"
    return "入る" if 0.5 * pred <= OBSERVED <= 2.0 * pred else "外"


def main():
    os.makedirs(OUT, exist_ok=True)
    stage = sys.argv[1]
    if stage == "check":
        res = {"provenance": provenance(), "controls": {}}
        for k, (cfgp, added) in CONTROLS.items():
            m = measure(cfgp); s = m["wall"]["sigma_median_m"]
            ok = (s < 0.005) if added == 0 else (abs(s - added) <= 0.3 * added)
            m.update({"added_m": added, "pass": bool(ok)}); res["controls"][k] = m
            print(k, "added", added, "wall σ median %.4f" % s, "planes", m["wall"]["n_planes"], "pass", ok)
        res["all_pass"] = all(v["pass"] for v in res["controls"].values())
        json.dump(res, open(os.path.join(OUT, "r42_noise_check.json"), "w"), ensure_ascii=False, indent=1)
        print("all_pass", res["all_pass"])
    elif stage == "real":
        chk = json.load(open(os.path.join(OUT, "r42_noise_check.json")))
        assert chk["all_pass"], "感度の確認が通っていない（R42 §3-3）。実データには進まない"
        res = {"provenance": provenance(), "scenes": {}}
        for s in COR + OTHER:
            res["scenes"][s] = measure("realdata/%s__E3" % s, seed=0)
            print(s, "wall σ median", res["scenes"][s]["wall"]["sigma_median_m"], "planes", res["scenes"][s]["wall"]["n_planes"])
        cor = [res["scenes"][s]["wall"]["sigma_median_m"] for s in COR]
        pooled = float(np.median(cor)); pp = predict(pooled)
        per = {s: {"sigma_m": v, "pred_deg": predict(v), "in": verdict_of(predict(v))} for s, v in zip(COR, cor)}
        v_pooled = verdict_of(pp); same = all(x["in"] == v_pooled for x in per.values())
        if same and v_pooled == "入る":
            v = "整合"
        elif same and v_pooled == "外":
            v = "整合を確認できない"
        else:
            v = "判断保留"
        res["judgment"] = {"sigma_real_m": pooled, "pred_deg": pp, "observed_deg": OBSERVED, "pooled_in": v_pooled,
                           "per_scene": per, "verdict": v}
        json.dump(res, open(os.path.join(OUT, "r42_noise_real.json"), "w"), ensure_ascii=False, indent=1)
        print(json.dumps(res["judgment"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

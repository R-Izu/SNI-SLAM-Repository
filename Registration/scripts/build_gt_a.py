"""R27 §2-2 — **GT-A** を作る：GT カメラ姿勢で深度を再投影し、正解が解析的に決まる source を作る。

なぜ要るか
----------
R26 §1 で分かったこと：**8 シーン中 7 シーンには凍結 T_gt が存在しなかった。**
83.8% は「各手法が自分の無摂動解に戻れたか」という自己一貫性である。
**GT 基準の数値が、合成データについて1つも無い。**

GT-A の考え方
-------------
1. Replica の **GT カメラ姿勢**（`traj.txt`。R27 §1 で 7/8 が配布物と sha256 一致）で
   深度画像を再投影し、**Replica 座標系で**点群 C を作る
2. **我々が自分で決めた既知の Sim(3) M** を掛けて source S = M·C とする
3. **正解は解析的に決まる**： `T_gt = M⁻¹`（source → 参照）

> **提案手法は一切出てこない。** 正解は我々が掛けた変換の逆であって、
> **どの手法の出力でもない。**

**M を恒等にしない。** 恒等だと「何も動かさない実装」が満点を取ってしまい、
**落ちようのない評価**になる（`_IMPL_OPERATING_RULES.md` F12）。

必ず出す量（R27 §2-3）
----------------------
**オラクル残差**＝ **source を正解 `T_gt` に置いたときの、参照との残差**。
正解が厳密でも、再投影の点群と参照メッシュが形として違えば残差は 0 にならない。
**その残差が、この課題の下限である。**
実データでは 0.139〜0.162 m で、「完璧な初期値からでも閾値を超える」ことを捕まえた。

自分で落ちる検査（F12 の教訓）
------------------------------
**再投影の座標変換を間違えていれば、C は参照と重ならない。**
そこで **M を掛ける前の C を、変換なしで参照と比べる**。
**ここで残差が大きければ、実装が間違っている。** 通ったときだけ先へ進む。

    conda activate sni-slam
    python Registration/scripts/build_gt_a.py --scene room_0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

import open3d as o3d                                        # noqa: E402
from failure_decomposition import provenance                # noqa: E402
from regbim import io_utils, metrics                        # noqa: E402
from regbim.labels import (LABEL_COLORS, NAME_TO_ID,          # noqa: E402
                           LabeledCloud, build_replica_to_six)

# 再投影が正しければ、変換なしで参照とこれ以下に収まるはず。**超えたら止める。**
#
# ★ **比べる相手を揃える。** 参照は `keep_classes`（壁・床・天井・扉・窓）だけで作られており、
#   什器を持たない。再投影の点群は什器を background として持つ（room_0 で 40%）。
#   chamfer は対称なので、**相手のいない什器の点がそのまま距離に乗る。**
#   最初の実装はこれで 0.1079 m を出し、自己検査が鳴った。
#   **閾値は動かさない。比べる対象を、参照と同じクラスに揃える。**
#   （これは「参照に無いクラスの点は、参照との距離を定義しない」という定義から出る修正であり、
#     通したい値に合わせた調整ではない。両方の数値を下に併記する）
SELFCHECK_CHAMFER_MAX = 0.10


def sha256(path: str, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def known_sim3(seed: int) -> np.ndarray:
    """**我々が決める既知の Sim(3)**。seed から決定的に作り、行列を来歴に残す。

    恒等にしない理由は docstring 冒頭のとおり。
    """
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.deg2rad(rng.uniform(20.0, 160.0))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    s = float(np.exp(rng.uniform(-0.25, 0.25)))
    t = rng.uniform(-2.0, 2.0, size=3)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def reproject(scene: str, cam: dict, frame_stride: int, pixel_stride: int,
              depth_max: float, classes_cfg: dict):
    """GT 姿勢で深度を再投影し、ラベル付き点群を **Replica 座標系で** 作る。"""
    seq = "data/replica/%s_official" % scene
    traj = np.loadtxt(os.path.join(seq, "traj.txt")).reshape(-1, 4, 4)
    H, W = int(cam["H"]), int(cam["W"])
    fx, fy, cx, cy = (float(cam[k]) for k in ("fx", "fy", "cx", "cy"))
    dscale = float(cam["png_depth_scale"])

    rep2six = build_replica_to_six(classes_cfg["replica_to_six"])
    lut = np.zeros(256, dtype=np.int64)          # Replica クラス id -> 6 クラス（既定 background=0）
    for rid, six in rep2six.items():
        if 0 <= int(rid) < 256:
            lut[int(rid)] = six

    v, u = np.mgrid[0:H:pixel_stride, 0:W:pixel_stride]
    u = u.ravel().astype(np.float64)
    v = v.ravel().astype(np.float64)
    # src/common.py:91 と同じ向き。姿勢は datasets.py:198 と同じ反転を掛けてから使う
    dx = (u - cx) / fx
    dy = -(v - cy) / fy
    dz = -np.ones_like(u)

    pts, labs = [], []
    idx = list(range(0, len(traj), frame_stride))
    for k, i in enumerate(idx):
        d = cv2.imread(os.path.join(seq, "depth", "depth_%d.png" % i),
                       cv2.IMREAD_UNCHANGED)
        sm = cv2.imread(os.path.join(seq, "semantic_class", "semantic_class_%d.png" % i),
                        cv2.IMREAD_UNCHANGED)
        if d is None or sm is None:
            raise FileNotFoundError("frame %d の depth/semantic が無い" % i)
        z = d[::pixel_stride, ::pixel_stride].ravel().astype(np.float64) / dscale
        lb = lut[sm[::pixel_stride, ::pixel_stride].ravel()]
        m = (z > 0) & (z < depth_max)
        if not m.any():
            continue
        c2w = traj[i].copy()
        c2w[:3, 1] *= -1               # datasets.py:198-199
        c2w[:3, 2] *= -1
        loc = np.stack([dx[m] * z[m], dy[m] * z[m], dz[m] * z[m]], axis=1)
        pts.append(loc @ c2w[:3, :3].T + c2w[:3, 3])
        labs.append(lb[m])
        if (k + 1) % 50 == 0:
            print("    %d / %d フレーム" % (k + 1, len(idx)), flush=True)
    return np.concatenate(pts), np.concatenate(labs), len(idx)


def voxel_dedup(pts: np.ndarray, labs: np.ndarray, voxel: float):
    """(ボクセル, ラベル) の組ごとに1点残す。

    **ラベルの多数決をしない**のは、クラス境界で片方を消すとその面が痩せるため。
    組ごとに残せば境界は両方残り、**面の位置は変わらない。**
    """
    key = np.floor(pts / voxel).astype(np.int64)
    comb = np.concatenate([key, labs[:, None]], axis=1)
    _, first = np.unique(comb, axis=0, return_index=True)
    first.sort()
    return pts[first], labs[first]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="room_0")
    ap.add_argument("--frame-stride", type=int, default=10)
    ap.add_argument("--pixel-stride", type=int, default=4)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--depth-max", type=float, default=10.0)
    ap.add_argument("--n-points", type=int, default=200000)
    ap.add_argument("--sim3-seed", type=int, default=20260915)
    ap.add_argument("--out", default="output/GT_A")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("Registration/configs/sectionC/%s.yaml" % args.scene))
    cam = yaml.safe_load(open("configs/Replica/replica.yaml"))["cam"]
    os.makedirs(args.out, exist_ok=True)

    print("再投影中… %s（フレーム %d 本おき / 画素 %d 本おき）"
          % (args.scene, args.frame_stride, args.pixel_stride), flush=True)
    pts, labs, n_frames = reproject(args.scene, cam, args.frame_stride,
                                    args.pixel_stride, args.depth_max, cfg["classes"])
    print("  生の点 %d" % len(pts))
    pts, labs = voxel_dedup(pts, labs, args.voxel)
    print("  %.0f mm ボクセル後 %d 点" % (args.voxel * 1000, len(pts)))

    rng = np.random.default_rng(0)
    if len(pts) > args.n_points:
        sel = rng.choice(len(pts), args.n_points, replace=False)
        sel.sort()
        pts, labs = pts[sel], labs[sel]
    print("  間引き後 %d 点  クラス内訳: %s"
          % (len(pts), {k: int((labs == v).sum())
                        for k, v in NAME_TO_ID.items() if (labs == v).any()}))

    # ---- 自分で落ちる検査：M を掛ける前に、変換なしで参照と比べる ----
    dst = io_utils.load_reference_cloud(cfg)
    keep = np.isin(labs, [NAME_TO_ID[n] for n in cfg["reference"]["keep_classes"]])
    ch_all = metrics.chamfer_distance(pts, dst.points)
    ch = metrics.chamfer_distance(pts[keep], dst.points)
    print("\n**自己検査：変換なしで参照と比べた chamfer**")
    print("  全点（什器の background を含む） %.4f m"
          "  ← 参照に相手がいない点を含むので大きく出る" % ch_all)
    print("  **参照と同じクラスだけ（%d 点） %.4f m**（上限 %.2f m）"
          % (int(keep.sum()), ch, SELFCHECK_CHAMFER_MAX))
    if ch > SELFCHECK_CHAMFER_MAX:
        print("**超えた。再投影の座標変換が間違っている可能性が高い。ここで止める。**")
        return 1
    print("  通った。再投影は参照と同じ座標系に落ちている")

    # ---- 既知の Sim(3) を掛けて source を作る ----
    M = known_sim3(args.sim3_seed)
    T_gt = np.linalg.inv(M)
    src_pts = metrics.apply_sim3(M, pts)
    R_m, _, s_m = metrics.decompose_sim3(M)
    print("\n既知の Sim(3) M：縮尺 %.5f / 回転 %.2f 度 / 並進 %s"
          % (s_m, metrics.rotation_error_deg(R_m, np.eye(3)),
             np.round(M[:3, 3], 3).tolist()))

    # ---- オラクル残差（R27 §2-3）----
    back = metrics.apply_sim3(T_gt, src_pts)
    oracle_ch = metrics.chamfer_distance(back, dst.points)
    oracle_ch_keep = metrics.chamfer_distance(back[keep], dst.points)
    tmp = LabeledCloud(points=src_pts, labels=labs, normals=None, meta={})
    oracle_inl = metrics.class_inlier_ratio(tmp, dst, T_gt,
                                            cfg["semantic_icp"]["max_corr_dist"])
    print("\n**オラクル残差（正解 T_gt に置いたときの参照との残差）**")
    print("  chamfer 全点               %.4f m  ← 什器を含む。参照に相手がいない" % oracle_ch)
    print("  **chamfer 参照と同じクラス %.4f m**  ← こちらが手法の到達下限" % oracle_ch_keep)
    print("  クラス内包率               %.4f（対応半径 %.2f m）"
          % (oracle_inl, cfg["semantic_icp"]["max_corr_dist"]))
    if oracle_inl > 0.999:
        print("    **注意：内包率が飽和している。**対応半径 %.2f m に対し残差が %.4f m と小さく、"
              % (cfg["semantic_icp"]["max_corr_dist"], oracle_ch_keep))
        print("    **この指標ではこの課題の難しさを測れない**（`_IMPL_OPERATING_RULES.md` F2）")
    print("  **手法がどれだけ正しくても、これ以下にはならない**")

    # ---- 保存 ----
    ply = os.path.join(args.out, "%s_source.ply" % args.scene)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(src_pts)
    pcd.colors = o3d.utility.Vector3dVector(LABEL_COLORS[labs] / 255.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=float(cfg["preprocess"]["normal_radius"]),
        max_nn=int(cfg["preprocess"]["normal_max_nn"])))
    o3d.io.write_point_cloud(ply, pcd)

    seq = "data/replica/%s_official" % args.scene
    man = {"provenance": provenance(), "scene": args.scene,
           "what": "GT-A: Replica の GT 姿勢で深度を再投影した source。正解は解析的",
           "uses_proposed_method": False,
           "traj_txt": os.path.join(seq, "traj.txt"),
           "traj_sha256_16": sha256(os.path.join(seq, "traj.txt")),
           "traj_provenance": "R27 §1 参照。7/8 シーンは配布アーカイブと sha256 一致",
           "camera": {k: cam[k] for k in ("H", "W", "fx", "fy", "cx", "cy",
                                          "png_depth_scale")},
           "pose_convention": "datasets.py:198-199 の反転 + common.py:91 の光線方向",
           "frame_stride": args.frame_stride, "n_frames_used": n_frames,
           "pixel_stride": args.pixel_stride, "voxel_m": args.voxel,
           "depth_max_m": args.depth_max, "n_points": int(len(src_pts)),
           "subsample_seed": 0,
           "sim3_seed": args.sim3_seed,
           "M_applied": M.tolist(), "T_gt": T_gt.tolist(),
           "T_gt_is_analytic": True,
           "selfcheck_chamfer_no_transform_m": float(ch),
           "selfcheck_chamfer_all_points_m": float(ch_all),
           "selfcheck_note": "参照は keep_classes のみで作られる。什器を含む全点の値も併記",
           "selfcheck_chamfer_all_points_m": float(ch_all),
           "selfcheck_note": "参照は keep_classes のみ。什器を含む全点の値も併記",
           "selfcheck_limit_m": SELFCHECK_CHAMFER_MAX,
           "oracle_chamfer_m": float(oracle_ch),
           "oracle_chamfer_keep_classes_m": float(oracle_ch_keep),
           "oracle_class_inlier_ratio": float(oracle_inl),
           "oracle_max_corr_dist_m": float(cfg["semantic_icp"]["max_corr_dist"]),
           "reference": {k: cfg["reference"][k]
                         for k in ("type", "mesh_path", "n_points", "keep_classes")},
           "source_ply": ply}
    with open(os.path.join(args.out, "%s_manifest.json" % args.scene), "w") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "T_gt_%s.json" % args.scene), "w") as f:
        json.dump({"T_gt": T_gt.tolist()}, f, indent=2)
    print("\n書き出した: %s" % ply)
    print("           %s/%s_manifest.json" % (args.out, args.scene))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

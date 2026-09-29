"""R39 §4 — 正解の回転の幅。G1〜G6（R10）に G7・G8 を足し、基準どうしの食い違いを測る。

**G7・G8 の定式化は、結果を見る前にこのファイルに書いて commit した**（R39 §6）。
**G1 は主基準のまま変えない。G7・G8 は幅を測るためだけの追加である。どちらが正しいかは決めない。**

共通（R10 の `criterion_spread.py` と同じ扱い）
- 初期値：手動 GT（ICP 前、`output/GT_alignment/T_gt_manual`）
- 参照：`Registration/output/ifc/m3_ifc_all.npz` の室内面（`is_inner`）
- source：`output/GT_alignment/source/<scene>.ply` のメッシュ頂点。クラスは頂点色から `color_to_label`

G7（壁だけの ICP）
- G1 と同じ：点対面・段階ゲート 0.30→0.15→0.08・voxel 0.03・各段 60 反復・剛体・BIM から 0.6 m 以内の source 点だけ
- 違い：**source・参照とも wall クラスだけ**を使う（0.6 m の近さも wall の参照に対して測る）

G8（壁の向きで回転を決める）
- ① 回転：手動 GT で置いた source の点のうち、参照（室内面）から水平距離 0.3 m 以内（R38 §4-1 と同じ規則。
  **領域は手動 GT の置き方で決める**。G1 に依存させないため）。
  - 傾き：その領域の floor・ceiling の頂点法線を、上向きにそろえて主軸を取り、+Z へ回す最小の回転
  - ヨー：傾きを直した後の、その領域の wall の頂点法線（水平成分 0.3 超）を 90° で畳んで [−45°, 45°) に置き、
    bin 0.25° で投票。ピーク ±5° 内の平均 ψ を 0° へ回す
  - 手動 GT で置いた領域の点の重心は動かさない
- ② 回転を固定し、**並進と縮尺**を解く：G1 と同じ前処理（voxel 0.03、BIM から 0.6 m 以内）・同じ段階ゲート
  0.30→0.15→0.08 で、点対面の最小二乗 Σ(n·(sRp + t − q))² を各段 60 反復（自前の実装。Open3D は回転固定を持たない）

**G8 は手法のヨーの決め方（壁の法線の投票）と同じ原理である。手法が G8 に近いことは、手法の正しさを示さない。**

出すもの（R39 §4-3）
- シーンごとに G1〜G8 の回転（傾き・ヨー）の G1 に対する差、幅（G1〜G6、G1〜G8）
- 基準対の d_Ω（Ω＝`omega.npz`）と d_{Ω_BIM}（Ω のうち G1 で置いて参照から水平 0.3 m 以内）
- `m3_cor_b/c/d`：G8 で置いたときの 411 の北・南の壁との離れ（R38 §2-6 と同じ）
- R37 の 160 対の 3 区分（6 基準・8 基準）
- PNG：シーンごとの G1〜G8 のヨーの点図 1 枚

確認：定式化 1 を同じ手順で回し直し、保存済みの G1 と一致するか（差の最大）を出す。
"""
from __future__ import annotations

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
import open3d as o3d                                          # noqa: E402

from criterion_verdict import d_omega                         # noqa: E402
from failure_decomposition import provenance                  # noqa: E402
from r36_rotation_diag import tilt_yaw                        # noqa: E402
from regbim import io_utils, metrics, preprocess              # noqa: E402
from regbim.labels import NAME_TO_ID, color_to_label          # noqa: E402

KIT = "output/GT_alignment"
NPZ = "Registration/output/ifc/m3_ifc_all.npz"
NEAR_BIM = 0.60
STAGES = [0.30, 0.15, 0.08]
VOXEL = 0.03
ROOM_DIST = 0.3
BIN, WIN = 0.25, 5.0
TH = {"rot_deg": 5.0, "trans": 0.1, "scale_ratio": 0.05}
WALL = NAME_TO_ID["wall"]


def load_ref(wall_only=False):
    z = np.load(NPZ, allow_pickle=False)
    m = z["is_inner"].astype(bool)
    if wall_only:
        m &= z["labels"] == WALL
    return z["points"][m], z["normals"][m], z["labels"][m]


def load_src(scene):
    mesh = o3d.io.read_triangle_mesh(os.path.join(KIT, "source", "%s.ply" % scene))
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    return v, np.asarray(mesh.vertex_normals), color_to_label(np.asarray(mesh.vertex_colors))


def pcd(p, n=None):
    c = o3d.geometry.PointCloud()
    c.points = o3d.utility.Vector3dVector(np.asarray(p, dtype=np.float64))
    if n is not None:
        c.normals = o3d.utility.Vector3dVector(np.asarray(n, dtype=np.float64))
    return c


def prep(src_pts, T0, ref_pts, ref_nrm):
    """G1 と同じ前処理：T0 で置いて voxel 0.03、参照から 0.6 m 以内の点だけ。"""
    ref = pcd(ref_pts, ref_nrm).voxel_down_sample(VOXEL)
    ref.normalize_normals()
    s = pcd(src_pts); s.transform(T0); s = s.voxel_down_sample(VOXEL)
    q = np.asarray(s.points)
    d, _ = cKDTree(np.asarray(ref.points)).query(q, k=1, workers=-1)
    return q[d <= NEAR_BIM], ref


def icp_plane(near_pts, ref):
    """定式化 1 と同じ Open3D の点対面 ICP（段階ゲート）。T0 で置いた点に対する補正を返す。"""
    near = pcd(near_pts)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    T = np.eye(4)
    for g in STAGES:
        r = o3d.pipelines.registration.registration_icp(
            near, ref, g, T, est, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T = np.asarray(r.transformation)
    return T


def fixed_rot_st(near_pts, ref, R, t0):
    """回転 R（T0 で置いた点に対する補正）を固定し、s・t を点対面で解く（段階ゲート、各段 60 反復）。

    初期値は s=1・t=t0。near_pts は領域の重心を原点に移した座標なので、t0 には重心を渡す
    （＝「手動 GT で置いた領域の重心は動かさない」の初期値）。
    """
    rp, rn = np.asarray(ref.points), np.asarray(ref.normals)
    tree = cKDTree(rp)
    p = near_pts @ R.T                       # 回転は固定
    s, t = 1.0, np.asarray(t0, dtype=np.float64).copy()
    for g in STAGES:
        for _ in range(60):
            x = s * p + t
            d, i = tree.query(x, k=1, workers=-1)
            k = d < g
            if k.sum() < 10:
                break
            n, q, pk = rn[i[k]], rp[i[k]], p[k]
            A = np.hstack([np.sum(n * pk, axis=1, keepdims=True), n])      # [n·p, n]
            b = np.sum(n * q, axis=1)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            ds = abs(sol[0] - s) + np.linalg.norm(sol[1:] - t)
            s, t = float(sol[0]), sol[1:]
            if ds < 1e-9:
                break
    return s, t


def rot_min(a, b):
    """単位ベクトル a を b へ回す最小の回転。"""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def yaw_vote(n):
    h = n[:, :2]; mag = np.linalg.norm(h, axis=1); h = h[mag > 0.3]
    a = (np.degrees(np.arctan2(h[:, 1], h[:, 0])) + 45.0) % 90.0 - 45.0
    hist, e = np.histogram(a, bins=int(90 / BIN), range=(-45, 45))
    pk = e[int(np.argmax(hist))] + BIN / 2
    d = (a - pk + 45.0) % 90.0 - 45.0
    return float(pk + d[np.abs(d) <= WIN].mean()), int(len(a))


def g8(scene, T0, src, ref_pts, ref_nrm):
    v, vn, lab = src
    R0 = T0[:3, :3] / np.cbrt(np.linalg.det(T0[:3, :3]))
    P0 = metrics.apply_sim3(T0, v)
    room = cKDTree(ref_pts[:, :2]).query(P0[:, :2], k=1, workers=-1)[0] <= ROOM_DIST
    n0 = vn @ R0.T
    fc = room & np.isin(lab, [NAME_TO_ID["floor"], NAME_TO_ID["ceiling"]])
    nf = n0[fc].copy(); nf[nf[:, 2] < 0] *= -1
    w, V = np.linalg.eigh(nf.T @ nf); up = V[:, -1]; up *= np.sign(up[2])
    Rt = rot_min(up, np.array([0.0, 0.0, 1.0]))
    psi, nw = yaw_vote((n0[room & (lab == WALL)]) @ Rt.T)
    c, s_ = np.cos(np.radians(-psi)), np.sin(np.radians(-psi))
    Ry = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    Rc = Ry @ Rt                                      # T0 で置いた点に対する回転の補正
    cen = P0[room].mean(axis=0)
    # 領域の重心を動かさない回転：x → Rc (x - cen) + cen
    near, ref = prep(v, T0, ref_pts, ref_nrm)
    near_c = near - cen
    s, t = fixed_rot_st(near_c, ref, Rc, cen)
    # 合成：x_src → T0 → (−cen) → s Rc (·) + t
    M = np.eye(4); M[:3, :3] = s * Rc; M[:3, 3] = t - s * Rc @ np.zeros(3)
    Tc = np.eye(4); Tc[:3, 3] = -cen
    return M @ Tc @ T0, {"up_tilt_deg": float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))),
                         "psi_room_deg": psi, "n_wall": nw, "n_floor_ceiling": int(fc.sum()),
                         "scale": s}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Registration/output/diag/r39_criteria.json")
    args = ap.parse_args()
    spread = json.load(open("Registration/output/diag/criterion_spread.json"))["scenes"]
    rp, rn, rl = load_ref(False)
    wp, wn, _ = load_ref(True)
    om_all = np.load("Registration/output/diag/omega.npz")
    res = {"provenance": provenance(), "scenes": {}}
    for scene in sorted(spread):
        man = os.path.join(KIT, "T_gt_manual", "T_gt_%s.json" % scene)
        if not os.path.exists(man):
            continue
        T0 = np.asarray(json.load(open(man))["T_gt"], dtype=np.float64).reshape(4, 4)
        src = load_src(scene)
        Gs = {f["id"]: np.asarray(f["T"], dtype=np.float64) for f in spread[scene]["results"] if "T" in f}
        # 確認：定式化 1 を回し直す
        near, ref = prep(src[0], T0, rp, rn)
        G1r = icp_plane(near, ref) @ T0
        chk = float(np.abs(G1r - Gs[1]).max())
        # G7：壁だけ
        wmask = src[2] == WALL
        near7, ref7 = prep(src[0][wmask], T0, wp, wn)
        Gs[7] = icp_plane(near7, ref7) @ T0
        # G8：壁の向き
        Gs[8], info8 = g8(scene, T0, src, rp, rn)
        R1 = metrics.decompose_sim3(Gs[1])[0]
        rot = {j: tilt_yaw(metrics.decompose_sim3(G)[0], R1) for j, G in Gs.items()}
        om = om_all[scene] if scene in om_all.files else None
        omb = None
        if om is not None:
            P = metrics.apply_sim3(Gs[1], om)
            omb = om[cKDTree(rp[:, :2]).query(P[:, :2], k=1, workers=-1)[0] <= ROOM_DIST]
        pair = {}
        for a in sorted(Gs):
            for b in sorted(Gs):
                if a < b and om is not None:
                    pair["%d-%d" % (a, b)] = {"d": d_omega(Gs[a], Gs[b], om), "d_bim": d_omega(Gs[a], Gs[b], omb)}
        yaw6 = [rot[j]["yaw_deg"] for j in range(1, 7) if j in rot]
        yaw8 = [rot[j]["yaw_deg"] for j in rot]
        res["scenes"][scene] = {"g1_rerun_maxabs": chk, "T": {j: G.tolist() for j, G in Gs.items()},
                                "rot_vs_g1": rot, "g8": info8, "pairs": pair,
                                "yaw_range_6": [min(yaw6), max(yaw6)], "yaw_range_8": [min(yaw8), max(yaw8)],
                                "scale": {j: metrics.decompose_sim3(G)[2] for j, G in Gs.items()}}
        print("%-11s G1再計算差 %.1e | ヨー差(G1基準) %s | 幅6 %.2f° 幅8 %.2f° | G8 縮尺 %.4f"
              % (scene, chk, " ".join("G%d %+.2f" % (j, rot[j]["yaw_deg"]) for j in sorted(rot)),
                 max(yaw6) - min(yaw6), max(yaw8) - min(yaw8), info8["scale"]), flush=True)
        json.dump(res, open(args.out, "w"), ensure_ascii=False, default=float)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

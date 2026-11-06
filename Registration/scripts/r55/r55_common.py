"""R55 の共通の定義（sni-slam の環境）。**回す前に commit する。** 指示書 §3-1〜§3-4。

試行：実データ 4 スキャン × 参照 E2（411）・E3（410＋411）× 摂動 k 0〜9（R37 の P 系列、`r49_run.perturbations`）× 観測 f ∈ {1p0, 0p6}
- source：f 1p0 は `r49_run.config("real", target, "full")`、f 0p6 は `r52_run.config(target, "0p6")`（slam_mesh、200,000 点、seed 0）
- 条件 A（同じ情報）：R46 の実データの STSL の入力と同じ作り方（`r46_stsl/r46_real_inputs.py`）。
  RG = SLAM の y 上 → z 上の固定の回転（ARKit の重力）。Pz = RG P RGᵀ のヨー ψ（`tilt_yaw(R, I)`）と水平の並進 (t_x, t_y) だけの A。
  **P_A = A RG**（傾き 0・縮尺 1・z の並進 0。全方式に同じ入力 q = P_A p を渡す）
- 条件 B（元の設定）：本手法は R49 の完全（P のまま。記録を使う：f 1p0 は `r49_data/results/real/full`、f 0p6 は `r52_data/results/f0p6`）。
  FPFH＋RANSAC・FGR は R49 と同じく P の縮尺を外す（`make_sim3(R, t/s, 1)`）。STSL は R46 の設定で、入力の作り方は条件 A と同じ
- 正解 G1（`real_criteria`）、Ω は `omega.npz`、Ω_BIM は `r39_omega_bim.npz`。誤差は `r41_rescore.per_trial(T, P_eff, G1, Ω, Ω_BIM)`
- 共通の局所調整（§3-4）：入力 q を T で置いた点（全点）と、参照（`load_reference_cloud(cfg)`、全点）に、Open3D 0.13 の点対面 ICP（剛体）。
  voxel 0.05 m、法線は半径 0.10 m・max_nn 30（両方の点）、対応の距離 0.3 m、反復の上限 50、relative_fitness = relative_rmse = 1e-6、
  クラスは使わない。結果 T' = U T（U は ICP の剛体の更新。T の縮尺は変えない）
"""
import copy, json, os, sys
import numpy as np

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49", "Registration/scripts/r52"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from regbim import io_utils, metrics                            # noqa: E402
from r36_rotation_diag import tilt_yaw                          # noqa: E402
import r49_run as RR                                            # noqa: E402
import r52_run as R52                                           # noqa: E402

OUT = os.path.expanduser("~/rizu/r55_data")
STSL_ROOT = os.path.expanduser("~/rizu/r46_stsl_data")
FS = ["1p0", "0p6"]
RG = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])
FLOOR = {"E2": "M3_411_both", "E3": "M3_410_411_both"}
ICP = {"voxel": 0.05, "normal_radius": 0.10, "normal_max_nn": 30, "dist": 0.3, "max_iter": 50, "rel": 1e-6}


def trials():
    return [(t, f, k) for t in RR.REAL for f in FS for k in range(10)]


def cfg_of(target, f):
    return copy.deepcopy(RR.config("real", target, "full") if f == "1p0" else R52.config(target, f))


def P_of(target, k):
    return np.asarray(RR.perturbations("real", target)[k], dtype=np.float64)


def cond_A(P):
    Pz = RG @ P @ RG.T
    R, t, s = metrics.decompose_sim3(Pz)
    psi = tilt_yaw(R, np.eye(3))["yaw_deg"]
    c, sn = np.cos(np.radians(psi)), np.sin(np.radians(psi))
    A = np.eye(4); A[:3, :3] = [[c, -sn, 0], [sn, c, 0], [0, 0, 1]]; A[:2, 3] = t[:2]
    return A @ RG, A, float(psi)


def strip_scale(P):
    R, t, s = metrics.decompose_sim3(P)
    return metrics.make_sim3(R, t / s, 1.0)


def common_icp(q_pts, dst_pts, T):
    import open3d as o3d
    def pc(x):
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(x, dtype=np.float64))).voxel_down_sample(ICP["voxel"])
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=ICP["normal_radius"], max_nn=ICP["normal_max_nn"]))
        return p
    s, d = pc(metrics.apply_sim3(np.asarray(T, dtype=np.float64), q_pts)), pc(dst_pts)
    r = o3d.pipelines.registration.registration_icp(
        s, d, ICP["dist"], np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=ICP["rel"], relative_rmse=ICP["rel"], max_iteration=ICP["max_iter"]))
    return np.asarray(r.transformation, dtype=np.float64) @ np.asarray(T, dtype=np.float64), float(r.fitness), float(r.inlier_rmse)

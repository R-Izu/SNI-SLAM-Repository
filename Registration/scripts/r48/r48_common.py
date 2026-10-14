"""R48 段階1 の共通の定義（numpy だけ。sni-slam・libim の両方の環境から読む）。指示書 §5（2026-10-14）に従う。**回す前に commit する。**

- 参照：段階0 の部分参照 23 個（`~/rizu/r48_data/BIM/<scene>_n<k>`）＋ GT-A の 8 シーン（参照 = 部屋全体）= 31 個
- 入力の範囲：複数部屋は r ∈ {0.5, 3 m, 切らない}（その参照の床の点からの水平距離）。GT-A は切らない
- 試行：参照 × 範囲ごとに 20。摂動 P = GT-A の config と同じ生成器（`metrics.random_sim3`、default_rng(0)、
  rot_deg [0, 180]（任意の軸）・trans ±2 m・log_scale ±0.4）の先頭 20。GT-A では R45 の trial_matrices の trial 0〜19 と同じ値（確かめる）
- 既知の Sim(3) M：`build_gt_a.known_sim3(20260915)`（GT-A と同じ seed）。正解 G = M⁻¹（source → Replica 座標）
- STSL への入力（実寸）：Rg' = 縮尺 s_G と、G の回転から求めた上向きを e_z に移す回転（R46 §8-4 の Rg に縮尺を足したもの）。
  A = (Rz(ψ), (t_x, t_y, 0), 縮尺 1)（ψ・t は Rg' P Rg'⁻¹ の z まわりの twist と水平の並進）。入力 q = A Rg' p、T_true = G (A Rg')⁻¹（剛体）
- dev：入力の壁の向きの軸からのずれ。参照の BIM 線分の向き（mod 90°、長さで重みをつけた 1° 刻みの最頻）θ0 と、
  T_true の z まわりの回転 φ から、入力の壁の向き θ0 − φ、dev = |((θ0 − φ + 45) mod 90) − 45|
- 区分（§5-5）：BIM 側の角 ≤ 12 少ない、13〜30 中、> 30 多い
"""
import os
import numpy as np

D = os.path.expanduser("~/rizu/r48_data")
SCENES = ["apartment_0", "apartment_1", "apartment_2", "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
          "frl_apartment_3", "frl_apartment_4", "frl_apartment_5", "hotel_0"]
GTA = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
CUTS = ["0p5", "3", "inf"]
CUT_M = {"0p5": 0.5, "3": 3.0, "inf": None}
N_TRIALS = 20
N_PC = 10
SIM3_SEED = 20260915
PERTURB = {"rot_deg": [0.0, 180.0], "trans": [-2.0, 2.0], "log_scale": [-0.4, 0.4]}
COVER_MIN = 0.80
WALL, DOOR, FLOOR, WINDOW, CEIL = 1, 2, 3, 4, 5


def refs_multi():
    out = []
    for s in SCENES:
        k = 1
        while os.path.exists(os.path.join(D, "BIM", "%s_n%d" % (s, k))):
            out.append("%s_n%d" % (s, k)); k += 1
    return out


def all_refs():
    return refs_multi() + ["GTA_" + s for s in GTA]


def cuts_of(ref):
    return ["inf"] if ref.startswith("GTA_") else CUTS


def scene_of(ref):
    return ref[4:] if ref.startswith("GTA_") else ref.rsplit("_n", 1)[0]


def category(n_corners):
    return "few" if n_corners <= 12 else ("mid" if n_corners <= 30 else "many")


def make_sim3(R, t, s):
    T = np.eye(4); T[:3, :3] = s * np.asarray(R); T[:3, 3] = t
    return T


def decompose_sim3(T):
    A = np.asarray(T, dtype=np.float64)[:3, :3]; s = float(np.cbrt(np.linalg.det(A)))
    return A / s, np.asarray(T, dtype=np.float64)[:3, 3].copy(), s


def apply(T, p):
    return np.asarray(p) @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def known_sim3(seed=SIM3_SEED):
    """`build_gt_a.known_sim3` の写し"""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    ang = np.deg2rad(rng.uniform(20.0, 160.0))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    s = float(np.exp(rng.uniform(-0.25, 0.25))); t = rng.uniform(-2.0, 2.0, size=3)
    return make_sim3(R, t, s)


def perturbations(n=N_TRIALS):
    """`metrics.random_sim3` を default_rng(0) で順に n 回（benchmark の摂動の列と同じ）"""
    rng = np.random.default_rng(0); out = []
    for _ in range(n):
        axis = rng.normal(size=3); axis /= (np.linalg.norm(axis) + 1e-12)
        ang = np.radians(rng.uniform(*PERTURB["rot_deg"]))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        t = rng.uniform(PERTURB["trans"][0], PERTURB["trans"][1], size=3)
        s = float(np.exp(rng.uniform(*PERTURB["log_scale"])))
        out.append(make_sim3(R, t, s))
    return out


def rot_to_z(u):
    u = u / np.linalg.norm(u); z = np.array([0.0, 0.0, 1.0]); v = np.cross(u, z); c = float(u @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + V + V @ V * (1.0 / (1.0 + c))


def twist_z(R):
    """R の z まわりの twist（swing-twist、`r36_rotation_diag.tilt_yaw(R, I)` の yaw と同じ式）[°]"""
    tr = np.trace(R); w = np.sqrt(max(0.0, 1.0 + tr)) / 2.0
    z = (R[1, 0] - R[0, 1]) / (4.0 * w) if w > 1e-8 else np.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) / 2.0
    psi = float(np.degrees(2.0 * np.arctan2(z, w)))
    return (psi + 180.0) % 360.0 - 180.0


def rg_metric(G):
    """Rg'：source を、z が上・実寸の座標に移す（回転は R46 の Rg と同じ、縮尺 s_G を足す）"""
    RG, _, sG = decompose_sim3(G)
    return make_sim3(rot_to_z(RG.T @ np.array([0.0, 0.0, 1.0])), np.zeros(3), sG)


def stsl_trial(P, G):
    Rg = rg_metric(G)
    R, t, _ = decompose_sim3(Rg @ P @ np.linalg.inv(Rg))
    psi = twist_z(R); c, sn = np.cos(np.radians(psi)), np.sin(np.radians(psi))
    A = make_sim3(np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]]), np.array([t[0], t[1], 0.0]), 1.0)
    return {"A": A, "Rg": Rg, "psi": psi, "T_true": G @ np.linalg.inv(A @ Rg)}


def dominant_angle(segs):
    if len(segs) == 0:
        return 0.0
    d = segs[:, 2:4] - segs[:, 0:2]; L = np.hypot(d[:, 0], d[:, 1]); a = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90.0
    h = np.bincount(np.minimum(a.astype(int), 89), weights=L, minlength=90)
    return float(np.argmax(h) + 0.5)


def dev_of(T_true, theta0):
    phi = twist_z(decompose_sim3(T_true)[0])
    return float(abs(((theta0 - phi + 45.0) % 90.0) - 45.0))

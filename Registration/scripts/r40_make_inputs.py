"""R40 §4-1 — BIM 由来の source に誤差を 1 種類ずつ足した入力を作る。**結果を見る前に固定して commit する。**

元：`output/R39_BIM/bim_points_bimframe.npz`（R39 §5 と同じ 99,227 点。BIM 座標の点・クラス・法線・410 の部分の印）。
誤差は **BIM 座標で**足し、最後に R39 と同じ M を掛ける（正解は R39 と同じ M^-1）。

| 記号 | 水準 | 作り方（乱数 seed） |
|---|---|---|
| N | σ = 1・2・5 cm | 点ごとに、面の法線方向へ N(0, σ²)（seed 41） |
| L | 5・10・20% | 構造 5 クラスの点から無作為に選び、**他の 4 クラスのどれかに一様に**付け替える（seed 42） |
| O-bg | source の点数の 10・25・50% | 下の「箱」の中に体積一様に点を置き、**background（0。`match_classes` に無い唯一のクラス）**を付ける |
| O-wall | 同上 | 同じ置き方・同じ点で、**wall** を付ける |
| S | 0.5・1・2° | **410 の部分**（410 の範囲＋0.35 m かつ 411 の内法の外）の点だけを、その重心まわりにヨーで回す |

外れ値の箱（家具を模す。結果を見る前に固定）
- 数：**12 個**。中心は source の floor 点から一様に選ぶ（seed 40）
- 大きさ：幅・奥行き U[0.5, 1.5] m、高さ U[0.7, 2.0] m（seed 40 の続き）。向きは BIM の軸に平行
- 高さ方向：箱の底を床の上面（floor 点の z の中央値）に置く。床の上 0〜2 m に収まる
- 点：外れ値の総数を 12 個の箱に**体積に比例して**割り振り、箱の中に体積一様（seed 43）。O-bg と O-wall は同じ点で、ラベルだけ違う
- 水準ごとの点は、50% の点集合の先頭から取る（10% ⊂ 25% ⊂ 50%）

    conda activate sni-slam
    python Registration/scripts/r40_make_inputs.py
"""
import json, os, sys
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
import ifc_export as ie                                        # write_ply と CLASS_NAMES（ifcopenshell は使わない）

SRC = "output/R39_BIM/bim_points_bimframe.npz"
OUT = "output/R40_BIM"
STRUCT = [1, 2, 3, 4, 5]                                       # wall door floor window ceiling
WALL, FLOOR, BG = 1, 3, 0
N_BOX = 12


def save(name, pts_bim, lab, M, info, man):
    src = pts_bim @ M[:3, :3].T + M[:3, 3]
    ie.write_ply(os.path.join(OUT, "%s.ply" % name), src, lab)
    man[name] = dict(info, n_points=int(len(src)),
                     class_counts={ie.CLASS_NAMES[c]: int((lab == c).sum()) for c in np.unique(lab)})
    print(name, man[name]["n_points"], man[name]["class_counts"], flush=True)


def main():
    z = np.load(SRC)
    P, L, Nn, in410, M = z["points"], z["labels"], z["normals"], z["in410"].astype(bool), z["M"]
    n = len(P)
    os.makedirs(OUT, exist_ok=True)
    man = {"source": SRC, "n_source": int(n), "M": M.tolist(), "T_gt": "output/R39_BIM/T_gt_bim.json"}
    save("zero", P, L, M, {"err": "none"}, man)

    rng = np.random.default_rng(41)
    for s in (0.01, 0.02, 0.05):
        save("N_%dcm" % round(s * 100), P + Nn * rng.normal(0, s, size=(n, 1)), L, M, {"err": "N", "sigma_m": s}, man)

    rng = np.random.default_rng(42)
    idx_s = np.flatnonzero(np.isin(L, STRUCT))
    perm = rng.permutation(idx_s)
    shift = rng.integers(1, 5, size=len(perm))                 # 他の 4 クラスへ
    for f in (0.05, 0.10, 0.20):
        k = perm[: int(round(f * len(idx_s)))]
        L2 = L.copy()
        cur = np.array([STRUCT.index(c) for c in L[k]])
        L2[k] = np.array(STRUCT)[(cur + shift[: len(k)]) % 5]
        save("L_%d" % round(f * 100), P, L2, M, {"err": "L", "frac": f}, man)

    # 箱
    rng = np.random.default_rng(40)
    fl = P[L == FLOOR]
    zf = float(np.median(fl[:, 2]))
    ctr = fl[rng.choice(len(fl), N_BOX, replace=False)][:, :2]
    wd = rng.uniform(0.5, 1.5, size=(N_BOX, 2))
    h = rng.uniform(0.7, 2.0, size=N_BOX)
    boxes = [{"center_xy": ctr[i].tolist(), "size_xy": wd[i].tolist(), "height": float(h[i]), "z0": zf}
             for i in range(N_BOX)]
    man["boxes"] = boxes
    vol = wd[:, 0] * wd[:, 1] * h
    n_max = int(round(0.50 * n))
    rng = np.random.default_rng(43)
    cnt = rng.multinomial(n_max, vol / vol.sum())
    pts = []
    for i in range(N_BOX):
        u = rng.random((cnt[i], 3))
        lo = np.array([ctr[i, 0] - wd[i, 0] / 2, ctr[i, 1] - wd[i, 1] / 2, zf])
        pts.append(lo + u * np.array([wd[i, 0], wd[i, 1], h[i]]))
    out_all = np.concatenate(pts)[rng.permutation(n_max)]
    for f in (0.10, 0.25, 0.50):
        o = out_all[: int(round(f * n))]
        for tag, lab in (("Obg", BG), ("Owall", WALL)):
            save("%s_%d" % (tag, round(f * 100)), np.vstack([P, o]),
                 np.concatenate([L, np.full(len(o), lab)]), M,
                 {"err": tag, "frac_of_source": f, "n_outliers": int(len(o))}, man)

    c410 = P[in410].mean(axis=0)
    for a in (0.5, 1.0, 2.0):
        t = np.radians(a)
        Rz = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
        P2 = P.copy()
        P2[in410] = (P[in410] - c410) @ Rz.T + c410
        save("S_%s" % str(a).replace(".", "p"), P2, L, M,
             {"err": "S", "yaw_deg": a, "n_rotated": int(in410.sum()), "center_bim": c410.tolist()}, man)
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

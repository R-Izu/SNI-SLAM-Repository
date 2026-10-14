"""R48 段階1 §5-2：source 案 (i) — 軌跡からの raycast（libim の環境、Open3D の RaycastingScene）。**登録の結果を見る前に固定する。**

規則（固定）
1. 部屋分けは段階0 の `r48_rooms.segment` のまま。軌跡を置く領域 = 残した部屋（2 m² 以上）の格子の和
2. 骨格：その領域の skimage.morphology.skeletonize。視点：骨格の格子を行優先の順に見て、既に選んだ視点から 0.5 m 以上離れていれば選ぶ
3. 視点の高さ：主の階の床 z0 ＋ 1.4 m
4. 光線：方位 0〜359.5°（0.5° 刻み、720 本）× 仰角 −40〜+40°（0.5° 刻み、161 本）。当たった最初の面の点（10 m まで）。
   ラベル = 当たった三角形のラベル（`r48_mesh.py`。全クラス、家具は background）
5. 点の重複除去：GT-A（`build_gt_a.py`）と同じ (0.02 m ボクセル, ラベル) ごとに 1 点、seed 0 で 200,000 点に間引く
出力：`~/rizu/r48_data/source_raw/<scene>.npz`（C：Replica 座標の点・ラベル、視点）、軌跡の図 `~/rizu/r48_data/fig/traj_<scene>.png`
"""
import json, os, sys
import numpy as np
import open3d as o3d
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r48_rooms as rr                                           # noqa: E402

D = os.path.expanduser("~/rizu/r48_data")
H_EYE, STEP, AZ, EL, DMAX, VOX, NPTS = 1.4, 0.5, 0.5, 0.5, 10.0, 0.02, 200000


def viewpoints(g):
    region = np.isin(g["rooms"], g["keep"])
    sk = np.argwhere(skeletonize(region))                         # 行優先の順
    xy = sk * rr.CELL + g["xy0"] + rr.CELL / 2
    sel = []
    for p in xy:
        if not sel or np.min(np.linalg.norm(np.asarray(sel) - p, axis=1)) >= STEP:
            sel.append(p)
    return np.asarray(sel), region


def dirs():
    az = np.radians(np.arange(0, 360, AZ)); el = np.radians(np.arange(-40, 40 + 1e-9, EL))
    A, E = np.meshgrid(az, el, indexing="ij")
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3).astype(np.float32)


def voxel_dedup(pts, labs, voxel):
    key = np.floor(pts / voxel).astype(np.int64)
    _, first = np.unique(np.concatenate([key, labs[:, None].astype(np.int64)], axis=1), axis=0, return_index=True)
    first.sort()
    return pts[first], labs[first]


def main():
    os.makedirs(os.path.join(D, "source_raw"), exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ("/mnt/c/Windows/Fonts/arial.ttf", "/home/student/plotconda/meiryo_ui.ttf"):
        fm.fontManager.addfont(f)
    plt.rcParams["font.family"] = ["Arial", "Meiryo UI"]; plt.rcParams["font.size"] = 8
    Dr = dirs()
    for scene in sys.argv[1:] or rr.SCENES:
        out = os.path.join(D, "source_raw", scene + ".npz")
        if os.path.exists(out):
            continue
        g = rr.segment(scene); vp, region = viewpoints(g)
        m = np.load(os.path.join(D, "mesh", scene + ".npz"))
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(m["V"]), o3d.core.Tensor(m["F"].astype(np.uint32)))
        pts, labs = [], []
        for p in vp:
            o = np.array([p[0], p[1], g["z0"] + H_EYE], np.float32)
            rays = np.concatenate([np.broadcast_to(o, Dr.shape), Dr], 1)
            ans = sc.cast_rays(o3d.core.Tensor(rays))
            t = ans["t_hit"].numpy(); pid = ans["primitive_ids"].numpy()
            ok = np.isfinite(t) & (t < DMAX)
            pts.append(o + Dr[ok] * t[ok, None]); labs.append(m["label"][pid[ok]])
        P = np.concatenate(pts).astype(np.float64); L = np.concatenate(labs).astype(np.int64)
        n_raw = len(P)
        P, L = voxel_dedup(P, L, VOX); n_vox = len(P)
        rng = np.random.default_rng(0)
        if len(P) > NPTS:
            sel = np.sort(rng.choice(len(P), NPTS, replace=False)); P, L = P[sel], L[sel]
        np.savez_compressed(out, points=P.astype(np.float32), labels=L.astype(np.int8), viewpoints=vp, z_eye=g["z0"] + H_EYE)
        info = {"scene": scene, "n_viewpoints": int(len(vp)), "n_rays_per_viewpoint": int(len(Dr)), "n_hits": int(n_raw),
                "n_after_voxel": int(n_vox), "n_points": int(len(P)), "class_counts": np.bincount(L, minlength=6).tolist()}
        json.dump(info, open(os.path.join(D, "source_raw", scene + ".json"), "w"))
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(np.where(region, 0.85, 1.0).T, origin="lower", cmap="gray", vmin=0, vmax=1,
                  extent=[g["xy0"][0], g["xy0"][0] + g["shp"][0] * rr.CELL, g["xy0"][1], g["xy0"][1] + g["shp"][1] * rr.CELL])
        sub = P[::10]; ax.scatter(sub[:, 0], sub[:, 1], s=0.05, c="#999999")
        ax.plot(vp[:, 0], vp[:, 1], ".", color="#d0021b", ms=3)
        ax.set_aspect("equal"); ax.set_title("%s：視点 %d（赤）、source の点（灰、1/10）、軌跡の領域（薄い灰）" % (scene, len(vp)), loc="left", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(D, "fig", "traj_%s.png" % scene), dpi=90); plt.close(fig)
        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()

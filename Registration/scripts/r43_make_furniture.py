"""R43 §2-1 — R42 §4-3 の置き方から 1 点だけ変えた、壁に沿った家具を模した外れ値。**置く前に固定して commit する。**

R42 からの変更（この 1 点だけ）：**箱の位置を、source の wall 点のうち 411 の室内面に属する点（`in410` でない点）から選ぶ。**
理由：R42 では 410 の壁にも置いていた。410 は参照（411）に無い部屋なので、そこに置いた箱の点は参照から
0.3 m 以内に入りようがなく、感度の確認（R42 §4-3）が 34.5% で通らなかった（R42 回す前の報告 §3-2）。
数・大きさ・隙間・6 面への点の打ち方・ラベル・水準・seed（420〜422）は R42 と同じ。

以下は R42 の説明（位置の候補の範囲だけが上のとおり変わる）。


元：`output/R39_BIM/bim_points_bimframe.npz`（R39 §5・R40 §4 と同じ 99,227 点、410＋411 の室内面）。
外れ値は **BIM 座標で**足し、最後に R39 と同じ M を掛ける（正解は R39 と同じ M^-1）。

箱の置き方（固定）
- 数：**12 個**（R40 と同じ）
- 位置：source の **wall 点（R43：411 の室内面に属する点だけ）**から一様に 12 点を選ぶ（seed 420）。その点の法線を水平に投影し、**部屋の内側**を向く向きにする
  （p + 0.5 n̂ の水平 0.3 m 以内に floor 点がある向き。両方とも無ければ、その点は捨てて次の候補を使う）
- 大きさ（seed 421）：幅（壁に沿う向き）U[0.5, 2.0] m、奥行き U[0.3, 0.6] m、高さ U[0.7, 2.0] m
- 壁との隙間（seed 421 の続き）：U[0.05, 0.25] m。箱の背面が壁の点から内側へ隙間だけ離れる
- 高さ方向：箱の底を床の上面（floor 点の z の中央値）に置く
- 点（seed 422）：箱の **6 面すべて**に、面積に比例して一様に打つ（指示書 §4-3「箱の表面に点を打つ」をそのまま読む。
  底面と背面は実際の SLAM では見えないが、ここでは入れる）。外れ値の総数を 12 個の箱に**表面積に比例して**割り振る
- ラベル：**background（0）**
- 水準：source の点数の **10・25・50%**。50% の点集合の先頭から取る（10% ⊂ 25% ⊂ 50%）
- 箱が部屋の角を越えたり、ほかの箱と重なったりしても、切らない

確認（§4-3。本番の前に報告する）
1. 正解の置き方（BIM 座標）で、外れ値の点のうち、参照（411、`r39_bim/bim_E2_plan_correlate.yaml` の参照）の点から
   0.3 m 以内にあるものの割合。**50% 以上**で本番に進む（水準ごとに出し、判定は 3 水準すべてで見る）
2. 実データ廊下 4 シーン（source seed 0、`preprocess.prepare` 後）の background の点の割合と、そのうち G1 の置き方で
   参照（E2＝411、E3＝411＋410）から 0.3 m 以内にあるものの割合（記述）

    conda activate sni-slam
    python Registration/scripts/r43_make_furniture.py
"""
import json, os, sys
import numpy as np
import yaml
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration")); sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
import ifc_export as ie                                        # noqa: E402  write_ply と CLASS_NAMES
from regbim import io_utils, preprocess, metrics               # noqa: E402
from failure_decomposition import provenance                   # noqa: E402

SRC = "output/R39_BIM/bim_points_bimframe.npz"
OUT = "output/R43_BIM"
DIAG = "Registration/output/diag/r42"
WALL, FLOOR, BG = 1, 3, 0
N_BOX = 12
LEVELS = (0.10, 0.25, 0.50)
COR = ["m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d"]


def inward(p, n, fl_tree):
    h = np.array([n[0], n[1], 0.0]); L = np.linalg.norm(h)
    if L < 0.5:
        return None
    h /= L
    for s in (1.0, -1.0):
        q = p + s * 0.5 * h
        if fl_tree.query_ball_point(q[:2], 0.3):
            return s * h
    return None


def box_faces(o, t, n, w, d, h):
    """o：背面の左下（床の上）、t：壁に沿う単位ベクトル、n：内向きの単位ベクトル。面 = (原点, 辺1, 辺2)"""
    z = np.array([0.0, 0.0, 1.0])
    W, D, H = t * w, n * d, z * h
    return [(o, W, H), (o + D, W, H),            # 背面・前面
            (o, D, H), (o + W, D, H),            # 側面 2
            (o, W, D), (o + H, W, D)]            # 底面・上面


def main():
    z = np.load(SRC)
    P, L, Nn, M = z["points"], z["labels"], z["normals"], z["M"]
    in410 = z["in410"].astype(bool)
    n_src = len(P)
    os.makedirs(OUT, exist_ok=True); os.makedirs(DIAG, exist_ok=True)
    fl = P[L == FLOOR]; zf = float(np.median(fl[:, 2])); fl_tree = cKDTree(fl[:, :2])
    wall_idx = np.flatnonzero((L == WALL) & ~in410)              # R43：411 の室内面の wall 点だけ

    rng = np.random.default_rng(420)
    order = rng.permutation(wall_idx)
    anchors = []
    for i in order:
        hn = inward(P[i], Nn[i], fl_tree)
        if hn is not None:
            anchors.append((P[i], hn))
        if len(anchors) == N_BOX:
            break
    rng = np.random.default_rng(421)
    w = rng.uniform(0.5, 2.0, N_BOX); d = rng.uniform(0.3, 0.6, N_BOX); h = rng.uniform(0.7, 2.0, N_BOX)
    gap = rng.uniform(0.05, 0.25, N_BOX)
    boxes, faces = [], []
    for k, (p, hn) in enumerate(anchors):
        t = np.cross([0.0, 0.0, 1.0], hn)
        o = np.array([p[0], p[1], zf]) + gap[k] * hn - 0.5 * w[k] * t
        fs = box_faces(o, t, hn, w[k], d[k], h[k])
        faces.append(fs)
        boxes.append({"anchor_wall_point": p.tolist(), "inward": hn.tolist(), "width": float(w[k]), "depth": float(d[k]),
                      "height": float(h[k]), "gap": float(gap[k]), "origin": o.tolist()})
    area = np.array([[np.linalg.norm(np.cross(a, b)) for (_, a, b) in fs] for fs in faces])   # (12, 6)
    n_max = int(round(0.50 * n_src))
    rng = np.random.default_rng(422)
    cnt = rng.multinomial(n_max, (area / area.sum()).ravel()).reshape(area.shape)
    pts = []
    for k, fs in enumerate(faces):
        for f, (o, a, b) in enumerate(fs):
            u = rng.random((cnt[k, f], 2))
            pts.append(o + u[:, :1] * a + u[:, 1:] * b)
    out_all = np.concatenate(pts)[rng.permutation(n_max)]

    man = {"provenance": provenance(), "source": SRC, "n_source": int(n_src), "M": M.tolist(),
           "T_gt": "output/R39_BIM/T_gt_bim.json", "floor_z": zf, "anchor_pool": "wall & ~in410 (411)", "boxes": boxes,
           "box_surface_area_m2": float(area.sum()), "levels": {}}
    cfg = yaml.safe_load(open("Registration/configs/r39_bim/bim_E2_plan_correlate.yaml"))
    ref = io_utils.load_reference_cloud(cfg)
    ref_tree = cKDTree(ref.points)
    for f in LEVELS:
        o = out_all[: int(round(f * n_src))]
        dist, _ = ref_tree.query(o)
        frac = float((dist < 0.3).mean())
        src = np.vstack([P, o]) @ M[:3, :3].T + M[:3, 3]
        lab = np.concatenate([L, np.full(len(o), BG)])
        name = "F_%d" % round(f * 100)
        ie.write_ply(os.path.join(OUT, "%s.ply" % name), src, lab)
        man["levels"][name] = {"frac_of_source": f, "n_outliers": int(len(o)), "frac_within_0p3m_of_ref": frac,
                               "dist_to_ref_q": {q: float(np.percentile(dist, q)) for q in (5, 25, 50, 75, 95)}}
        print(name, len(o), "within 0.3 m of ref: %.3f" % frac, flush=True)
    man["check1_pass"] = all(v["frac_within_0p3m_of_ref"] >= 0.5 for v in man["levels"].values())
    print("check1_pass", man["check1_pass"])

    # 実データ（記述）
    sp = json.load(open("Registration/output/diag/criterion_spread.json"))["scenes"]
    man["realdata"] = {}
    for s in COR:
        G1 = next(np.asarray(r["T"]) for r in sp[s]["results"] if int(r["id"]) == 1)
        ent = {}
        for e in ("E2", "E3"):
            c = yaml.safe_load(open("Registration/configs/realdata/%s__%s.yaml" % (s, e)))
            spec = dict(c["source"], seed=0)
            src = preprocess.prepare(io_utils.load_source_cloud(dict(c, source=spec)), c)
            bg = src.points[src.labels == BG]
            r_tree = cKDTree(io_utils.load_reference_cloud(c).points)
            dd, _ = r_tree.query(metrics.apply_sim3(G1, bg))
            ent[e] = {"n_source_prepared": int(len(src.points)), "n_background": int(len(bg)),
                      "frac_background": float(len(bg) / len(src.points)), "frac_bg_within_0p3m_of_ref": float((dd < 0.3).mean())}
        man["realdata"][s] = ent
        print(s, ent, flush=True)
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(man, open(os.path.join(DIAG, "r43_furniture_check.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

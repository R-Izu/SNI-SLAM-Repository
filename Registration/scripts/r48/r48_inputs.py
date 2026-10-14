"""R48 段階1：入力・参照・STSL の入力・幾何の確認（被覆率）・陽性対照の入力を作る（libim の環境）。**登録はしない。回す前に commit する。**

複数部屋のシーンごと
1. 自己検査（GT-A と同じ考え方）：M を掛ける前の C（参照と同じ構造 5 クラスの点）を、変換なしで参照（シーン全体、1,000,000 点）と比べる。
   C → 参照の平均距離が 0.10 m を超えたら止める（参照にはどの視点からも見えない面＝apartment_0 の上の階など があるので、
   対称の chamfer は併記だけにする）
2. source S = M C（M = `known_sim3(20260915)`）。法線は GT-A と同じ（Open3D、半径 0.15、最大 50 近傍）。Ω = S の wall・floor・ceiling の点を
   順に 20,000 点に間引いたもの（`build_omega` と同じ規則）
3. 部分参照：段階0 の規則（`r48_rooms`）で作り直し、段階0 の点の数と一致することを確かめる → `refs/<ref>.npz`
4. 入力の範囲：C の点のうち、その参照の floor の点（0.05 m で間引き）から水平距離 ≤ r のもの（r = 0.5・3 m）、切らない = 全部
   → `inputs_sni/<ref>__<cut>.npz`（source 座標の点・ラベル・法線）
5. STSL：`r48_common.stsl_trial` の A・Rg' で q = A Rg' p → `inputs/<ref>__<cut>/NNNNNN.npy` と `spec.json`（P・A・Rg・T_true・dev）
6. 幾何の確認（§5-4）：参照の wall の点のうち床の上 0.3〜1.8 m の点で、BIM 線分から水平 0.10 m 以内の割合（被覆率）
7. 陽性対照（§5-4、記述だけ）：参照そのものの構造 5 クラスの点（Replica 座標、z は上）に、摂動の先頭 10 の水平化したヨーと水平の並進
   （Rg = I、縮尺なし）→ `inputs/pc_<ref>/`、T_true = A⁻¹
GT-A：参照・source・G は `r48_gta_dump.py` の出力。BIM 線分は `lines_from_walls_v2`（床の高さ = 参照の floor の z の中央値、R46 段階3 と同じ）
→ `BIM/GTA_<scene>/`。STSL の入力・被覆率・陽性対照は上と同じ。
出力のまとめ：`~/rizu/r48_data/inputs_manifest.json`
"""
import hashlib, json, os, sys
import numpy as np
import open3d as o3d
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r48_common as C                                          # noqa: E402
import r48_rooms as rr                                          # noqa: E402

SELFCHECK_MAX = 0.10


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def normals(P):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=50))
    return np.asarray(pc.normals)


def omega(P, L):
    pts = P[np.isin(L, [C.WALL, C.FLOOR, C.CEIL])]
    return pts[np.linspace(0, len(pts) - 1, 20000).astype(int)] if len(pts) > 20000 else pts


def coverage(refP, refL, z0, segs):
    w = (refL == C.WALL) & (refP[:, 2] > z0 + 0.3) & (refP[:, 2] < z0 + 1.8)
    Q = refP[w, :2]
    if len(Q) == 0 or len(segs) == 0:
        return 0.0, int(len(Q))
    best = np.full(len(Q), np.inf)
    for x in segs:
        a, b = x[:2], x[2:4]; d = b - a; L2 = float(d @ d)
        t = np.clip(((Q - a) @ d) / L2, 0, 1) if L2 > 0 else np.zeros(len(Q))
        best = np.minimum(best, np.linalg.norm(Q - (a + t[:, None] * d), axis=1))
    return float(np.mean(best <= 0.10)), int(len(Q))


def partial_refs(scene):
    """段階0（`r48_rooms.main`）と同じ規則で部分参照の点を作り直す"""
    g = rr.segment(scene); P, L, N, z0 = g["P"], g["L"], g["N"], g["z0"]
    order = [g["keep"][0]] if g["keep"] else []; steps = [list(order)]
    while len(order) < len(g["keep"]):
        cand = sorted({q for r in order for q in g["adj"][r]} - set(order), key=lambda q: -g["areas"][q])
        if not cand:
            break
        order.append(cand[0]); steps.append(list(order))
    if steps and len(steps[-1]) < len(g["keep"]):
        steps.append(list(g["keep"]))
    cell = g["ij"](P); inb = (cell[:, 0] >= 0) & (cell[:, 1] >= 0) & (cell[:, 0] < g["shp"][0]) & (cell[:, 1] < g["shp"][1])
    lvl = (P[:, 2] > z0 - 0.3) & (P[:, 2] < z0 + 3.0); out = []
    for n, st in enumerate(steps, 1):
        region = ndi.binary_dilation(np.isin(g["rooms"], st), iterations=3)
        sel = np.zeros(len(P), bool); sel[inb] = region[cell[inb, 0], cell[inb, 1]]; sel &= lvl
        out.append(("%s_n%d" % (scene, n), P[sel], L[sel], N[sel], z0))
    return out


def stsl_set(name, Ssrc, G, theta0):
    d = os.path.join(C.D, "inputs", name); os.makedirs(d, exist_ok=True)
    spec = {"set": name, "trials": []}
    for k, P in enumerate(C.perturbations()):
        t = C.stsl_trial(P, G)
        np.save(os.path.join(d, "%06d.npy" % k), C.apply(t["A"] @ t["Rg"], Ssrc).astype(np.float32))
        spec["trials"].append({"index": k, "P": P.tolist(), "A": t["A"].tolist(), "Rg": t["Rg"].tolist(), "psi": t["psi"],
                               "T_true": t["T_true"].tolist(), "dev": C.dev_of(t["T_true"], theta0),
                               "true_scale_of_input": float(C.decompose_sim3(t["T_true"])[2])})
    json.dump(spec, open(os.path.join(d, "spec.json"), "w"))
    return spec


def pc_set(ref, refP, theta0):
    d = os.path.join(C.D, "inputs", "pc_" + ref); os.makedirs(d, exist_ok=True)
    spec = {"set": "pc_" + ref, "trials": []}
    for k, P in enumerate(C.perturbations(C.N_PC)):
        t = C.stsl_trial(P, np.eye(4))
        np.save(os.path.join(d, "%06d.npy" % k), C.apply(t["A"], refP).astype(np.float32))
        Tt = np.linalg.inv(t["A"])
        spec["trials"].append({"index": k, "A": t["A"].tolist(), "T_true": Tt.tolist(), "dev": C.dev_of(Tt, theta0)})
    json.dump(spec, open(os.path.join(d, "spec.json"), "w"))


def ref_entry(refP, refL, z0, segs):
    theta0 = C.dominant_angle(segs); cov, nw = coverage(refP, refL, z0, segs)
    return theta0, {"n_ref_points": int(len(refP)), "z_floor": float(z0), "n_segments": int(len(segs)), "theta0_deg": theta0,
                    "coverage": cov, "n_wall_points_cov": nw, "stsl_included": bool(cov >= C.COVER_MIN), "inputs": {}}


def do_ref(man, ref, refP, refL, refN, z0, S, Cp, srcL, srcN, G, segs):
    np.savez_compressed(os.path.join(C.D, "refs", ref + ".npz"), points=refP.astype(np.float32), labels=refL.astype(np.int8),
                        normals=refN.astype(np.float32))
    theta0, ent = ref_entry(refP, refL, z0, segs)
    fl = refP[refL == C.FLOOR, :2]
    fl = fl[np.unique(np.floor(fl / 0.05).astype(np.int64), axis=0, return_index=True)[1]]
    tree = cKDTree(fl)
    for cut in C.CUTS:
        if C.CUT_M[cut] is None:
            m = np.ones(len(Cp), bool)
        else:
            m = tree.query(Cp[:, :2])[0] <= C.CUT_M[cut]
        name = "%s__%s" % (ref, cut)
        np.savez_compressed(os.path.join(C.D, "inputs_sni", name + ".npz"), points=S[m].astype(np.float32),
                            labels=srcL[m].astype(np.int8), normals=srcN[m].astype(np.float32))
        spec = stsl_set(name, S[m], G, theta0)
        ent["inputs"][cut] = {"n_points": int(m.sum()), "sha256_16": sha(S[m].astype(np.float32)),
                              "dev": [t["dev"] for t in spec["trials"]]}
    pc_set(ref, refP, theta0)
    man["refs"][ref] = ent
    print(ref, "points", len(refP), "segs", len(segs), "theta0 %.1f" % theta0, "coverage %.3f" % ent["coverage"],
          "inputs", {c: v["n_points"] for c, v in ent["inputs"].items()}, flush=True)


def main():
    for d in ("refs", "inputs_sni", "inputs"):
        os.makedirs(os.path.join(C.D, d), exist_ok=True)
    mp = os.path.join(C.D, "inputs_manifest.json")
    man = {"refs": {}, "scenes": {}}
    M = C.known_sim3(); G = np.linalg.inv(M); man["M"] = M.tolist(); man["G"] = G.tolist()
    om = {}
    for scene in C.SCENES:
        z = np.load(os.path.join(C.D, "source_raw", scene + ".npz")); Cp = z["points"].astype(np.float64); L = z["labels"].astype(np.int64)
        RP = np.load(os.path.join(C.D, "ref", scene + ".npz"))["points"].astype(np.float64)
        keep = np.isin(L, [C.WALL, C.DOOR, C.FLOOR, C.WINDOW, C.CEIL])
        d_cr = cKDTree(RP).query(Cp[keep])[0]; d_rc = cKDTree(Cp[keep]).query(RP)[0]
        sc = {"selfcheck_c_to_ref_mean_m": float(d_cr.mean()), "selfcheck_chamfer_sym_m": float((d_cr.mean() + d_rc.mean()) / 2),
              "selfcheck_c_to_ref_p95_m": float(np.percentile(d_cr, 95)), "selfcheck_limit_m": SELFCHECK_MAX,
              "n_source": int(len(Cp)), "class_counts": np.bincount(L, minlength=6).tolist()}
        if sc["selfcheck_c_to_ref_mean_m"] > SELFCHECK_MAX:
            raise SystemExit("自己検査を超えた：%s %s" % (scene, sc))
        S = C.apply(M, Cp); N = normals(S)
        sc["oracle_c_to_ref_mean_m"] = float(cKDTree(RP).query(C.apply(G, S)[keep])[0].mean())
        print(scene, sc, flush=True)
        om[scene] = omega(S, L); man["scenes"][scene] = sc
        stage0 = json.load(open(os.path.join(C.D, "rooms", scene + ".json")))["steps"]
        for i, (rname, rp, rl, rn, z0) in enumerate(partial_refs(scene)):
            if len(rp) != stage0[i]["n_points"]:
                raise SystemExit("段階0 と点の数が違う：%s %d vs %d" % (rname, len(rp), stage0[i]["n_points"]))
            segs = np.loadtxt(os.path.join(C.D, "BIM", rname, "bimLineSeg.txt")).reshape(-1, 4)
            do_ref(man, rname, rp, rl, rn, z0, S, Cp, L, N, G, segs)
        json.dump(man, open(mp, "w"), indent=1)
    gta = np.load(os.path.join(C.D, "gta.npz"))
    for s in C.GTA:
        r = np.load(os.path.join(C.D, "refs", "GTA_%s.npz" % s))
        RP, RL, RN = r["points"].astype(np.float64), r["labels"].astype(np.int64), r["normals"].astype(np.float64)
        z0 = float(np.median(RP[RL == C.FLOOR, 2]))
        w = (RL == C.WALL) & (np.abs(RN[:, 2]) < 0.3)
        segs = rr.lines_from_walls_v2(RP[w] - [0, 0, z0], RN[w])
        bd = os.path.join(C.D, "BIM", "GTA_" + s); os.makedirs(bd, exist_ok=True)
        np.savetxt(os.path.join(bd, "bimLineSeg.txt"), segs, fmt="%.4f")
        W = RP[RL == C.WALL] - [0, 0, z0]; rr.write_pcd(os.path.join(bd, "bim.pcd"), W)
        flat = np.c_[W[:, :2], np.zeros(len(W))]
        key = np.unique(np.floor(flat[:, :2] / 0.05).astype(np.int64), axis=0, return_index=True)[1]
        rr.write_pcd(os.path.join(bd, "bim2d.pcd"), flat[np.sort(key)])
        json.dump({"z_floor": z0, "n_segments": int(len(segs)), "reference": "gt_a_plan_a/%s (seed 0, 200000)" % s},
                  open(os.path.join(bd, "meta.json"), "w"))
        src = np.load(os.path.join(C.D, "inputs_sni", "GTA_%s__inf.npz" % s)); Sg = src["points"].astype(np.float64)
        om[s] = gta["omega_" + s]
        theta0, ent = ref_entry(RP, RL, z0, segs)
        spec = stsl_set("GTA_%s__inf" % s, Sg, gta["G_" + s], theta0); pc_set("GTA_" + s, RP, theta0)
        ent["inputs"]["inf"] = {"n_points": int(len(Sg)), "sha256_16": sha(src["points"]), "dev": [t["dev"] for t in spec["trials"]]}
        man["refs"]["GTA_" + s] = ent
        print("GTA_" + s, "segs", len(segs), "theta0 %.1f" % theta0, "coverage %.3f" % ent["coverage"], flush=True)
    np.savez_compressed(os.path.join(C.D, "omega.npz"), **om)
    json.dump(man, open(mp, "w"), indent=1)


if __name__ == "__main__":
    main()

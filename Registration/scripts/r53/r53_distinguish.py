"""R53：取り違えは区別できたか（sni-slam の環境、ブランチ `exp/r53`）。**回す前に commit する。位置合わせは回さない。** 指示書 §3。

記録にある候補の変換を採点し直すだけである。変換の出どころ（§3-1）
- R51 の試行ごとの記録 `~/rizu/r51_data/stage0_trials.json`・R52 の `~/rizu/r52_data/trials.json` には変換が無い（正誤・順位・ρ だけ）。
  変換はその元の結果ファイルの `rows[k]`：T_est・P・diag.after_icp（通常の精緻化の後の 4 候補）・diag.scores（その ρ、小数 5 桁）・
  diag.short（短い精緻化の後の 16 候補）。R52 の f < 1 は diag.score_short（16 候補の ρ）もある
  - R49 の完全（R51 の合成・実データ、R52 の f 1.0）：`~/rizu/r49_data/results/<synth|real>/full/<target>.json`
  - R52 の f 0.2〜0.8：`~/rizu/r52_data/results/f<f>/<target>.json`
  - R50 の採点だけクラスなし（陽性対照）：`~/rizu/r50_data/results/synth/score_only/<target>.json`
- 変換 T は「P で動かした source を前処理（`preprocess.prepare`）した点」→ 参照。source の元の座標での正解は G、P で動かした座標では G P⁻¹

計算（指示書 §3-3。ここで足した細部は ＊）
- 点：source は `prepare(transform_cloud(src, P), cfg)`、参照は `prepare(dst, cfg)`（どちらも ρ と同じ点。＊）。構造＝壁・床・天井
- ρ(T) = `metrics.class_inlier_ratio(src_p, dst_p, T, τ)`、τ = cfg の `semantic_icp.max_corr_dist`。試行ごとに記録の scores と比べる（＊ 検算）
- 参照の内側：参照の床の点の xy を 0.1 m の格子に落とし、3×3 の正方の構造要素で 1 回膨らませた（＊ 8 近傍）マスの範囲で、
  z が [参照の床の z の最小, 参照の天井の z の最大] の範囲（`inner_region`）。上向きは z（`tilt_yaw` と同じ）
- C(T)：source の構造の点のうち、T で置くと内側に入り、かつ同じクラスの参照の点までの距離が τ より大きいものの数 ÷ source の構造の点の数（＊ 分母は構造の点のすべて）
- 観測した範囲 O：参照の構造の点のうち、G P⁻¹ で置いた source の点（すべてのクラス、＊）までの距離が 0.5 m 以下のもの
- V(T)：O の点のうち、T で置いた source の同じクラスの点までの距離が τ 以下のものの割合
- S：O の点 x を A = T_w T_c⁻¹（4×4 の Sim(3)）で移し、同じクラスの参照の点までの距離が τ 以下のものの割合
- 参考：ρ(T_c)・ρ(T_w)・Δρ = ρ(T_w) − ρ(T_c)（＊ 符号）、ヨーの差・並進の差・縮尺の比（`r51_stage0.rel` と同じ定義。並進の差は Ω の重心で測る）、
  G P⁻¹ で置いた source の点（すべて）のうち参照の内側の外に出る割合
試行の群（§3-2）：正しい＝`r51_stage0.ok(per_trial(·, P, G1, Ω))`
- S：4 候補に正しい候補があり、記録の scores の 1 位（同じ値は記録の順、R51 と同じ）が正しくない。T_c＝正しい 4 候補のうち ρ 最大、T_w＝1 位
- N：16 候補（diag.short）に正しい候補があり、4 候補には無い。T_c＝正しい 16 候補のうち ρ 最大、T_w＝4 候補のうち ρ 最大
- 陰性対照：大きな失敗でない試行から、データ × f ごとに `default_rng(0).choice(n, min(n,100), replace=False)`（試行は target・k の順に並べる、＊）。
  T_c＝G P⁻¹、T_w＝T_est
- 陽性対照：R50 score_only の合成の大きな失敗で Δρ = 0（R51 の記録）の試行から、`default_rng(0)` で 100。T_w＝その方式の 1 位、
  T_c＝正しい 4 候補のうち ρ（クラスあり）最大
対象のデータは実データと合成（指示書 §3-2・§3-4。R49 の bim は対象に無い）
出力：試行ごと `~/rizu/r53_data/<mode>.json`、まとめ `Registration/output/diag/r53/r53_summary.json`
使い方：python r53_distinguish.py controls [並列数] → lines → main [並列数] → summary
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49", "Registration/scripts/r50", "Registration/scripts/r51",
          "Registration/scripts/r52"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from regbim import io_utils, metrics, preprocess                # noqa: E402
from regbim.labels import NAME_TO_ID                            # noqa: E402
from failure_decomposition import provenance                    # noqa: E402
from r41_rescore import per_trial                               # noqa: E402
import r49_run as RR                                            # noqa: E402
from r49_score import truth, big                                # noqa: E402
from r51_stage0 import ok, rel                                  # noqa: E402
import r52_run as R52                                           # noqa: E402

OUT = os.path.expanduser("~/rizu/r53_data")
DIAG = "Registration/output/diag/r53"
STRUCT = [NAME_TO_ID[n] for n in ("wall", "floor", "ceiling")]
GRID, OBS_R = 0.1, 0.5
FRACS = ["0p2", "0p4", "0p6", "0p8", "1p0"]


# --------------------------------------------------------------------------- 記録の読み出し
def result_path(src, data, f, target):
    if src == "r50_score_only":
        return os.path.expanduser("~/rizu/r50_data/results/synth/score_only/%s.json" % target)
    if data == "synth" or f == "1p0":
        return RR.path(data, "full", target)
    return os.path.join(R52.OUT, "f" + f, target + ".json")


def cfg_for(data, f, target):
    if data == "synth":
        return RR.config("synth", target, "full")
    return RR.config("real", target, "full") if f == "1p0" else R52.config(target, f)


def stable_top(scores):
    return int(np.argsort(-np.asarray(scores, dtype=float), kind="stable")[0])


# --------------------------------------------------------------------------- 対象の試行
def enumerate_trials():
    """(data, f, target, k) と、その試行の記録（T・P・正誤）を返す。f は synth では '-'"""
    out = []
    for target in RR.SYNTH:
        res = json.load(open(result_path("full", "synth", "-", target)))["rows"]
        out += [("synth", "-", target, k, res[str(k)]) for k in range(RR.n_trials("synth"))]
    for f in FRACS:
        for target in RR.REAL:
            res = json.load(open(result_path("full", "real", f, target)))["rows"]
            out += [("real", f, target, k, res[str(k)]) for k in range(RR.n_trials("real"))]
    return out


_TRUTH = {}


def gt(data, target):
    if (data, target) not in _TRUTH:
        Gs, om, _ = truth(data, target)
        _TRUTH[(data, target)] = (Gs["1"], om)
    return _TRUTH[(data, target)]


def classify(data, target, r):
    """群（S・N・success・other）と、T_c・T_w の番号を記録だけから決める（ρ は使わない段）"""
    G, om = gt(data, target); P = np.asarray(r["P"], dtype=np.float64); d = r["diag"]
    c4 = [ok(per_trial(T, P, G, om, None)) for T in d["after_icp"]]
    c16 = [ok(per_trial(T, P, G, om, None)) for T in d["short"]]
    top = stable_top(d["scores"])
    fail = big(per_trial(r["T_est"], P, G, om, None))
    if any(c4) and not c4[top]:
        grp = "S"
    elif any(c16) and not any(c4):
        grp = "N"
    else:
        grp = "fail_other" if fail else "success"
    return {"group": grp, "fail": bool(fail), "c4": c4, "c16": c16, "top": top,
            "T_est_is_top": bool(np.array_equal(np.asarray(r["T_est"]), np.asarray(d["after_icp"][top])))}


# --------------------------------------------------------------------------- 幾何の量
class Ref:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tau = float(cfg["semantic_icp"]["max_corr_dist"])
        dst = io_utils.load_reference_cloud(cfg)
        self.dst_p = preprocess.prepare(dst, cfg)
        pts, lab = self.dst_p.points, self.dst_p.labels
        self.trees = {c: cKDTree(pts[lab == c]) for c in STRUCT if (lab == c).any()}
        self.struct_mask = np.isin(lab, STRUCT)
        fl = pts[lab == NAME_TO_ID["floor"]]; ce = pts[lab == NAME_TO_ID["ceiling"]]
        self.zlo = float(fl[:, 2].min()); self.zhi = float(ce[:, 2].max()) if len(ce) else None
        self.o = np.floor(fl[:, :2].min(0) / GRID) * GRID - 2 * GRID
        ij = np.floor((fl[:, :2] - self.o) / GRID).astype(int)
        occ = np.zeros(ij.max(0) + 3, dtype=bool); occ[ij[:, 0], ij[:, 1]] = True
        self.occ = binary_dilation(occ, structure=np.ones((3, 3), dtype=bool))
        self.info = {"tau": self.tau, "z_floor_min": self.zlo, "z_ceiling_max": self.zhi, "n_cells_floor": int(occ.sum()),
                     "n_cells_dilated": int(self.occ.sum()), "classes_present": sorted(int(c) for c in self.trees),
                     "n_points": int(len(pts)), "n_struct": int(self.struct_mask.sum())}

    def inside(self, x):
        ij = np.floor((x[:, :2] - self.o) / GRID).astype(int)
        m = (ij >= 0).all(1) & (ij[:, 0] < self.occ.shape[0]) & (ij[:, 1] < self.occ.shape[1])
        res = np.zeros(len(x), dtype=bool)
        res[m] = self.occ[ij[m, 0], ij[m, 1]]
        return res & (x[:, 2] >= self.zlo) & (x[:, 2] <= (self.zhi if self.zhi is not None else -np.inf))

    def nn_same(self, x, lab):
        d = np.full(len(x), np.inf)
        for c, tr in self.trees.items():
            m = lab == c
            if m.any():
                d[m] = tr.query(x[m], k=1, workers=1)[0]
        return d


def quantities(ref, src_p, Tc, Tw, Tgt):
    tau = ref.tau
    sp, sl = src_p.points, src_p.labels
    sm = np.isin(sl, STRUCT)
    out = {}

    def C(T):
        x = metrics.apply_sim3(T, sp[sm])
        far = ref.nn_same(x, sl[sm]) > tau
        return float((ref.inside(x) & far).sum() / max(sm.sum(), 1)), int(ref.inside(x).sum())
    out["C_c"], out["n_inside_c"] = C(Tc); out["C_w"], out["n_inside_w"] = C(Tw)
    out["dC"] = out["C_w"] - out["C_c"]
    # 観測した範囲 O
    xg = metrics.apply_sim3(Tgt, sp)
    rp, rl = ref.dst_p.points[ref.struct_mask], ref.dst_p.labels[ref.struct_mask]
    dO = cKDTree(xg).query(rp, k=1, workers=1)[0]
    O, Ol = rp[dO <= OBS_R], rl[dO <= OBS_R]
    out["n_O"] = int(len(O))

    def V(T):
        x = metrics.apply_sim3(T, sp)
        d = np.full(len(O), np.inf)
        for c in STRUCT:
            ms, mo = sl == c, Ol == c
            if ms.any() and mo.any():
                d[mo] = cKDTree(x[ms]).query(O[mo], k=1, workers=1)[0]
        return float((d <= tau).mean()) if len(O) else None
    out["V_c"], out["V_w"] = V(Tc), V(Tw)
    out["dV"] = None if out["V_c"] is None else out["V_c"] - out["V_w"]
    A = np.asarray(Tw) @ np.linalg.inv(np.asarray(Tc))
    out["S"] = float((ref.nn_same(metrics.apply_sim3(A, O), Ol) <= tau).mean()) if len(O) else None
    out["out_gt"] = float(1.0 - ref.inside(xg).mean())
    out["n_src"], out["n_src_struct"] = int(len(sp)), int(sm.sum())
    return out


# --------------------------------------------------------------------------- 1 試行
_C = {}


def env_for(data, f, target, src_kind):
    key = (data, f, target)
    if key not in _C:
        cfg = cfg_for(data, f, target)
        _C.clear(); _C[key] = (cfg, io_utils.load_source_cloud(cfg), Ref(cfg))
    return _C[key]


def job(a):
    kind, data, f, target, k, Tc_sel, src_kind = a
    cfg, src, ref = env_for(data, f, target, src_kind)
    r = json.load(open(result_path(src_kind, data, f, target)))["rows"][str(k)]
    P = np.asarray(r["P"], dtype=np.float64); d = r["diag"]
    G, om = gt(data, target)
    Tgt = G @ np.linalg.inv(P)
    src_p = preprocess.prepare(metrics.transform_cloud(src, P), cfg)
    rho = lambda T: float(metrics.class_inlier_ratio(src_p, ref.dst_p, np.asarray(T, dtype=np.float64), ref.tau))
    rho4 = [rho(T) for T in d["after_icp"]]
    rec = {"kind": kind, "data": data, "f": f, "target": target, "scan": target.split("__")[0] if data == "real" else target,
           "trial": k, "source": src_kind, "rho4": rho4, "scores_rec": d["scores"]}
    if src_kind != "r50_score_only":                       # 検算：記録の ρ（小数 5 桁）と同じか
        rec["rho4_max_absdiff_vs_rec"] = float(np.max(np.abs(np.round(rho4, 5) - np.asarray(d["scores"], dtype=float))))
    c4 = [ok(per_trial(T, P, G, om, None)) for T in d["after_icp"]]
    if kind == "neg":
        Tc, Tw = Tgt, np.asarray(r["T_est"], dtype=np.float64)
        rec["sel"] = {"Tc": "G P^-1", "Tw": "T_est"}
    elif kind in ("S", "pos"):
        top = stable_top(d["scores"])
        ic = max((i for i in range(4) if c4[i]), key=lambda i: rho4[i])
        Tc, Tw = np.asarray(d["after_icp"][ic]), np.asarray(d["after_icp"][top])
        rec["sel"] = {"Tc": "after_icp[%d]" % ic, "Tw": "after_icp[%d]" % top}
    else:                                                  # N
        c16 = [ok(per_trial(T, P, G, om, None)) for T in d["short"]]
        rho16 = {i: rho(d["short"][i]) for i in range(len(d["short"])) if c16[i]}
        ic = max(rho16, key=rho16.get); iw = int(np.argmax(rho4))
        Tc, Tw = np.asarray(d["short"][ic]), np.asarray(d["after_icp"][iw])
        rec["sel"] = {"Tc": "short[%d]" % ic, "Tw": "after_icp[%d]" % iw}
        rec["rho16_correct"] = rho16
        if "score_short" in d:
            rec["rho16_max_absdiff_vs_rec"] = float(max(abs(rho16[i] - float(d["score_short"][i])) for i in rho16))
    Tc, Tw = np.asarray(Tc, dtype=np.float64), np.asarray(Tw, dtype=np.float64)
    rec.update(quantities(ref, src_p, Tc, Tw, Tgt))
    rec["rho_c"], rec["rho_w"] = rho(Tc), rho(Tw); rec["drho"] = rec["rho_w"] - rec["rho_c"]
    a_, y_, t_, s_ = rel(Tw, Tc, P, om.mean(0))
    rec.update(rot_deg=a_, yaw_deg=y_, trans_m=t_, scale_rel=s_)
    _, _, sw = metrics.decompose_sim3(Tw @ P); _, _, sc = metrics.decompose_sim3(Tc @ P)
    rec["scale_ratio"] = float(sw / sc)
    rec["ref_info"] = ref.info
    return rec


def run(jobs, name, npar):
    os.makedirs(OUT, exist_ok=True)
    jobs = sorted(jobs, key=lambda a: (a[1], a[2], a[3], a[4]))
    with Pool(npar) as pool:
        recs = []
        for i, rec in enumerate(pool.imap(job, jobs, chunksize=1)):
            recs.append(rec)
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                json.dump({"provenance": provenance(), "rows": recs}, open(os.path.join(OUT, name + ".json"), "w"), default=float)
                print(name, i + 1, len(jobs), flush=True)


def sel_controls():
    trials = enumerate_trials()
    neg = []
    by = {}
    for data, f, target, k, r in trials:
        c = classify(data, target, r)
        if not c["fail"]:
            by.setdefault((data, f), []).append((target, k))
    picked = {}
    for key in sorted(by):
        lst = sorted(by[key]); n = len(lst)
        idx = np.random.default_rng(0).choice(n, size=min(n, 100), replace=False)
        picked["%s|%s" % key] = {"n_success": n, "n_picked": len(idx)}
        neg += [("neg", key[0], key[1], lst[i][0], lst[i][1], None, "full") for i in sorted(idx)]
    r51 = json.load(open(os.path.expanduser("~/rizu/r51_data/stage0_trials.json")))
    pos_all = sorted((r["target"], r["trial"]) for r in r51
                     if r["method"] == "score_only" and r["data"] == "synth" and r["fail"] and r["drho"] == 0)
    idx = np.random.default_rng(0).choice(len(pos_all), size=min(len(pos_all), 100), replace=False)
    pos = [("pos", "synth", "-", pos_all[i][0], pos_all[i][1], None, "r50_score_only") for i in sorted(idx)]
    picked["pos"] = {"n_candidates": len(pos_all), "n_picked": len(idx)}
    return neg, pos, picked


def sel_main():
    jobs, counts = [], {}
    for data, f, target, k, r in enumerate_trials():
        c = classify(data, target, r)
        if c["group"] in ("S", "N"):
            jobs.append((c["group"], data, f, target, k, None, "full"))
            key = "%s|%s|%s" % (c["group"], data, f); counts[key] = counts.get(key, 0) + 1
    return jobs, counts


if __name__ == "__main__":
    mode = sys.argv[1]; npar = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    os.makedirs(DIAG, exist_ok=True)
    if mode == "check":                                    # 1 試行ずつ、検算だけ（ρ の再計算が記録と合うか・内側の範囲）
        for a in [("neg", "real", "1p0", "m3_cor_a__E2", 0, None, "full"), ("neg", "real", "0p6", "m3_cor_b__E3", 0, None, "full"),
                  ("neg", "synth", "-", "room_0", 0, None, "full")]:
            rec = job(a)
            print({k: rec[k] for k in ("data", "f", "target", "rho4", "scores_rec", "rho4_max_absdiff_vs_rec", "C_c", "C_w", "V_c", "V_w",
                                       "S", "n_O", "out_gt", "n_src_struct")}, rec["ref_info"])
    elif mode == "controls":
        neg, pos, picked = sel_controls()
        json.dump({"provenance": provenance(), "picked": picked}, open(os.path.join(DIAG, "controls_picked.json"), "w"), indent=1)
        print(picked, flush=True)
        run(neg + pos, "controls", npar)
    elif mode == "main":
        jobs, counts = sel_main()
        json.dump({"provenance": provenance(), "counts": counts}, open(os.path.join(DIAG, "main_picked.json"), "w"), indent=1)
        print(counts, flush=True)
        run(jobs, "main", npar)

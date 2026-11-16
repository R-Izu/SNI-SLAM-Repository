"""R63 §7-2 の 3：条件 A の本手法を ConSLAM で回す（sni-slam の環境、ブランチ `exp/r63`）。**回す前に commit する。**

- 条件 A（R55 と同じ考え方：全方式に同じ情報）：入力は R62 で STSL のために作った水平の入力と同じ作り方 q = A_P · Rg · p
  （Rg は正解 A の上向きを e_z に移す回転、A_P は摂動 P から取ったヨーと水平の並進だけ。`r62_stsl.py inputs` と同じ式）。
  ここでは意味ラベル付きの入力（`~/rizu/conslam/inputs/b2_f<f>.ply`）に P_eff = A_P · Rg を `transform_cloud` でかける
- 方式：R55 の条件 A の本手法の設定。R62 の config（パラメータは変えない）に、`proposed_fixed_scale`（縮尺 1）と
  `proposed.plan_correlate: {scale_lo: 1, scale_hi: 1, scale_n: 1}`（`r46_real_sni.make(…, "planA_fixed")` と同じ）。
  採点だけクラスなし・ICP だけクラスなしは、さらに `ablation.score_single_class`・`ablation.icp_single_class`（`_with_ablation` で合わさる）
- 摂動 P：`r49_run.perturbations("real", "m3_cor_a__E2")` の k 0〜9
- 誤差（実行の時点で記録）：`per_trial(T_est, P_eff, A, Ω_b2)`（T は q → IFC、正解は A · P_eff⁻¹）
- 段階：s5（full・f 1p0 の 40 試行）→ 基準内 0/40 なら止める（§7-2 の S5）→ rest（full の f 0p2〜0p8 の 160、score_only・icp_only の f 1p0・0p6 の 160）
出力：`~/rizu/r63_data/resultsA/<方式>/b2_f<f>__<参照>.json`
使い方：python r63_condA.py s5 [並列数] ／ rest [並列数]
"""
import copy, json, os, sys, time
from multiprocessing import Pool
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
for p in ("Registration", "Registration/scripts", "Registration/scripts/r49"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
from regbim import io_utils, metrics                           # noqa: E402
from regbim.methods import get_method                          # noqa: E402
from failure_decomposition import provenance                   # noqa: E402
from r41_rescore import per_trial                              # noqa: E402
from r36_rotation_diag import tilt_yaw                         # noqa: E402
import r49_run as RR                                           # noqa: E402

OUT = os.path.expanduser("~/rizu/r63_data/resultsA")
CO = os.path.expanduser("~/rizu/conslam/out")
REFS = ["S1_small", "S1_large", "S3_small", "S3_large"]
A = np.array(json.load(open(os.path.join(CO, "extra.json")))["b_normalized"]["A"])
Rg = np.array(json.load(open(os.path.join(CO, "r62_stsl.json")))["Rg"])
OM = np.load(os.path.join(CO, "r62_omega.npz"))["omega_b2"]
PS = [np.asarray(P, dtype=np.float64) for P in RR.perturbations("real", "m3_cor_a__E2")]


def P_eff(k):
    R, t, s = metrics.decompose_sim3(Rg @ PS[k] @ Rg.T)
    psi = tilt_yaw(R, np.eye(3))["yaw_deg"]; c, sn = np.cos(np.radians(psi)), np.sin(np.radians(psi))
    AP = np.eye(4); AP[:3, :3] = [[c, -sn, 0], [sn, c, 0], [0, 0, 1]]; AP[:2, 3] = t[:2]
    return AP @ Rg


def jobs(stage):
    if stage == "s5":
        return [("full", "1p0", r, k) for r in REFS for k in range(10)]
    js = [("full", f, r, k) for f in ("0p2", "0p4", "0p6", "0p8") for r in REFS for k in range(10)]
    return js + [(m, f, r, k) for m in ("score_only", "icp_only") for f in ("1p0", "0p6") for r in REFS for k in range(10)]


def config(m, f, ref):
    cfg = yaml.safe_load(open("Registration/configs/conslam/b2_f%s__%s.yaml" % (f, ref)))
    p = cfg.setdefault("proposed", {}); p["translation_init"] = "plan_correlate"; p["plan_correlate"] = {"scale_lo": 1.0, "scale_hi": 1.0, "scale_n": 1}
    if m == "score_only":
        cfg["ablation"] = {"score_single_class": True}
    elif m == "icp_only":
        cfg["ablation"] = {"icp_single_class": True}
    return cfg


def path(m, f, ref):
    return os.path.join(OUT, m, "b2_f%s__%s.json" % (f, ref))


_C = {}


def job(a):
    m, f, ref, k = a
    if (f, ref) not in _C:
        cfg = config("full", f, ref)
        _C.clear(); _C[(f, ref)] = (io_utils.load_source_cloud(cfg), io_utils.load_reference_cloud(cfg))
    src, dst = _C[(f, ref)]
    Pe = P_eff(k); meth = get_method("proposed_fixed_scale")
    t0 = time.perf_counter(); T = meth.register(metrics.transform_cloud(src, Pe), dst, config(m, f, ref)); dt = time.perf_counter() - t0
    pd, yd = meth.last_plan_diag, meth.last_yaw_diag
    diag = {"gen": [c["T_gen"] for c in pd["stage_candidates"]], "short": [c["T_short"] for c in pd["stage_candidates"]],
            "after_icp": yd["candidate_T"], "scores": yd["candidate_scores"]}
    e = per_trial(np.asarray(T), Pe, A, OM, None)
    return a, {"T_est": np.asarray(T).tolist(), "P": Pe.tolist(), "P_orig": PS[k].tolist(), "time_s": dt, "diag": diag,
               "run_eO": float(e["eO"]), "run_eR": float(e["eR"]), "run_v": bool(e["v"])}


def main():
    stage = sys.argv[1]; npar = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    have, todo = {}, []
    for a in jobs(stage):
        key = a[:3]
        if key not in have:
            p = path(*key); have[key] = json.load(open(p)) if os.path.exists(p) else {"provenance": provenance(), "rows": {}}
        if str(a[3]) not in have[key]["rows"]:
            todo.append(a)
    todo.sort(key=lambda a: (a[1], a[2], a[0], a[3])); print("todo", len(todo), flush=True); t0 = time.time()
    with Pool(npar) as pool:
        for i, (a, r) in enumerate(pool.imap_unordered(job, todo, chunksize=1)):
            key = a[:3]; have[key]["rows"][str(a[3])] = r
            os.makedirs(os.path.dirname(path(*key)), exist_ok=True); json.dump(have[key], open(path(*key), "w"))
            print("done", i + 1, len(todo), a, "%.0fs" % r["time_s"], "eO %.3f eR %.1f" % (r["run_eO"], r["run_eR"]), "elapsed %.0fs" % (time.time() - t0), flush=True)
    if stage == "s5":
        ok = sum(bool(r["run_v"] and r["run_eR"] < 45 and r["run_eO"] < 1.0) for ref in REFS for r in json.load(open(path("full", "1p0", ref)))["rows"].values())
        print("S5 within-line", ok, "/ 40", "STOP" if ok == 0 else "continue")


if __name__ == "__main__":
    main()

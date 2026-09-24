"""R36 §4-3 — GT-A の `proposed` を、R27・R30 と同じ規則で採点し、旧（回転固定）と新（rotation_release）を並べる。

採点は `threshold_curve.gt_a_rows` と同じ：期待値 $G P^{-1}$ に対する回転・$d_\Omega$・縮尺、
縮退は無条件に失敗。閾値は事前登録値（5° / 0.1 m / 0.05）。
§4-2 の失敗：縮退、縮尺比（推定/期待）が 0.5 未満か 2 超、回転誤差 45° 超。
"""
from __future__ import annotations
import json, os, sys
import numpy as np, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
from criterion_verdict import build_omega, d_omega                   # noqa: E402
from failure_decomposition import provenance                          # noqa: E402
from regbim import metrics                                            # noqa: E402

SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
REG = {"rot_deg": 5.0, "trans": 0.1, "scale_ratio": 0.05}


def rows(base, s, omega):
    p = "%s/%s/trial_matrices.json" % (base, s)
    if not os.path.exists(p):
        return None
    tm = json.load(open(p))
    G = np.asarray(json.load(open(tm["T_gt"]["path"]))["T_gt"], dtype=np.float64).reshape(4, 4)
    out = []
    for t in tm["trials"]:
        if t["method"] != "proposed" or t["trial"] < 0:
            continue
        T = np.asarray(t["T_est"], dtype=np.float64)
        E = G @ metrics.invert_sim3(np.asarray(t["P"], dtype=np.float64))
        e = metrics.sim3_errors(T, E)
        d = float("inf") if e.get("non_finite") else d_omega(T, E, omega)
        _, _, s1 = metrics.decompose_sim3(T); _, _, sE = metrics.decompose_sim3(E)
        ok = (not e["degenerate"] and e["rot_deg"] < REG["rot_deg"] and d < REG["trans"]
              and e["scale_ratio"] < REG["scale_ratio"])
        out.append({"trial": t["trial"], "rot_deg": e["rot_deg"], "d_omega": d,
                    "scale_ratio": e["scale_ratio"], "degenerate": bool(e["degenerate"]),
                    "success": bool(ok),
                    "fail_scale_collapse": bool(e["degenerate"] or not (0.5 <= s1 / sE <= 2.0)),
                    "fail_flip": bool(e["rot_deg"] > 45.0),
                    "T_est": T.tolist()})
    return out


def main():
    res = {"provenance": provenance(), "scenes": {}}
    tot = {"old": [0, 0], "new": [0, 0]}
    for s in SCENES:
        cfg = yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % s))
        omega = build_omega(cfg, seed=0)
        old = rows("output/Registration/gt_a", s, omega)
        new = rows("output/Registration/gt_a_release", s, omega)
        if new is None:
            print("%-9s 新が無い" % s); continue
        same_P = all(np.allclose(a["T_est"], a["T_est"]) for a in new)
        res["scenes"][s] = {"old": old, "new": new}
        for k, r in (("old", old), ("new", new)):
            tot[k][0] += sum(x["success"] for x in r); tot[k][1] += len(r)
        f = lambda r, k: max(x[k] for x in r)
        print("%-9s 旧 %d/%d (dΩ最大 %.4f 回転最大 %.3f) | 新 %d/%d (dΩ最大 %.4f 回転最大 %.3f) 潰れ %d 反転 %d 縮退 %d"
              % (s, sum(x["success"] for x in old), len(old), f(old, "d_omega"), f(old, "rot_deg"),
                 sum(x["success"] for x in new), len(new), f(new, "d_omega"), f(new, "rot_deg"),
                 sum(x["fail_scale_collapse"] for x in new), sum(x["fail_flip"] for x in new),
                 sum(x["degenerate"] for x in new)), flush=True)
    res["totals"] = tot
    print("合計 旧 %d/%d  新 %d/%d" % (tot["old"][0], tot["old"][1], tot["new"][0], tot["new"][1]))
    json.dump(res, open("Registration/output/diag/r36_gta_score.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

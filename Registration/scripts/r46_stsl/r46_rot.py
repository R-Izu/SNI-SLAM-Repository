"""R46 §9-3：回転への依存の確認（小さく）。**パラメータは変えない。**

inputs：陽性対照と同じ入力（R39 §5、410＋411、誤差なし）に、並進なし・ヨーだけ 0°〜350° を 10° ごと（36 通り）。
回し方：`r46_stsl_pipeline.py rot <A|B> 5 M3_410_411`（設定 A・B、K = 5）
summary：ヨーごとに、source 側で取れた角の数、候補の数、STSL の判定、単位行列（候補なし）の件数。図 `rot_corners.png`

使い方：python r46_rot.py inputs   （sni-slam の環境）
        python r46_rot.py summary  （libim か plotconda の環境）
"""
import json, os, re, sys
import numpy as np

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
YAWS = list(range(0, 360, 10))


def inputs():
    P0 = np.load(os.path.expanduser("~/rizu/SNI-SLAM/output/R39_BIM/bim_points_bimframe.npz"))["points"].astype(np.float64)
    out = os.path.join(ROOT, "inputs", "rot"); os.makedirs(out, exist_ok=True)
    spec = {"source": "output/R39_BIM/bim_points_bimframe.npz", "trials": []}
    for i, y in enumerate(YAWS):
        c, s = np.cos(np.radians(y)), np.sin(np.radians(y))
        A = np.eye(4); A[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        np.save(os.path.join(out, "%06d.npy" % i), (P0 @ A[:3, :3].T).astype(np.float32))
        spec["trials"].append({"index": i, "yaw_deg": float(y), "t_xy": [0.0, 0.0], "T_true": np.linalg.inv(A).tolist()})
    json.dump(spec, open(os.path.join(out, "spec.json"), "w"), indent=1)


def jl(line):
    return json.loads(re.sub(r"(?<![A-Za-z])(-?)inf(?![A-Za-z])", lambda m: m.group(1) + "Infinity", line).replace("nan", "NaN"))


def summary():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, st in zip(axes, "AB"):
        rows = {k: [jl(l) for l in open(os.path.join(ROOT, "results", "rot", "%s_k%d_M3_410_411.jsonl" % (st, k)))] for k in range(1, 6)}
        per = []
        for i, y in enumerate(YAWS):
            rs = [rows[k][i] for k in range(1, 6)]
            per.append({"yaw": y, "corners": [r["n_corners"] for r in rs], "n_cand": [len(r["candidates"]) for r in rs],
                        "success_stsl": [r["success_stsl"] for r in rs], "no_candidate": [len(r["candidates"]) == 0 for r in rs]})
        res[st] = per
        cm = np.array([np.median(p["corners"]) for p in per]); ok = np.array([sum(p["success_stsl"]) >= 3 and sum(p["no_candidate"]) < 3 for p in per])
        ax.bar(YAWS, cm, width=8, color=np.where(ok, "#2a78d6", "#eb6834"))
        ax.set_xlabel("ヨー [°]" if False else "yaw [deg]"); ax.set_title("setting %s: corners (median of 5); blue = registered, orange = not" % st, fontsize=8)
        print(st, "registered (majority, with candidates) %d/36; no-candidate (majority) %d/36; corners median range %d-%d"
              % (ok.sum(), sum(sum(p["no_candidate"]) >= 3 for p in per), cm.min(), cm.max()))
    axes[0].set_ylabel("corners")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "results", "rot", "rot_corners.png"), dpi=110)
    json.dump(res, open(os.path.join(ROOT, "results", "rot", "rot_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    {"inputs": inputs, "summary": summary}[sys.argv[1]]()

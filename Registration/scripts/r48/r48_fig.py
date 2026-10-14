"""R48 段階1 の図：参照ごとの BIM 側の角の数（横、対数）と大きな失敗の割合（縦）。方式ごと・範囲ごと。結果を読む語は書かない。"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ("/mnt/c/Windows/Fonts/meiryo.ttc",):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Meiryo", "DejaVu Sans"]; plt.rcParams["font.size"] = 9

d = json.load(open(os.path.expanduser("~/rizu/SNI-SLAM/Registration/output/diag/r48/r48_score.json")))
rates = d["main"]["rates"]
CUTS = [("0p5", "r = 0.5 m", "#2a6fdb", "o"), ("3", "r = 3 m", "#e08a00", "s"), ("inf", "切らない（複数部屋）", "#555555", "D")]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
rng = np.random.default_rng(0)
for ax, (m, title) in zip(axs, [("STSL", "STSL（設定 A、K = 5 の多数決）"), ("planA", "案A"), ("centroid", "重心（参考）")]):
    for x0 in (12.5, 30.5):
        ax.axvline(x0, color="#bbbbbb", lw=0.8, ls="--")
    for cut, lab, col, mk in CUTS:
        xs, ys = [], []
        for r, e in d["refs"].items():
            if r.startswith("GTA_") or cut not in e["cells"]:
                continue
            xs.append(e["n_corners_bim"]); ys.append(rates["%s|%s" % (r, cut)][m])
        j = np.exp(rng.uniform(-0.04, 0.04, len(xs)))
        ax.scatter(np.asarray(xs) * j, ys, s=28, marker=mk, facecolor="none", edgecolor=col, lw=1.3, label=lab)
    gx = [e["n_corners_bim"] for r, e in d["refs"].items() if r.startswith("GTA_")]
    gy = [rates["%s|inf" % r][m] for r in d["refs"] if r.startswith("GTA_")]
    ax.scatter(np.asarray(gx) * np.exp(rng.uniform(-0.04, 0.04, len(gx))), gy, s=34, marker="^", color="#d94a2b", label="GT-A（1 部屋、切らない）")
    ax.set_xscale("log"); ax.set_xticks([1, 3, 10, 30, 100]); ax.set_xticklabels(["1", "3", "10", "30", "100"])
    ax.set_xlabel("参照の BIM 側の角の数（破線：区分の境 12・30）"); ax.set_title(title, loc="left")
    ax.grid(axis="y", color="#eeeeee")
axs[0].set_ylabel("大きな失敗の割合（参照 × 範囲ごと、20 試行）")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(sys.argv[1], dpi=100)

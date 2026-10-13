"""R47 B-2 の図：シーン×切り出しごとに、ヨー ψ（横）と source 側の角の数の中央値（縦）、STSL・案A の大きな失敗の印。結果を読む語は書かない。"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ("/mnt/c/Windows/Fonts/meiryo.ttc",):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Meiryo", "DejaVu Sans"]
d = json.load(open("Registration/output/diag/r47/r47_score.json"))["b2"]["rows"]
scenes = sorted({r["scene"] for r in d}); cuts = ["0p5", "inf"]
fig, axs = plt.subplots(len(cuts), len(scenes), figsize=(16, 6.5), sharex=True, sharey=True)
for i, c in enumerate(cuts):
    for j, s in enumerate(scenes):
        ax = axs[i, j]; rs = sorted([r for r in d if r["scene"] == s and r["cut"] == c], key=lambda r: r["yaw"])
        for x in (0, 90, 180, 270):
            ax.axvspan(x - 5, x + 5, color="#e8e8e8", lw=0)
        y = [r["yaw"] for r in rs]; n = [r["source_corners_median"] for r in rs]
        ax.plot(y, n, color="#888888", lw=1, zorder=1)
        ok = [r for r in rs if not r["stsl_fail"]]; ng = [r for r in rs if r["stsl_fail"]]
        ax.scatter([r["yaw"] for r in ok], [r["source_corners_median"] for r in ok], s=22, facecolor="white", edgecolor="#2a6fdb", lw=1.2, zorder=2, label="STSL 成功")
        ax.scatter([r["yaw"] for r in ng], [r["source_corners_median"] for r in ng], s=26, color="#d94a2b", marker="x", lw=1.6, zorder=3, label="STSL 大きな失敗")
        pa = [r for r in rs if r["planA_fail"]]
        ax.scatter([r["yaw"] for r in pa], [-1.5] * len(pa), s=30, color="#7a3fbf", marker="v", zorder=3, label="案A 大きな失敗")
        ax.set_title("%s  r=%s   STSL 失敗 %d/36・案A 失敗 %d/36" % (s, c.replace("p", "."), len(ng), len(pa)), fontsize=9)
        ax.set_xticks(range(0, 360, 90)); ax.grid(axis="y", color="#eeeeee")
        if j == 0: ax.set_ylabel("source 側の角の数（K=5 の中央値）")
        if i == len(cuts) - 1: ax.set_xlabel("入力のヨー ψ [°]（灰の帯：dev < 5°）")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False, fontsize=10)
fig.suptitle("R47 B-2：入力の向き・source 側の角の数・大きな失敗（STSL both_A、案A）", y=0.995, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(sys.argv[1], dpi=110)

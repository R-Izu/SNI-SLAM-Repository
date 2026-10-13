"""R46 段階1の図（`r46_score.json` と `rot_summary.json` だけを読む。採点はしない）。

1. 同時到達割合 F(a=5°, b, c=0.05)（G1、e_Ω）を、切り出しの段ごとに方式で比べる。STSL は採った設定の全 5 回（無効は失敗に数える）
2. 回転への依存（§9-3）：ヨーごとの角の数（5 回の中央）と、登録できたか
実行：~/plotconda/bin/python r46_plots.py <出力ディレクトリ>
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in ("/mnt/c/Windows/Fonts/arial.ttf", "/home/student/plotconda/meiryo_ui.ttf"):
    fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Arial", "Meiryo UI"]; plt.rcParams["font.size"] = 8
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
GRID = "#e1e0d9"
COL = {"planA": "#2a78d6", "planA_fixed": "#1baf7a", "centroid": "#898781", "STSL_best": "#eb6834"}
LAB = {"planA": "案A", "planA_fixed": "案A・縮尺固定", "centroid": "重心（参考）", "STSL_best": "STSL（条件ごとに良い方の設定、全 5 回）"}
CUTL = {"0p5": "r = 0.5 m", "3": "r = 3 m", "8": "r = 8 m", "inf": "切らない"}

d = json.load(open(os.path.expanduser("~/rizu/SNI-SLAM/Registration/output/diag/r46/r46_score.json")))
B = np.logspace(np.log10(0.01), np.log10(30.0), 61)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for ax, cut in zip(axes, ("0p5", "3", "8", "inf")):
    for m in ("centroid", "planA_fixed", "planA", "STSL_best"):
        ax.plot(B, d["curves_G1"][cut][m]["5"], color=COL[m], lw=1.6, label=LAB[m], drawstyle="steps-post")
    ax.set_xscale("log"); ax.set_xlim(0.01, 30); ax.set_ylim(-0.02, 1.02); ax.grid(True, color=GRID, lw=0.6)
    ax.axvline(1.0, color="#52514e", lw=0.8, ls=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("参照の床から " + CUTL[cut] if cut != "inf" else "切らない", fontsize=9, loc="left")
    ax.set_xlabel("距離の閾値 b [m]（e_Ω、G1）")
axes[0].set_ylabel("同時到達割合 F(5°, b, 0.05)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
fig.suptitle("R46 段階1：廊下 8 条件（E2・E3）× 10 試行。案A・重心は各 80 試行、STSL は 400 回（候補なし・前処理の失敗は失敗に数える）。"
             "縮尺 < 0.05 の断面。点線 = 1 m（大きな失敗の線）", fontsize=8.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.08, 1, 0.92)); fig.savefig(os.path.join(OUT, "r46_curves.png"), dpi=130); plt.close(fig)

r = json.load(open(os.path.expanduser("~/rizu/r46_stsl_data/results/rot/rot_summary.json")))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
for ax, st in zip(axes, "AB"):
    yaw = [p["yaw"] for p in r[st]]; cm = [np.median(p["corners"]) for p in r[st]]
    ok = [sum(p["success_stsl"]) >= 3 and sum(p["no_candidate"]) < 3 for p in r[st]]
    ax.bar(yaw, cm, width=8, color=["#2a78d6" if o else "#eb6834" for o in ok])
    for y, c, o in zip(yaw, cm, ok):
        if c == 0:
            ax.plot([y], [0.15], "v", color="#eb6834", ms=5, mec="none")
    ax.set_xticks(range(0, 360, 30)); ax.set_xlabel("入力のヨー [°]（並進なし）")
    ax.set_title("設定 %s（%s）" % (st, "2f_office_02" if st == "A" else "1f_office_03"), fontsize=9, loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("source 側で取れた角の数（5 回の中央）")
fig.suptitle("R46 §9-3：陽性対照と同じ入力（410＋411、誤差なし）をヨーだけ回す。青 = 登録できた（5 回中 3 回以上）、"
             "橙 = できなかった（▼ = 角 0）", fontsize=8.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(OUT, "r46_rotation.png"), dpi=130); plt.close(fig)
for f in sorted(os.listdir(OUT)):
    print(f, os.path.getsize(os.path.join(OUT, f)) // 1024, "KB")

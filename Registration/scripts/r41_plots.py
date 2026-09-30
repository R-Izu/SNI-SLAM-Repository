"""R41 の図。`r41_rescore.json` だけを読む（採点はしない）。

- 曲線：同時到達割合 F(a, b, c=0.05)。横軸 b（対数）、線 = 回転の閾値 a（5 本）。縮尺 0.05 の断面
- 実データ：e_Ω と e_ΩBIM を別パネル。灰色の帯 = G1〜G8 の最小〜最大（a = 5° のみ）＝基準感度帯（信頼帯ではない）
- 累積分布：e_R・e_Ω・q（有効な試行の値。基準は G1 / 解析的な正解）
実行：~/plotconda/bin/python Registration/scripts/r41_plots.py <出力先ディレクトリ>
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

MEIRYO_UI = "/home/student/plotconda/meiryo_ui.ttf"
for f in ("/mnt/c/Windows/Fonts/arial.ttf", MEIRYO_UI):
    fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Arial", "Meiryo UI"]
plt.rcParams["font.size"] = 8

INK2, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"
A_COL = {"1": "#0b3d91", "2": "#2a78d6", "5": "#1baf7a", "10": "#eb6834", "45": "#b8336a"}
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#b8336a", "#6b4fbb", "#c9a227", "#52514e", "#0b3d91"]

J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "diag", "r41", "r41_rescore.json")))
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
B = np.asarray(J["grid"]["b"])
NOTE = "縮尺 < 0.05 に固定した断面。曲線は点ごとの値で、同時帯ではない"


def style(ax, title, ylabel=True):
    ax.set_xscale("log"); ax.set_xlim(0.01, 30); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axvline(0.1, color=MUTED, lw=0.8, ls=":")
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlabel("距離の閾値 b [m]")
    if ylabel:
        ax.set_ylabel("同時到達割合 F(a, b, 0.05)")


def draw_curves(ax, curves, band=None):
    if band is not None:
        ax.fill_between(B, band[0], band[1], color="#c3c2b7", alpha=0.5, lw=0, label="基準感度帯 G1〜G8（a=5°）")
    for a in ("1", "2", "5", "10", "45"):
        ax.plot(B, curves[a], color=A_COL[a], lw=1.4, label="a = %s°" % a, drawstyle="steps-post")


def real_fig(keys, titles, fname, suptitle):
    fig, axes = plt.subplots(2, len(keys), figsize=(max(3.0 * len(keys), 7.0), 5.6), squeeze=False)
    for c, (k, t) in enumerate(zip(keys, titles)):
        e = J["real"][k]; pc = e["per_criterion"]
        for r, key in enumerate(("curve_eO", "curve_eOB")):
            ax = axes[r, c]
            cur = pc["1"][key]
            if cur is None:
                ax.text(0.5, 0.5, "Ω_BIM の保存が無い条件を含む\n（下の別図に 8 条件だけで出す）", ha="center", va="center", transform=ax.transAxes, color=MUTED)
                style(ax, "%s  %s" % (t, "e_ΩBIM"), ylabel=(c == 0)); continue
            allc = np.array([pc[str(j)][key]["5"] for j in range(1, 9)])
            draw_curves(ax, cur, band=(allc.min(0), allc.max(0)))
            style(ax, "%s  %s（N = %d）" % (t, "e_Ω" if r == 0 else "e_ΩBIM", e["N"]), ylabel=(c == 0))
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(suptitle, fontsize=9, x=0.01, ha="left")
    fig.text(0.01, 0.005, NOTE + "。点線 = 0.1 m。\n灰色の帯は基準感度帯（G1〜G8 の最小〜最大）で、信頼帯ではない。重なった線は、後に描いた線（a の大きい方）だけが見える", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95)); fig.savefig(os.path.join(OUT, fname), dpi=130); plt.close(fig)


def ecdf(ax, vals, label, col, xlabel, xlog=True):
    v = np.sort(np.asarray([x for x in vals if x is not None and np.isfinite(x)]))
    if len(v) == 0:
        return
    if xlog:
        v = np.maximum(v, 1e-5)
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=col, lw=1.3, label="%s（n=%d）" % (label, len(v)))
    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylim(0, 1.02); ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def ecdf_fig(groups, fname, suptitle):
    """groups: [(label, rows)]、rows は試行ごとの量の dict のリスト"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    for i, (lab, rows) in enumerate(groups):
        col = CAT[i % len(CAT)]
        ecdf(axes[0], [r["eR"] for r in rows if r.get("o") != "invalid"], lab, col, "回転誤差 e_R [°]")
        ecdf(axes[1], [r["eO"] for r in rows if r.get("o") != "invalid"], lab, col, "e_Ω [m]")
        ecdf(axes[2], [r["q"] for r in rows if r.get("o") != "invalid"], lab, col, "縮尺の相対誤差 q")
    axes[0].set_ylabel("累積割合（有効な試行）")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=4)
    fig.suptitle(suptitle, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06 * (1 + (len(groups) - 1) // 4), 1, 0.93)); fig.savefig(os.path.join(OUT, fname), dpi=130); plt.close(fig)


# ---- D1・D2・D3（実データ）
real_fig(["D1|default|"], ["D1 実データ・既定 40 条件"], "r41_D1_curves.png", "D1：実データ・既定（R30 の 40 条件）。基準 G1")
real_fig(["D2|default|", "D2|planA|"], ["D2 既定", "D2 案A"], "r41_D2_curves.png", "D2：実データ・R35 の 8 条件（既定と案A）。基準 G1")
real_fig(["D3|off|S", "D3|on|S", "D3|off|P", "D3|on|P"], ["D3 S・off", "D3 S・on", "D3 P・off", "D3 P・on"], "r41_D3_curves.png",
         "D3：実データ・R37 の 160 対（8 条件 × S 10・P 10）。off/on は同じ試行の対。基準 G1")

# D1 の e_ΩBIM（Ω_BIM の保存がある 8 条件だけ）— D1 の cor 8 条件の点は D2 既定と同じ出力
pts = [p for p in J["real"]["D1|default|"]["all_points"] if p["eOB"] is not None]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for i, p in enumerate(sorted(pts, key=lambda p: p["eOB"])):
    ax.plot([p["eOB"]], [i], "o", color=CAT[0], ms=5, mec="none")
    ax.plot([p["eO"]], [i], "o", color=CAT[1], ms=5, mec="none")
    ax.text(0.012, i, p["stratum"].replace("m3_", ""), va="center", fontsize=6.5, color=INK2)
ax.plot([], [], "o", color=CAT[0], mec="none", label="e_ΩBIM"); ax.plot([], [], "o", color=CAT[1], mec="none", label="e_Ω")
ax.set_xscale("log"); ax.set_xlim(0.01, 30); ax.set_yticks([]); ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.set_xlabel("[m]（基準 G1）"); ax.grid(True, axis="x", color=GRID, lw=0.6)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.set_title("D1 のうち Ω_BIM がある 8 条件（全点）", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "r41_D1_omega_bim_points.png"), dpi=130); plt.close(fig)

ap = lambda k: J["real"][k].get("all_points")
ecdf_fig([("D1 既定", ap("D1|default|")), ("D2 既定", ap("D2|default|")), ("D2 案A", ap("D2|planA|"))], "r41_D1_D2_ecdf.png",
         "D1・D2：全点の累積分布（基準 G1）。D2 既定の 8 点は D1 の cor 8 条件と同じ出力")

# ---- D4（シーン等重みの全体。シーン別は JSON）
meths = list(J["D4_pooled_equal_scene_weight"])
fig, axes = plt.subplots(2, 4, figsize=(12, 5.6))
for ax, m in zip(axes.flat, meths):
    draw_curves(ax, J["D4_pooled_equal_scene_weight"][m])
    style(ax, "%s（8 シーン × 100、シーン等重み）" % m, ylabel=(ax in axes[:, 0]))
axes[0, 0].legend(fontsize=7, frameon=False, loc="center left")
fig.suptitle("D4：合成 GT-A（R27 の全手法）。解析的な正解・元の Ω", fontsize=9, x=0.01, ha="left")
fig.text(0.01, 0.005, NOTE + "。点線 = 0.1 m。重なった線は、後に描いた線（a の大きい方）だけが見える", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(os.path.join(OUT, "r41_D4_curves.png"), dpi=120); plt.close(fig)

TR = J["trials_real"]; TS = J["trials_synth"]
g1 = lambda cond: [r["per"]["1"] for r in TR if cond(r)]
ecdf_fig([("S・off", g1(lambda r: r["dataset"] == "D3" and r["series"] == "S" and r["method"] == "off")),
          ("S・on", g1(lambda r: r["dataset"] == "D3" and r["series"] == "S" and r["method"] == "on")),
          ("P・off", g1(lambda r: r["dataset"] == "D3" and r["series"] == "P" and r["method"] == "off")),
          ("P・on", g1(lambda r: r["dataset"] == "D3" and r["series"] == "P" and r["method"] == "on"))],
         "r41_D3_ecdf.png", "D3：R37 の 160 対の累積分布（基準 G1）。各 80 = 8 条件 × 10")
ecdf_fig([(m, [r["per"] for r in TS if r["dataset"] == "D4" and r["method"] == m]) for m in meths], "r41_D4_ecdf.png",
         "D4：合成 GT-A の累積分布（各手法 8 シーン × 100。各シーン同数なので合算＝シーン等重み）")
sel = ["R39_E2_planA", "R39_E3_planA", "R40_zero", "R40_N_1cm", "R40_N_2cm", "R40_N_5cm", "R40_E2p_centroid", "R40_E2p_planA"]
ecdf_fig([(st + ("（交絡）" if "E2p" in st else ""), [r["per"] for r in TS if r["dataset"] == "D5" and r["method"] == "proposed" and r["stratum"] == st]) for st in sel],
         "r41_D5_ecdf.png", "D5：BIM 由来の対照、proposed の累積分布（一部の入力。全入力は JSON）。実データの精度ではない")
# ---- D5（proposed のみの曲線。入力ごと）
d5 = [(k, e) for k, e in J["synth"].items() if k.startswith("D5|proposed|")]
order = ["R39_E2_centroid", "R39_E2_planA", "R39_E3_centroid", "R39_E3_planA", "R40_zero", "R40_N_1cm", "R40_N_2cm", "R40_N_5cm",
         "R40_L_5", "R40_L_10", "R40_L_20", "R40_Obg_10", "R40_Obg_25", "R40_Obg_50", "R40_Owall_10", "R40_Owall_25", "R40_Owall_50",
         "R40_S_0p5", "R40_S_1p0", "R40_S_2p0", "R40_E2p_centroid", "R40_E2p_planA"]
d5 = dict((k.split("|")[2], e) for k, e in d5)
fig, axes = plt.subplots(4, 6, figsize=(15, 9.6))
for ax, st in zip(axes.flat, order):
    e = d5[st]
    draw_curves(ax, e["curve_eO"])
    style(ax, "%s%s（N=%d）" % (st, "  ※交絡" if e["confounded"] else "", e["N"]), ylabel=(ax in axes[:, 0]))
    ax.set_xlabel("")
for ax in list(axes.flat)[len(order):]:
    ax.axis("off")
h, l = axes[0, 0].get_legend_handles_labels()
list(axes.flat)[-1].legend(h, l, fontsize=8, frameon=False, loc="center")
fig.suptitle("D5：BIM 由来の対照（proposed のみ。R39 §5・R40 §4、E2' は R40 §3 で条件が交絡）。理想面・廊下なしの合成入力で、実データの精度ではない", fontsize=9, x=0.01, ha="left")
fig.text(0.01, 0.005, NOTE + "。横軸 = 距離の閾値 b [m]。点線 = 0.1 m。重なった線は、後に描いた線（a の大きい方）だけが見える", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 0.96)); fig.savefig(os.path.join(OUT, "r41_D5_curves.png"), dpi=110); plt.close(fig)
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        print(f, os.path.getsize(os.path.join(OUT, f)) // 1024, "KB")

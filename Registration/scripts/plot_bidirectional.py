"""R14 §3-1 / R16 §3-1 の図を出す。

図1：横軸 rho_P・縦軸 rho_Q（960 候補、正しい候補と誤答を色分け）＋ 縮退解
図2：rho_P 対 s/s_G と rho_Q 対 s/s_G（拡大を報酬するかを見る）
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(REPO)

SC = "Registration/output/diag/bidirectional_scores.json"
DG = "Registration/output/diag/degenerate_rho.json"
OUT = "Registration/output/diag"

recs = json.load(open(SC))["records"]
deg = json.load(open(DG))["cells"]

cor = [r for r in recs if r["is_correct"]]
wrg = [r for r in recs if not r["is_correct"]]
win = [r for r in recs if r["is_winner"]]

fig, ax = plt.subplots(figsize=(7.2, 5.6))
ax.scatter([r["rho_P"] for r in wrg], [r["rho_Q"] for r in wrg],
           s=14, c="#b0b0b0", label="wrong candidate (n=%d)" % len(wrg))
ax.scatter([r["rho_P"] for r in win], [r["rho_Q"] for r in win],
           s=26, facecolors="none", edgecolors="#d95f02", linewidths=1.1,
           label="winner (n=%d)" % len(win))
ax.scatter([r["rho_P"] for r in cor], [r["rho_Q"] for r in cor],
           s=42, c="#1b9e77", marker="^", label="correct candidate (n=%d)" % len(cor))
d_deg = [c for c in deg if c["degenerate"]]
ax.scatter([c["rho_P_method"] for c in d_deg], [c["rho_Q_method"] for c in d_deg],
           s=90, c="#e7298a", marker="X", label="degenerate solution (n=%d)" % len(d_deg))
ax.scatter([c["rho_P_ref"] for c in deg], [c["rho_Q_ref"] for c in deg],
           s=60, c="#7570b3", marker="s", label="full-coverage control (n=%d)" % len(deg))
ax.set_xlabel(r"$\rho_P$  (source-side explained ratio)")
ax.set_ylabel(r"$\rho_Q$  (reference-side explained ratio, denominator $Q_{room}$)")
ax.set_title("Bidirectional scores of all saved candidates\n"
             "correct candidates do not separate on either axis")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_rhoP_rhoQ.png"), dpi=160)

fig2, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
s = np.array([r["s_over_sG"] for r in recs])
for a, tag, lab in ((axes[0], "rho_P", r"$\rho_P$"), (axes[1], "rho_Q", r"$\rho_Q$")):
    v = np.array([r[tag] for r in recs])
    m = np.array([r["is_correct"] for r in recs])
    a.scatter(s[~m], v[~m], s=12, c="#b0b0b0", label="wrong")
    a.scatter(s[m], v[m], s=40, c="#1b9e77", marker="^", label="correct")
    a.set_xlabel(r"$s / s_G$")
    a.set_ylabel(lab)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
axes[0].set_title(r"$\rho_P$ vs scale ratio")
axes[1].set_title(r"$\rho_Q$ vs scale ratio")
fig2.suptitle("Real-data candidates only: s/s_G spans %.3f-%.3f, so this does not "
              "probe the collapse regime" % (s.min(), s.max()), fontsize=9)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "fig_rho_vs_scale.png"), dpi=160)

print("s/s_G の範囲: %.4f 〜 %.4f (n=%d)" % (s.min(), s.max(), len(s)))
print("wrote %s/fig_rhoP_rhoQ.png, %s/fig_rho_vs_scale.png" % (OUT, OUT))

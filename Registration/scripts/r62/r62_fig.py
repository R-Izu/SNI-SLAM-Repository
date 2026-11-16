"""R62 §5 の図（sni-slam の環境）。**回す前に commit する。**
1. `r62_refs_input.png`：参照 4 つ（箱ごとの色）と、入力 b-2 の f 1.0（A で IFC の座標に置いた 200,000 点、灰）を真上から
2. `r62_input_labels.png`：入力 b-2 の f 1.0（normalized の座標のまま）をクラスの色で、真上から（左）と横から（右、x-z）
出力：`~/rizu/conslam/out/`
"""
import json, os
import numpy as np
import open3d as o3d

OUT = os.path.expanduser("~/rizu/conslam/out")
REF = os.path.expanduser("~/rizu/conslam/refs")
PAL = np.array([[128, 128, 128], [255, 64, 64], [255, 200, 64], [180, 220, 255], [64, 200, 255], [200, 100, 255]]) / 255.0
NAMES = ["background", "wall", "door", "floor", "window", "ceiling"]


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    A = np.array(json.load(open(os.path.join(OUT, "extra.json")))["b_normalized"]["A"])
    pc = o3d.io.read_point_cloud(os.path.expanduser("~/rizu/conslam/inputs/b2_f1p0.ply"))
    P = np.asarray(pc.points); C = np.asarray(pc.colors)
    lab = np.argmin(((C[:, None, :] - PAL[None]) ** 2).sum(2), 1)
    X = P @ A[:3, :3].T + A[:3, 3]
    boxes = {"S1_small": (136.5, 83.5, 146.5, 91.5), "S1_large": (136.5, 83.5, 156.5, 91.5),
             "S3_small": (177.5, 82.0, 187.5, 90.0), "S3_large": (167.5, 82.0, 187.5, 90.0)}
    col = {"S1_small": "#D55E00", "S1_large": "#E69F00", "S3_small": "#0072B2", "S3_large": "#56B4E9"}
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(X[:, 0], X[:, 1], s=0.2, c="0.6", label="input b-2 f1.0 (placed by A)")
    for r, b in boxes.items():
        z = np.load(os.path.join(REF, r + ".npz")); p = z["points"]
        ax.scatter(p[::5, 0], p[::5, 1], s=0.2, c=col[r])
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col[r], lw=2, label=r))
    ax.set_aspect("equal"); ax.legend(markerscale=10, loc="upper left"); ax.set_title("R62 references (boxes) and input b-2 f1.0")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r62_refs_input.png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    order = np.argsort(lab == 0)[::-1]
    ax[0].scatter(P[order, 0], P[order, 1], s=0.2, c=PAL[lab[order]]); ax[0].set_aspect("equal"); ax[0].set_title("b-2 f1.0, top (normalized frame)")
    ax[1].scatter(P[order, 0], P[order, 2], s=0.2, c=PAL[lab[order]]); ax[1].set_aspect("equal"); ax[1].set_title("b-2 f1.0, side x-z (normalized frame)")
    fig.legend(handles=[Patch(color=PAL[i], label=NAMES[i]) for i in range(6)], loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(OUT, "r62_input_labels.png"), dpi=100); plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()

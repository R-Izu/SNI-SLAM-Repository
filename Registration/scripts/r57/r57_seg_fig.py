"""R57 §3-4 の図：`r57_seg.py` が保存したラベル（`out/seg20/<時刻>.png`）と元の RGB から、4 枚の重ね合わせを作る（sni-slam の環境）。
seg2d の環境に matplotlib が無いので分けた。4 枚は 20 枚のうち 1・7・14・20 番目（`r57_seg.py` と同じ）。出力 `out/seg20_overlay.png`
"""
import json, os, zipfile
import numpy as np
import cv2

Z = "/mnt/d/rizu/SNI-SLAM/data/ConSLAM/sequence2/Copy of rgb.zip"
OUT = os.path.expanduser("~/rizu/conslam/out")
NAMES = ["background", "wall", "door", "floor", "window", "ceiling"]
PAL = np.array([[128, 128, 128], [255, 64, 64], [255, 200, 64], [180, 220, 255], [64, 200, 255], [200, 100, 255]], dtype=np.uint8)
res = json.load(open(os.path.join(OUT, "seg20.json")))
zf = zipfile.ZipFile(Z)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
fig, ax = plt.subplots(2, 4, figsize=(20, 8))
for j, idx in enumerate([0, 6, 13, 19]):
    stamp = res["per_image"][idx]["stamp"]
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(zf.read("rgb_2058x1533/%s.png" % stamp), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    cls = cv2.imread(os.path.join(OUT, "seg20", stamp + ".png"), cv2.IMREAD_GRAYSCALE)
    ax[0][j].imshow(rgb); ax[0][j].set_title(stamp); ax[0][j].axis("off")
    ax[1][j].imshow((0.5 * rgb + 0.5 * PAL[cls]).astype(np.uint8)); ax[1][j].axis("off")
fig.legend(handles=[Patch(color=PAL[i] / 255, label=NAMES[i]) for i in range(6)], loc="lower center", ncol=6)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(OUT, "seg20_overlay.png"), dpi=80)
print("ok", res["total_frac"], res["n_images_with_class"])

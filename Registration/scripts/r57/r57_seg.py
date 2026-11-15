"""R57 §3-4：意味ラベルの推定を系列 2 の RGB 20 枚に当てる（seg2d の環境）。**回す前に commit する。手で正誤を付けない。**

- 推定：`tools/realdata/gen_labels_ade20k.py` の `AdeSegmenter`（既定のモデル `facebook/mask2former-swin-large-ade-semantic`、
  R2 以後の実データの意味ラベルを作ったもの）と、対応表 `tools/realdata/label_map_ade20k.yaml`（ADE20K の 0 壁・3 床・5 天井・8 窓・14 扉、ほかは背景）
- 画像：`Copy of rgb.zip` の `rgb_2058x1533/*.png` を時刻の順に並べ、`np.linspace(0, n-1, 20)` の位置の 20 枚。回転・歪みの補正はかけない
  （`gen_labels_ade20k.py` の回転の補正は iPad の縦持ちのためのもので、ConSLAM の画像には当てない）
- 出力：クラスごとの画素の割合（1 枚ごとと 20 枚の合計）`out/seg20.json`、ラベルの PNG `out/seg20/<時刻>.png`（画素値は 0〜5 のクラスの番号）、
  4 枚（20 枚のうち 1・7・14・20 番目）の重ね合わせ `out/seg20_overlay.png`
"""
import io, json, os, sys, zipfile
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "tools", "realdata"))
os.chdir(REPO)
from gen_labels_ade20k import AdeSegmenter                     # noqa: E402

Z = "/mnt/d/rizu/SNI-SLAM/data/ConSLAM/sequence2/Copy of rgb.zip"
OUT = os.path.expanduser("~/rizu/conslam/out")
NAMES = ["background", "wall", "door", "floor", "window", "ceiling"]
PAL = np.array([[128, 128, 128], [255, 64, 64], [255, 200, 64], [180, 220, 255], [64, 200, 255], [200, 100, 255]], dtype=np.uint8)


def main():
    import cv2
    lm = yaml.safe_load(open("tools/realdata/label_map_ade20k.yaml"))
    lut = np.zeros(256, dtype=np.uint8)
    for ade, cls in lm["mapping"].items():
        lut[int(ade)] = NAMES.index(cls)
    zf = zipfile.ZipFile(Z)
    names = sorted([n for n in zf.namelist() if n.endswith(".png") and n.count("/") == 1], key=lambda n: int(os.path.basename(n)[:-4]))
    pick = [names[int(round(x))] for x in np.linspace(0, len(names) - 1, 20)]
    seg = AdeSegmenter("facebook/mask2former-swin-large-ade-semantic")
    order = seg.check_label_order()
    os.makedirs(os.path.join(OUT, "seg20"), exist_ok=True)
    res = {"model": "facebook/mask2former-swin-large-ade-semantic", "label_order_check": order, "n_images": len(pick), "per_image": []}
    tot = np.zeros(6); imgs = []
    for n in pick:
        bgr = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ade = seg.predict([rgb])[0]
        cls = lut[np.clip(ade, 0, 255)]
        cnt = np.bincount(cls.ravel(), minlength=6).astype(float)
        tot += cnt
        stamp = os.path.basename(n)[:-4]
        cv2.imwrite(os.path.join(OUT, "seg20", stamp + ".png"), cls)
        res["per_image"].append({"stamp": stamp, "size_hw": list(cls.shape), "frac": {NAMES[i]: float(cnt[i] / cnt.sum()) for i in range(6)}})
        imgs.append((stamp, rgb, cls))
        print(stamp, {NAMES[i]: round(float(cnt[i] / cnt.sum()), 3) for i in range(6)}, flush=True)
    res["total_frac"] = {NAMES[i]: float(tot[i] / tot.sum()) for i in range(6)}
    res["n_images_with_class"] = {NAMES[i]: int(sum(r["frac"][NAMES[i]] > 0 for r in res["per_image"])) for i in range(6)}
    json.dump(res, open(os.path.join(OUT, "seg20.json"), "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    for j, idx in enumerate([0, 6, 13, 19]):
        stamp, rgb, cls = imgs[idx]
        ax[0][j].imshow(rgb); ax[0][j].set_title(stamp); ax[0][j].axis("off")
        ax[1][j].imshow((0.5 * rgb + 0.5 * PAL[cls]).astype(np.uint8)); ax[1][j].axis("off")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=PAL[i] / 255, label=NAMES[i]) for i in range(6)], loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(OUT, "seg20_overlay.png"), dpi=80); plt.close(fig)
    print("total", res["total_frac"])


if __name__ == "__main__":
    main()

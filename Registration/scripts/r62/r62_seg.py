"""R62 §3-1 の 2：使う走査に時刻が最も近い RGB 画像に、意味ラベルを付ける（seg2d の環境）。**回す前に commit する。**

- 使う走査：b-2 は `normalized/` に同じ時刻（μs）の姿勢がある走査、b-1 は S2 の GT に同じ時刻の行がある走査。その和
- 各走査の時刻に最も近い RGB 画像（`Copy of rgb.zip` の `rgb_2058x1533/<μs>.png`）を選ぶ
- 推定：`tools/realdata/gen_labels_ade20k.py` の `AdeSegmenter`（`facebook/mask2former-swin-large-ade-semantic`）と `label_map_ade20k.yaml`（R57 §6 と同じ）。
  元の画像（歪みはそのまま）に当てる。歪みは投影の側で扱う（`r62_build.py`）
- 出力：`~/rizu/conslam/work/seg/<RGB の時刻>.png`（画素値 0〜5）、`~/rizu/conslam/work/scan_rgb.json`（走査の時刻 → RGB の時刻・時間差、どちらの入力で使うか）
"""
import glob, json, os, sys, zipfile
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "tools", "realdata"))
os.chdir(REPO)
from gen_labels_ade20k import AdeSegmenter                     # noqa: E402

D = "/mnt/d/rizu/SNI-SLAM/data/ConSLAM"
W = os.path.expanduser("~/rizu/conslam/work")
NAMES = ["background", "wall", "door", "floor", "window", "ceiling"]


def main():
    import cv2
    lid = sorted(int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(W, "lidar-pcd", "*.pcd")))
    nz = {int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(W, "normalized", "*.txt"))}
    gt = {int(round(t * 1e6)) for t in np.loadtxt(os.path.join(D, "m1743877", "S2_All_LiDAR_poses_v5_denser_T31.tum"))[:, 0]}
    b2 = [s for s in lid if s in nz]; b1 = [s for s in lid if s in gt]
    zf = zipfile.ZipFile(os.path.join(D, "sequence2", "Copy of rgb.zip"))
    rgb = np.array(sorted(int(os.path.basename(n)[:-4]) for n in zf.namelist() if n.endswith(".png") and n.count("/") == 1))
    use = sorted(set(b2) | set(b1))
    m = {}
    for s in use:
        j = int(np.argmin(np.abs(rgb - s)))
        m[str(s)] = {"rgb": int(rgb[j]), "dt_ms": float((rgb[j] - s) / 1000.0), "b2": s in nz, "b1": s in gt}
    json.dump({"n_b2": len(b2), "n_b1": len(b1), "n_both": len(set(b2) & set(b1)), "n_union": len(use), "scans": m},
              open(os.path.join(W, "scan_rgb.json"), "w"), indent=1)
    lm = yaml.safe_load(open("tools/realdata/label_map_ade20k.yaml"))
    lut = np.zeros(256, dtype=np.uint8)
    for ade, cls in lm["mapping"].items():
        lut[int(ade)] = NAMES.index(cls)
    seg = AdeSegmenter("facebook/mask2former-swin-large-ade-semantic"); seg.check_label_order()
    os.makedirs(os.path.join(W, "seg"), exist_ok=True)
    todo = sorted({v["rgb"] for v in m.values()})
    for i, r in enumerate(todo):
        out = os.path.join(W, "seg", "%d.png" % r)
        if os.path.exists(out):
            continue
        bgr = cv2.imdecode(np.frombuffer(zf.read("rgb_2058x1533/%d.png" % r), np.uint8), cv2.IMREAD_COLOR)
        cls = lut[np.clip(seg.predict([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)])[0], 0, 255)]
        cv2.imwrite(out, cls)
        if i % 50 == 0:
            print(i, len(todo), flush=True)
    print("n_b2", len(b2), "n_b1", len(b1), "both", len(set(b2) & set(b1)), "images", len(todo))


if __name__ == "__main__":
    main()

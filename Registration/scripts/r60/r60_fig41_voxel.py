"""R60 §3-5：R59 (B) の点群を voxel で間引いて写す（sni-slam の環境）。**回す前に commit する。**
- 入力 `Registration/output/fig/r59_fig41/`、出力 `Registration/output/fig/r59_fig41_v<voxel>/`（同じ構成）
- 各 PLY を Open3D の `voxel_down_sample(voxel)` で間引く。色は 1 ファイルに 1 色なので、間引いた後も同じ色を入れ直す（青またはオレンジ）
- `transforms.csv`・`preview_top.png` はそのまま写す。`_counts.json` に各ファイルの点の数と、フォルダごとの容量
使い方：python r60_fig41_voxel.py 0.10
"""
import json, os, shutil, sys
import numpy as np
import open3d as o3d

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts", "r59"))
os.chdir(REPO)
from r59_fig41 import write_ply                                  # noqa: E402

SRC = "Registration/output/fig/r59_fig41"


def main(voxel):
    dst = "%s_v%s" % (SRC, ("%.2f" % voxel).replace(".", "p"))
    shutil.rmtree(dst, ignore_errors=True)
    counts = {}
    for folder in sorted(os.listdir(SRC)):
        sp = os.path.join(SRC, folder)
        if not os.path.isdir(sp):
            continue
        os.makedirs(os.path.join(dst, folder))
        for fn in sorted(os.listdir(sp)):
            if fn.endswith(".ply"):
                pc = o3d.io.read_point_cloud(os.path.join(sp, fn))
                rgb = tuple(int(round(v * 255)) for v in np.asarray(pc.colors)[0])
                ds = pc.voxel_down_sample(voxel)
                write_ply(os.path.join(dst, folder, fn), np.asarray(ds.points), rgb)
                counts["%s/%s" % (folder, fn)] = {"before": len(pc.points), "after": len(ds.points), "rgb": rgb}
            else:
                shutil.copy(os.path.join(sp, fn), os.path.join(dst, folder, fn))
    sizes = {f: sum(os.path.getsize(os.path.join(dst, f, x)) for x in os.listdir(os.path.join(dst, f))) for f in os.listdir(dst)}
    json.dump({"voxel": voxel, "counts": counts, "folder_bytes": sizes, "total_bytes": sum(sizes.values())}, open(os.path.join(dst, "_counts.json"), "w"), indent=1)
    print(dst, "total MB", sum(sizes.values()) / 1e6)


if __name__ == "__main__":
    main(float(sys.argv[1]))

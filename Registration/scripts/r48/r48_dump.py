"""R48 段階0：Replica の複数部屋のシーンから、構造 5 クラスの参照の点を取り出す（sni-slam の環境）。**登録はしない。**

GT-A の参照と同じ読み方（`replica_gt`、`replica_to_six`、構造 5 クラス）。点の数は 1,000,000（複数部屋なので GT-A の 200,000 より多く）、
標本化の seed 0。出力：`~/rizu/r48_data/ref/<scene>.npz`（points・labels・normals）
"""
import os, sys
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); os.chdir(REPO)
from regbim import io_utils                                       # noqa: E402

SCENES = ["apartment_0", "apartment_1", "apartment_2", "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
          "frl_apartment_3", "frl_apartment_4", "frl_apartment_5", "hotel_0"]
OUT = os.path.expanduser("~/rizu/r48_data/ref")


def main():
    os.makedirs(OUT, exist_ok=True)
    base = yaml.safe_load(open("Registration/configs/gt_a/room_0.yaml"))
    for s in SCENES:
        if os.path.exists(os.path.join(OUT, s + ".npz")):
            continue
        cfg = dict(base, reference=dict(base["reference"], mesh_path="data/replica/%s/habitat/mesh_semantic.ply" % s,
                                        info_path="data/replica/%s/habitat/info_semantic.json" % s, n_points=1000000, seed=0))
        ref = io_utils.load_reference_cloud(cfg)
        np.savez_compressed(os.path.join(OUT, s + ".npz"), points=ref.points.astype(np.float32), labels=ref.labels.astype(np.int8),
                            normals=ref.normals.astype(np.float32))
        print(s, len(ref.points), np.bincount(ref.labels, minlength=6), "z range %.2f..%.2f" % (ref.points[:, 2].min(), ref.points[:, 2].max()), flush=True)


if __name__ == "__main__":
    main()

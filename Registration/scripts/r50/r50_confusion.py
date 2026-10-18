"""R50 段階0 §3-5：参照のクラスの混同行列（点の数）。元のクラス（行）× 幾何から作ったクラス（列、`io_utils.geometric_reference_classes`）。
参照：GT-A 8 シーン（`gt_a_plan_a/<scene>.yaml`、seed 0）、410＋411（`realdata/m3_cor_a__E3.yaml`）、411（`realdata/m3_cor_a__E2.yaml`）。
IFC の参照は、どのスキャンでも同じ（`cache_path` が同じ）。出力：`Registration/output/diag/r50/confusion.json`
"""
import json, os, sys
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration")); os.chdir(REPO)
from regbim import io_utils                                     # noqa: E402
from regbim.labels import NAME_TO_ID                            # noqa: E402

ROWS = ["background", "wall", "door", "floor", "window", "ceiling"]
COLS = ["background", "wall", "floor", "ceiling"]
REFS = [("GTA_" + s, "Registration/configs/gt_a_plan_a/%s.yaml" % s)
        for s in ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]]
REFS += [("BIM_410_411", "Registration/configs/realdata/m3_cor_a__E3.yaml"), ("BIM_411", "Registration/configs/realdata/m3_cor_a__E2.yaml")]


def main():
    out = {}
    for name, c in REFS:
        cfg = yaml.safe_load(open(c))
        ref = io_utils.load_reference_cloud(cfg)
        g = io_utils.geometric_reference_classes(ref).labels
        M = [[int(((ref.labels == NAME_TO_ID[r]) & (g == NAME_TO_ID[k])).sum()) for k in COLS] for r in ROWS]
        out[name] = {"config": c, "n_points": int(len(ref)), "rows_original": ROWS, "cols_geometric": COLS, "matrix": M}
        print(name, len(ref))
        for r, row in zip(ROWS, M):
            if sum(row):
                print("   %-10s" % r, row)
    os.makedirs("Registration/output/diag/r50", exist_ok=True)
    json.dump(out, open("Registration/output/diag/r50/confusion.json", "w"), indent=1)


if __name__ == "__main__":
    main()

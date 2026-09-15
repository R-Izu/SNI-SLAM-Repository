"""R27 §2-2 — GT-A 用の config を 8 シーンぶん作る。

**sectionC の config をそのまま写し、source と t_gt_path だけ差し替える。**
手法のパラメータ・摂動の範囲・成功閾値は **1 つも変えない**。
変えると「GT 基準にしたから変わった」のか「設定を変えたから変わった」のか分からなくなる。

差し替えるのは3か所だけ：

| 何を | 旧 | 新 |
|---|---|---|
| `source.type` | `slam_mesh` | `points_ply` |
| `source.mesh_path` | SLAM の再構成メッシュ | `output/GT_A/<scene>_source.ply` |
| `eval.t_gt_path` | 無い（7 シーン）／提案手法の解（room_0） | `output/GT_A/T_gt_<scene>.json`（**解析的な正解**） |

`source.seed` と `source.n_points` は**削る**。points_ply は何も標本化しないので、
残すと「決まっている」と誤読される（`io_utils._load_points_ply` が例外で止める）。

    python Registration/scripts/gen_gt_a_configs.py
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(REPO)

SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1",
          "office_2", "office_3", "office_4"]

HEADER = """# GT-A 評価用の config（R27 §2-2。自動生成。手で編集しない）
#
# **sectionC/{scene}.yaml の写しで、source と t_gt_path だけ差し替えてある。**
# 手法のパラメータ・摂動範囲・成功閾値は sectionC と同一。
#
# 正解 T_gt は **我々が掛けた既知の Sim(3) の逆行列**であり、提案手法の出力ではない。
# 作り方と来歴は output/GT_A/{scene}_manifest.json にある。
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="Registration/configs/gt_a")
    ap.add_argument("--gt-a-dir", default="output/GT_A")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    made = []
    for s in SCENES:
        src_ply = os.path.join(args.gt_a_dir, "%s_source.ply" % s)
        t_gt = os.path.join(args.gt_a_dir, "T_gt_%s.json" % s)
        if not (os.path.exists(src_ply) and os.path.exists(t_gt)):
            print("%-10s **飛ばす**（GT-A がまだ無い）" % s)
            continue
        cfg = yaml.safe_load(open("Registration/configs/sectionC/%s.yaml" % s))
        cfg["source"] = {"type": "points_ply", "mesh_path": src_ply}
        cfg["eval"] = dict(cfg["eval"])
        cfg["eval"]["t_gt_path"] = t_gt
        cfg["eval"]["out_dir"] = "output/Registration/gt_a/%s" % s
        p = os.path.join(args.out_dir, "%s.yaml" % s)
        with open(p, "w") as f:
            f.write(HEADER.format(scene=s))
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False,
                           default_flow_style=False)
        made.append(p)
        print("%-10s -> %s" % (s, p))

    print("\n%d 件。摂動・閾値は sectionC と同一であることを確認した:" % len(made))
    if made:
        a = yaml.safe_load(open(made[0]))
        b = yaml.safe_load(open("Registration/configs/sectionC/%s.yaml"
                                % os.path.basename(made[0])[:-5]))
        for k in ("perturb", "success", "trials", "seed"):
            same = a["eval"].get(k) == b["eval"].get(k)
            print("  eval.%-10s %s" % (k, "同一" if same else "**違う**"))
        for k in ("semantic_icp", "rotation", "preprocess", "baseline_open3d"):
            print("  %-16s %s" % (k, "同一" if a.get(k) == b.get(k) else "**違う**"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

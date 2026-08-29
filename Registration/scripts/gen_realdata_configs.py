"""R5 Q8 — 実データ位置合わせの config を作る（source = TSDF メッシュ、reference = IFC）。

**結果は出さない。** GT 位置合わせが先に要るため、ここでやるのは配線と疎通確認だけ
（R5 §7-2 で「結果を出さない前提で承認」）。

条件（R3 指示書の付録）
-----------------------
| 条件 | reference | 見たいこと |
|---|---|---|
| E1 | **410（直方体）** | 北西端の似た部屋へ誤収束するか＝室の取り違え |
| E2 | **411（L字）** | L字は識別性が高い → E1 の対照 |
| E3 | 411 + 410 | 未対応領域を廊下＋対向室のみに減らしたときの誤差 |
| E4 | E1 を**幾何のみ**（意味なし） | 什器・廊下を落とせないと何が起きるか |

E4 は `ablation.single_class` で意味を潰す（既存の仕組み。手法は変えない）。

    conda activate sni-slam
    python Registration/scripts/gen_realdata_configs.py
    python Registration/scripts/gen_realdata_configs.py --smoke   # 読み込みだけ確認
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Dict, List

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))

SCENES = ["m3_room_a", "m3_room_b", "m3_block_a", "m3_block_b", "m3_block_c",
          "m3_block_d", "m3_cor_a", "m3_cor_b", "m3_cor_c", "m3_cor_d"]

# 条件名 -> (spaces, cache 名, 意味を潰すか)
CONDITIONS: Dict[str, tuple] = {
    "E1": (["410"], "m3_ifc_410.npz", False),
    "E2": (["411"], "m3_ifc_411.npz", False),
    "E3": (["411", "410"], "m3_ifc_411_410.npz", False),
    "E4": (["410"], "m3_ifc_410.npz", True),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="Registration/configs/realdata")
    ap.add_argument("--base", default="Registration/configs/m3_ifc.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="生成した config で source/reference が読めるかだけ確認する")
    args = ap.parse_args()
    os.chdir(REPO)

    base = yaml.safe_load(open(args.base))
    os.makedirs(args.out, exist_ok=True)
    made: List[str] = []

    for scene in SCENES:
        mesh = "output/RealData/_TSDF/%s/run1/mesh/final_mesh_semantic_projected.ply" % scene
        if not os.path.exists(mesh):
            print("  %-12s 投影済みメッシュ無し。飛ばす" % scene)
            continue
        for cond, (spaces, cache, no_sem) in CONDITIONS.items():
            cfg = copy.deepcopy(base)
            # ★ source は **投影済み**の TSDF メッシュ。生メッシュは頂点色が写真の RGB で、
            #   6クラスのパレットとして読むと無意味なラベルになる（R4追補6 §C-2 で一度踏んだ）。
            cfg["source"]["mesh_path"] = mesh
            cfg["reference"]["spaces"] = spaces
            cfg["reference"]["cache_path"] = "Registration/output/ifc/" + cache
            if no_sem:
                # E4: 意味を使わない対照。手法の実装は変えず既存の ablation を使う
                cfg["ablation"] = {"single_class": True}
            cfg["eval"]["out_dir"] = "Registration/output/realdata/%s__%s" % (scene, cond)
            cfg["eval"]["t_gt_path"] = "output/GT_alignment/T_gt/T_gt_%s.json" % scene
            cfg.setdefault("diagnostics", {})["record_yaw"] = True
            p = os.path.join(args.out, "%s__%s.yaml" % (scene, cond))
            with open(p, "w") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            made.append(p)
    print("生成 %d 件 -> %s" % (len(made), args.out))

    if args.smoke:
        from regbim import io_utils
        print("\n--- 疎通確認（読み込みのみ。位置合わせは回さない）---")
        seen = set()
        for p in made:
            cfg = yaml.safe_load(open(p))
            key = (cfg["source"]["mesh_path"], tuple(cfg["reference"]["spaces"]))
            if key in seen:
                continue
            seen.add(key)
            try:
                s = io_utils.load_source_cloud(cfg)
                r = io_utils.load_reference_cloud(cfg)
                print("  %-28s source %s 点 %s / reference %s 点 %s"
                      % (os.path.basename(p), "{:,}".format(len(s)),
                         io_utils.class_counts(s), "{:,}".format(len(r)),
                         io_utils.class_counts(r)))
            except Exception as e:
                print("  %-28s ×  %s: %s" % (os.path.basename(p), type(e).__name__, e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

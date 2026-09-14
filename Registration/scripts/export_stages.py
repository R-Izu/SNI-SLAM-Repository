"""段階ごとの位置合わせ結果を書き出す（本人の目視要望。R24 追補）。

> **method_solution は reference に対して反時計回りに 90 度ずれている。**
> **位置合わせをいくつかの段階で分けて出力し、どこでミスっているのかデバッグしたい**

提案手法は 5 段階ある。**どこで正解から外れるかを、段階ごとの点群で見る。**

| 段階 | 何をするか |
|---|---|
| S1 種 | 重力で正立させ、マンハッタン拘束でヨー候補を 4 つ作り、各候補に縮尺と並進を与える |
| S2 候補 ICP | 候補ごとに回転を固定して ICP |
| S3 勝者 | クラス内包率 $\\rho_P$ が最大の候補を選ぶ |
| S4 重心再初期化 | 勝者の (R, s) を固定して並進だけ入れ直し、もう一度 ICP |
| S5 最終 | S3 と S4 を Chamfer で比べて近い方を採る |

**GT も同じ場所に出す**ので、どの段階で GT から離れるかが並べて見える。

    conda activate sni-slam
    python Registration/scripts/export_stages.py --target m3_cor_c__E2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from criterion_verdict import build_omega, d_omega                 # noqa: E402
from export_for_review import C_REF, C_SOURCE, sha256, write_ply   # noqa: E402
from failure_decomposition import provenance                       # noqa: E402
from regbim import io_utils, metrics                               # noqa: E402
from regbim.methods import get_method                              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="m3_cor_c__E2")
    ap.add_argument("--gt-kit", default="output/GT_alignment_probe",
                    help="既定は新 GT。旧 GT で見るなら output/GT_alignment")
    ap.add_argument("--scale-init", default=None,
                    choices=["median_axes", "vertical_only"],
                    help="R25 §4 の診断用に seed の縮尺の決め方を上書きする。"
                         "**config は書き換えない。既定は config の値（median_axes）**")
    ap.add_argument("--no-ply", action="store_true",
                    help="点群を書かず数値だけ出す（複数条件を掃くとき）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suffix = "" if args.scale_init is None else "__" + args.scale_init
    out = args.out or os.path.join("output", "stages_%s%s" % (args.target, suffix))
    os.makedirs(out, exist_ok=True)

    scene = args.target.split("__")[0]
    cfg = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % args.target))
    cfg["source"] = dict(cfg["source"], seed=0)
    cfg.setdefault("diagnostics", {})
    cfg["diagnostics"]["record_yaw"] = True
    cfg["diagnostics"]["record_stages"] = True
    if args.scale_init is not None:
        cfg.setdefault("proposed", {})["scale_init"] = args.scale_init

    src = io_utils.load_source_cloud(cfg)
    dst = io_utils.load_reference_cloud(cfg)
    G = np.asarray(json.load(open(os.path.join(
        args.gt_kit, "T_gt", "T_gt_%s.json" % scene)))["T_gt"],
        dtype=np.float64).reshape(4, 4)
    RG, _, S_GT = metrics.decompose_sim3(G)
    omega = build_omega(cfg, seed=0)

    m = get_method("proposed")
    print("実行中… %s（GT: %s）" % (args.target, args.gt_kit), flush=True)
    T_final = m.register(src, dst, cfg)
    st = m.last_stage_diag
    yaw = m.last_yaw_diag

    files: List[Dict] = []

    def emit(name: str, T, note: str):
        """1 段階ぶんを記録する。

        **縮尺比** は |s_est / s_GT − 1|。**GT と同じ縮尺なら 0。**
        **d_Ω** は固定評価点集合 Ω の RMS 変位（R17 §5）。
        **並進ベクトルの差は使わない**（R10 §3-2 で禁じている）。
        """
        T = np.asarray(T, dtype=np.float64)
        R_e, _, s_e = metrics.decompose_sim3(T)
        rot = metrics.rotation_error_deg(R_e, RG)
        sr = abs(s_e / S_GT - 1.0)
        dw = d_omega(T, G, omega)
        if not args.no_ply:
            write_ply(os.path.join(out, name + ".ply"),
                      metrics.apply_sim3(T, src.points),
                      np.tile(C_SOURCE, (len(src), 1)))
        files.append({"file": name + ".ply", "stage": note,
                      "gt_rot_deg": round(rot, 2),
                      "scale_ratio": round(sr, 4),
                      "d_omega_m": round(dw, 3),
                      "T": T.tolist()})
        print("  %-32s 回転 %7.2f 度 / 縮尺比 %7.4f / d_Ω %9.3f m   %s"
              % (name, rot, sr, dw, note))

    # 参照と GT
    if not args.no_ply:
        write_ply(os.path.join(out, "00_bim_reference.ply"), dst.points,
                  np.tile(C_REF, (len(dst), 1)))
    files.append({"file": "00_bim_reference.ply", "stage": "BIM 参照（変換なし）"})
    emit("01_gt", G, "GT の解（正解）")

    # S1 種・S2 候補 ICP 後
    for i, (seed_T, icp_T, sc) in enumerate(zip(st["seeds"], st["after_icp"] or [],
                                                st["scores"] or [])):
        emit("S1_seed_cand%d" % i, seed_T, "段階1 種（ICP 前）候補 %d" % i)
        emit("S2_afterICP_cand%d_score%.4f" % (i, sc), icp_T,
             "段階2 候補 %d の ICP 後（スコア %.4f）" % (i, sc))

    emit("S3_winner", st["winner_plane_T"],
         "段階3 勝者（候補 %d。スコア最大）" % yaw["winner"])
    emit("S4a_refine_init", st["refine_init_T"], "段階4 重心で並進を入れ直した初期値")
    emit("S4b_refine_after", st["refine_T"], "段階4 その ICP 後")
    emit("S5_final", T_final,
         "段階5 最終（Chamfer 比較：勝者 %.4f / 再初期化 %.4f → %s を採用）"
         % (st["chamfer_plane"], st["chamfer_refine"],
            "再初期化" if st["final_is_refine"] else "勝者"))

    meta = {"provenance": provenance(), "target": args.target, "gt_kit": args.gt_kit,
            "scale_init": cfg.get("proposed", {}).get("scale_init", "median_axes"),
            "scale_init_overridden": args.scale_init is not None,
            "s_gt": S_GT, "omega_n": int(len(omega)),
            "source_mesh": cfg["source"]["mesh_path"],
            "source_sha256_16": sha256(cfg["source"]["mesh_path"]),
            "source_seed": 0,
            "reference_cache": cfg["reference"]["cache_path"],
            "reference_spaces": cfg["reference"].get("spaces"),
            "candidate_scores": yaw["candidate_scores"],
            "winner": yaw["winner"], "margin": yaw["margin"],
            "chamfer_plane": st["chamfer_plane"], "chamfer_refine": st["chamfer_refine"],
            "files": files}
    with open(os.path.join(out, "STAGES.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n候補のスコア: %s（勝者 %d、1位と2位の差 %s）"
          % (yaw["candidate_scores"], yaw["winner"], yaw["margin"]))
    best = min(range(len(files)), key=lambda i: files[i].get("gt_rot_deg", 1e9)
               if files[i]["file"].startswith("S2") else 1e9)
    print("**GT にいちばん近い候補は %s**" % files[best]["file"])
    print("\n%d ファイルを %s に書き出した" % (len(files), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

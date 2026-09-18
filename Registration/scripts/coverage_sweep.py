"""R30 §3-2 — **被覆の不一致だけを動かして、どこで壊れるかを測る。**

なぜ合成でやるか（R30 §3-4）
----------------------------
実データでは正解自体が 0.043〜0.187 m 動き、参照 BIM の精度もラベル品質も交絡する。
**合成なら、被覆の不一致だけを動かして他を固定できる。**
実データはその結果が実際の条件で成立するかの確認に使う。逆ではない。

作り方（R30 §3-2）
------------------
- **source は GT-A のまま変えない**
- **参照点群を、シーンの一部だけに切り縮める**（既存の `reference.clip`。R3 T3 で実装済み）
- 切り縮めた割合を振る

**正解 $T_{gt}$ は参照を切っても変わらない**（我々が掛けた既知の Sim(3) の逆行列である）。
**だから「正解が動くから比較できない」という問題が起きない。** ここが実データと違う。

測るもの
--------
| 量 | 定義 |
|---|---|
| **重なり率** | **source のうち、参照の範囲に入る点の割合**（R30 §3-2 の定義） |
| **構造重心の差** | 実データで測った量（0.16 m / 5.83〜9.33 m）と同じ定義 |
| **段階1の並進の初期値の誤差** | 実データの 9.0〜14.9 m と同じ定義（正解の向きを持つ候補の種の $d_\Omega$） |
| **最終の $d_\Omega$ と成功判定** | 結果 |

**⚠ 予測 Q1〜Q3 は R30 §3-3 に登録済みである。ここで登録し直さない。**
**境界は予測されていない（何割かは分からない）。**

    conda activate sni-slam
    python Registration/scripts/coverage_sweep.py --scene room_0
"""

from __future__ import annotations

import argparse
import copy
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

from criterion_verdict import build_omega, d_omega          # noqa: E402
from failure_decomposition import provenance                # noqa: E402
from regbim import io_utils, metrics                        # noqa: E402
from regbim.labels import NAME_TO_ID                        # noqa: E402
from regbim.methods import get_method                       # noqa: E402

STRUCT = ("wall", "floor", "ceiling")
# 事前登録済みの成功条件。**動かさない。**
SUCC = {"rot_deg": 5.0, "trans": 0.1, "scale_ratio": 0.05}
# 「正解の向きを持つ候補」の判定（既存の YAW_MODE_TOL_DEG と同じ）
ROT_TOL = 5.0


def struct_pts(cloud) -> np.ndarray:
    ids = [NAME_TO_ID[n] for n in STRUCT if n in NAME_TO_ID]
    return cloud.points[np.isin(cloud.labels, ids)]


def measure(cfg: Dict, keep_frac: float, anchor, omega, G) -> Dict:
    """1 つの被覆率について、機構の各段を測る。**摂動は掛けない**（機構を見るため）。"""
    c = copy.deepcopy(cfg)
    if keep_frac < 1.0:
        c["reference"] = dict(c["reference"], clip={"keep_frac": keep_frac,
                                                    "anchor": list(anchor)})
    c.setdefault("diagnostics", {})
    c["diagnostics"]["record_yaw"] = True
    c["diagnostics"]["record_stages"] = True

    src = io_utils.load_source_cloud(c)
    dst = io_utils.load_reference_cloud(c)
    clip_meta = dst.meta.get("clip", {})

    # --- 重なり率：source を正解で置いたとき、参照の XY 範囲に入る点の割合 ---
    src_in_ref = metrics.apply_sim3(G, src.points)
    lo = dst.points[:, :2].min(axis=0)
    hi = dst.points[:, :2].max(axis=0)
    inside = ((src_in_ref[:, :2] >= lo) & (src_in_ref[:, :2] <= hi)).all(axis=1)
    overlap = float(inside.mean())

    # --- 構造重心の差（実データと同じ定義）---
    sp = metrics.apply_sim3(G, struct_pts(src))
    dp = struct_pts(dst)
    centroid_diff = float(np.linalg.norm(sp.mean(axis=0) - dp.mean(axis=0)))

    RG = metrics.decompose_sim3(G)[0]
    m = get_method("proposed")
    T_final = m.register(src, dst, c)
    st, yaw = m.last_stage_diag, m.last_yaw_diag

    # --- 段階1：正解の向きを持つ候補の種の誤差 ---
    seeds = st["seeds"]
    rots = [metrics.rotation_error_deg(metrics.decompose_sim3(np.asarray(s))[0], RG)
            for s in seeds]
    k = int(np.argmin(rots))
    seed_ok = rots[k] < ROT_TOL
    seed_err = d_omega(np.asarray(seeds[k]), G, omega)

    e = metrics.sim3_errors(T_final, G)
    final_d = d_omega(T_final, G, omega)
    ok = (not e["degenerate"] and e["rot_deg"] < SUCC["rot_deg"]
          and final_d < SUCC["trans"] and e["scale_ratio"] < SUCC["scale_ratio"])

    return {"keep_frac_nominal": keep_frac, "anchor": list(anchor),
            "coverage_achieved": clip_meta.get("coverage_achieved", 1.0),
            "overlap_ratio": overlap,
            "centroid_diff_m": centroid_diff,
            "n_ref_points": int(len(dst)),
            "n_ref_wall": int((dst.labels == NAME_TO_ID["wall"]).sum()),
            "correct_rot_candidate_exists": bool(seed_ok),
            "correct_rot_candidate_deg": float(rots[k]),
            "stage1_seed_d_omega_m": float(seed_err),
            "final_d_omega_m": float(final_d),
            "final_rot_deg": float(e["rot_deg"]),
            "final_scale_ratio": float(e["scale_ratio"]),
            "success": bool(ok),
            "candidate_scores": yaw.get("candidate_scores")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="room_0")
    ap.add_argument("--keep", nargs="*", type=float,
                    default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    # ★ 隅は "x,y" を ";" で連ねた**1つの文字列**で受ける。
    #   nargs="*" にして `--anchors=-1,-1 --anchors=-1,1` と並べると
    #   **後のものが前を上書きし、最後の1つしか効かない**。
    #   実際それで隅を1つしか回しておらず、位置依存を見ないまま表を作りかけた。
    #   空白区切りにしないのは、先頭が負号の値が option と解釈されるためである。
    ap.add_argument("--anchors", default="-1,-1;-1,1;1,-1;1,1",
                    help='切り取る箱を寄せる隅を ";" で連ねる。'
                         '**複数出して位置依存を見る**')
    ap.add_argument("--out", default="Registration/output/diag/coverage_sweep.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("Registration/configs/gt_a/%s.yaml" % args.scene))
    omega = build_omega(cfg)
    G = np.asarray(json.load(open(cfg["eval"]["t_gt_path"]))["T_gt"],
                   dtype=np.float64).reshape(4, 4)

    rows: List[Dict] = []
    print("%-7s %-7s %9s %9s %10s %10s %10s %6s"
          % ("keep", "anchor", "重なり率", "重心差[m]", "段階1[m]", "最終[m]",
             "最終回転", "成功"))
    print("-" * 82)
    wanted = [a.strip() for a in args.anchors.split(";") if a.strip()]
    print("隅 %d 通り: %s\n" % (len(wanted), wanted))
    for kf in args.keep:
        anchors = ["0,0"] if kf >= 1.0 else wanted
        for a in anchors:
            anchor = [float(v) for v in a.split(",")]
            try:
                r = measure(cfg, kf, anchor, omega, G)
            except Exception as ex:                     # 1 点落ちても続ける
                r = {"keep_frac_nominal": kf, "anchor": anchor,
                     "error": "%s: %s" % (type(ex).__name__, ex)}
                print("%-7.2f %-7s **失敗** %s" % (kf, a, r["error"][:60]))
            else:
                print("%-7.2f %-7s %9.4f %9.3f %10.3f %10.3f %10.2f %6s"
                      % (kf, a, r["overlap_ratio"], r["centroid_diff_m"],
                         r["stage1_seed_d_omega_m"], r["final_d_omega_m"],
                         r["final_rot_deg"], "○" if r["success"] else "×"))
            rows.append(r)
            # **1 件ごとに書き出す**（R23 §4-6）
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump({"provenance": provenance(), "scene": args.scene,
                           "success_thresholds": SUCC,
                           "note": "摂動なしの直接解。予測 Q1〜Q3 は R30 §3-3 に登録済み",
                           "rows": rows}, f, indent=2, ensure_ascii=False)

    ok = [r for r in rows if "error" not in r]
    if ok:
        print("\n## 重なり率の順に並べ直す（**Q1・Q2 を読むための並び**）\n")
        print("%9s %10s %10s %10s %6s" % ("重なり率", "重心差[m]", "段階1[m]",
                                          "最終[m]", "成功"))
        for r in sorted(ok, key=lambda x: -x["overlap_ratio"]):
            print("%9.4f %10.3f %10.3f %10.3f %6s"
                  % (r["overlap_ratio"], r["centroid_diff_m"],
                     r["stage1_seed_d_omega_m"], r["final_d_omega_m"],
                     "○" if r["success"] else "×"))
        s = [r for r in ok if r["success"]]
        f = [r for r in ok if not r["success"]]
        if s and f:
            print("\n成功した最小の重なり率: %.4f" % min(r["overlap_ratio"] for r in s))
            print("失敗した最大の重なり率: %.4f" % max(r["overlap_ratio"] for r in f))
        elif not f:
            print("\n**全条件で成功。この範囲では壊れない**")
        else:
            print("\n**全条件で失敗**")
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

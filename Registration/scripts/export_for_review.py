"""R24 §2 — 本人が目視するための点群を書き出す。

出すもの（R24 §2 の表そのまま。**見栄えの良い条件だけを出さない**）：

1. `m3_block_b E2` …… 手法の解で変換した source ＋ BIM 参照
2. `m3_cor_c E2` …… **手法の解** と **GT の解** の2つ（179° 誤りを並べて見る）
3. `m3_room_b` …… **旧 GT** と **新 GT** の2つ（同じ source を両方で変換）
4. `is_inner` の色分け …… 壁・床・天井・**扉**を色分けした参照点群

共通の要求（R24 §2）：

- ファイル名に「何を・どの変換で写したか」を入れる
- **来歴を添える**（R23 §4-1）：コード版・入力ハッシュ・参照・GT の識別子
- source と参照が区別できる色。**クラス別の色は 4 だけ**
- 間引いてよいが、**間引き率を書く**

**印象を誘導する加工をしない**（不自然な色分け・視点の選択などをしない）。

    conda activate sni-slam
    python Registration/scripts/export_for_review.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Dict, List

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration"))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)

from failure_decomposition import provenance                  # noqa: E402
from regbim import io_utils, metrics                          # noqa: E402
from regbim.labels import NAME_TO_ID                          # noqa: E402

# source と参照を区別する色（R24：source と参照が区別できること）
C_SOURCE = (1.00, 0.55, 0.10)      # 橙：スキャン
C_REF = (0.35, 0.45, 0.90)         # 青：BIM 参照
# クラス別の色は 4 だけ
C_CLASS = {"wall": (0.65, 0.65, 0.65), "floor": (0.30, 0.70, 0.35),
           "ceiling": (0.55, 0.35, 0.75), "door": (0.95, 0.20, 0.20),
           "window": (0.20, 0.85, 0.90)}


def sha256(path: str, limit: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if f.tell() > limit:
                h.update(b"TRUNCATED")
                break
    return h.hexdigest()[:16]


def write_ply(path: str, pts: np.ndarray, colors: np.ndarray) -> None:
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    o3d.io.write_point_cloud(path, pc)


def decimate(n: int, keep: int, seed: int = 0):
    """決定的に間引く（乱数を使わず等間隔）。間引き率を返す。"""
    if n <= keep:
        return np.arange(n), 1.0
    idx = np.linspace(0, n - 1, keep).astype(int)
    return idx, keep / n


def gt_of(kit: str, scene: str) -> np.ndarray:
    p = os.path.join(kit, "T_gt", "T_gt_%s.json" % scene)
    return np.asarray(json.load(open(p))["T_gt"], dtype=np.float64).reshape(4, 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output/review_2026-09-13")
    ap.add_argument("--max-points", type=int, default=300000,
                    help="CloudCompare で開ける範囲に間引く")
    ap.add_argument("--direct", default="Registration/output/diag/realdata_direct_v3.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    est = {r["target"]: np.asarray(r["T_est"], dtype=np.float64)
           for r in json.load(open(args.direct))["results"]}
    manifest: List[Dict] = []

    def emit(fname: str, pts, colors, meta: Dict):
        idx, frac = decimate(len(pts), args.max_points)
        p = os.path.join(args.out, fname)
        write_ply(p, np.asarray(pts)[idx], np.asarray(colors)[idx])
        meta = dict(meta, file=fname, n_points_written=int(len(idx)),
                    n_points_original=int(len(pts)), decimation_kept_fraction=round(frac, 4))
        manifest.append(meta)
        print("  %-58s %7d 点（間引き率 %.3f）" % (fname, len(idx), frac))

    def load(target: str):
        cfg = yaml.safe_load(open("Registration/configs/realdata/%s.yaml" % target))
        cfg["source"] = dict(cfg["source"], seed=0)
        return cfg, io_utils.load_source_cloud(cfg), io_utils.load_reference_cloud(cfg)

    def src_meta(cfg, scene):
        return {"scene": scene,
                "source_mesh": cfg["source"]["mesh_path"],
                "source_sha256_16": sha256(cfg["source"]["mesh_path"]),
                "source_seed": 0, "source_n_points": cfg["source"]["n_points"],
                "reference_cache": cfg["reference"]["cache_path"],
                "reference_sha256_16": sha256(cfg["reference"]["cache_path"]),
                "reference_spaces": cfg["reference"].get("spaces"),
                "reference_inner_only": cfg["reference"].get("inner_only")}

    # ---- 1. m3_block_b E2：手法の解 ＋ BIM 参照 ----
    print("1. m3_block_b E2（手法の解と BIM 参照）")
    t = "m3_block_b__E2"
    cfg, src, dst = load(t)
    m = src_meta(cfg, "m3_block_b")
    emit("m3_block_b__E2__method_solution.ply",
         metrics.apply_sim3(est[t], src.points),
         np.tile(C_SOURCE, (len(src), 1)),
         dict(m, transform="手法の解（無摂動、seed 0）",
              note=("この条件は**旧 GT では成功（d_Ω 0.052 m）、新 GT では不合格（0.253 m）**。"
                    "どちらの GT で評価したかで結論が変わる。目視はその判定材料ではない。")))
    emit("m3_block_b__E2__bim_reference.ply", dst.points,
         np.tile(C_REF, (len(dst), 1)),
         dict(m, transform="変換なし（BIM 座標そのもの）"))

    # ---- 2. m3_cor_c E2：手法の解と GT の解 ----
    print("2. m3_cor_c E2（179° 誤り。手法の解と GT の解を並べる）")
    t = "m3_cor_c__E2"
    cfg, src, dst = load(t)
    m = src_meta(cfg, "m3_cor_c")
    emit("m3_cor_c__E2__method_solution.ply",
         metrics.apply_sim3(est[t], src.points),
         np.tile(C_SOURCE, (len(src), 1)),
         dict(m, transform="手法の解", note="GT からの回転誤差 89.6°（旧 GT）/ 90.5°（新 GT）"))
    for tag, kit in (("old_gt", "output/GT_alignment"), ("new_gt", "output/GT_alignment_probe")):
        emit("m3_cor_c__E2__gt_solution_%s.ply" % tag,
             metrics.apply_sim3(gt_of(kit, "m3_cor_c"), src.points),
             np.tile(C_SOURCE, (len(src), 1)),
             dict(m, transform="GT の解（%s）" % tag, gt_kit=kit))
    emit("m3_cor_c__E2__bim_reference.ply", dst.points,
         np.tile(C_REF, (len(dst), 1)), dict(m, transform="変換なし"))

    # ---- 3. m3_room_b：旧 GT と新 GT ----
    print("3. m3_room_b（旧 GT と新 GT。d_Ω で 0.499 m 差）")
    t = "m3_room_b__E1"
    cfg, src, dst = load(t)
    m = src_meta(cfg, "m3_room_b")
    for tag, kit in (("old_gt", "output/GT_alignment"), ("new_gt", "output/GT_alignment_probe")):
        emit("m3_room_b__%s.ply" % tag,
             metrics.apply_sim3(gt_of(kit, "m3_room_b"), src.points),
             np.tile(C_SOURCE, (len(src), 1)),
             dict(m, transform="GT（%s）で変換した同じ source" % tag, gt_kit=kit,
                  note="この 2 つの差が d_Ω で 0.499 m。10 シーンで最大である"))
    emit("m3_room_b__bim_reference.ply", dst.points,
         np.tile(C_REF, (len(dst), 1)), dict(m, transform="変換なし"))

    # ---- 4. is_inner の色分け ----
    print("4. is_inner の色分け（壁・床・天井・扉）")
    z = np.load("Registration/output/ifc/m3_ifc_all.npz", allow_pickle=False)
    inner = z["is_inner"].astype(bool)
    lab = z["labels"]
    meta4 = {"reference_npz": "Registration/output/ifc/m3_ifc_all.npz",
             "reference_sha256_16": sha256("Registration/output/ifc/m3_ifc_all.npz"),
             "ifc_spaces": "IfcSpace は 2 つのみ（'1' と '2'、LongName は「部屋」）。**廊下は無い**"}
    for tag, mask in (("inner", inner), ("outer", ~inner)):
        cols = np.zeros((int(mask.sum()), 3))
        sub = lab[mask]
        for name, c in C_CLASS.items():
            cols[sub == NAME_TO_ID[name]] = c
        emit("bim_is_inner__%s.ply" % tag, z["points"][mask], cols,
             dict(meta4, transform="変換なし",
                  note="色：壁=灰 / 床=緑 / 天井=紫 / 扉=赤 / 窓=水色",
                  is_inner=(tag == "inner")))
    # 扉だけ別ファイル（R24 §2：扉だけ別でも良い）
    dm = inner & (lab == NAME_TO_ID["door"])
    frac_door = float(dm.sum()) / max(int((lab == NAME_TO_ID["door"]).sum()), 1)
    emit("bim_is_inner__door_only.ply", z["points"][dm],
         np.tile(C_CLASS["door"], (int(dm.sum()), 1)),
         dict(meta4, transform="変換なし",
              note="扉の室内向き面のみ。**扉の is_inner 残存率 %.3f は未解決の問題である**"
                   % frac_door))

    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump({"provenance": provenance(), "files": manifest}, f,
                  indent=2, ensure_ascii=False)
    print("\n%d ファイルを %s に書き出した" % (len(manifest), args.out))
    print("MANIFEST.json に来歴（コード版・入力ハッシュ・参照・GT）を書いた")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

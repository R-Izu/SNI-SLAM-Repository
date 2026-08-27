"""追補5 §2 — 再構成品質の指標を確定させる。

なぜ集中度をやめるか
--------------------
壁法線ヨー角の集中度は **「再構成のノイズ」と「シーンの幾何的な豊かさ」を区別できない**。
値が低い理由が (a) 法線がノイズで散っている か (b) 部屋が本当に多方向の壁を持つ か
判別できず、しかも (b) は手法にとって**有利**な条件である。
つまり指標の向きが逆になる。実例：同じ 411（L字）を撮った2本が 5.16 と 2.41 で2倍違った。

差し替え
--------
主  : **深度支持率** — メッシュ頂点のうち実測 depth に許容差内で支持される割合。
      「メッシュが観測に裏付けられているか」そのもので、部屋の形に依存しない。
従  : **床面積あたり頂点数** — 偽の等値面の量を直接表す。

検証（追補4 §1 の一般則。指標自身以外で確かめる）
------------------------------------------------
1. **撮影者の目視評価をラベル付き検証セットとして使い**、順序を再現できるか見る
2. **Replica 参照**（`output/Replica/room0_official/260310_test4`）での値域
3. **床面積との相関**（交絡していないことの確認。depth 5 m 打ち切りのため
   広いシーンほど支持率が下がる可能性がある）

使い方
------
    conda activate sni-slam
    python tools/realdata/quality_metrics.py
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

# 撮影者による目視評価（2026-08-26 の振り返り §5-1 A）。
# 良い順のランク。同順位は同じ値。**これが指標の検証セットになる。**
VISUAL_RANK: Dict[str, int] = {
    "m3_cor_a": 1,     # かなりきれい
    "m3_cor_b": 1,     # かなりきれい
    "m3_block_a": 3,   # きれい（細部＝椅子机は汚い）
    "m3_block_b": 3,   # m3_block_a と同等
    "m3_room_a": 5,    # ある程度きれい
    "m3_room_b": 5,    # ある程度きれい
    "m3_block_c": 7,   # 411 は可、410 は片側の壁が崩壊
    "m3_block_d": 8,   # かなり崩壊。検証では使えない
}

# S0 の preflight で実測した床面積（α-shape）。scan_id -> m^2
SCENE_TO_SCAN: Dict[str, str] = {
    "m3_room_a": "707e94b9a5", "m3_room_b": "a499702124",
    "m3_block_a": "03d9034a38", "m3_block_b": "fdbbcc52f1",
    "m3_block_c": "b8ebcd8158", "m3_block_d": "d02315a5e8",
    "m3_cor_a": "5355665db1", "m3_cor_b": "5e2269a58d",
    "m3_cor_c": "6e4afbf6e6", "m3_cor_d": "b17452f252",
}


def read_json(p: str) -> Dict:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def spearman(a: List[float], b: List[float]) -> Optional[float]:
    """順位相関。scipy を使わずに済ませる（同順位は平均順位）。"""
    if len(a) < 3:
        return None

    def rank(x: List[float]) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x)
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # 同値は平均順位に
        for v in np.unique(x):
            m = x == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default="output/RealData/_D")
    ap.add_argument("--preflight", default="output/RealData/_preflight")
    ap.add_argument("--out", default="output/RealData/quality_metrics.json")
    args = ap.parse_args()

    rows: List[Dict[str, object]] = []
    for scene, scan in SCENE_TO_SCAN.items():
        run = os.path.join(args.out_root, scene, "run1")
        proj = read_json(os.path.join(
            run, "mesh", "final_mesh_semantic_projected_report.json"))
        pre = read_json(os.path.join(args.preflight, "preflight_%s.json" % scan))
        if not proj:
            rows.append({"scene": scene, "status": "no_projection"})
            continue
        area = pre.get("alpha_shape_area_m2")
        nv = proj.get("n_vertices")
        # 観測側を分母にした候補（全シーン同一条件で測定済み）
        mvd = read_json(os.path.join(run, "map_vs_depth_fixed.json"))
        rows.append({
            "scene": scene, "scan_id": scan, "status": "ok",
            "floor_area_m2": round(float(area), 1) if area else None,
            "n_vertices": nv,
            # 主指標
            "depth_support": proj.get("vertices_with_votes_frac"),
            # 従指標（低いほど良い＝偽の等値面が少ない）
            "verts_per_m2": round(nv / area) if (nv and area) else None,
            # 観測点 -> 最近傍メッシュ頂点。分母が観測点数なのでサイズ交絡を受けにくい
            "completion_median_m": mvd.get("completion_median_m"),
            "completion_within_10cm": mvd.get("completion_frac_within_10cm"),
            "visual_rank": VISUAL_RANK.get(scene),
        })

    ok = [r for r in rows if r.get("status") == "ok" and r.get("visual_rank")]
    ok.sort(key=lambda r: -float(r["depth_support"]))

    hdr = "%-12s %10s %12s %11s %13s %8s" % (
        "scene", "床面積m2", "頂点数", "深度支持率", "頂点数/m2", "目視順位")
    print(hdr)
    print("-" * len(hdr))
    for r in ok:
        print("%-12s %10s %12s %11.3f %13s %8s" % (
            r["scene"], r["floor_area_m2"], "{:,}".format(r["n_vertices"]),
            r["depth_support"], "{:,}".format(r["verts_per_m2"]), r["visual_rank"]))

    # --- 検証1: 目視評価との順位一致 ---
    vr = [float(r["visual_rank"]) for r in ok]
    ds = [float(r["depth_support"]) for r in ok]
    vp = [float(r["verts_per_m2"]) for r in ok]
    # 目視順位は小さいほど良い、深度支持率は大きいほど良い → 負の相関が期待値
    rho_ds = spearman(vr, ds)
    rho_vp = spearman(vr, vp)
    cm = [float(r["completion_median_m"]) for r in ok if r.get("completion_median_m")]
    cw = [float(r["completion_within_10cm"]) for r in ok if r.get("completion_within_10cm")]
    vr_c = [float(r["visual_rank"]) for r in ok if r.get("completion_median_m")]
    fa_c = [float(r["floor_area_m2"]) for r in ok if r.get("completion_median_m")]
    rho_cm = spearman(vr_c, cm) if len(cm) == len(vr_c) else None
    rho_cw = spearman(vr_c, cw) if len(cw) == len(vr_c) else None
    rho_cm_area = spearman(fa_c, cm) if len(cm) == len(fa_c) else None
    print("\n[検証1] 撮影者の目視評価との順位相関（Spearman, n=%d）" % len(ok))
    print("  深度支持率     rho = %+.3f  （目視順位は小さいほど良いので **負**が期待値）"
          % (rho_ds if rho_ds is not None else float("nan")))
    print("  頂点数/m2      rho = %+.3f  （偽の面が多いほど悪いので **正**が期待値）"
          % (rho_vp if rho_vp is not None else float("nan")))

    # --- 検証3: 床面積との交絡 ---
    fa = [float(r["floor_area_m2"]) for r in ok]
    rho_conf = spearman(fa, ds)
    print("\n[検証3] 床面積と深度支持率の順位相関 rho = %+.3f" % (
        rho_conf if rho_conf is not None else float("nan")))
    print("  （強い負なら『広いシーンほど低く出る』交絡があり、正規化が要る）")

    rep = {"rows": rows,
           "validation": {"spearman_visual_vs_depth_support": rho_ds,
                          "spearman_visual_vs_verts_per_m2": rho_vp,
                          "spearman_area_vs_depth_support": rho_conf,
                          "spearman_visual_vs_completion_median": rho_cm,
                          "spearman_visual_vs_completion_within_10cm": rho_cw,
                          "spearman_area_vs_completion_median": rho_cm_area,
                          "n": len(ok)}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""R39 §5-2 — BIM（410＋411）から、誤差の無い source を作る（段階1）。**bim-ifc env で実行する**（ifcopenshell）。

作り方（R39 §5-2。結果を見る前に固定）
1. `m3-411.ifc` を三角形分割（`ifc_export.triangulate`、階 4FL、構造 5 クラス）
2. 面積に比例して高密度（800 点/m²）に点を打ち（`ifc_export.sample_surface`、seed 0）、
   室 411・410 の範囲（内法＋0.35 m、`clip_to_spaces`）に切り、**室内側の面**（`mark_inner_per_space`）だけ残す
3. **クラスごとに、実データの面密度の中央値まで間引く**（`r39_real_density.json` の per_class_median。
   廊下 4 シーンの seed 0 の source から測った値）。間引きは乱数 seed 1 の非復元抽出
4. GT-A と同じ既知の Sim(3) M（`output/GT_A/room_0_manifest.json` の `M_applied`）を掛けて source とする。
   **正解 T_gt = M^-1**（解析的）
5. 廊下は作らない（建物全体の BIM が無いため。形を推測で足さない）

出力：`output/R39_BIM/bim_source.ply`（色でクラス）、`output/R39_BIM/T_gt_bim.json`、`output/R39_BIM/manifest.json`

    conda run -n bim-ifc python Registration/scripts/r39_bim_source.py
"""
import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Registration", "scripts"))
os.chdir(REPO)
import ifc_export as ie                                         # noqa: E402

IFC = "BIM_IFC_Extraction/input/m3-411.ifc"
RAW_DENSITY = 800.0          # 点/m²（間引く前）
SPACES = ["411", "410"]
MARGIN = 0.35
OUT = "output/R39_BIM"


def main():
    import ifcopenshell
    cfg = yaml.safe_load(open("Registration/configs/m3_ifc.yaml"))
    class_map = cfg["classes"]["ifc_class_map"]
    keep = cfg["reference"]["keep_classes"]
    storeys = cfg["reference"]["storeys"]
    dens = json.load(open("Registration/output/diag/r39_real_density.json"))["per_class_median"]
    M = np.asarray(json.load(open("output/GT_A/room_0_manifest.json"))["M_applied"], dtype=np.float64)

    f = ifcopenshell.open(IFC)
    verts, tris, tri_lab, info = ie.triangulate(f, class_map, storeys=storeys)
    keep_ids = [ie.NAME_TO_ID[n] for n in keep]
    m = np.isin(tri_lab, keep_ids)
    tris, tri_lab = tris[m], tri_lab[m]
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    A = float(area.sum())
    n_raw = int(round(RAW_DENSITY * A))
    p, l, nn = ie.sample_surface(verts, tris, tri_lab, n_raw, seed=0)

    recs = ie.space_records(f)
    alias = ie.name_spaces(recs)
    ids = [alias[s] for s in SPACES]
    k = ie.clip_to_spaces(p, recs, ids, MARGIN)
    p, l, nn = p[k], l[k], nn[k]
    z_fl = ie._horizontal_face_z(p, nn, l == ie.NAME_TO_ID["floor"], want_up=True)
    z_ce = ie._horizontal_face_z(p, nn, l == ie.NAME_TO_ID["ceiling"], want_up=False)
    zr = (z_fl, z_ce) if (z_fl is not None and z_ce is not None and z_ce > z_fl) else None
    per = ie.mark_inner_per_space(p, nn, recs, z_range=zr)
    inner = np.zeros(len(p), dtype=bool)
    for sid in ids:
        if sid in per:
            inner |= per[sid]
    p, l, nn = p[inner], l[inner], nn[inner]

    rng = np.random.default_rng(1)
    sel, stat = [], {}
    for c in keep:
        cid = ie.NAME_TO_ID[c]
        idx = np.flatnonzero(l == cid)
        inner_area = len(idx) / RAW_DENSITY           # 室内面の面積（高密度の点数から）
        n_want = int(round(dens[c] * inner_area))
        n_want = min(n_want, len(idx))
        pick = np.sort(rng.choice(idx, size=n_want, replace=False)) if n_want else idx[:0]
        sel.append(pick)
        stat[c] = {"inner_area_m2": inner_area, "target_density_per_m2": dens[c], "n_points": int(n_want)}
    sel = np.concatenate(sel)
    pts, lab, nrm = p[sel], l[sel], nn[sel]

    # R40：誤差を足すため、BIM 座標の点・クラス・法線と、410 の部分か（410 の範囲＋0.35 m かつ 411 の内法の外）を保存する。
    #      ply の中身は変えない（同じ seed・同じ順序）。
    in410 = (ie.clip_to_spaces(pts, recs, [alias["410"]], MARGIN)
             & ~ie.clip_to_spaces(pts, recs, [alias["411"]], 0.0))
    area_space = {}
    for sname in ("411", "410"):
        ks = ie.clip_to_spaces(p, recs, [alias[sname]], MARGIN)
        per_s = ie.mark_inner_per_space(p[ks], nn[ks], recs, z_range=zr).get(alias[sname])
        cnt = {c: int(((l[ks] == ie.NAME_TO_ID[c]) & per_s).sum()) if per_s is not None else 0 for c in keep}
        area_space[sname] = {c: cnt[c] / RAW_DENSITY for c in keep}

    R = M[:3, :3]
    src = pts @ R.T + M[:3, 3]
    os.makedirs(OUT, exist_ok=True)
    ie.write_ply(os.path.join(OUT, "bim_source.ply"), src, lab)
    np.savez_compressed(os.path.join(OUT, "bim_points_bimframe.npz"), points=pts, labels=lab, normals=nrm,
                        in410=in410, M=M)
    Minv = np.linalg.inv(M)
    json.dump({"T_gt": Minv.tolist(), "note": "解析的な正解：我々が掛けた M の逆行列"},
              open(os.path.join(OUT, "T_gt_bim.json"), "w"), indent=1)
    s = float(np.cbrt(np.linalg.det(R)))
    man = {"what": "R39 §5 段階1：410＋411 の BIM の室内面から作った誤差の無い source",
           "ifc": IFC, "storeys": storeys, "spaces": SPACES, "space_margin_m": MARGIN,
           "keep_classes": keep, "raw_density_per_m2": RAW_DENSITY, "total_area_all_surfaces_m2": A,
           "n_raw": n_raw, "n_after_clip_inner": int(inner.sum()), "per_class": stat,
           "n_points": int(len(src)), "M_applied": M.tolist(),
           "M_scale": s, "M_from": "output/GT_A/room_0_manifest.json（GT-A と同じ M）",
           "sample_seed": 0, "decimate_seed": 1, "corridor": "作らない（建物全体の BIM が無い）",
           "inner_area_by_space_m2": area_space, "n_in410": int(in410.sum())}
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in man.items() if k != "M_applied"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

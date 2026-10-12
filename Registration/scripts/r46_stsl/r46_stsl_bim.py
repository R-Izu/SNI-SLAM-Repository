"""R46 §8-2：IFC の 410・411 から、STSL が読む BIM の入力を作る（**公式と違う前処理**。公式の作り方は非公開）。

参照は R39 と同じ読み方（`io_utils.load_reference_cloud`、室内面のみ）：
- `M3_410_411`：`configs/r39_bim/bim_E3_plan_correlate.yaml` の参照（410＋411）
- `M3_411`    ：`configs/r39_bim/bim_E2_plan_correlate.yaml` の参照（411 のみ）

作り方（固定）
1. 壁（wall）の点のうち、法線がほぼ水平（|n_z| < 0.3）のもの
2. 法線の向き（0〜180°、2° 刻み）と、法線方向の位置（0.05 m 刻み）で面に分ける（隣の刻みは 1 つにまとめる）。50 点未満の面は捨てる
3. 各面の点を面内の水平方向に並べ、0.3 m を超える隙間で切る。長さ 0.3 m 以上の区間を 2D 線分（端点 = 区間の両端）とする
4. `bimLineSeg.txt`：1 行に「xa ya xb yb」（公式の `LineSegments.save_to_file` と同じ形式）
5. `bim.pcd`：壁の点（3D）。`bim2d.pcd`：壁の点を z = 0 に落とし 0.05 m で間引いたもの
6. **z は BIM の床の高さを 0 にずらす**（STSL は地面の平面を z = 0 に置くため。§8-4「地面の平面を BIM の床の高さに合わせる」）。
   ずらした量 z_floor（参照の floor の点の z の中央値）を `meta.json` に残し、STSL の変換を BIM 座標に戻すときに足す

出力：`$R46_STSL_ROOT/BIM/<名前>/`（R46_STSL_ROOT の既定 ~/rizu/r46_stsl_data）
"""
import json, os, sys
import numpy as np
import yaml

REPO = os.path.expanduser("~/rizu/SNI-SLAM")
sys.path.insert(0, os.path.join(REPO, "Registration"))
os.chdir(REPO)
from regbim import io_utils                                  # noqa: E402
from regbim.labels import NAME_TO_ID                         # noqa: E402

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
REFS = {"M3_410_411": "r39_bim/bim_E3_plan_correlate", "M3_411": "r39_bim/bim_E2_plan_correlate"}
ANG_BIN, OFF_BIN, MIN_PTS, GAP, MIN_LEN = 2.0, 0.05, 50, 0.3, 0.3


def write_pcd(path, pts):
    pts = np.asarray(pts, dtype=np.float32)
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
        f.write("WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n" % (len(pts), len(pts)))
        np.savetxt(f, pts, fmt="%.5f")


def merge_bins(keys):
    """整数の刻みの集合を、隣り合うものどうしでまとめる"""
    keys = sorted(set(keys)); groups, cur = [], []
    for k in keys:
        if cur and k - cur[-1] > 1:
            groups.append(cur); cur = []
        cur.append(k)
    if cur:
        groups.append(cur)
    return groups


def lines_from_walls(P, N):
    ang = np.degrees(np.arctan2(N[:, 1], N[:, 0])) % 180.0
    abin = np.floor(ang / ANG_BIN).astype(int)
    segs = []
    for ag in merge_bins(abin):
        ma = np.isin(abin, ag)
        a = np.radians(np.median(ang[ma])); n = np.array([np.cos(a), np.sin(a)]); t = np.array([-n[1], n[0]])
        off = P[ma, :2] @ n
        obin = np.floor(off / OFF_BIN).astype(int)
        for og in merge_bins(obin):
            mo = np.isin(obin, og)
            if mo.sum() < MIN_PTS:
                continue
            Q = P[ma][mo][:, :2]; d = float(np.median(Q @ n)); s = np.sort(Q @ t)
            cuts = np.flatnonzero(np.diff(s) > GAP)
            for lo, hi in zip(np.r_[0, cuts + 1], np.r_[cuts, len(s) - 1]):
                if s[hi] - s[lo] >= MIN_LEN:
                    pa, pb = d * n + s[lo] * t, d * n + s[hi] * t
                    segs.append([pa[0], pa[1], pb[0], pb[1]])
    return np.asarray(segs)


def main():
    for name, cfgname in REFS.items():
        cfg = yaml.safe_load(open("Registration/configs/%s.yaml" % cfgname))
        ref = io_utils.load_reference_cloud(cfg)
        z_floor = float(np.median(ref.points[ref.labels == NAME_TO_ID["floor"], 2]))
        w = (ref.labels == NAME_TO_ID["wall"]) & (np.abs(ref.normals[:, 2]) < 0.3)
        P = ref.points[w] - np.array([0.0, 0.0, z_floor]); Nn = ref.normals[w]
        segs = lines_from_walls(P, Nn)
        d = os.path.join(ROOT, "BIM", name); os.makedirs(d, exist_ok=True)
        np.savetxt(os.path.join(d, "bimLineSeg.txt"), segs, fmt="%.5f")
        write_pcd(os.path.join(d, "bim.pcd"), P)
        flat = np.c_[P[:, :2], np.zeros(len(P))]
        key = np.unique(np.floor(flat[:, :2] / 0.05).astype(np.int64), axis=0, return_index=True)[1]
        write_pcd(os.path.join(d, "bim2d.pcd"), flat[np.sort(key)])
        for f in ("bimHashTable.bin", "bimHashNeighTable.bin", "bimHashIntTable.bin"):   # 古いハッシュ表は消して作り直させる
            if os.path.exists(os.path.join(d, f)):
                os.remove(os.path.join(d, f))
        meta = {"reference_config": cfgname, "z_floor": z_floor, "n_wall_points": int(len(P)), "n_segments": int(len(segs)),
                "segment_length_total_m": float(np.sum(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))),
                "params": {"ANG_BIN": ANG_BIN, "OFF_BIN": OFF_BIN, "MIN_PTS": MIN_PTS, "GAP": GAP, "MIN_LEN": MIN_LEN}}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        print(name, meta)


if __name__ == "__main__":
    main()

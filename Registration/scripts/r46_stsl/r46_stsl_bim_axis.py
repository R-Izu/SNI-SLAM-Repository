"""R46 §11-2 B3：IFC の壁の基準線（Axis 表現）から STSL の BIM 線分を作る（bim-ifc の環境、ifcopenshell）。**回す前に作って報告する。**

公式の BIM 線分は、Revit の壁の base curve（location line、既定は壁の芯）である（STSL の issue #1 の回答）。
作り方（固定）
1. `BIM_IFC_Extraction/input/m3-411.ifc` の `IfcWall`（`IfcWallStandardCase` を含む）について、`Representation` のうち
   `RepresentationIdentifier == 'Axis'` の表現の曲線（IfcPolyline、または IfcTrimmedCurve の基の IfcLine を端点で切ったもの）を、
   `ifcopenshell.util.placement.get_local_placement(ObjectPlacement)` で世界座標にし、長さの単位を m にする
2. 参照に入っている壁だけを使う：B1 の両面の BIM（`M3_*_both/bim.pcd`、壁の点）のうち、その軸から水平に 0.25 m 以内にある点を
   軸に投影し、その範囲（最小〜最大）に軸を切る。50 点未満なら、その壁はこの参照に入っていないとして捨てる
3. Axis が無い壁は、B1 の両面の線分のうち、平行（2° 以内）で 0.1〜0.3 m 離れた対の中点の線で代える（壁ごとにどちらかを記録）
4. z_floor は段階1と同じ 0.5 m（BIM の座標から引く）。`bim.pcd` は線分の上の点（0.05 m おき）を高さ 0.2〜2.5 m（0.1 m おき）に
   並べたもの、`bim2d.pcd` は線分の上の点（0.05 m おき、z = 0）。公式の点群の作り方は確かめられないので、この作り方を書く
出力：`$ROOT/BIM/M3_411_axis`・`M3_410_411_axis`、図 `axis_compare.png`（3 通りの BIM を重ねる）
"""
import json, os, sys
import numpy as np

ROOT = os.environ.get("R46_STSL_ROOT", os.path.expanduser("~/rizu/r46_stsl_data"))
IFC = os.path.expanduser("~/rizu/SNI-SLAM/BIM_IFC_Extraction/input/m3-411.ifc")
Z_FLOOR = 0.5
NAMES = {"M3_411": "M3_411_both", "M3_410_411": "M3_410_411_both"}


def read_pcd(p):
    with open(p) as f:
        lines = f.readlines()
    i = [k for k, l in enumerate(lines) if l.startswith("DATA")][0]
    return np.loadtxt(lines[i + 1:])


def write_pcd(path, pts):
    pts = np.asarray(pts, dtype=np.float32)
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
        f.write("WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n" % (len(pts), len(pts)))
        np.savetxt(f, pts, fmt="%.5f")


def axes():
    import ifcopenshell, ifcopenshell.util.placement, ifcopenshell.util.unit
    f = ifcopenshell.open(IFC)
    unit = ifcopenshell.util.unit.calculate_unit_scale(f)
    out, no_axis = [], []
    for w in f.by_type("IfcWall"):
        reps = [r for r in (w.Representation.Representations if w.Representation else []) if r.RepresentationIdentifier == "Axis"]
        if not reps:
            no_axis.append(w.GlobalId); continue
        M = np.asarray(ifcopenshell.util.placement.get_local_placement(w.ObjectPlacement), dtype=np.float64)
        pts = []
        for it in reps[0].Items:
            if it.is_a("IfcPolyline"):
                pts = [list(p.Coordinates) for p in it.Points]
            elif it.is_a("IfcTrimmedCurve") and it.BasisCurve.is_a("IfcLine"):
                L = it.BasisCurve; p0 = np.asarray(L.Pnt.Coordinates, float); v = np.asarray(L.Dir.Orientation.DirectionRatios, float)
                v = v / np.linalg.norm(v) * L.Dir.Magnitude
                ts = [t.wrappedValue if hasattr(t, "wrappedValue") else (t[0] if isinstance(t, tuple) else t) for t in (it.Trim1[0], it.Trim2[0])]
                pts = [list(p0 + float(ts[0]) * v), list(p0 + float(ts[1]) * v)]
        if len(pts) < 2:
            no_axis.append(w.GlobalId); continue
        P = np.array([p + [0.0] * (3 - len(p)) for p in pts], float)
        Pw = (P @ M[:3, :3].T + M[:3, 3]) * unit
        for a, b in zip(Pw[:-1], Pw[1:]):
            out.append({"wall": w.GlobalId, "name": w.Name, "a": a[:2].tolist(), "b": b[:2].tolist()})
    return out, no_axis


def clip(seg, W):
    a, b = np.asarray(seg["a"]), np.asarray(seg["b"]); L = np.linalg.norm(b - a)
    if L < 1e-6:
        return None
    t = (b - a) / L; n = np.array([-t[1], t[0]])
    rel = W[:, :2] - a
    m = np.abs(rel @ n) <= 0.25
    if m.sum() < 50:
        return None
    s = rel[m] @ t
    lo, hi = max(float(s.min()), 0.0), min(float(s.max()), L)
    if hi - lo < 0.3:
        return None
    return [*(a + lo * t), *(a + hi * t)]


def midlines(both_segs):
    """B1 の両面の線分から、平行で 0.1〜0.3 m 離れた対の中点の線"""
    out = []
    for i in range(len(both_segs)):
        for j in range(i + 1, len(both_segs)):
            p, q = both_segs[i], both_segs[j]
            dp, dq = p[2:] - p[:2], q[2:] - q[:2]
            ang = np.degrees(np.arccos(min(1, abs(dp @ dq) / np.linalg.norm(dp) / np.linalg.norm(dq))))
            n = np.array([-dp[1], dp[0]]) / np.linalg.norm(dp)
            d = abs((q[:2] - p[:2]) @ n)
            if ang < 2 and 0.1 <= d <= 0.3:
                out.append((p + q[[0, 1, 2, 3]] if (dp @ dq) > 0 else p + q[[2, 3, 0, 1]]) / 2)
    return out


def main():
    segs, no_axis = axes()
    print("IfcWall axis segments", len(segs), "walls without Axis", len(no_axis), flush=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (name, both) in zip(axs, NAMES.items()):
        W = read_pcd(os.path.join(ROOT, "BIM", both, "bim.pcd")) + np.array([0, 0, Z_FLOOR])   # BIM 座標に戻す
        lines, rec = [], []
        for s in segs:
            c = clip(s, W)
            if c is not None:
                lines.append(c); rec.append({"wall": s["wall"], "name": s["name"], "source": "Axis"})
        if no_axis:
            for m in midlines(np.loadtxt(os.path.join(ROOT, "BIM", both, "bimLineSeg.txt"))):
                lines.append(list(m)); rec.append({"wall": None, "name": None, "source": "midline_of_B1_pair"})
        L = np.asarray(lines)
        d = os.path.join(ROOT, "BIM", name + "_axis"); os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        np.savetxt(os.path.join(d, "bimLineSeg.txt"), L, fmt="%.5f")
        pts2, pts3 = [], []
        for x in L:
            a, b = x[:2], x[2:]; n = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.05)) + 1)
            for u in np.linspace(0, 1, n):
                p = a + u * (b - a); pts2.append([p[0], p[1], 0.0])
                for z in np.arange(0.2, 2.5001, 0.1):
                    pts3.append([p[0], p[1], z])
        write_pcd(os.path.join(d, "bim2d.pcd"), pts2); write_pcd(os.path.join(d, "bim.pcd"), pts3)
        tot = float(np.sum(np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])))
        meta = {"z_floor": Z_FLOOR, "n_segments": int(len(L)), "segment_length_total_m": tot, "walls": rec,
                "n_walls_without_axis_in_ifc": len(no_axis), "source": "IfcWall Axis representation (clipped to the reference walls of %s)" % both}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        print(name + "_axis", "segments", len(L), "total %.1f m" % tot, "sources", {k: sum(r["source"] == k for r in rec) for k in ("Axis", "midline_of_B1_pair")}, flush=True)
        for variant, col, lw in ((name, "#2a78d6", 2.4), (both, "#eb6834", 1.2), (name + "_axis", "#111111", 1.6)):
            for x in np.atleast_2d(np.loadtxt(os.path.join(ROOT, "BIM", variant, "bimLineSeg.txt"))):
                ax.plot([x[0], x[2]], [x[1], x[3]], color=col, lw=lw)
        ax.plot([], [], color="#2a78d6", lw=2.4, label="室内面（段階1）"); ax.plot([], [], color="#eb6834", lw=1.2, label="両面（B1）")
        ax.plot([], [], color="#111111", lw=1.6, label="基準線 Axis（B3）")
        ax.set_aspect("equal"); ax.set_title(name, loc="left"); ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "BIM", "axis_compare.png"), dpi=110)


if __name__ == "__main__":
    main()

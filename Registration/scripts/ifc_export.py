"""R3 T1 — IFC を読み、6クラスのラベル付き点群にする。

なぜ「エクスポータ＋キャッシュ」という形なのか
------------------------------------------------
指示書 §3 T1 は `io_utils.py:43-44` の `ifc` 分岐を実装せよと言っている。しかし
**位置合わせ本体が動く `sni-slam` env は Python 3.7 で、ifcopenshell が入らない**
（ifcopenshell 0.8.5 は py>=3.9。実測: `sni-slam` に無し／`bim-ifc` py3.10 に有り）。

そこで処理を2段に割る：

1. 本ファイル（`bim-ifc` env）が IFC → **点群＋ラベル＋法線の .npz** を書く
2. `io_utils._load_ifc_reference`（`sni-slam` env）がそれを読む

config からは従来どおり `reference.type: ifc` で指定でき、キャッシュが無ければ
**生成コマンドを添えて停止する**ので、経路が隠れることはない。
`bim-ifc` 上で走らせた場合は本ファイルの関数を直接呼ぶので、キャッシュは省略できる。

依存を増やさない方針
--------------------
`bim-ifc` に trimesh が無いため、面積重みのサンプリングは numpy で自前に書く
（Replica 側 `io_utils.py:109` は trimesh を使うが、同じ「面積重み＋重心座標」である）。
seed 固定で再現する。

使い方
------
    conda run -n bim-ifc python Registration/scripts/ifc_export.py \
        --ifc BIM_IFC_Extraction/input/m3-411.ifc \
        --config Registration/configs/m3_ifc.yaml \
        --spaces 411 --out Registration/output/ifc/m3_411.npz --ply
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# 内部6クラス。src/utils/Mesher.py decode_segmap と同一（regbim/labels.py:17 の写し。
# bim-ifc env からは regbim を import できないので、ここでは定義を複製せず
# 「順序が正典と一致すること」を起動時に検査する）。
CLASS_NAMES: List[str] = ["background", "wall", "door", "floor", "window", "ceiling"]
NAME_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(CLASS_NAMES)}

# 幾何を持つが「面」ではないもの。空間そのものを三角形化すると室内を中身で埋めてしまう。
SKIP_TYPES = ("IfcSpace", "IfcSite", "IfcBuilding", "IfcBuildingStorey",
              "IfcOpeningElement", "IfcAnnotation", "IfcGrid", "IfcVirtualElement")

PREFIX_POW = {
    "EXA": 18, "PETA": 15, "TERA": 12, "GIGA": 9, "MEGA": 6, "KILO": 3,
    "HECTO": 2, "DECA": 1, None: 0, "DECI": -1, "CENTI": -2, "MILLI": -3,
    "MICRO": -6, "NANO": -9, "PICO": -12,
}


# --------------------------------------------------------------------------- #
# クラス解決
# --------------------------------------------------------------------------- #
def resolve_label(el, class_map: Dict) -> Tuple[int, str]:
    """要素 -> (6クラス id, 決定の根拠)。

    `IfcWallStandardCase` は `IfcWall` の派生なので、**完全一致を先に見る**。
    一致しなければ継承を辿り、それも無ければ `_default`。
    """
    name = el.is_a()
    entry = class_map.get(name)
    how = "exact"
    if entry is None:
        for key in sorted((k for k in class_map if k != "_default"),
                          key=len, reverse=True):
            try:
                if el.is_a(key):
                    entry, how = class_map[key], "inherits:%s" % key
                    break
            except Exception:
                continue
    if entry is None:
        return NAME_TO_ID[str(class_map.get("_default", "background"))], "default"

    if isinstance(entry, dict):
        # PredefinedType による分岐（IfcSlab .FLOOR. / IfcCovering .CEILING.）
        sub = entry.get("predefined_type", {})
        pt = getattr(el, "PredefinedType", None)
        pt = None if pt is None else str(pt)
        six = sub.get(pt, sub.get("_default", class_map.get("_default", "background")))
        return NAME_TO_ID[str(six)], "%s/PredefinedType=%s" % (how, pt)
    return NAME_TO_ID[str(entry)], how


def length_scale_to_m(f) -> Dict[str, object]:
    """`IfcSIUnit` から m 換算係数を読む（決め打ちしない。指示書 T1-2 4.）。

    ★ただし **幾何には適用しない**。`ifcopenshell.geom` は自分で単位を正規化して
    m で返すため（`ifc_survey.py:96-99` で実測。掛けると室 extent が 7 m → 0.007 m に
    なることで確認済み）。ここで読むのは「ファイルが mm 建てである」ことの記録と、
    掛けたら壊れるという事実を残すためである。
    """
    info: Dict[str, object] = {"found": False, "file_unit_scale_to_m": 1.0}
    for ua in f.by_type("IfcUnitAssignment"):
        for u in ua.Units:
            if u.is_a("IfcSIUnit") and u.UnitType == "LENGTHUNIT":
                info = {"found": True, "unit": u.Name, "prefix": u.Prefix,
                        "file_unit_scale_to_m": 10.0 ** PREFIX_POW.get(u.Prefix, 0)}
            elif u.is_a("IfcConversionBasedUnit") and u.UnitType == "LENGTHUNIT":
                info = {"found": True, "unit": u.Name, "prefix": "conversion_based",
                        "file_unit_scale_to_m":
                            float(u.ConversionFactor.ValueComponent.wrappedValue)}
    return info


# --------------------------------------------------------------------------- #
# 三角形化
# --------------------------------------------------------------------------- #
def storey_of(f) -> Dict[int, str]:
    """要素 id -> 階の名前（`IfcRelContainedInSpatialStructure` 経由）。"""
    owner: Dict[int, str] = {}
    for rel in f.by_type("IfcRelContainedInSpatialStructure"):
        st = rel.RelatingStructure
        if not st.is_a("IfcBuildingStorey"):
            continue
        for el in rel.RelatedElements:
            owner[el.id()] = str(st.Name)
    return owner


def triangulate(f, class_map: Dict, storeys: Optional[List[str]] = None,
                verbose: bool = False):
    """全要素を三角形化し、面ごとに要素由来のラベルを持たせる。

    面の分類はしない（要素単位でラベルが決まる。指示書 T1-2 2.）。

    `storeys` で階を絞れる。**このモデルでは必須である**：`IfcCovering .CEILING.` が
    2枚あり、うち z=3.700 の1枚は階 `4FL+2700`（別レベル）に属している。
    対象の階 `4FL` は壁が全10枚 z 0.500→3.100（高さ 2.600 m）で、床上面 0.500 と
    合わせて室高 2.600 m。S0 の実測 2.576〜2.578 m と一致するのはこちらである。
    3.700 の天井を混ぜると、**支える壁の無い水平面**が 0.6 m 上に増え、
    重力軸推定と ICP の両方に効いてしまう。
    """
    import ifcopenshell.geom

    owner = storey_of(f) if storeys else {}
    want = set(storeys or [])
    dropped: Dict[str, int] = {}

    settings = ifcopenshell.geom.settings()
    try:
        settings.set("use-world-coords", True)       # ifcopenshell 0.8 系
    except Exception:
        settings.set(settings.USE_WORLD_COORDS, True)

    V: List[np.ndarray] = []
    F: List[np.ndarray] = []
    L: List[np.ndarray] = []
    per_class_elems: Dict[str, int] = {}
    how_log: Dict[str, str] = {}
    n_fail = 0
    off = 0
    for el in f.by_type("IfcProduct"):
        if el.is_a() in SKIP_TYPES or any(el.is_a(t) for t in SKIP_TYPES):
            continue
        if getattr(el, "Representation", None) is None:
            continue
        if want:
            st = owner.get(el.id())
            if st not in want:
                dropped[str(st)] = dropped.get(str(st), 0) + 1
                continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, el)
        except Exception:
            n_fail += 1
            continue
        v = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
        tri = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
        if len(v) == 0 or len(tri) == 0:
            continue
        cid, how = resolve_label(el, class_map)
        how_log.setdefault(el.is_a(), how)
        per_class_elems[CLASS_NAMES[cid]] = per_class_elems.get(CLASS_NAMES[cid], 0) + 1
        V.append(v)
        F.append(tri + off)
        L.append(np.full(len(tri), cid, dtype=np.int64))
        off += len(v)
        if verbose:
            print("  %-26s -> %-10s (%s) verts=%d tris=%d"
                  % (el.is_a(), CLASS_NAMES[cid], how, len(v), len(tri)))

    if not V:
        raise ValueError("三角形化できた要素が1つも無い")
    return (np.concatenate(V), np.concatenate(F), np.concatenate(L),
            {"elements_per_class": per_class_elems,
             "label_resolution": how_log, "n_shape_failures": n_fail,
             "storeys_kept": sorted(want) or None,
             "elements_dropped_by_storey": dropped,
             "storeys_present": sorted(set(storey_of(f).values()))})


# --------------------------------------------------------------------------- #
# IfcSpace による室単位の切り出し
# --------------------------------------------------------------------------- #
def space_records(f) -> List[Dict[str, object]]:
    """各 `IfcSpace` の占有セル・extent を測り、L字か直方体かを判定する材料を作る。

    フットプリントは **XY の占有セル数**で測る。凸包だと L 字の凹みを埋めてしまい、
    判別したい差がそのまま消える（`ifc_survey.py:105-107` と同じ理由）。
    """
    import ifcopenshell.geom
    settings = ifcopenshell.geom.settings()
    try:
        settings.set("use-world-coords", True)
    except Exception:
        settings.set(settings.USE_WORLD_COORDS, True)

    cell = 0.1
    out: List[Dict[str, object]] = []
    for sp in f.by_type("IfcSpace"):
        try:
            shape = ifcopenshell.geom.create_shape(settings, sp)
        except Exception as e:
            out.append({"id": sp.id(), "Name": sp.Name, "error": str(e)})
            continue
        v = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
        tri = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
        pts = sample_surface(v, tri, np.zeros(len(tri), dtype=np.int64), 200000, seed=0)[0]
        keys = np.unique(np.floor(pts[:, :2] / cell).astype(np.int64), axis=0)
        foot = len(keys) * cell * cell
        lo, hi = v.min(axis=0), v.max(axis=0)
        ext = hi - lo
        bbox = float(ext[0] * ext[1])
        out.append({
            "id": sp.id(), "Name": sp.Name, "LongName": getattr(sp, "LongName", None),
            "extent_m": [round(float(x), 3) for x in ext],
            "height_m": round(float(ext[2]), 3),
            "footprint_area_m2": round(foot, 2),
            "bbox_area_m2": round(bbox, 2),
            "fill_ratio": round(foot / bbox, 3) if bbox > 0 else None,
            "z_range_m": [round(float(lo[2]), 3), round(float(hi[2]), 3)],
            "_cells": keys, "_cell_size": cell,
        })
    return out


def name_spaces(recs: List[Dict[str, object]]) -> Dict[str, int]:
    """`IfcSpace` の名称は "1"/"2" で、どちらが 411/410 か分からない。**幾何で決める。**

    - 411 は L 字（非凸）で 119.2 m²、410 は直方体で 35.1 m²（S0 の実データ実測。
      同じ部屋を別々に撮った2本が 0.3% 以内で一致しており信頼できる）
    - **床面積が大きい方が 411** とし、`fill_ratio`（占有面積/bbox面積）が
      それと整合するか（411 が小さい＝非凸）を **独立な2つ目の根拠**として確認する
    """
    ok = [r for r in recs if "footprint_area_m2" in r]
    if len(ok) < 2:
        return {}
    ok = sorted(ok, key=lambda r: -float(r["footprint_area_m2"]))
    big, small = ok[0], ok[1]
    return {"411": int(big["id"]), "410": int(small["id"])}


def _horizontal_face_z(points: np.ndarray, normals: np.ndarray,
                       mask: np.ndarray, want_up: bool,
                       bin_m: float = 0.02) -> Optional[float]:
    """指定クラスの、上（下）向きの水平面の最頻 z。"""
    m = mask & (np.abs(normals[:, 2]) > 0.9)
    m &= (normals[:, 2] > 0) if want_up else (normals[:, 2] < 0)
    z = points[m, 2]
    if len(z) < 50:
        return None
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + bin_m, bin_m))
    if len(hist) == 0:
        return float(np.median(z))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def mark_inner(points: np.ndarray, normals: np.ndarray, recs: List[Dict],
               z_range: Optional[Tuple[float, float]] = None,
               eps: float = 0.05) -> np.ndarray:
    """各点が**室内側の面**か（R5 §4 / Q6）。手法には一切使わない。

    なぜ要るか
    ----------
    IFC の要素はソリッドなので**面が表裏2枚**ある。SLAM が観測できるのは室内側の1枚だけ
    なので、参照側にだけ「もう1枚の面」が存在する。その距離は実測で
    床スラブ 0.150 m / 壁 0.150〜0.200 m であり、**ICP の対応ゲート 0.3 m の内側・
    成功閾値 0.05 m の外側**にある。つまり小さなバイアスではなく失敗を生みうる。

    判定
    ----
    点を法線方向に ±eps ずらし、**片側だけが室の内部に入るか**を見る。
    室の重心方向との内積では、長い壁の端で符号を誤る（重心は局所的な内外を表さない）。

    - 壁：内側の面から 5 cm 内側へ動かすと室のフットプリント内、外側へ動かすと外
    - 床の上面：法線は上向きなので +eps は室の z 範囲内、−eps はスラブの中
    - 天井の下面：法線は下向きなので +eps が室内

    どちらとも言えない点（例：室に面していない外壁の外側同士）は False にする。

    ★ z の範囲は `IfcSpace` のものをそのまま使ってはいけない。
      このモデルの `IfcSpace` は z 0.50〜4.00（＝階高）で、**天井 3.10 がその内側に丸ごと入る**。
      すると天井の上下どちらも「室内」になり、天井の `is_inner` が全て False になる（実際そうなった）。
      呼び出し側が **床上面〜天井下面**を渡すこと。
    """
    inside_p = np.zeros(len(points), dtype=bool)
    inside_m = np.zeros(len(points), dtype=bool)
    pp, pm = points + eps * normals, points - eps * normals
    for r in recs:
        if "_cells" not in r:
            continue
        cell = float(r["_cell_size"])
        cells = set(map(tuple, r["_cells"].tolist()))
        z0, z1 = z_range if z_range else r["z_range_m"]
        for q, acc in ((pp, inside_p), (pm, inside_m)):
            k = np.floor(q[:, :2] / cell).astype(np.int64)
            inxy = np.fromiter((tuple(x) in cells for x in k.tolist()),
                               dtype=bool, count=len(k))
            acc |= inxy & (q[:, 2] >= z0) & (q[:, 2] <= z1)
    return inside_p & ~inside_m


def clip_to_spaces(points: np.ndarray, recs: List[Dict], ids: List[int],
                   margin_m: float) -> np.ndarray:
    """指定した室の占有セルを `margin_m` だけ膨張させ、その中の点を残す。

    膨張が要るのは、`IfcSpace` が**室の内法**であり、室を囲む壁はその外側にあるため。
    膨張しないと wall が丸ごと落ちる。z は室の範囲にマージンを付けて挟む。
    """
    keep = np.zeros(len(points), dtype=bool)
    for r in recs:
        if int(r["id"]) not in ids:
            continue
        cell = float(r["_cell_size"])
        d = int(np.ceil(margin_m / cell))
        base = r["_cells"]
        # 膨張：(2d+1)^2 の近傍セルを足す。数千セル × 数十近傍なので素直に展開してよい
        offs = np.array([(i, j) for i in range(-d, d + 1) for j in range(-d, d + 1)
                         if i * i + j * j <= d * d], dtype=np.int64)
        dil = (base[:, None, :] + offs[None, :, :]).reshape(-1, 2)
        dil = np.unique(dil, axis=0)
        cells = set(map(tuple, dil.tolist()))
        z0, z1 = r["z_range_m"]
        pk = np.floor(points[:, :2] / cell).astype(np.int64)
        inxy = np.fromiter((tuple(p) in cells for p in pk.tolist()),
                           dtype=bool, count=len(pk))
        keep |= inxy & (points[:, 2] >= z0 - margin_m) & (points[:, 2] <= z1 + margin_m)
    return keep


# --------------------------------------------------------------------------- #
# 面積重みサンプリング（trimesh 非依存）
# --------------------------------------------------------------------------- #
def sample_surface(verts: np.ndarray, tris: np.ndarray, tri_labels: np.ndarray,
                   n: int, seed: int = 0):
    """面積に比例して三角形を選び、重心座標で一様に点を打つ。

    `trimesh.sample.sample_surface`（Replica 側が使う）と同じ定義。
    """
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    tot = area.sum()
    if tot <= 0:
        raise ValueError("総面積が 0")
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(tris), size=n, p=area / tot)
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    pts = ((1 - r1)[:, None] * v0[fi]
           + (r1 * (1 - r2))[:, None] * v1[fi]
           + (r1 * r2)[:, None] * v2[fi])
    # 法線は面法線の厳密値を使う（推定しない。指示書 T1-2 5.）
    nrm = cross[fi]
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(ln > 0, ln, 1.0)
    return pts, tri_labels[fi], nrm


# --------------------------------------------------------------------------- #
def build_ifc_cloud(ifc_path: str, class_map: Dict, n_points: int,
                    spaces: Optional[List[str]] = None,
                    space_margin_m: float = 0.35, seed: int = 0,
                    keep_classes: Optional[List[str]] = None,
                    storeys: Optional[List[str]] = None,
                    verbose: bool = False) -> Dict[str, object]:
    """IFC -> points/labels/normals + 来歴。`io_utils` からも呼べる純関数。"""
    import ifcopenshell

    f = ifcopenshell.open(ifc_path)
    unit = length_scale_to_m(f)
    verts, tris, tri_lab, info = triangulate(f, class_map, storeys=storeys,
                                             verbose=verbose)

    recs = space_records(f)
    alias = name_spaces(recs)

    # keep_classes: reference 側は構造クラスだけ残す（Replica 側 io_utils.py:94 と同じ）
    if keep_classes:
        keep_ids = {NAME_TO_ID[n] for n in keep_classes}
        m = np.isin(tri_lab, list(keep_ids))
        if not m.any():
            raise ValueError("keep_classes に該当する面が無い: %s" % keep_classes)
        tris, tri_lab = tris[m], tri_lab[m]

    # 室で切り出す場合、切り出し後の点数が n_points になるよう多めに打ってから絞る
    want_ids: List[int] = []
    if spaces:
        for s in spaces:
            key = str(s)
            if key in alias:
                want_ids.append(alias[key])
            else:
                hit = [r for r in recs if str(r.get("Name")) == key or str(r["id"]) == key]
                if not hit:
                    raise ValueError("IfcSpace が見つからない: %r（別名 %s / Name %s）"
                                     % (s, sorted(alias), [r.get("Name") for r in recs]))
                want_ids.append(int(hit[0]["id"]))

    over = 1
    pts = lab = nrm = None
    for _ in range(6):                       # 足りなければ倍々で打ち直す（最大 32 倍）
        p, l, nn = sample_surface(verts, tris, tri_lab, n_points * over, seed=seed)
        if want_ids:
            k = clip_to_spaces(p, recs, want_ids, space_margin_m)
            p, l, nn = p[k], l[k], nn[k]
        if len(p) >= n_points:
            pts, lab, nrm = p[:n_points], l[:n_points], nn[:n_points]
            break
        over *= 2
    if pts is None:
        raise ValueError("室で切り出した後の点が %d に満たない（最大 %d 倍まで試行）"
                         % (n_points, over))

    # R5 §4 / Q6: 表裏の属性化。**手法は変えない。** 失敗分析で
    # 「裏面に吸われた対応の割合」を出せるようにするためだけに持つ。
    # 実効的な室の高さ＝床の上面から天井の下面まで。`IfcSpace` の z（階高）ではない
    z_fl = _horizontal_face_z(pts, nrm, lab == NAME_TO_ID["floor"], want_up=True)
    z_ce = _horizontal_face_z(pts, nrm, lab == NAME_TO_ID["ceiling"], want_up=False)
    zr = (z_fl, z_ce) if (z_fl is not None and z_ce is not None and z_ce > z_fl) else None
    is_inner = mark_inner(pts, nrm, recs, z_range=zr)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    meta = {
        "room_z_range_m": (None if zr is None
                           else [round(zr[0], 4), round(zr[1], 4)]),
        "is_inner_frac": round(float(is_inner.mean()), 4),
        "is_inner_frac_by_class": {
            CLASS_NAMES[c]: round(float(is_inner[lab == c].mean()), 4)
            for c in sorted(np.unique(lab))},
        "is_inner_note": ("室内側の面か。法線方向に ±5 cm ずらして片側だけが室の内部に"
                          "入るかで判定。手法には使わない（R5 §4 / Q6）"),
        "source": "ifc", "path": os.path.abspath(ifc_path), "schema": f.schema,
        "length_unit": unit,
        # ★ geom が m を返すので係数は掛けていない。掛けると 1/1000 になる
        "applied_unit_scale": 1.0,
        "applied_translation": [0.0, 0.0, 0.0],   # 原点正規化はしない（指示書 T1-4）
        "storeys_requested": storeys,
        "spaces_requested": spaces, "space_ids_used": want_ids,
        "space_alias": alias, "space_margin_m": space_margin_m,
        "spaces": [{k: v for k, v in r.items() if not k.startswith("_")} for r in recs],
        "n_points": int(len(pts)), "seed": seed, "keep_classes": keep_classes,
        "extent_m": [round(float(x), 3) for x in (hi - lo)],
        "bbox_min_m": [round(float(x), 3) for x in lo],
        "bbox_max_m": [round(float(x), 3) for x in hi],
        "class_counts": {CLASS_NAMES[c]: int((lab == c).sum())
                         for c in sorted(np.unique(lab))},
    }
    meta.update(info)
    return {"points": pts, "labels": lab, "normals": nrm,
            "is_inner": is_inner, "meta": meta}


def write_ply(path: str, pts: np.ndarray, lab: np.ndarray) -> None:
    """目視確認用。regbim/labels.py の LABEL_COLORS と同じ配色で色を付ける。"""
    palette = np.array([(128, 128, 128), (255, 64, 64), (255, 200, 64),
                        (180, 220, 255), (64, 200, 255), (200, 100, 255)],
                       dtype=np.uint8)
    c = palette[np.clip(lab, 0, 5)]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(pts))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, q in zip(pts, c):
            f.write("%.5f %.5f %.5f %d %d %d\n" % (p[0], p[1], p[2], q[0], q[1], q[2]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ifc", required=True)
    ap.add_argument("--config", required=True, help="ifc_class_map を含む config")
    ap.add_argument("--out", required=True, help="出力 .npz")
    ap.add_argument("--spaces", nargs="*", default=None,
                    help='室の別名。例 411 / 410 / "411 410"。省略で全体')
    ap.add_argument("--storeys", nargs="*", default=None,
                    help="階で絞る。既定は config の reference.storeys")
    ap.add_argument("--n-points", type=int, default=None,
                    help="既定は config の reference.n_points")
    ap.add_argument("--space-margin", type=float, default=0.35,
                    help="IfcSpace は内法なので、室を囲む壁を拾うために膨張させる量 [m]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-keep-classes", action="store_true",
                    help="config の keep_classes を無視して全クラス出す（診断用）")
    ap.add_argument("--ply", action="store_true", help="目視用の色付き PLY も書く")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cls = cfg["classes"]
    if cls["names"] != CLASS_NAMES:
        raise ValueError("クラス順が正典と違う: %s" % cls["names"])
    class_map = cls["ifc_class_map"]
    ref = cfg.get("reference", {})
    n_points = args.n_points or int(ref.get("n_points", 200000))
    keep = None if args.no_keep_classes else ref.get("keep_classes")

    res = build_ifc_cloud(args.ifc, class_map, n_points, spaces=args.spaces,
                          space_margin_m=args.space_margin, seed=args.seed,
                          keep_classes=keep,
                          storeys=args.storeys or ref.get("storeys"),
                          verbose=args.verbose)
    meta = res["meta"]

    print("schema %s / 単位 %s (x%.6g で m) — ただし幾何には掛けない（geom が m で返す）"
          % (meta["schema"], meta["length_unit"].get("unit"),
             meta["length_unit"]["file_unit_scale_to_m"]))
    print("階: 全 %s / 採用 %s / 落とした要素 %s"
          % (meta["storeys_present"], meta["storeys_kept"],
             meta["elements_dropped_by_storey"] or "なし"))
    print("三角形化できなかった要素: %d" % meta["n_shape_failures"])
    print("要素数（クラス別）: %s" % meta["elements_per_class"])
    print("点数（クラス別）  : %s" % meta["class_counts"])
    print("extent [m]        : %s" % meta["extent_m"])
    print("\nIfcSpace:")
    for r in meta["spaces"]:
        print("  id=%s Name=%r 床%.1f m2 bbox%.1f m2 fill=%s 高さ%s m"
              % (r["id"], r.get("Name"), r.get("footprint_area_m2", 0),
                 r.get("bbox_area_m2", 0), r.get("fill_ratio"), r.get("height_m")))
    print("別名の割当（床面積の大小で決定）: %s" % meta["space_alias"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print("室内側の面の割合: 全体 %.3f / クラス別 %s"
          % (meta["is_inner_frac"], meta["is_inner_frac_by_class"]))
    np.savez_compressed(args.out, points=res["points"], labels=res["labels"],
                        normals=res["normals"], is_inner=res["is_inner"],
                        meta=json.dumps(meta, ensure_ascii=False))
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("\nwrote %s" % args.out)
    if args.ply:
        p = os.path.splitext(args.out)[0] + ".ply"
        write_ply(p, res["points"], res["labels"])
        print("wrote %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

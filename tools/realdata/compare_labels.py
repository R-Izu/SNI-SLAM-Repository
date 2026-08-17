"""S2 A/B ゲート — 解像度案・ラベル案を並べて比較する。

指示書 §5 S2 の受入条件
  - 解像度①アナモルフィック／②レターボックスの比較結果と採用理由
  - wall / floor / ceiling の3クラスが**いずれも一定割合で出ている**ことの確認

出力
----
<out>/contact_<tag>.png     RGB と重畳を横に並べたコンタクトシート
<out>/compare_labels.json   variant ごとのクラス別画素率・3クラス被覆・不一致率

使い方
------
    conda activate sni-slam
    python tools/realdata/compare_labels.py \
        --variant anamorphic=data/realdata_abtest/_abtest_anamorphic \
        --variant letterbox=data/realdata_abtest/_abtest_letterbox \
        --out output/RealData/_abtest --samples 6
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Replica 生ID -> (名前, BGR)
CLASS_COLORS: Dict[int, Tuple[str, Tuple[int, int, int]]] = {
    0:  ("background", (60, 60, 60)),
    93: ("wall",       (200, 120, 40)),
    37: ("door",       (40, 140, 240)),
    40: ("floor",      (60, 190, 90)),
    97: ("window",     (230, 220, 60)),
    31: ("ceiling",    (60, 60, 220)),
}


def frame_index(path: str) -> int:
    m = re.findall(r"\d+", os.path.basename(path))
    return int(m[0]) if m else -1


def colorize(lab: np.ndarray) -> np.ndarray:
    out = np.zeros(lab.shape + (3,), dtype=np.uint8)
    for v, (_, bgr) in CLASS_COLORS.items():
        out[lab == v] = bgr
    return out


def pixel_fraction(labels: List[np.ndarray]) -> Dict[str, float]:
    counts = np.zeros(256, dtype=np.int64)
    for lab in labels:
        counts += np.bincount(lab.reshape(-1), minlength=256)
    total = max(int(counts.sum()), 1)
    return {name: round(float(counts[v] / total), 5)
            for v, (name, _) in CLASS_COLORS.items()}


def load_variant(scene_dir: str, sem_name: str, idxs: List[int]
                 ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    rgbs, labs = [], []
    for i in idxs:
        r = cv2.imread(os.path.join(scene_dir, "rgb", "rgb_%d.png" % i), cv2.IMREAD_COLOR)
        l = cv2.imread(os.path.join(scene_dir, sem_name, "semantic_class_%d.png" % i),
                       cv2.IMREAD_UNCHANGED)
        if r is None or l is None:
            raise SystemExit("missing frame %d in %s (rgb=%s, label=%s)" % (
                i, scene_dir, r is not None, l is not None))
        rgbs.append(r)
        labs.append(l)
    return rgbs, labs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", required=True,
                    help="name=path 形式。複数指定可")
    ap.add_argument("--sem-name", default="semantic_class")
    ap.add_argument("--out", default="output/RealData/_abtest")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.55, help="重畳の不透明度")
    args = ap.parse_args()

    # name=path または name=path:semantic_dir_name
    variants: Dict[str, str] = {}
    sem_names: Dict[str, str] = {}
    for v in args.variant:
        name, _, spec = v.partition("=")
        if not spec:
            raise SystemExit("--variant は name=path[:sem_dir] 形式で指定すること: %r" % v)
        path, sep, sem = spec.partition(":")
        variants[name] = path
        sem_names[name] = sem if sep else args.sem_name
    os.makedirs(args.out, exist_ok=True)

    # 全 variant に共通して存在するフレームから等間隔にサンプルする
    common = None
    for name, path in variants.items():
        ids = {frame_index(p) for p in
               glob.glob(os.path.join(path, sem_names[name], "semantic_class_*.png"))}
        common = ids if common is None else (common & ids)
    if not common:
        raise SystemExit("variant 間で共通するフレームが無い")
    ordered = sorted(common)
    idxs = [ordered[int(round(t))] for t in
            np.linspace(0, len(ordered) - 1, min(args.samples, len(ordered)))]
    print("comparing frames %s" % idxs)

    report: Dict[str, object] = {"variants": {}, "frames": idxs,
                                 "sem_names": sem_names}
    rows: List[np.ndarray] = []
    loaded: Dict[str, List[np.ndarray]] = {}

    for name, path in variants.items():
        rgbs, labs = load_variant(path, sem_names[name], idxs)
        loaded[name] = labs
        frac = pixel_fraction(labs)
        present = {c: bool(frac[c] > 0.001) for c in ("wall", "floor", "ceiling")}

        # 指示書追補 Q6 の判定：フレームごとに wall/floor/ceiling の画素率を出し、
        # 「3クラスのいずれかが 2% 未満」のフレームが過半を占めるかを見る。
        per_frame = [pixel_fraction([l]) for l in labs]
        thr = 0.02
        below = [any(pf[c] < thr for c in ("wall", "floor", "ceiling")) for pf in per_frame]
        cls_ok_rate = {c: round(float(np.mean([pf[c] >= thr for pf in per_frame])), 3)
                       for c in ("wall", "floor", "ceiling")}
        report["variants"][name] = {
            "path": path, "sem_dir": sem_names[name], "pixel_fraction": frac,
            "structural_classes_present": present,
            "all_three_present": all(present.values()),
            "per_frame_class_ok_rate_at_2pct": cls_ok_rate,
            "frames_with_any_class_below_2pct": int(sum(below)),
            "n_frames_checked": len(labs),
            "q6_fail_majority_below_2pct": bool(sum(below) > len(labs) / 2.0),
        }
        tiles = []
        for r, l in zip(rgbs, labs):
            ov = cv2.addWeighted(r, 1.0 - args.alpha, colorize(l), args.alpha, 0)
            cv2.putText(ov, name, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(ov, name, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
            tiles.append(cv2.resize(ov, (480, 272)))
        rows.append(np.hstack(tiles))

    # 参考: 素の RGB 行を先頭に置く
    first = list(variants)[0]
    rgb0, _ = load_variant(variants[first], sem_names[first], idxs)
    rows.insert(0, np.hstack([cv2.resize(r, (480, 272)) for r in rgb0]))

    sheet = os.path.join(args.out, "contact_%s.png" % "_vs_".join(variants))
    cv2.imwrite(sheet, np.vstack(rows))

    # variant 間の不一致率（同一解像度・同一フレームのときのみ意味がある）
    names = list(variants)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = loaded[names[i]], loaded[names[j]]
            if a[0].shape != b[0].shape:
                continue
            dis = float(np.mean([np.mean(x != y) for x, y in zip(a, b)]))
            report.setdefault("disagreement", {})["%s_vs_%s" % (names[i], names[j])] = round(dis, 5)

    with open(os.path.join(args.out, "compare_labels.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    hdr = "%-14s %10s %10s %10s %10s %10s %10s  %s" % (
        "variant", "background", "wall", "door", "floor", "window", "ceiling", "3クラス")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name in variants:
        f_ = report["variants"][name]["pixel_fraction"]
        print("%-14s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f  %s" % (
            name, f_["background"], f_["wall"], f_["door"], f_["floor"],
            f_["window"], f_["ceiling"],
            "OK" if report["variants"][name]["all_three_present"] else "NG"))
    print("\n[Q6 判定] フレーム単位で wall/floor/ceiling が 2% 以上ある割合")
    print("%-14s %8s %8s %8s   %s" % ("variant", "wall", "floor", "ceiling",
                                      "3クラス未満が過半 (=案B移行)"))
    for name in variants:
        v = report["variants"][name]
        r = v["per_frame_class_ok_rate_at_2pct"]
        print("%-14s %8.3f %8.3f %8.3f   %s (%d/%d frames)" % (
            name, r["wall"], r["floor"], r["ceiling"],
            "FAIL" if v["q6_fail_majority_below_2pct"] else "PASS",
            v["frames_with_any_class_below_2pct"], v["n_frames_checked"]))
    if "disagreement" in report:
        print("\ndisagreement: %s" % report["disagreement"])
    print("\nwrote %s" % sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 2 案A — ADE20K 学習済みセグメンタで semantic_class/*.png を作る。

背景
----
`seg/dinov2_replica.pth` の分類層は 52 クラス版で、`n_classes: 6` と形が合わず
起動時に捨てられる（[[2026-08-10_r2_realdata_slam_pipeline]] §2-B #3）。
そのため SNI-SLAM のオンライン意味推論は使えず、**意味ラベルを外部から供給する**必要がある。
本スクリプトは既製の ADE20K セグメンタで推論し、画素値を Replica 生ID
`[0,93,37,40,97,31]` で書き出す。ローダは無改造で通る。

縦持ち撮影への対応（★）
-----------------------
今回の10本はすべて iPad を**縦持ち**で撮っており、`rgb.mp4` の横長フレームの中で
内容が 90° 回っている。ADE20K のモデルは正立画像で学習されているため、
そのまま入れると wall/floor/ceiling の精度が落ちる。

回転量は目視ではなく**重力方向から決める**：世界の上方向 (+Y) を各フレームの
カメラ座標へ移すと、画像平面内での「上」の向きが出る。これが画像の上
(0,-1) に最も近づく 90° の倍数を採用する。scene 全体の中央値で1つに決め、
推論時に正立へ回してから、ラベルを逆回転して元の向きに戻して保存する。

使い方
------
    conda activate seg2d
    python tools/realdata/gen_labels_ade20k.py --scene-dir data/realdata/m3_cor_d \
        --model facebook/mask2former-swin-large-ade-semantic --batch 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

ADE_UPRIGHT_SIZE = None   # 既定はプロセッサ既定サイズ


# ---------------------------------------------------------------------------
# 回転量の決定
# ---------------------------------------------------------------------------

def load_c2w_opencv(traj_path: str) -> np.ndarray:
    """traj.txt (16 値/行) を読み、OpenCV c2w に戻す。

    ``convert_stray.py`` は書き出し時に col 1,2 を反転している
    （ローダ `src/utils/datasets.py:198-199` が再反転するため）。
    ここでは同じ反転をもう一度かけて OpenCV c2w に戻す（反転は対合）。
    """
    rows = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    m = np.asarray(rows, dtype=np.float64).reshape(-1, 4, 4)
    m[:, :3, 1] *= -1
    m[:, :3, 2] *= -1
    return m


def rot_vec_like_rot90(v: Tuple[float, float], k: int) -> Tuple[float, float]:
    """``np.rot90(img, k)`` が画像内のベクトル (x=列方向, y=行方向) をどう動かすかを返す。

    ``np.rot90(a, 1)`` は out[i][j] = a[j][W-1-i]、すなわち画素 (row r, col c) を
    (W-1-c, r) へ移す。変位ベクトル (dx, dy) = (dcol, drow) はこれにより
    (dy, -dx) へ移る。角度の符号規約で間違えないよう、この写像を直接繰り返し適用する。
    """
    x, y = v
    for _ in range(k % 4):
        x, y = y, -x
    return x, y


def decide_rotation_k(c2w: np.ndarray) -> Tuple[int, Dict[str, float]]:
    """画像を正立にするための ``np.rot90`` の回転数 k を返す。

    世界の上 (0,1,0) をカメラ座標へ移した u_cam の画像平面成分 (u_x, u_y) が、
    回転後に画像の上 (0,-1) を向くような k を選ぶ。
    OpenCV 画像座標は x 右・y 下。

    ★ 角度計算で回転の向きを取り違えると floor と ceiling が 180° 入れ替わるため
      （実際に一度それで誤ラベルを出した）、角度ではなく上の写像を総当たりする。
    """
    R = c2w[:, :3, :3]
    up_world = np.array([0.0, 1.0, 0.0])
    u = np.einsum("nji,j->ni", R, up_world)       # R^T @ up = ワールド上方向のカメラ座標表現
    uv = u[:, :2]
    nrm = np.linalg.norm(uv, axis=1, keepdims=True)
    ok = nrm[:, 0] > 1e-6                          # 真上/真下を向いた frame は寄与させない
    uv = uv[ok] / nrm[ok]
    mean_u = uv.mean(axis=0)
    conc = float(np.linalg.norm(mean_u))           # 0..1 のまとまり具合

    scores = [-rot_vec_like_rot90(tuple(mean_u), kk)[1] for kk in range(4)]
    k = int(np.argmax(scores))
    return k, {
        "k_rot90": k,
        "up_in_image_mean": [round(float(v), 4) for v in mean_u],
        "concentration": round(conc, 3),
        "alignment_after_rot": round(float(scores[k] / max(conc, 1e-9)), 3),
        "frames_used": int(ok.sum()),
    }


def _self_check_rotation() -> None:
    """規約の自己検査。上方向が画像左 (-1,0) にあるなら k=3（時計回り）が正解。"""
    assert rot_vec_like_rot90((-1.0, 0.0), 3) == (0.0, -1.0), "rot90 の写像規約が壊れている"
    assert rot_vec_like_rot90((-1.0, 0.0), 1) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

class AdeSegmenter:
    def __init__(self, model_id: str, device: str = "cuda", fp16: bool = True):
        from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation
        from transformers import AutoModelForSemanticSegmentation
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.is_mask2former = "mask2former" in model_id.lower() or "maskformer" in model_id.lower()
        if self.is_mask2former:
            self.model = AutoModelForUniversalSegmentation.from_pretrained(model_id)
        else:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
        self.device = device
        self.dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        self.model = self.model.to(device=device, dtype=self.dtype).eval()
        self.id2label = getattr(self.model.config, "id2label", {}) or {}

    def check_label_order(self) -> Dict[str, str]:
        """ADE20K の並び（0=wall, 3=floor, 5=ceiling, 8=windowpane, 14=door）を確認する。"""
        want = {0: "wall", 3: "floor", 5: "ceiling", 8: "window", 14: "door"}
        got = {}
        for k, v in want.items():
            name = str(self.id2label.get(k, self.id2label.get(str(k), "?"))).lower()
            got[k] = name
            if v not in name:
                raise SystemExit(
                    "ADE20K のラベル並びが想定と違う: id %d は %r で、%r を含まない。"
                    "label_map_ade20k.yaml を確認すること。" % (k, name, v))
        return {str(k): v for k, v in got.items()}

    @torch.no_grad()
    def predict(self, images_rgb: List[np.ndarray]) -> List[np.ndarray]:
        """RGB (H,W,3) uint8 のリスト -> ADE20K id マップ (H,W) int32 のリスト。"""
        sizes = [(im.shape[0], im.shape[1]) for im in images_rgb]
        inputs = self.processor(images=images_rgb, return_tensors="pt")
        inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point()
                      else v.to(self.device)) for k, v in inputs.items()}
        out = self.model(**inputs)
        if self.is_mask2former:
            maps = self.processor.post_process_semantic_segmentation(
                out, target_sizes=sizes)
            return [m.cpu().numpy().astype(np.int32) for m in maps]
        logits = torch.nn.functional.interpolate(
            out.logits.float(), size=sizes[0], mode="bilinear", align_corners=False)
        return [m for m in logits.argmax(1).cpu().numpy().astype(np.int32)]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def frame_index(path: str) -> int:
    m = re.findall(r"\d+", os.path.basename(path))
    return int(m[0]) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", required=True,
                    help="convert_stray.py が作ったシーンディレクトリ")
    ap.add_argument("--model", default="facebook/mask2former-swin-large-ade-semantic")
    ap.add_argument("--label-map", default="tools/realdata/label_map_ade20k.yaml")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rotate-k", type=int, default=None,
                    help="正立化の回転量を手で指定（既定は重力方向から自動判定）")
    ap.add_argument("--no-rotate", action="store_true", help="回転補正を無効化（A/B 比較用）")
    ap.add_argument("--out-name", default="semantic_class")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--fix-aspect", dest="fix_aspect", action="store_true", default=True,
                    help="anamorphic シーンで、推論の前だけ縦横比を自然に戻す（既定 ON）。"
                         "ADE20K のモデルは自然な縦横比で学習されているため、"
                         "横に伸びた画像をそのまま入れると精度が落ちる。"
                         "ラベルは最近傍で元の anamorphic 格子へ戻すので、"
                         "SLAM が使う幾何（無効画素ゼロ）は変わらない")
    ap.add_argument("--no-fix-aspect", dest="fix_aspect", action="store_false")
    ap.add_argument("--pad-left", type=int, default=None,
                    help="letterbox の左黒帯の幅。指定すると黒帯を background に強制する"
                         "（既定は conversion_report.json から読む）")
    args = ap.parse_args()

    t0 = time.time()
    scene_dir = args.scene_dir
    rgb_paths = sorted(glob.glob(os.path.join(scene_dir, "rgb", "rgb_*.png")),
                       key=frame_index)
    if args.limit:
        rgb_paths = rgb_paths[:args.limit]
    if not rgb_paths:
        raise SystemExit("no rgb_*.png under %s/rgb" % scene_dir)
    out_dir = os.path.join(scene_dir, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(args.label_map) as f:
        lm = yaml.safe_load(f)
    rid = lm["replica_ids"]
    lut = np.zeros(256, dtype=np.uint8)          # ADE20K id -> Replica 生ID
    lut[:] = rid["background"]
    for ade_id, cls in lm["mapping"].items():
        lut[int(ade_id)] = rid[cls]

    # --- 回転量 ---
    _self_check_rotation()

    # --- 変換時の情報（mode・黒帯位置）を読む ---
    conv: Dict[str, object] = {}
    conv_path = os.path.join(scene_dir, "conversion_report.json")
    if os.path.exists(conv_path):
        with open(conv_path) as f:
            conv = json.load(f)
    mode = str(conv.get("mode", "anamorphic"))
    pad_left = args.pad_left
    inner_w = None
    if pad_left is None and mode == "letterbox":
        pad_left = int(conv.get("intrinsics_target", {}).get("pad_left", 0))
        inner_w = int(conv.get("intrinsics_target", {}).get("inner_w", 0)) or None

    # 推論時だけ自然な縦横比に戻す（anamorphic のみ）
    unstretch_w = None
    if args.fix_aspect and mode == "anamorphic":
        h, w = 680, 1200
        unstretch_w = int(round(h * 1920.0 / 1440.0))     # = 907
        print("aspect fix: %dx%d -> %dx%d for inference (labels mapped back)"
              % (w, h, unstretch_w, h))

    rot_info: Dict[str, object] = {"applied": False}
    k = 0
    if not args.no_rotate:
        if args.rotate_k is not None:
            k, rot_info = args.rotate_k % 4, {"k_rot90": args.rotate_k % 4, "source": "manual"}
        else:
            c2w = load_c2w_opencv(os.path.join(scene_dir, "traj.txt"))
            k, rot_info = decide_rotation_k(c2w)
            rot_info["source"] = "gravity"
        rot_info["applied"] = bool(k != 0)
    print("rotation for upright inference: k_ccw=%d  %s" % (k, rot_info))

    seg = AdeSegmenter(args.model, fp16=not args.fp32)
    label_check = seg.check_label_order()
    print("ADE20K label order OK: %s" % label_check)

    counts = np.zeros(256, dtype=np.int64)
    n_done = 0
    for s in range(0, len(rgb_paths), args.batch):
        chunk = rgb_paths[s:s + args.batch]
        imgs, shapes = [], []
        for p in chunk:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            shapes.append(rgb.shape[:2])
            if unstretch_w:                       # 推論の前だけ縦横比を戻す
                rgb = cv2.resize(rgb, (unstretch_w, rgb.shape[0]),
                                 interpolation=cv2.INTER_AREA)
            imgs.append(np.rot90(rgb, k).copy() if k else rgb)
        maps = seg.predict(imgs)
        for p, m, (h0, w0) in zip(chunk, maps, shapes):
            if k:
                m = np.rot90(m, -k).copy()       # 元の向きへ戻す
            if unstretch_w:                       # ラベルを元の anamorphic 格子へ戻す
                m = cv2.resize(m.astype(np.int32), (w0, h0),
                               interpolation=cv2.INTER_NEAREST)
            lab = lut[np.clip(m, 0, 255)]
            if pad_left:                          # letterbox の黒帯は background に固定
                lab[:, :pad_left] = rid["background"]
                right = (pad_left + inner_w) if inner_w else (lab.shape[1] - pad_left)
                lab[:, right:] = rid["background"]
            counts += np.bincount(lab.reshape(-1), minlength=256)
            cv2.imwrite(os.path.join(out_dir, "semantic_class_%d.png" % frame_index(p)), lab)
        n_done += len(chunk)
        if n_done % 200 < args.batch:
            el = time.time() - t0
            print("  %d/%d  %.1fs  (%.3f s/frame, ETA %.1f min)" % (
                n_done, len(rgb_paths), el, el / n_done,
                (len(rgb_paths) - n_done) * el / n_done / 60), flush=True)

    total = counts.sum()
    name_of = {v: k2 for k2, v in rid.items()}
    pixel_frac = {name_of[v]: round(float(counts[v] / total), 5)
                  for v in sorted(set(rid.values()))}
    report = {
        "scene_dir": scene_dir, "model": args.model, "n_frames": len(rgb_paths),
        "mode": mode, "aspect_fix_width": unstretch_w, "pad_left_forced_bg": pad_left,
        "rotation": rot_info, "label_map": args.label_map,
        "ade20k_label_check": label_check,
        "pixel_fraction": pixel_frac,
        "structural_classes_present": {
            c: bool(pixel_frac.get(c, 0) > 0.001) for c in ("wall", "floor", "ceiling")},
        "sec_per_frame": round((time.time() - t0) / max(len(rgb_paths), 1), 4),
        "elapsed_s": round(time.time() - t0, 1),
        "out_dir": out_dir,
    }
    with open(os.path.join(scene_dir, "label_report_%s.json" % args.out_name), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\npixel fraction: %s" % pixel_frac)
    print("wall/floor/ceiling present: %s" % report["structural_classes_present"])
    print("%.4f s/frame, total %.1f s -> %s" % (
        report["sec_per_frame"], report["elapsed_s"], out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

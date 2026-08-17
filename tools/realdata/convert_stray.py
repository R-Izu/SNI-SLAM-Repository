"""Phase 1 — Stray Scanner のスキャンを SNI-SLAM の入力形式へ変換する。

生成物（``--out``/<scene>/ 以下）::

    rgb/rgb_<i>.png              1200x680 BGR
    depth/depth_<i>.png          1200x680 uint16 (mm), png_depth_scale=1000
    traj.txt                     1 行 16 値の c2w（ローダが col 1,2 を再反転する前提）
    conversion_report.json       採用した姿勢規約・内部パラメータ・bound・検証値
    configs/RealData/<scene>.yaml（リポジトリ側に生成）

``semantic_class/`` は本スクリプトでは作らない（Phase 2 の gen_labels_*.py が作る）。

解像度の2案（指示書 S2 の A/B ゲート対象）
-----------------------------------------
anamorphic : 1920x1440 を 1200x680 へ非等方縮小。幾何は厳密に保存され無効画素が出ない。
             画は横に伸びる。fx,fy を別々にスケールする。
letterbox  : アスペクトを保って 907x680 に縮小し、左右へ黒帯を入れて 1200x680 にする。
             画は自然だが左右 146/147 px が無効（depth=0）になる。

姿勢規約
--------
``odometry.csv`` の姿勢を OpenCV / OpenGL の2通りに解釈し、2 フレーム間 NN 距離と
床法線を実測して**良い方を自動採用**する（採用理由と両方の数値を必ず記録する）。
``src/utils/datasets.py:198-199`` が読み込み時に col 1,2 を反転するため、
``traj.txt`` には**採用した c2w の col 1,2 を反転して**書く。

使い方
------
    conda activate sni-slam
    python tools/realdata/convert_stray.py --scan b17452f252 --scene m3_cor_d \
        --mode anamorphic --max-depth 5.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stray_io  # noqa: E402
import preflight  # noqa: E402

TARGET_W, TARGET_H = 1200, 680      # configs/Replica/replica.yaml の cam.W / cam.H
BOUND_MARGIN_M = 0.3
NN_MEDIAN_MAX_M = 0.08              # 指示書 S1 受入条件
FLOOR_NORMAL_MAX_DEG = 3.0          # 同上


# ---------------------------------------------------------------------------
# 内部パラメータ
# ---------------------------------------------------------------------------

def target_intrinsics(intr_rgb_med: np.ndarray, mode: str) -> Dict[str, float]:
    """RGB 解像度基準の (fx,fy,cx,cy) を 1200x680 の目標解像度へ換算する。"""
    fx, fy, cx, cy = [float(v) for v in intr_rgb_med]
    if mode == "anamorphic":
        sx = TARGET_W / float(stray_io.RGB_W)
        sy = TARGET_H / float(stray_io.RGB_H)
        return {"fx": fx * sx, "fy": fy * sy, "cx": cx * sx, "cy": cy * sy,
                "inner_w": TARGET_W, "pad_left": 0}
    if mode == "letterbox":
        s = TARGET_H / float(stray_io.RGB_H)
        inner_w = int(round(stray_io.RGB_W * s))
        pad_left = (TARGET_W - inner_w) // 2
        return {"fx": fx * s, "fy": fy * s, "cx": cx * s + pad_left, "cy": cy * s,
                "inner_w": inner_w, "pad_left": pad_left}
    raise ValueError("mode must be 'anamorphic' or 'letterbox'")


def resize_color(img: np.ndarray, mode: str, geom: Dict[str, float]) -> np.ndarray:
    if mode == "anamorphic":
        return cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    inner_w, pad_left = int(geom["inner_w"]), int(geom["pad_left"])
    small = cv2.resize(img, (inner_w, TARGET_H), interpolation=cv2.INTER_AREA)
    out = np.zeros((TARGET_H, TARGET_W, 3), dtype=img.dtype)
    out[:, pad_left:pad_left + inner_w] = small
    return out


def resize_depth(depth_mm: np.ndarray, mode: str, geom: Dict[str, float]) -> np.ndarray:
    """depth は補間で値を混ぜてはいけないので必ず INTER_NEAREST。黒帯は 0（無効）。"""
    if mode == "anamorphic":
        return cv2.resize(depth_mm, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
    inner_w, pad_left = int(geom["inner_w"]), int(geom["pad_left"])
    small = cv2.resize(depth_mm, (inner_w, TARGET_H), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((TARGET_H, TARGET_W), dtype=depth_mm.dtype)
    out[:, pad_left:pad_left + inner_w] = small
    return out


# ---------------------------------------------------------------------------
# bound
# ---------------------------------------------------------------------------

def compute_bound(
    scan_dir: str, odo: Dict[str, np.ndarray], depth_files: List[str],
    conf_files: List[str], c2w: np.ndarray, conf_min: int, max_depth: float,
    stride: int = 10,
) -> Tuple[List[List[float]], Dict[str, object]]:
    """10 フレームおきに逆投影し、各軸 1/99 パーセンタイル ±0.3 m で bound を作る。

    ``src/SNI_SLAM.py:159-173`` の bound を外れるとレンダリングの far が負になり
    トラッキングが発散するため、**カメラ軌跡が完全に内側に入ること**を検証する。
    """
    pts = []
    for i in range(0, len(depth_files), stride):
        d = stray_io.read_depth_m(depth_files[i])
        cf = stray_io.read_confidence(conf_files[i]) if conf_files else None
        pts.append(stray_io.backproject_frame(
            d, cf, odo["intr"][i], c2w[i], conf_min=conf_min,
            pix_stride=2, max_depth=max_depth))
    p = np.concatenate([x for x in pts if len(x)], axis=0)
    lo = np.percentile(p, 1, axis=0) - BOUND_MARGIN_M
    hi = np.percentile(p, 99, axis=0) + BOUND_MARGIN_M

    # 軌跡が必ず内側に入るよう、はみ出したぶんだけ広げる
    tr = odo["pos"]
    lo = np.minimum(lo, tr.min(axis=0) - BOUND_MARGIN_M)
    hi = np.maximum(hi, tr.max(axis=0) + BOUND_MARGIN_M)

    bound = [[round(float(lo[k]), 3), round(float(hi[k]), 3)] for k in range(3)]
    info = {
        "n_points_used": int(len(p)),
        "traj_inside_bound": bool(
            np.all(tr >= lo + 1e-9) and np.all(tr <= hi - 1e-9)),
        "point_extent": [[round(float(np.percentile(p[:, k], 1)), 3),
                          round(float(np.percentile(p[:, k], 99)), 3)] for k in range(3)],
    }
    return bound, info


# ---------------------------------------------------------------------------
# config 生成
# ---------------------------------------------------------------------------

CONFIG_TEMPLATE = """# 自動生成: tools/realdata/convert_stray.py
# scan_id: {scan_id} / mode: {mode} / generated: frame {n_img}
# 手で編集しないこと（再生成すると上書きされる）。
inherit_from: configs/Replica/replica.yaml
{tracking_block}mapping:
  bound: [[{b00},{b01}],[{b10},{b11}],[{b20},{b21}]]
  marching_cubes_bound: [[{b00},{b01}],[{b10},{b11}],[{b20},{b21}]]
  # 実データは数千フレームあり、keyframe_every=4 だと keyframe の color/depth だけで
  # {kf_gb:.1f} GB になる（{n_kf} keyframe x 13.06 MB）。31 GB の実機では OOM するので
  # RAM に保持せず読み直す。数値は保持版と同一（src/Mapper.py: LazyKeyframe）。
  keyframe_store: reload
  # 中間メッシュは毎回**全 keyframe** を TSDF 積分するため、フレーム数の二乗で効いてくる。
  # 下流（precheck / Registration）が使うのは最終メッシュだけなので中間生成を止める。
  # final_mesh_semantic.ply は idx == n_img-1 で mesh_freq に関係なく生成される。
  mesh_freq: {mesh_freq}
cam:
  H: {H}
  W: {W}
  fx: {fx:.4f}
  fy: {fy:.4f}
  cx: {cx:.4f}
  cy: {cy:.4f}
  png_depth_scale: 1000.0
  crop_edge: 0
data:
  input_folder: {input_folder}
  output: {output}
"""


def write_config(path: str, scan_id: str, scene: str, mode: str, n_img: int,
                 bound: List[List[float]], intr: Dict[str, float],
                 input_folder: str) -> None:
    # letterbox は左右に無効帯が出るので、トラッキングの画素サンプルを端から避ける
    tracking_block = ""
    if mode == "letterbox":
        tracking_block = "tracking:\n  ignore_edge_W: 150\n"
    n_kf = max(n_img // 4, 1)              # configs/SNI-SLAM.yaml: keyframe_every: 4
    txt = CONFIG_TEMPLATE.format(
        scan_id=scan_id, mode=mode, n_img=n_img, tracking_block=tracking_block,
        n_kf=n_kf, kf_gb=n_kf * 13.06 / 1024.0, mesh_freq=max(n_img * 10, 100000),
        b00=bound[0][0], b01=bound[0][1], b10=bound[1][0], b11=bound[1][1],
        b20=bound[2][0], b21=bound[2][1], H=TARGET_H, W=TARGET_W,
        fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"],
        input_folder=input_folder, output="output/RealData/%s/run1" % scene)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(txt)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True, help="Stray の scan_id")
    ap.add_argument("--scene", required=True, help="出力シーン名")
    ap.add_argument("--root", default="Real-data")
    ap.add_argument("--out", default="data/realdata")
    ap.add_argument("--config-dir", default="configs/RealData")
    ap.add_argument("--mode", default="anamorphic",
                    choices=["anamorphic", "letterbox"])
    ap.add_argument("--conf-min", type=int, default=1,
                    help="この confidence 未満の画素を depth=0（無効）にする")
    ap.add_argument("--max-depth", type=float, default=5.0,
                    help="この距離を超える depth を 0（無効）にする。iPad LiDAR の実測レンジは "
                         "約 5 m で、それ以遠は ARKit の外挿値（報告 Q5 参照）")
    ap.add_argument("--limit", type=int, default=0,
                    help="先頭 N フレームだけ変換（0 で全件）")
    ap.add_argument("--sample", type=int, default=0,
                    help="スキャン全体から N フレームを等間隔で抜き出す（A/B ゲート用）。"
                         "先頭 N 枚だけだとカメラの向きが偏り、天井が1枚も写らないことがある。"
                         "出力の frame 番号は 0..N-1 に振り直され、SLAM の入力には使えない")
    ap.add_argument("--png-compress", type=int, default=3)
    ap.add_argument("--skip-images", action="store_true",
                    help="traj/config/report だけ作り直す")
    ap.add_argument("--depth-only", metavar="REF_SCENE_DIR", default=None,
                    help="depth だけを別条件で作り直す。RGB は再エンコードせず、"
                         "参照シーンの conversion_report.json からフレーム構成を引き継ぐ。"
                         "T-A3（depth 制限あり／なしの比較）用")
    ap.add_argument("--regen-config", action="store_true",
                    help="既存の conversion_report.json から config だけ書き直す。"
                         "bound も画像も作り直さない（config テンプレートを変えたとき用）")
    args = ap.parse_args()

    t0 = time.time()
    scan_dir = os.path.join(args.root, args.scan)

    if args.regen_config:
        out_dir = os.path.join(args.out, args.scene)
        with open(os.path.join(out_dir, "conversion_report.json")) as f:
            prev = json.load(f)
        cfg_path = os.path.join(args.config_dir, "%s.yaml" % args.scene)
        write_config(cfg_path, prev["scan_id"], args.scene, prev["mode"],
                     int(prev["n_frames_final"]), prev["bound"],
                     prev["intrinsics_target"], input_folder=out_dir)
        print("regenerated %s (n_img=%d, mode=%s)"
              % (cfg_path, prev["n_frames_final"], prev["mode"]))
        return 0
    out_dir = os.path.join(args.out, args.scene)
    rgb_dir, dep_dir = os.path.join(out_dir, "rgb"), os.path.join(out_dir, "depth")
    for d in (rgb_dir, dep_dir):
        os.makedirs(d, exist_ok=True)

    rep: Dict[str, object] = {
        "scan_id": args.scan, "scene": args.scene, "mode": args.mode,
        "conf_min": args.conf_min, "max_depth_m": args.max_depth,
        "target_hw": [TARGET_H, TARGET_W],
    }

    # --- 1. 読み込みとフレーム数の整合 ---
    odo = stray_io.read_odometry(scan_dir)
    depth_files = stray_io.list_depth_files(scan_dir)
    conf_files = stray_io.list_confidence_files(scan_dir)
    n_meta = stray_io.rgb_frame_count(scan_dir)
    n = min(len(odo["pos"]), len(depth_files))
    if not (len(odo["pos"]) == len(depth_files) == len(conf_files) == n_meta):
        raise SystemExit("frame count mismatch: odo=%d depth=%d conf=%d rgb_meta=%d"
                         % (len(odo["pos"]), len(depth_files), len(conf_files), n_meta))
    if args.limit:
        n = min(n, args.limit)
    # 変換対象のソース frame 番号。--sample のときだけ非連続になる
    if args.sample:
        sel = [int(round(t)) for t in np.linspace(0, n - 1, min(args.sample, n))]
        sel = sorted(set(sel))
    else:
        sel = list(range(n))
    rep["n_frames_source"] = int(len(depth_files))
    rep["n_frames_converted"] = int(len(sel))
    rep["sampled"] = bool(args.sample)
    if args.sample:
        rep["source_frame_indices"] = sel

    # --- 2. 姿勢規約の自動判定 ---
    pc = preflight.check_pose_convention(
        scan_dir, odo, depth_files, conf_files, conf_min=args.conf_min)
    if not pc:
        raise SystemExit("pose convention check produced no result")
    best = min(pc, key=lambda k: pc[k]["nn_median_m"])
    rep["pose_convention"] = {"adopted": best, "measurements": pc,
                              "reason": "2 フレーム間 NN 距離中央値が最小の解釈を採用"}
    nn = pc[best]["nn_median_m"]
    ang = pc[best]["floor_normal_vs_up_deg"]
    rep["nn_median_m"], rep["floor_normal_vs_up_deg"] = nn, ang
    rep["accept_nn_median"] = bool(nn < NN_MEDIAN_MAX_M)
    rep["accept_floor_normal"] = bool(ang < FLOOR_NORMAL_MAX_DEG)
    print("pose convention: %s (NN %.4f m, floor %.3f deg)" % (best, nn, ang))
    for k, v in pc.items():
        print("   %-8s NN=%.4f m  floor=%.3f deg" % (k, v["nn_median_m"],
                                                     v["floor_normal_vs_up_deg"]))
    c2w = stray_io.build_c2w(odo, convention=best)

    # --- 3. 内部パラメータ ---
    intr_med = np.median(odo["intr"][:n], axis=0)
    geom = target_intrinsics(intr_med, args.mode)
    rep["intrinsics_rgb_median"] = [round(float(v), 4) for v in intr_med]
    rep["intrinsics_target"] = {k: round(float(v), 4) for k, v in geom.items()}
    rep["intrinsics_rgb_spread"] = [
        round(float(odo["intr"][:n, k].max() - odo["intr"][:n, k].min()), 4)
        for k in range(4)]

    # --- 4. bound ---
    # --max-depth 0 は「depth を切らない」の意。逆投影側にそのまま渡すと
    # 「0 m 未満だけ採用」になって点が1つも残らないため、健全な上限に読み替える
    # （validate_scene_data.py の SANE_DEPTH_RANGE_M 上限と揃える）。
    bp_max_depth = args.max_depth if args.max_depth > 0 else 20.0
    rep["backproject_max_depth_m"] = bp_max_depth
    bound, binfo = compute_bound(scan_dir, odo, depth_files[:n], conf_files[:n],
                                 c2w, args.conf_min, bp_max_depth)
    rep["bound"], rep["bound_info"] = bound, binfo
    print("bound: %s (traj inside: %s)" % (bound, binfo["traj_inside_bound"]))

    # --- 5. 画像 ---
    # traj.txt は「実際に書けた RGB フレーム」に合わせる必要があるので、画像を先に書く。
    # rgb.mp4 のコンテナメタデータ (CAP_PROP_FRAME_COUNT) は実デコード数と 1 ずれることがある
    # （b17452f252 で実測: メタデータ 8453 / 実デコード 8452）。
    written_src: List[int] = list(sel)

    if args.depth_only:
        # 参照シーンと同じフレーム構成で depth だけを別条件で書き直す
        with open(os.path.join(args.depth_only, "conversion_report.json")) as f:
            ref = json.load(f)
        written_src = list(ref.get("source_frame_indices")
                           or range(int(ref["n_frames_final"])))
        rep["depth_only_ref"] = args.depth_only
        rep["n_frames_final"] = len(written_src)
        zero_frac = []
        png = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compress]
        for out_i, i in enumerate(written_src):
            d16 = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
            cf = cv2.imread(conf_files[i], cv2.IMREAD_UNCHANGED)
            bad = (cf < args.conf_min) | (d16 == 0)
            if args.max_depth > 0:
                bad |= d16 > int(args.max_depth * stray_io.DEPTH_SCALE)
            d16 = d16.copy()
            d16[bad] = 0
            dd = resize_depth(d16, args.mode, geom)
            zero_frac.append(float(np.count_nonzero(dd == 0) / dd.size))
            cv2.imwrite(os.path.join(dep_dir, "depth_%d.png" % out_i), dd, png)
            if (out_i + 1) % 1000 == 0:
                print("  depth %d/%d (%.0fs)" % (out_i + 1, len(written_src),
                                                 time.time() - t0), flush=True)
        rep["depth_invalid_frac_mean"] = round(float(np.mean(zero_frac)), 4)
        os.rmdir(rgb_dir) if not os.listdir(rgb_dir) else None
    elif not args.skip_images:
        cap = cv2.VideoCapture(os.path.join(scan_dir, "rgb.mp4"))
        png = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compress]
        want = {v: j for j, v in enumerate(sel)}
        n_decoded, n_written, zero_frac = 0, 0, []
        written_src = []
        try:
            for i in range(n):
                ok, frame = cap.read()
                if not ok:
                    break
                n_decoded += 1
                if i not in want:
                    continue
                out_i = want[i] if args.sample else i
                cv2.imwrite(os.path.join(rgb_dir, "rgb_%d.png" % out_i),
                            resize_color(frame, args.mode, geom), png)

                d16 = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
                cf = cv2.imread(conf_files[i], cv2.IMREAD_UNCHANGED)
                bad = (cf < args.conf_min) | (d16 == 0)
                if args.max_depth > 0:
                    bad |= d16 > int(args.max_depth * stray_io.DEPTH_SCALE)
                d16 = d16.copy()
                d16[bad] = 0
                dd = resize_depth(d16, args.mode, geom)
                zero_frac.append(float(np.count_nonzero(dd == 0) / dd.size))
                cv2.imwrite(os.path.join(dep_dir, "depth_%d.png" % out_i), dd, png)
                n_written += 1
                written_src.append(i)
                if n_written % 500 == 0:
                    print("  %d/%d frames (%.0fs)" % (n_written, len(sel),
                                                      time.time() - t0), flush=True)
        finally:
            cap.release()
        rep["n_frames_decoded"] = n_decoded
        rep["n_frames_written"] = n_written
        rep["n_frames_dropped_tail"] = int(n - n_decoded)
        rep["rgb_meta_vs_decoded_match"] = bool(n_decoded == n)
        rep["depth_invalid_frac_mean"] = round(float(np.mean(zero_frac)), 4) if zero_frac else None
        # コンテナのメタデータと実デコード数が数フレームずれるのは H.264 では珍しくない。
        # 末尾を捨てて詰めれば済むが、大きくずれるのは破損なので止める。
        shortfall = n - n_decoded
        if shortfall > max(5, int(0.005 * n)):
            raise SystemExit(
                "rgb decode produced only %d of %d frames (shortfall %d) — 動画が壊れている可能性"
                % (n_decoded, n, shortfall))
        if shortfall:
            print("  note: rgb.mp4 のメタデータは %d frame だが実デコードは %d frame。"
                  "末尾 %d frame を捨てて揃える。" % (n, n_decoded, shortfall))
        written_src = written_src
        rep["n_frames_final"] = len(written_src)

    # --- 6. traj.txt（実際に書けた RGB フレームに合わせる）---
    # ローダ (src/utils/datasets.py:198-199) が col 1,2 を反転するので、先に反転して書く
    with open(os.path.join(out_dir, "traj.txt"), "w") as f:
        for i in written_src:
            m = c2w[i].copy()
            m[:3, 1] *= -1
            m[:3, 2] *= -1
            f.write(" ".join("%.9f" % v for v in m.reshape(-1)) + "\n")
    rep["n_traj_lines"] = len(written_src)

    # --- 7. config ---
    cfg_path = os.path.join(args.config_dir, "%s.yaml" % args.scene)
    write_config(cfg_path, args.scan, args.scene, args.mode, len(written_src), bound, geom,
                 input_folder=os.path.join(args.out, args.scene))
    rep["config_path"] = cfg_path
    rep["input_folder"] = out_dir
    rep["elapsed_s"] = round(time.time() - t0, 1)

    with open(os.path.join(out_dir, "conversion_report.json"), "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)

    ok = rep["accept_nn_median"] and rep["accept_floor_normal"] and binfo["traj_inside_bound"]
    print("\n%s -> %s  [%s]  %.1fs" % (args.scan, out_dir,
                                       "ACCEPT" if ok else "CHECK", rep["elapsed_s"]))
    print("  NN median      %.4f m  (< %.2f: %s)" % (nn, NN_MEDIAN_MAX_M, rep["accept_nn_median"]))
    print("  floor normal   %.3f deg (< %.1f: %s)" % (ang, FLOOR_NORMAL_MAX_DEG,
                                                      rep["accept_floor_normal"]))
    print("  traj in bound  %s" % binfo["traj_inside_bound"])
    print("  config         %s" % cfg_path)
    print("  次に semantic_class/ を作ること: tools/realdata/gen_labels_ade20k.py")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

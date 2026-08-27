"""保存済み checkpoint からメッシュだけを作り直す（SLAM を再実行しない）。

なぜ要るか
----------
`meshing.resolution` が `configs/Replica/replica.yaml` の 0.01（1 cm）のままだと、
廊下を含む大きな bound（例 43.7 x 3.26 x 17.3 m）で **24.6 億ボクセル**になり、
最終メッシュ生成が終わらない。トラッキング自体は完走しており checkpoint も残っているので、
**解像度だけ変えてメッシュを作り直せば 4 時間の再実行を避けられる。**

`src/utils/Logger.py:49-68` が decoder の state_dict と 9 組のプレーンを全て保存しているため、
それらを読み戻せば `Mesher.get_mesh` をそのまま呼べる。

使い方
------
    conda activate sni-slam
    python tools/realdata/mesh_from_ckpt.py \
        --config configs/RealData/_D/m3_cor_b.yaml \
        --run output/RealData/_D/m3_cor_b/run1 \
        --resolution 0.025
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)

from src import config as sni_config          # noqa: E402
from src.SNI_SLAM import SNI_SLAM             # noqa: E402
from src.Mapper import LazyKeyframe           # noqa: E402


def rss_gb() -> float:
    """このプロセスの常駐メモリ [GB]。段階ごとの確保量を切り分けるために使う。"""
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) / 1048576.0
    except Exception:
        pass
    return float("nan")


def stage(msg: str) -> None:
    print("[RSS %6.2f GB] %s" % (rss_gb(), msg), flush=True)


class _Args:
    """SNI_SLAM が期待する argparse 名前空間の最小構成。"""
    def __init__(self, cfg_path: str, out: str, input_folder=None):
        self.config = cfg_path
        self.output = out
        self.input_folder = input_folder


def grid_points(bound: np.ndarray, res: float) -> float:
    n = 1.0
    for k in range(3):
        n *= max((bound[k][1] - bound[k][0]) / res, 1.0)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="output/.../runN（ckpts を含む）")
    ap.add_argument("--resolution", type=float, default=None,
                    help="marching cubes の解像度 [m]。既定は config の値を使う")
    ap.add_argument("--auto-resolution", action="store_true",
                    help="config を使わず bound の体積から解像度を決める（診断用）")
    ap.add_argument("--trace-mesher", action="store_true",
                    help="Mesher の各段でメモリを測る。src/utils/Mesher.py は書き換えず、"
                         "実行時にラップするだけ（コアに副作用を残さない）")
    ap.add_argument("--target-grid-points", type=float, default=4.0e7,
                    help="自動決定時のグリッド点数の目安。configs/Replica/room0_official.yaml "
                         "が 0.02 m で約 3900 万点であり、それが実績のある規模")
    ap.add_argument("--kf-stride", type=int, default=1,
                    help="Mesher.get_bound_from_frames に渡す keyframe の間引き。"
                         "この関数は voxel_length 7.8mm の TSDF に全 keyframe を積むので、"
                         "廊下のような大きなシーンではメモリを食い切る")
    ap.add_argument("--ckpt", default=None, help="既定は ckpts の最後")
    ap.add_argument("--no-cull", action="store_true",
                    help="cull_mesh を実行しない。下流（precheck / Registration）が使うのは "
                         "final_mesh_semantic.ply であって culled 版ではないので、"
                         "再メッシュ時は省いてよい（全フレーム走査するので時間がかかる）")
    args = ap.parse_args()

    os.chdir(REPO)
    cfg = sni_config.load_config(args.config, "configs/SNI-SLAM.yaml")

    bound = np.array(cfg["mapping"]["marching_cubes_bound"], dtype=np.float64)
    vol = float(np.prod(bound[:, 1] - bound[:, 0]))
    # ★既定は **config の値をそのまま使う**。
    # 以前はここで bound 体積から再計算しており、config が全シーン一律 0.04 なのに
    # m3_cor_d では 0.035 が選ばれてグリッド点数が 46.1 M（m3_cor_b の 38.2 M より多い）に
    # なっていた。「bound が小さいのに失敗する」という矛盾はこれが原因。
    # 追補3 §1 の「設定を凍結して全シーンに同じ config」にも反していた。
    if args.resolution is not None:
        res = args.resolution
    elif args.auto_resolution:
        res = max(0.01, (vol / args.target_grid_points) ** (1.0 / 3.0))
        res = round(res / 0.005) * 0.005          # 5 mm 刻みに丸める
    else:
        res = float(cfg["meshing"]["resolution"])
    print("bound volume %.1f m^3 / resolution %.3f m -> %.1f M grid points"
          % (vol, res, grid_points(bound, res) / 1e6))
    cfg["meshing"]["resolution"] = res

    # SNI_SLAM を構築すると decoders / planes / mesher が一式そろう
    stage("start")
    sni = SNI_SLAM(cfg, _Args(args.config, args.run))
    stage("SNI_SLAM 構築後（init_planes / get_dataset / Mesher を含む）")

    ck_dir = os.path.join(args.run, "ckpts")
    ck_path = args.ckpt or os.path.join(
        ck_dir, sorted(f for f in os.listdir(ck_dir) if f.endswith(".tar"))[-1])
    print("loading %s" % ck_path)
    ck = torch.load(ck_path, map_location="cpu")
    stage("checkpoint 読み込み後")

    device = cfg["mapping"]["device"] if "device" in cfg["mapping"] else "cuda:0"
    decoders = sni.shared_decoders
    decoders.load_state_dict(ck["decoder_state_dict"])
    decoders = decoders.to(device)
    stage("decoder を GPU へ転送後")

    def to_dev(planes: List[torch.Tensor]) -> List[torch.Tensor]:
        return [p.to(device) for p in planes]

    all_planes = (
        to_dev(ck["planes_xy"]), to_dev(ck["planes_xz"]), to_dev(ck["planes_yz"]),
        to_dev(ck["c_planes_xy"]), to_dev(ck["c_planes_xz"]), to_dev(ck["c_planes_yz"]),
        to_dev(ck["s_planes_xy"]), to_dev(ck["s_planes_xz"]), to_dev(ck["s_planes_yz"]),
    )

    # Mesher.get_bound_from_frames は keyframe ごとの color/depth/est_c2w を要求する。
    # color/depth はデータセットから読み直す（LazyKeyframe）。
    frame_reader = sni.mesher.frame_reader if hasattr(sni.mesher, "frame_reader") else None
    if frame_reader is None:
        from src.utils.datasets import get_dataset
        frame_reader = get_dataset(cfg, _Args(args.config, args.run), 1.0, device="cpu")
    est = ck["estimate_c2w_list"]
    kf_list = ck["keyframe_list"]
    kf_sel = [int(i) for i in kf_list if int(i) < len(est)][::max(args.kf_stride, 1)]
    keyframe_dict = [
        LazyKeyframe({"idx": i, "color": None, "depth": None,
                      "est_c2w": est[i].clone(), "gt_c2w": est[i].clone()},
                     frame_reader)
        for i in kf_sel
    ]
    print("keyframes for bound: %d of %d (stride %d)"
          % (len(keyframe_dict), len(kf_list), args.kf_stride))
    stage("plane を GPU へ転送し keyframe_dict を構築後")

    if args.trace_mesher:
        # Mesher の主要メソッドを実行時にラップして、入出力の規模と RSS を出す。
        # どの段で 30 GB を確保しているかを、コアを触らずに特定するため。
        import types

        def wrap(obj, name):
            orig = getattr(obj, name)

            def wrapped(*a, **kw):
                stage("  -> %s 開始" % name)
                r = orig(*a, **kw)
                try:
                    if isinstance(r, dict) and "grid_points" in r:
                        extra = " grid_points=%s" % (r["grid_points"].shape,)
                    elif hasattr(r, "shape"):
                        extra = " out.shape=%s" % (r.shape,)
                    else:
                        extra = ""
                except Exception:
                    extra = ""
                stage("  <- %s 終了%s" % (name, extra))
                return r
            setattr(obj, name, wrapped)

        for m in ("get_bound_from_frames", "get_grid_uniform", "eval_points"):
            if hasattr(sni.mesher, m):
                wrap(sni.mesher, m)

    mesh_dir = os.path.join(args.run, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    out_color = os.path.join(mesh_dir, "final_mesh_color.ply")
    out_sem = os.path.join(mesh_dir, "final_mesh_semantic.ply")

    t0 = time.time()
    sni.mesher.get_mesh(out_color, all_planes, decoders, keyframe_dict, device,
                        mesh_out_semantic=out_sem, color=True, semantic=True)
    stage("get_mesh 完了")
    print("meshing done in %.1f min" % ((time.time() - t0) / 60))

    if not args.no_cull:
        from src.tools.cull_mesh import cull_mesh
        cull_mesh(out_color, cfg, _Args(args.config, args.run), device,
                  estimate_c2w_list=est)
    print("wrote %s / %s" % (out_sem, out_color))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# npz_to_cloudcompare.py
# labeled_pointcloud.npz を CloudCompare 用 PLY（スカラーフィールド付き）に変換する
import argparse
import sys
from pathlib import Path
import numpy as np

LABEL_NAMES = {0: "wall", 1: "door", 2: "slab", 3: "window", 4: "ceiling", 5: "column"}


def save_ply_with_scalar(xyz, labels, colors, output_path, hide_labels=()):
    """
    CloudCompare が読める binary PLY を書き出す。
    ・RGB カラー付き
    ・label プロパティをスカラーフィールドとして保持
    hide_labels に含まれる label ID の点は除外する。
    """
    mask = np.ones(len(xyz), dtype=bool)
    for lid in hide_labels:
        mask &= (labels != lid)

    xyz    = xyz[mask].astype(np.float32)
    labels = labels[mask]
    colors = colors[mask].astype(np.uint8)
    n      = len(xyz)

    if n == 0:
        print("警告: フィルタ後に点が残りませんでした", file=sys.stderr)
        return

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float label\n"
        "end_header\n"
    )

    # 1頂点 = float32*3 + uint8*3 + float32*1 = 19 bytes（パディングなし）
    dt = np.dtype({
        "names":    ["x",    "y",    "z",    "red", "green", "blue", "label"],
        "formats":  ["<f4",  "<f4",  "<f4",  "u1",  "u1",    "u1",   "<f4"],
        "offsets":  [0,      4,      8,      12,    13,      14,     15],
        "itemsize": 19,
    })
    data = np.empty(n, dtype=dt)
    data["x"]     = xyz[:, 0]
    data["y"]     = xyz[:, 1]
    data["z"]     = xyz[:, 2]
    data["red"]   = colors[:, 0]
    data["green"] = colors[:, 1]
    data["blue"]  = colors[:, 2]
    data["label"] = labels.astype(np.float32)

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    print(f"  → {output_path}  ({n:,} 点)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="labeled_pointcloud.npz を CloudCompare 用 PLY（スカラーフィールド付き）に変換する"
    )
    parser.add_argument("npz_path", help="入力 NPZ ファイルのパス")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="出力 PLY ファイルのパス（省略時は NPZ と同じディレクトリに _cc.ply を生成）",
    )
    parser.add_argument(
        "--hide-labels",
        type=int,
        nargs="+",
        default=[4],
        metavar="LABEL_ID",
        help=(
            "除外するラベル ID（スペース区切りで複数指定可）\n"
            "デフォルト: 4 (ceiling)\n"
            f"利用可能なラベル: {LABEL_NAMES}"
        ),
    )
    parser.add_argument(
        "--all-labels",
        action="store_true",
        help="全ラベルを含める（--hide-labels を無効化）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        print(f"エラー: ファイルが見つかりません: {npz_path}", file=sys.stderr)
        sys.exit(1)

    data   = np.load(str(npz_path))
    xyz    = data["xyz"]
    labels = data["label"]
    colors = data["rgb"]

    hide = [] if args.all_labels else args.hide_labels

    if args.output:
        output_ply = Path(args.output)
    else:
        suffix = "_cc.ply" if hide else "_cc_all.ply"
        output_ply = npz_path.with_name(npz_path.stem + suffix)

    print(f"入力: {npz_path}  ({len(xyz):,} 点)")
    if hide:
        names = [LABEL_NAMES.get(l, str(l)) for l in hide]
        print(f"除外ラベル: {hide} ({', '.join(names)})")
    else:
        print("全ラベルを含めて出力")

    save_ply_with_scalar(xyz, labels, colors, output_ply, hide_labels=hide)

    print("\nCloudCompare での操作:")
    print("  1. PLY を開く")
    print("  2. Edit > Scalar Fields > Set SF as active で 'label' を選択")
    print("  3. Edit > Scalar Fields > Filter by value でラベル値を指定してフィルタリング")


if __name__ == "__main__":
    main()

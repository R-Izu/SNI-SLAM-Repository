# BIM_IFC_Extraction

IFC（BIM）ファイルから、SNI-SLAM のセマンティッククラスに対応したラベル付き点群を生成する前処理ツール群。
SNI-SLAM 本体とは独立して動作し、本体の `sni-slam` conda 環境・スクリプトには一切影響しない。

## 構成

- `src/inspect_ifc.py` — IFC → ラベル付き点群（`*_labeled_pointcloud.npz` / `.ply`）。
  6クラス（0:wall, 1:door, 2:slab, 3:window, 4:ceiling, 5:column）を
  ポアソンディスクサンプリング（既定 500点/m²）で抽出する。依存: `ifcopenshell`, `open3d`, `numpy`。
- `src/npz_to_cloudcompare.py` — NPZ → CloudCompare 用スカラーフィールド付き binary PLY。依存: `numpy` のみ。
- `input/` — 入力 IFC（`m3-411.ifc`）。
- `ouput/` — 生成済み参照出力（NPZ/PLY）。

## 環境

`inspect_ifc.py` が必要とする `ifcopenshell` は、本体 `sni-slam` 環境（Python 3.7 / libgcc-ng 11 系に固定）と
依存が衝突して導入できない。そのため **IFC 抽出専用の conda 環境を別途用意する**。
`npz_to_cloudcompare.py` は numpy のみなので `sni-slam` でもそのまま動く。

### 専用環境の作成（初回のみ）

```bash
conda create -n bim-ifc -c conda-forge python=3.10 ifcopenshell open3d numpy -y
```

動作確認済みバージョン: ifcopenshell 0.8.5 / open3d 0.19.0 / numpy 2.2.6。

## 使い方

作業ディレクトリは本フォルダ（`SNI-SLAM/BIM_IFC_Extraction`）を想定。

### 1. IFC → ラベル付き点群

```bash
conda run -n bim-ifc python src/inspect_ifc.py input/m3-411.ifc -o ouput
# オプション: --density で 1m² あたりの点数を変更（既定 500）
```

→ `ouput/m3-411_labeled_pointcloud.npz` と `.ply` を生成。
クラス別要素数・合計点数がログに出力される。

> 注: 実行時に `No module named 'lark'` という警告が出ることがあるが、
> これは ifcopenshell の IFC ストリーム解析（任意機能）の警告で、点群生成には影響しない。

### 2. NPZ → CloudCompare 用 PLY

```bash
conda run -n sni-slam python src/npz_to_cloudcompare.py ouput/m3-411_labeled_pointcloud.npz
# 既定で ceiling(label 4) を除外。全ラベル含めるなら --all-labels
```

→ `ouput/m3-411_labeled_pointcloud_cc.ply`（label をスカラーフィールドとして保持）を生成。
CloudCompare で `Edit > Scalar Fields > Set SF as active` から `label` を選択するとクラス別に色分け・フィルタできる。

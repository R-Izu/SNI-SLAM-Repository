## SNI-SLAM: Semantic Neural Implicit SLAM
Siting Zhu*, Guangming Wang*, Hermann Blum, Jiuming Liu, Liang Song, Marc Pollefeys, Hesheng Wang  
<div align="center">
  <h3>CVPR 2024 [<a href="https://arxiv.org/pdf/2311.11016.pdf">論文</a>] [<a href="https://drive.google.com/file/d/1oRKoly8cxple0Z3CcgbBvC_8wYQhOtR3/view?usp=drive_link">補足資料</a>]</h3>
</div>

## システムフロー

<p align="center">
  <a href="">
    <img src="./demo/System_flow.png" alt="System flow" width="80%">
  </a>
</p>

## インストール

まず必要な依存関係をすべて満たしていることを確認してください。  
最も簡単な方法は [anaconda](https://www.anaconda.com/) を利用することです。

Linux では、環境を作成する前に **libopenexr-dev** をインストールし、その後 `sni` という名前の conda 環境を作成します。

```bash
sudo apt-get install libopenexr-dev
conda env create -f environment.yaml
conda activate sni
```

## 実行方法

### Replica データセット

1. セマンティックラベル付きの Replica データを  
   [Google Drive](https://drive.google.com/drive/u/0/folders/1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h) からダウンロードし、`./data/replica` フォルダに配置します。  
   このリポジトリでは Replica データセットの一部のみを提供しています。  
   完全な Replica データ生成手順については、`data_generation` ディレクトリを参照してください。

2. 学習済みセグメンテーションネットワークを  
   [Google Drive](https://drive.google.com/drive/u/0/folders/1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h) からダウンロードし、`./seg` フォルダに保存します  
   （`seg/facebookresearch_dinov2_main.zip` を展開して配置）。

3. SNI-SLAM を以下のコマンドで実行します。

```bash
python -W ignore run.py configs/Replica/room1.yaml
```

評価用のメッシュは  
`$OUTPUT_FOLDER/mesh/final_mesh_eval_rec_culled.ply`  
として保存されます。

## 評価

### 平均軌跡誤差（Average Trajectory Error）

平均軌跡誤差を評価するには、対象シーンに対応する config ファイルを指定して以下を実行します。

```bash
# Replica の room1 の例
python src/tools/eval_ate.py configs/Replica/room1.yaml
```

### 再構成精度（Reconstruction Metrics）

再構成の評価には、以下のコードをベースにしたスクリプトを利用しています。  
[neural_slam_eval](https://github.com/JingwenWang95/neural_slam_eval)

## SNI-SLAM の結果の可視化

SNI-SLAM の結果を可視化するには、  
`configs/SNI-SLAM.yaml` 内の `mesh_freq` を `40` に設定し、最初から SNI-SLAM を再実行することを推奨します。

学習完了後、以下のコマンドで可視化を行います。

```bash
python visualizer.py configs/Replica/room1.yaml --top_view --save_rendering
```

可視化結果の動画は `output/Replica/room1/vis.mp4` に保存されます。  
緑の軌跡が真値（Ground Truth）の軌跡、赤の軌跡が SNI-SLAM による推定軌跡を表します。

### Visualizer のコマンドライン引数

- `--output $OUTPUT_FOLDER`  
  出力フォルダを指定します（config ファイルの設定を上書き）。
- `--top_view`  
  カメラをトップビューに設定します。指定しない場合は、シーケンスの最初のフレームから見た視点になります。
- `--save_rendering`  
  レンダリングした動画を `vis.mp4` として出力フォルダに保存します。
- `--no_gt_traj`  
  真値の軌跡を描画しないようにします。

## 論文の引用

本コードや論文が有用であれば、以下の BibTeX を用いて引用してください。

```BibTeX
@inproceedings{zhu2024sni,
  title={Sni-slam: Semantic neural implicit slam},
  author={Zhu, Siting and Wang, Guangming and Blum, Hermann and Liu, Jiuming and Song, Liang and Pollefeys, Marc and Wang, Hesheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21167--21177},
  year={2024}
}
```

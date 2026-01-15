## Replica データ生成

### Replica データセットのダウンロード

3D モデルと情報ファイルを  
[Replica](https://github.com/facebookresearch/Replica-Dataset) からダウンロードします。

### 3D オブジェクトメッシュの抽出

`./data_generation/extract_inst_obj.py` 内の入力パスを、手元の Replica データ（3D メッシュやアノテーション）の保存場所に合わせて書き換え、その後以下を実行します。

```angular2html
python ./data_generation/extract_inst_obj.py
```

### カメラ軌跡の生成

カメラ軌跡の詳細な生成方法については、  
[Semantic-NeRF](https://github.com/Harry-Zhi/semantic_nerf/issues/25#issuecomment-1340595427) を参照してください。  
ランダム軌跡生成コードは **単一部屋シーン** のみを想定しています。複数部屋のシーンでは、障害物との衝突判定などの追加処理が必要です（コントリビュート歓迎）。

### 2D 画像のレンダリング

カメラ軌跡 \(t_{wc}\)（config 内の `pose_file` を変更）を与えると、  
[Habitat-Sim](https://github.com/facebookresearch/habitat-sim) を用いて RGB / Depth / Semantic / Instance 画像をレンダリングします。

#### Habitat-Sim 0.2.1 のインストール

Habitat-Sim 0.2.1 のインストールには conda を使うことを推奨します。

```angular2html
conda create -n habitat python=3.8.12 cmake=3.14.0
conda activate habitat
conda install habitat-sim=0.2.1 withbullet -c conda-forge -c aihabitat 
conda install numba=0.54.1
```

#### 設定ファイルを用いたレンダリングの実行

```angular2html
python ./data_generation/habitat_renderer.py --config ./data_generation/replica_render_config_vMAP.yaml 
```

HDR 画像を取得したい場合は、`semantic_mesh.ply` ではなく `mesh.ply` を使用してください（config 内のパスを変更）。  
また、露光の高い既存の RGB を置き換えるために、新たに生成した `rgb` フォルダをコピーして上書きします。

```angular2html
python ./data_generation/habitat_renderer.py --config ./data_generation/replica_render_config_vMAP.yaml 
```
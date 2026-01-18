# 🔄 Cursor.md - SLAM&BIM Registration Project Documentation

## 📋 Project Overview

SNI-SLAM-BIM は、SNI-SLAM (Semantic Neural Implicit SLAM) をベースに、DINOv2 によるセマンティックセグメンテーションと BIM (Building Information Modeling) データを統合した高精度な位置推定・可視化システムです。Semantic ICP を用いた密結合（Tight Coupling）により、リアルタイムでのドリフト補正と、BIM座標系への正確な自動位置合わせを実現します。

## 🏗️ Architecture
```mermaid
graph TD
    %% Define Styles
    classDef slam fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px;
    classDef sem fill:#d5e8d4,stroke:#82b366,stroke-width:2px;
    classDef reg fill:#ffe6cc,stroke:#d79b00,stroke-width:2px;
    classDef vis fill:#e1d5e7,stroke:#9673a6,stroke-width:2px;
    classDef input fill:#f5f5f5,stroke:#666666,stroke-width:1px;

    %% Nodes
    subgraph Inputs
        RGBD[RGB-D Camera Stream]:::input
        BIM_File[BIM Data File <br/>(IFC/OBJ)]:::input
    end

    subgraph SLAM_Core ["SLAM Core (SNI-SLAM)"]
        Tracker[Tracking Module<br/>(Pose Estimation)]:::slam
        Mapper[Mapping Module<br/>(Keyframe & Depth)]:::slam
    end

    subgraph Semantic_Proc ["Semantic Processing"]
        DINO[DINOv2 Segmentation<br/>(src/networks/dinov2_seg.py)]:::sem
        SemCloudGen[Semantic Point Cloud<br/>Generator]:::sem
    end

    subgraph Reg_System ["Registration & BIM"]
        BIMProc[BIM Processor]:::reg
        RegMod[Registration Module<br/>(Semantic ICP)]:::reg
    end

    subgraph Visualization
        Viewer[BIM Visualizer<br/>(visualizer_with_BIM.py)]:::vis
    end

    %% Data Flow
    RGBD --> Tracker
    Tracker -->|Current Pose Twc| Mapper
    
    %% Semantic Loop
    Mapper -->|Keyframe RGB| DINO
    Mapper -->|Keyframe Depth & Pose| SemCloudGen
    DINO -->|Semantic Mask| SemCloudGen
    SemCloudGen -->|Local Semantic Point Cloud<br/>(SLAM Coords)| RegMod

    %% BIM Processing
    BIM_File --> BIMProc
    BIMProc -->|Reference Semantic BIM Data<br/>(BIM Coords)| RegMod
    BIMProc -->|BIM Geometry Model| Viewer

    %% Feedback Loop (Tight Coupling)
    RegMod -->|Correction Transform &<br/>Drift Feedback| Tracker

    %% Visualization Flow
    RegMod -->|Correction Transform| Viewer
    Tracker -->|Corrected Pose| Viewer
    DINO -->|Global Point Cloud| Viewer

```
### 🎯 Core Concept
🌍 Global Semantic Map: DINOv2によってラベル付けされた点群と、BIMモデルが統合された絶対座標系。

📍 Semantic ICP: 点群の幾何形状に加え、セマンティックラベル（壁、床など）の一致を拘束条件とした高精度な位置合わせアルゴリズム。

🔗 Tight Coupling Feedback: BIMとの位置合わせで得られた補正行列を、リアルタイムに Tracker へフィードバックして自己位置推定のドリフトを解消する仕組み。

🏢 BIM-Anchored Visualization: BIMモデルを背景（Ground Truth）として固定し、その上にSLAMの構築結果をリアルタイムに重畳表示する可視化プロセス。

🧩 主要コンポーネント
SLAMコア: src/SNI_SLAM.py をベースに実装。ニューラル暗黙的表現を使用。外部からの姿勢補正を受け付けるように修正済み。

セグメンテーション: src/networks/dinov2_seg.py。キーフレーム用のセマンティックマスクを提供。

登録モジュール: （新規）ローカルSLAM点群と参照BIM点群の間でSemantic ICPを実行。T_bim_slamを計算。

可視化ツール: visualizer_with_BIM.py（新規）。visualizer.pyをベースに、BIMモデルを背景のGround Truthとしてレンダリングし、リアルタイムのSLAM軌跡/点群をそれに合わせて配置。

📚 AIアシスタント向けの注意事項
コード生成やデバッグ時に、以下のガイドラインに従ってください：

1. コード標準 & Python環境
PyTorch & CUDA: システムはCUDA操作に大きく依存します。テンソルデバイスが正しく管理されていることを確認してください（.to(device)）。

マルチプロセッシング: SNI_SLAM.pyはtorch.multiprocessingをspawnメソッドで使用します。picklingを壊すグローバル状態の変更を避けてください。

型ヒント: すべての新規関数にPythonの型ヒントを使用してください。特に幾何学的変換については必須です（例：def align(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:）。

2. セマンティック登録戦略
アルゴリズム: Semantic ICPを使用。同じセマンティッククラスを持つ点同士のみマッチングしてください（例：壁の点と壁の点）。

フィードバック機構: 計算されたドリフト補正（変換）は、Trackerにフィードバックして現在の姿勢推定を更新し、「密結合（Tight Coupling）」を確保する必要があります。

3. ファイル処理
BIMデータ: 入力BIMデータはメッシュ（OBJ/PLY）またはラベル付き点群として読み込めることを想定。

設定: すべてのパラメータ（DINOv2マッピング、ICP閾値）は、既存のYAML設定システム（configs/）で管理してください。

4. 実装の優先順位
非破壊的: 元のSNI-SLAMロジックを壊さないでください。登録をMapperまたはTrackerループに接続されたアドオンモジュールとして実装してください。

可視化: visualizer_with_BIM.pyはリアルタイム更新をサポートする必要があります。Open3Dのノンブロッキング可視化の使用を検討してください。

5. テストチェックリスト
[ ] DINOv2が入力フレームを定義されたBIMクラス（壁、床など）に正しくセグメント化する。

[ ] セマンティック点群ジェネレータがDepth + Maskから正確なローカル点群を作成する。

[ ] 登録モジュールが収束し、有効な変換行列を出力する。

[ ] visualizer_with_BIM.pyがBIMモデルと整列されたSLAM点群の両方を表示する。

[ ] ベースラインSNI-SLAMと比較してドリフトが観察可能に減少している。
"""
DINOv2セグメンテーション学習スクリプト
- 別PCで実行することを想定
- 必要なファイル: seg/facebookresearch_dinov2_main/ (DINOv2ソースコード)
- データ形式:
    dataset/
      rgb/       # RGB画像 (*.png or *.jpg)
      semantic/  # セマンティックマスクPNG (H x W, グレースケール、ピクセル値=クラスID)

使い方:
    python train_dinov2_seg.py \
        --data_dir /path/to/dataset \
        --n_classes 52 \
        --img_h 680 \
        --img_w 1200 \
        --output_path ./seg/dinov2_custom.pth \
        --epochs 50 \
        --batch_size 4 \
        --lr_backbone 1e-5 \
        --lr_head 1e-4

Replica (room0)向け:
    python train_dinov2_seg.py \
        --data_dir data/replica/room_0_official \
        --n_classes 52 \
        --img_h 680 \
        --img_w 1200 \
        --output_path ./seg/dinov2_replica_room0.pth \
        --epochs 30 \
        --batch_size 2
"""

import os
import sys
import argparse
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# DINOv2ソースコードのパスを追加
parent_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(parent_dir, 'seg', 'facebookresearch_dinov2_main')
sys.path.insert(0, module_path)

from src.networks.dinov2_seg import DINO2SEG


# ------------------------------------------------------------------ #
#  Dataset
# ------------------------------------------------------------------ #
class SegDataset(Dataset):
    """
    RGB画像とセマンティックマスクをロードするデータセット。
    セマンティックマスクはグレースケールPNG（ピクセル値=クラスID）。
    """
    def __init__(self, data_dir: str, img_h: int, img_w: int):
        self.rgb_paths = sorted(glob.glob(os.path.join(data_dir, 'rgb', '*.png')) +
                                glob.glob(os.path.join(data_dir, 'rgb', '*.jpg')))
        self.sem_paths = sorted(glob.glob(os.path.join(data_dir, 'semantic', '*.png')))

        if len(self.rgb_paths) == 0:
            raise FileNotFoundError(f"RGB画像が見つかりません: {data_dir}/rgb/")
        if len(self.sem_paths) == 0:
            raise FileNotFoundError(f"セマンティックマスクが見つかりません: {data_dir}/semantic/")
        if len(self.rgb_paths) != len(self.sem_paths):
            raise ValueError(
                f"RGB枚数({len(self.rgb_paths)})とマスク枚数({len(self.sem_paths)})が一致しません"
            )

        self.img_transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.img_h = img_h
        self.img_w = img_w

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int):
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        sem = Image.open(self.sem_paths[idx])

        rgb_tensor = self.img_transform(rgb)  # (3, H, W)

        # セマンティックマスク: グレースケール → (H, W) のLongテンソル
        sem_array = np.array(sem.resize((self.img_w, self.img_h),
                                        resample=Image.NEAREST), dtype=np.int64)
        sem_tensor = torch.from_numpy(sem_array)  # (H, W)

        return rgb_tensor, sem_tensor


# ------------------------------------------------------------------ #
#  Training helper: forward で logits を取得するラッパー
# ------------------------------------------------------------------ #
class DINO2SEGTrainer(nn.Module):
    """
    DINO2SEGをラップして、訓練時に logits (クラスlogit map) を返す。
    DINO2SEGのforwardはmode='train'のときargmaxを返すため、
    CrossEntropyLoss用にlogitsを返すよう上書きする。
    """
    def __init__(self, base_model: DINO2SEG):
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB画像テンソル
        Returns:
            logits: (B, num_class, H_out, W_out)
        """
        x = self.base.upsample(x)
        bs = x.shape[0]
        mask_dim = (x.shape[2] / self.base.patch_size,
                    x.shape[3] / self.base.patch_size)

        out = self.base.backbone.forward_features(x.float())
        out = out["x_norm_patchtokens"]
        out = out.reshape(bs, self.base.embedding_size,
                          int(mask_dim[0]), int(mask_dim[1]))

        # segmentation_conv全段を適用してlogitsを返す（argmaxしない）
        logits = self.base.segmentation_conv(out)  # (B, num_class, H_out, W_out)
        return logits


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser(description='DINOv2セグメンテーション学習')
    p.add_argument('--data_dir', type=str, required=True,
                   help='データセットディレクトリ (rgb/ と semantic/ を含む)')
    p.add_argument('--n_classes', type=int, default=52,
                   help='セマンティッククラス数 (Replica=52)')
    p.add_argument('--img_h', type=int, default=680, help='入力画像の高さ')
    p.add_argument('--img_w', type=int, default=1200, help='入力画像の幅')
    p.add_argument('--crop_edge', type=int, default=0,
                   help='DINO2SEGのedgeパラメータ (SNI-SLAMのcrop_edgeに合わせる)')
    p.add_argument('--c_dim', type=int, default=16,
                   help='特徴次元数 (SNI-SLAMのmodel.c_dimに合わせる)')
    p.add_argument('--output_path', type=str, default='./seg/dinov2_custom.pth',
                   help='学習済みモデルの保存先')
    p.add_argument('--pretrained', type=str, default=None,
                   help='既存のpthファイルから初期化する場合のパス')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr_backbone', type=float, default=1e-5,
                   help='DINOv2 backbone (blocks.4以降) の学習率')
    p.add_argument('--lr_head', type=float, default=1e-4,
                   help='segmentation_conv の学習率')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--val_split', type=float, default=0.1,
                   help='検証データの割合 (0=検証なし)')
    p.add_argument('--save_every', type=int, default=10,
                   help='何エポックごとにチェックポイントを保存するか')
    p.add_argument('--dry_run', action='store_true',
                   help='データ形状チェックのみ実行（実際には学習しない）')
    return p.parse_args()


def build_optimizer(model: DINO2SEGTrainer, lr_backbone: float, lr_head: float):
    """backbone と segmentation_conv で学習率を分ける"""
    backbone_params = [
        p for n, p in model.base.backbone.named_parameters()
        if p.requires_grad
    ]
    head_params = list(model.base.segmentation_conv.parameters())
    return optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params,     'lr': lr_head},
    ], weight_decay=1e-4)


def train_one_epoch(model: DINO2SEGTrainer,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    epoch: int) -> float:
    model.train()
    total_loss = 0.0
    for i, (rgb, sem) in enumerate(loader):
        rgb = rgb.to(device)
        sem = sem.to(device, dtype=torch.long)  # CrossEntropyLoss expects Long

        logits = model(rgb)  # (B, C, H_out, W_out)

        # マスクをlogitsのサイズにリサイズ
        if logits.shape[-2:] != sem.shape[-2:]:
            sem = nn.functional.interpolate(
                sem.unsqueeze(1).float(),
                size=logits.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()

        loss = criterion(logits, sem)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"  [Epoch {epoch}] step {i+1}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model: DINO2SEGTrainer,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    for rgb, sem in loader:
        rgb = rgb.to(device)
        sem = sem.to(device, dtype=torch.long)
        logits = model(rgb)
        if logits.shape[-2:] != sem.shape[-2:]:
            sem = nn.functional.interpolate(
                sem.unsqueeze(1).float(),
                size=logits.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()
        total_loss += criterion(logits, sem).item()
    return total_loss / len(loader)


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    # データセット構築
    dataset = SegDataset(args.data_dir, args.img_h, args.img_w)
    print(f"データ数: {len(dataset)}")

    if args.dry_run:
        rgb, sem = dataset[0]
        print(f"[dry_run] rgb shape: {rgb.shape}, sem shape: {sem.shape}")
        print(f"[dry_run] sem unique values (first 10): {torch.unique(sem)[:10]}")
        print("[dry_run] 完了。--dry_run を外すと実際に学習します。")
        return

    # Train/Val 分割
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    if n_val > 0:
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    else:
        train_set = dataset
        val_set = None

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers) if val_set else None

    # モデル構築
    base_model = DINO2SEG(
        img_h=args.img_h,
        img_w=args.img_w,
        num_cls=args.n_classes,
        edge=args.crop_edge,
        dim=args.c_dim,
    )
    if args.pretrained:
        print(f"事前学習モデルをロード: {args.pretrained}")
        base_model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))

    model = DINO2SEGTrainer(base_model).to(device)

    optimizer = build_optimizer(model, args.lr_backbone, args.lr_head)
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # 255は無効ラベル

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f}", end='')

        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"  val_loss={val_loss:.4f}", end='')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.base.state_dict(), args.output_path.replace('.pth', '_best.pth'))
                print(" [best saved]", end='')

        print()

        if epoch % args.save_every == 0:
            ckpt_path = args.output_path.replace('.pth', f'_ep{epoch:03d}.pth')
            torch.save(model.base.state_dict(), ckpt_path)
            print(f"  → チェックポイント保存: {ckpt_path}")

    # 最終モデルを保存
    torch.save(model.base.state_dict(), args.output_path)
    print(f"\n学習完了。最終モデル保存: {args.output_path}")
    print(f"SNI-SLAMで使うには configs/SNI-SLAM.yaml の "
          f"model.cnn.pretrained_model_path を {args.output_path} に設定してください。")


if __name__ == '__main__':
    main()

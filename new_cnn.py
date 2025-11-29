# cnn_poster_regression.py
# Train ResNet18 / ResNet50 / EfficientNet_B0 on movie poster -> IMDB score regression.

import os
import random
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from torchvision import models


# --------------------------------------------------------
# Utils
# --------------------------------------------------------

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------
# Config
# --------------------------------------------------------

class CNNConfig:
    def __init__(
        self,
        model_name: str = "resnet18",   # "resnet18", "resnet50", "efficientnet_b0"
        img_size: int = 224,
        batch_size: int = 128,
        val_batch_size: int = 64,
        epochs: int = 20,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        seed: int = 42,
        num_workers: int = 2,
        dropout: float = 0.3,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        pretrained_weights: bool = True,
    ):
        self.model_name = model_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.num_workers = num_workers
        self.dropout = dropout
        self.mean = mean
        self.std = std
        self.pretrained_weights = pretrained_weights


# --------------------------------------------------------
# Dataset / DataModule
# --------------------------------------------------------

class PosterDataset(Dataset):
    """
    Dataset for MovieGenre.csv + downloaded posters.

    Expects DataFrame with columns:
      - 'local_path': path to poster image file (jpg)
      - 'IMDB Score': float rating
    """
    def __init__(self, frame: pd.DataFrame, transform: T.Compose):
        frame = frame.copy()
        frame["IMDB Score"] = frame["IMDB Score"].astype(float)
        self.df = frame.reset_index(drop=True)
        self.tfm = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        img_path = row["local_path"]

        # Load image
        img = Image.open(img_path).convert("RGB")
        x = self.tfm(img)  # C x H x W

        # Target: IMDB score
        y = torch.tensor(row["IMDB Score"], dtype=torch.float32)
        return x, y


class CNNDataModule:
    def __init__(self, df: pd.DataFrame, cfg: CNNConfig):
        # Keep only rows with image path and score
        self.df = df.dropna(subset=["local_path", "IMDB Score"]).reset_index(drop=True)
        # Also filter to existing files, in case some images failed to download
        self.df = self.df[self.df["local_path"].apply(os.path.exists)].reset_index(drop=True)

        self.cfg = cfg
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        self.train_tf = T.Compose([
            T.Resize((cfg.img_size, cfg.img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=cfg.mean, std=cfg.std),
        ])
        self.eval_tf = T.Compose([
            T.Resize((cfg.img_size, cfg.img_size)),
            T.ToTensor(),
            T.Normalize(mean=cfg.mean, std=cfg.std),
        ])

    def setup(self, val_size: float = 0.1, test_size: float = 0.1):
        from sklearn.model_selection import train_test_split
        seed_all(self.cfg.seed)

        train_df, tmp_df = train_test_split(
            self.df, test_size=val_size + test_size,
            random_state=self.cfg.seed, shuffle=True
        )
        rel_val = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.5
        val_df, test_df = train_test_split(
            tmp_df, test_size=1 - rel_val,
            random_state=self.cfg.seed, shuffle=True
        )
        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df

        print(f"Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")

    def train_dataloader(self) -> DataLoader:
        ds = PosterDataset(self.train_df, self.train_tf)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        ds = PosterDataset(self.val_df, self.eval_tf)
        return DataLoader(
            ds,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        ds = PosterDataset(self.test_df, self.eval_tf)
        return DataLoader(
            ds,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
        )


# --------------------------------------------------------
# Model
# --------------------------------------------------------

class CNNRegressor(nn.Module):
    """
    Wraps an ImageNet-pretrained CNN and replaces the head for 1-D regression (IMDB score).
    """
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone, in_feats = self._build_backbone(cfg.model_name, cfg.pretrained_weights)
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(in_feats, 1),
        )

    def _build_backbone(self, name: str, pretrained_weights: bool = True):
        if name == "resnet18":
            if pretrained_weights:
                m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                m = models.resnet18(weights=None)
            in_feats = m.fc.in_features
            m.fc = nn.Identity()

        elif name == "resnet50":
            if pretrained_weights:
                m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                m = models.resnet50(weights=None)
            in_feats = m.fc.in_features
            m.fc = nn.Identity()

        elif name == "efficientnet_b0":
            if pretrained_weights:
                m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                m = models.efficientnet_b0(weights=None)
            in_feats = m.classifier[1].in_features
            m.classifier = nn.Identity()

        else:
            raise ValueError(f"Unsupported model_name: {name}")

        return m, in_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        out = self.head(feats)      # (B, 1)
        return out.squeeze(1)       # (B,)


# --------------------------------------------------------
# Trainer
# --------------------------------------------------------

class CNNTrainer:
    def __init__(self, cfg: CNNConfig, model: nn.Module):
        self.cfg = cfg
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.criterion = nn.L1Loss()  # MAE loss

        run_name = f"{cfg.model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_bs{cfg.batch_size}_ep{cfg.epochs}"
        log_dir = os.path.join("runs", run_name)
        self.writer = SummaryWriter(log_dir=log_dir)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        abs_err, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            abs_err += torch.abs(pred - y).sum().item()
            n += y.size(0)
        mae = abs_err / max(n, 1)
        return {"mae": mae}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        best = {"val_mae": float("inf")}
        for ep in range(1, self.cfg.epochs + 1):
            self.model.train()
            running, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad(set_to_none=True)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.opt.step()
                running += loss.item() * y.size(0)
                n += y.size(0)

            train_mae = running / max(n, 1)
            val_metrics = self.evaluate(val_loader)
            val_mae = val_metrics["mae"]

            print(f"Epoch {ep}/{self.cfg.epochs} | train MAE: {train_mae:.4f} | val MAE: {val_mae:.4f}")

            self.writer.add_scalar("MAE/train", train_mae, ep)
            self.writer.add_scalar("MAE/val", val_mae, ep)

            if val_mae < best["val_mae"]:
                best["val_mae"] = val_mae
                best["state_dict"] = {k: v.cpu() for k, v in self.model.state_dict().items()}

        if "state_dict" in best:
            self.model.load_state_dict(best["state_dict"])

        self.writer.close()
        return best

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        preds = []
        for x, _ in loader:
            x = x.to(self.device)
            p = self.model(x).detach().cpu().numpy()
            preds.append(p)
        return np.concatenate(preds, axis=0)

if __name__ == "__main__":
    from new_kaggleloader import KaggleLoader  # your loader that adds 'local_path'

    # Load dataset from local Kaggle cache
    loader = KaggleLoader("neha1703/movie-genre-from-its-poster")
    df = loader.df()  
    df = df.dropna(subset=["local_path", "IMDB Score"])

    model_name = "resnet18"  # "resnet50", "efficientnet_b0"
 
    # Config: train ResNet18
    cfg = CNNConfig(
        model_name=model_name,
        batch_size=128,
        epochs=50,
        pretrained_weights=True,
    )
 
    seed_all(cfg.seed)

    dm = CNNDataModule(df, cfg)
    dm.setup(val_size=0.1, test_size=0.1)

    model = CNNRegressor(cfg)
    trainer = CNNTrainer(cfg, model)

    best = trainer.fit(train_loader=dm.train_dataloader(),
                       val_loader=dm.val_dataloader())
    print("Best val MAE:", best["val_mae"])

    test_metrics = trainer.evaluate(dm.test_dataloader())
    print("Test MAE:", round(test_metrics["mae"], 4))

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/imdb_reg_{cfg.model_name}.pth")

# cnn_poster_genre.py
# Train ResNet18 / ResNet50 / EfficientNet_B0 on movie poster -> Genre (multi-label classification).

import os
import random
from datetime import datetime
from typing import Dict, List

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
        dropout: float = 0.5,
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

        # will be filled later by DataModule
        self.num_classes: int | None = None
        self.classes: List[str] | None = None


# --------------------------------------------------------
# Dataset / DataModule
# --------------------------------------------------------

class PosterDataset(Dataset):
    """
    Dataset for MovieGenre.csv + downloaded posters.

    Expects DataFrame with columns:
      - 'local_path': path to poster image file (jpg)
      - 'Genre': string like "Action|Adventure|Family"
    """
    def __init__(self, frame: pd.DataFrame, transform: T.Compose,
                 genre2idx: Dict[str, int], num_classes: int):
        self.df = frame.reset_index(drop=True).copy()
        self.tfm = transform
        self.genre2idx = genre2idx
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.df)

    def _encode_genres(self, genre_str: str) -> torch.Tensor:
        """
        Convert "Action|Adventure|Family" -> multi-hot vector [1,1,1,0,...]
        """
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        if isinstance(genre_str, str):
            for g in genre_str.split("|"):
                g = g.strip()
                if g and g in self.genre2idx:
                    y[self.genre2idx[g]] = 1.0
        return y

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        img_path = row["local_path"]

        # Load image
        img = Image.open(img_path).convert("RGB")
        x = self.tfm(img)  # C x H x W

        # Target: multi-label genre vector
        y = self._encode_genres(row["Genre"])
        return x, y


class CNNDataModule:
    def __init__(self, df: pd.DataFrame, cfg: CNNConfig):
        # Keep only rows with image path and genre
        self.df = df.dropna(subset=["local_path", "Genre"]).reset_index(drop=True)
        # Filter to existing files
        self.df = self.df[self.df["local_path"].apply(os.path.exists)].reset_index(drop=True)

        self.cfg = cfg
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        # --- build genre vocabulary ---
        all_genres = set()
        for g in self.df["Genre"].dropna():
            for t in str(g).split("|"):
                t = t.strip()
                if t:
                    all_genres.add(t)
        self.classes = sorted(all_genres)
        self.genre2idx = {g: i for i, g in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        print(f"Found {self.num_classes} genres:", self.classes)

        # propagate to cfg
        self.cfg.num_classes = self.num_classes
        self.cfg.classes = self.classes

        self.train_tf = T.Compose([
            T.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
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
        ds = PosterDataset(self.train_df, self.train_tf, self.genre2idx, self.num_classes)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        ds = PosterDataset(self.val_df, self.eval_tf, self.genre2idx, self.num_classes)
        return DataLoader(
            ds,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        ds = PosterDataset(self.test_df, self.eval_tf, self.genre2idx, self.num_classes)
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

class CNNClassifier(nn.Module):
    """
    CNN backbone + linear head for multi-label genre classification.
    Output shape: (B, num_classes) logits.
    """
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        assert cfg.num_classes is not None, "cfg.num_classes must be set by CNNDataModule first."
        self.cfg = cfg
        self.backbone, in_feats = self._build_backbone(cfg.model_name, cfg.pretrained_weights)
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(in_feats, cfg.num_classes),
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
        out = self.head(feats)      # (B, num_classes) logits
        return out


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
        # multi-label classification loss
        self.criterion = nn.BCEWithLogitsLoss()

        run_name = f"{cfg.model_name}_genres_{datetime.now().strftime('%Y%m%d-%H%M%S')}_bs{cfg.batch_size}_ep{cfg.epochs}_pretrained"
        log_dir = os.path.join("runs", run_name)
        self.writer = SummaryWriter(log_dir=log_dir)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        all_preds = []
        all_targets = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits)
            all_preds.append(probs.detach().cpu())
            all_targets.append(y.detach().cpu())

        avg_loss = total_loss / max(n_batches, 1)

        # Simple micro-F1-like metric using threshold 0.5
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        binary_preds = (preds >= 0.3).float()

        eps = 1e-8
        tp = (binary_preds * targets).sum().item()
        fp = (binary_preds * (1 - targets)).sum().item()
        fn = ((1 - binary_preds) * targets).sum().item()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        return {"loss": avg_loss, "f1": f1}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        best = {"val_f1": 0.0}
        for ep in range(1, self.cfg.epochs + 1):
            self.model.train()
            running_loss, n_batches = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.opt.step()
                running_loss += loss.item()
                n_batches += 1

            train_loss = running_loss / max(n_batches, 1)

            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]
            val_f1 = val_metrics["f1"]

            print(
                f"Epoch {ep}/{self.cfg.epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"val F1: {val_f1:.4f}"
            )

            self.writer.add_scalar("Loss/train", train_loss, ep)
            self.writer.add_scalar("Loss/val", val_loss, ep)
            self.writer.add_scalar("F1/val", val_f1, ep)

            if val_f1 > best["val_f1"]:
                best["val_f1"] = val_f1
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
            logits = self.model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds.append(probs)
        return np.concatenate(preds, axis=0)
# --------------------------------------------------------
# Main
# --------------------------------------------------------

if __name__ == "__main__":
    from new_kaggleloader import KaggleLoader  # must add 'local_path' and keep 'Genre'

    # Load dataset from local Kaggle cache
    loader = KaggleLoader("neha1703/movie-genre-from-its-poster")
    df = loader.df()  # needs columns: 'Genre', 'local_path'

    option = input("Choose model (1: resnet18, 2: resnet50, 3: efficientnet_b0): ")
    if option == "1":
        model_name = "resnet18" # "resnet18"        
    elif option == "2":
        model_name = "resnet50"  # "resnet50"              
    elif option == "3":
        model_name = "efficientnet_b0"  # "efficientnet_b0" 
    else:
        print("Invalid option. Defaulting to resnet18.")
        exit()

    print("\r\n Using model:", model_name)

    cfg = CNNConfig(
        model_name=model_name,
        batch_size=128,
        epochs=25,
        pretrained_weights=True,
    )

    seed_all(cfg.seed)

    dm = CNNDataModule(df, cfg)
    dm.setup(val_size=0.1, test_size=0.1)

    model = CNNClassifier(cfg)
    trainer = CNNTrainer(cfg, model)

    best = trainer.fit(train_loader=dm.train_dataloader(),
                       val_loader=dm.val_dataloader())
    print("Best val F1:", best["val_f1"])

    test_metrics = trainer.evaluate(dm.test_dataloader())
    print("Test loss:", round(test_metrics["loss"], 4))
    print("Test F1:", round(test_metrics["f1"], 4))

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{cfg.model_name}_genres_pretrained.pth")

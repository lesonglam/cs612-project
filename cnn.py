import random, pandas as pd, numpy as np, torch, torch.nn as nn
 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms   as T
from torchvision import models as models

from typing import Tuple, Dict


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


 
class CNNConfig:
    img_size: int = 224
    batch_size: int = 256
    val_batch_size: int = 64
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 2
    model_name: str = "resnet18"  # swap to "efficientnet_b0" etc. if desired
    dropout: float = 0.2
    # normalization for ImageNet-pretrained backbones
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

class PosterDataset(Dataset):
    """
    Expects a DataFrame with columns:
    - 'feature': nested list (H x W x 3) RGB poster
    - 'overall': float rating
    """
    def __init__(self, frame: pd.DataFrame, transform: T.Compose):
        self.df = frame.reset_index(drop=True).copy()
        self.tfm = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        arr = np.array(self.df.loc[idx, "feature"], dtype=np.uint8)  # H x W x 3
        img = Image.fromarray(arr, mode="RGB")
        x = self.tfm(img)                                            # C x H x W
        y = torch.tensor(self.df.loc[idx, "overall"], dtype=torch.float32)
        return x, y
    
    
class CNNDataModule:
    def __init__(self, df: pd.DataFrame, cfg: CNNConfig):
        self.df = df.dropna(subset=["feature", "overall"]).reset_index(drop=True)
        self.cfg = cfg
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        self.train_tf = T.Compose([
            T.ToTensor(),
            T.Resize((cfg.img_size, cfg.img_size)),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=cfg.mean, std=cfg.std),
        ])
        self.eval_tf = T.Compose([
            T.ToTensor(),
            T.Resize((cfg.img_size, cfg.img_size)),
            T.Normalize(mean=cfg.mean, std=cfg.std),
        ])

    def setup(self, val_size: float = 0.1, test_size: float = 0.1):
        from sklearn.model_selection import train_test_split
        seed_all(self.cfg.seed)

        train_df, tmp_df = train_test_split(
            self.df, test_size=val_size + test_size, random_state=self.cfg.seed, shuffle=True
        )
        rel_val = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.5
        val_df, test_df = train_test_split(
            tmp_df, test_size=1 - rel_val, random_state=self.cfg.seed, shuffle=True
        )
        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df

    def train_dataloader(self) -> DataLoader:
        ds = PosterDataset(self.train_df, self.train_tf)
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        ds = PosterDataset(self.val_df, self.eval_tf)
        return DataLoader(ds, batch_size=self.cfg.val_batch_size, shuffle=False,
                          num_workers=self.cfg.num_workers, pin_memory=False)

    def test_dataloader(self) -> DataLoader:
        ds = PosterDataset(self.test_df, self.eval_tf)
        return DataLoader(ds, batch_size=self.cfg.val_batch_size, shuffle=False,
                          num_workers=self.cfg.num_workers, pin_memory=False)
    

class CNNRegressor(nn.Module):
    """
    Wraps an ImageNet-pretrained CNN and replaces the head for 1-D regression (rating).
    """
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone, in_feats = self._build_backbone(cfg.model_name)
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(in_feats, 1)
        )

    def _build_backbone(self, name: str):
        if name == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_feats = m.fc.in_features
            m.fc = nn.Identity() 
        else:
            raise ValueError(f"Unsupported model_name: {name}")
        return m, in_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        out = self.head(feats)           # (B, 1)
        return out.squeeze(1)            # (B,)




class CNNTrainer:
    def __init__(self, cfg: CNNConfig, model: CNNRegressor):
        self.cfg = cfg
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        # Train directly on MAE for stability with ratings
        self.criterion = nn.L1Loss()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        abs_err, n = 0.0, 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
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
                x = x.to(self.device)
                y = y.to(self.device)
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

            if val_mae < best["val_mae"]:
                best["val_mae"] = val_mae
                best["state_dict"] = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # load best
        if "state_dict" in best:
            self.model.load_state_dict(best["state_dict"])
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


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Assume you already have a DataFrame `df` with columns: feature (HxWx3 nested list), overall (float)
    # from your Kaggle dataset load.
    # df = ...  # prepare it before running this file

    from kaggleloader import KaggleLoader
    df = KaggleLoader("shivamardeshna/movies-dataset").df()


    # Minimal demo (uncomment after providing df)
    cfg = CNNConfig()
    
    seed_all(cfg.seed)
    
    dm = CNNDataModule(df, cfg)
    dm.setup(val_size=0.1, test_size=0.1)
    
    model = CNNRegressor(cfg)
    trainer = CNNTrainer(cfg, model)
    
    best = trainer.fit(dm.train_dataloader(), dm.val_dataloader())
    print("Best val MAE:", best["val_mae"])
    
    test_metrics = trainer.evaluate(dm.test_dataloader())
    print("Test MAE:", round(test_metrics["mae"], 4))
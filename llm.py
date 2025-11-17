# llm_oop.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments
import torch, inspect


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# -----------------------------
# Utils / config
# -----------------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class LLMConfig:
    model_name: str = "distilbert-base-uncased"
    max_len: int = 256
    train_batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 3
    lr: float = 5e-5
    weight_decay: float = 0.01
    seed: int = 42
    fp16: Optional[bool] = True  # None -> auto: use True if CUDA available
    output_dir: str = "distilbert-reg"
    text_col: str = "reviewText"  # or "original_reviewText"
    label_col: str = "overall"


# -----------------------------
# Dataset
# -----------------------------
class ReviewDataset(Dataset):
    """
    Expects a DataFrame with text and numeric label columns.
    Tokenization happens on-the-fly to avoid storing big tensors in RAM.
    """
    def __init__(self, frame: pd.DataFrame, tokenizer: AutoTokenizer, cfg: LLMConfig):
        self.df = frame.reset_index(drop=True).copy()
        self.tok = tokenizer
        self.cfg = cfg

        # Strict types
        self.df[self.cfg.text_col] = self.df[self.cfg.text_col].astype(str)
        self.df[self.cfg.label_col] = self.df[self.cfg.label_col].astype(float)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        text = self.df.at[idx, self.cfg.text_col]
        label = float(self.df.at[idx, self.cfg.label_col])
        enc = self.tok(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float32)  # regression target
        return item


# -----------------------------
# DataModule (splits + collator)
# -----------------------------
class LLMDataModule:
    def __init__(self, df: pd.DataFrame, cfg: LLMConfig):
        self.cfg = cfg
        self.df = df.dropna(subset=[cfg.text_col, cfg.label_col]).reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.data_collator = DataCollatorWithPadding(self.tokenizer, padding="longest")

        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def setup(self, val_size: float = 0.1, test_size: float = 0.1):
        from sklearn.model_selection import train_test_split
        seed_all(self.cfg.seed)
        train_df, tmp = train_test_split(
            self.df, test_size=val_size + test_size, random_state=self.cfg.seed, shuffle=True
        )
        rel_val = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.5
        val_df, test_df = train_test_split(
            tmp, test_size=1 - rel_val, random_state=self.cfg.seed, shuffle=True
        )
        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df

    def train_dataset(self) -> ReviewDataset:
        return ReviewDataset(self.train_df, self.tokenizer, self.cfg)

    def val_dataset(self) -> ReviewDataset:
        return ReviewDataset(self.val_df, self.tokenizer, self.cfg)

    def test_dataset(self) -> ReviewDataset:
        return ReviewDataset(self.test_df, self.tokenizer, self.cfg)


# -----------------------------
# Model
# -----------------------------
class LLMRegressor:
    """
    Thin wrapper around a Hugging Face regression head (num_labels=1).
    Provides train/eval using HF Trainer.
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=1
        )

    @staticmethod
    def compute_metrics(eval_pred) -> Dict[str, float]:
        preds, labels = eval_pred
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        mae = float(np.mean(np.abs(preds - labels)))
        mse = float(np.mean((preds - labels) ** 2))
        return {"mae": mae, "mse": mse}
    
    def _training_args(self) -> TrainingArguments:
        base = dict(
            output_dir=self.cfg.output_dir,
            learning_rate=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            per_device_train_batch_size=self.cfg.train_batch_size,
            per_device_eval_batch_size=self.cfg.eval_batch_size,
            num_train_epochs=self.cfg.epochs,
            logging_steps=50,
            seed=self.cfg.seed,
            fp16=(torch.cuda.is_available() if self.cfg.fp16 is None else bool(self.cfg.fp16)),
            eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
            save_strategy="epoch",
            report_to=[],  # Disable external logging
        )
        return TrainingArguments(**base)
    
    def fit(
        self,
        dm: LLMDataModule,
    ) -> Dict[str, float]:
        trainer = Trainer(
            model=self.model,
            args=self._training_args(),
            train_dataset=dm.train_dataset(),
            eval_dataset=dm.val_dataset(),
            tokenizer=dm.tokenizer,
            data_collator=dm.data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate(eval_dataset=dm.val_dataset())
        # Keep best weights already loaded by Trainer
        return {"val_mae": metrics.get("eval_mae", float("inf")), **metrics}

    def evaluate(self, dm: LLMDataModule) -> Dict[str, float]:
        trainer = Trainer(
            model=self.model,
            args=self._training_args(),
            eval_dataset=dm.test_dataset(),
            tokenizer=dm.tokenizer,
            data_collator=dm.data_collator,
            compute_metrics=self.compute_metrics,
        )
        test_metrics = trainer.evaluate()
        # rename keys for clarity
        out = {k.replace("eval_", "test_"): v for k, v in test_metrics.items()}
        return out

    def predict_scores(self, dm: LLMDataModule) -> np.ndarray:
        # Create separate args for prediction without evaluation strategy
        pred_args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            per_device_eval_batch_size=self.cfg.eval_batch_size,
            seed=self.cfg.seed,
            fp16=(torch.cuda.is_available() if self.cfg.fp16 is None else bool(self.cfg.fp16)),
            report_to=[],
            # No evaluation strategy for prediction
        )
        
        trainer = Trainer(
            model=self.model,
            args=pred_args,
            tokenizer=dm.tokenizer,
            data_collator=dm.data_collator,
        )
        preds = trainer.predict(test_dataset=dm.test_dataset()).predictions
        return preds.reshape(-1)
    
    def save_model(self, save_path: str):
        """Save the trained model and tokenizer"""
        self.model.save_pretrained(save_path)
        # Note: You'll need access to the tokenizer from somewhere

    def save_pretrained(self, save_path: str, tokenizer=None):
        """Save model and optionally tokenizer"""
        self.model.save_pretrained(save_path)
        if tokenizer:
            tokenizer.save_pretrained(save_path)

 
if __name__ == "__main__":
    a= 0
    # df must contain columns:
    #   - text (default cfg.text_col = 'reviewText')
    #   - label (default cfg.label_col = 'overall')
    # plus your other columns. Ensure no NaNs in these.
    # df = ...  # load your DataFrame here
    from kaggleloader import KaggleLoader
    df = KaggleLoader("shivamardeshna/movies-dataset").df()

    # Minimal demo (uncomment after providing df)
    cfg = LLMConfig(
        model_name="distilbert-base-uncased",
        epochs=1,   # for demo purposes
        train_batch_size=16,
        eval_batch_size=32,
        text_col="reviewText",
        label_col="overall",
    )
    seed_all(cfg.seed)
    
    dm = LLMDataModule(df, cfg)
    dm.setup(val_size=0.1, test_size=0.1)
    
    llm = LLMRegressor(cfg)
    best = llm.fit(dm)
    print("Best (val) metrics:", best)


    # Save the trained model
    model_save_path = "trained_distilbert_regressor"
    llm.model.save_pretrained(model_save_path)
    dm.tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

    
    test_metrics = llm.evaluate(dm)
    print("Test metrics:", test_metrics)
    
    preds = llm.predict_scores(dm)  # raw regression outputs
    print("Predictions shape:", preds.shape)

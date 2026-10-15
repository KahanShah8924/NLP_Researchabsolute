"""Offline Training and Temperature Calibration for XLM-RoBERTa Sentiment Head.

Runs separately from interactive inference to maintain minimal memory consumption.
Trains the lightweight classification head on top of the shared XLM-RoBERTa encoder
and tunes temperature scaling on the validation set for confidence calibration.

Usage:
    python offline_finetune.py --train_samples 1200 --val_samples 400 --epochs 5 --lr 1e-3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.single_transformer import (
    SENTIMENT_LABELS,
    SENTIMENT_LABEL_TO_ID,
    SingleTransformerMultitask,
    get_optimal_device,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


class TextSentimentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return str(self.texts[idx]), int(self.labels[idx])


def train_sentiment_head(
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/validation.csv",
    output_path: str = "models/xlmr_sentiment_head.pt",
    train_samples: int = 1500,
    val_samples: int = 500,
    batch_size: int = 8,
    epochs: int = 4,
    lr: float = 8e-4,
    seed: int = 42,
) -> dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_optimal_device()
    LOGGER.info("Device selected for training: %s", device)

    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    if train_samples > 0 and train_samples < len(train_df):
        train_df = train_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), train_samples // 3), random_state=seed)
        ).reset_index(drop=True)

    if val_samples > 0 and val_samples < len(val_df):
        val_df = val_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), val_samples // 3), random_state=seed)
        ).reset_index(drop=True)

    LOGGER.info("Train dataset size: %d | Validation dataset size: %d", len(train_df), len(val_df))

    # Initialize single model
    model = SingleTransformerMultitask(device=device, max_length=128)
    model.encoder.eval()  # Freeze heavy backbone to save memory & training time

    train_texts = train_df["text"].tolist()
    train_labels = [SENTIMENT_LABEL_TO_ID[l] for l in train_df["label"]]
    val_texts = val_df["text"].tolist()
    val_labels = [SENTIMENT_LABEL_TO_ID[l] for l in val_df["label"]]

    # Pre-extract CLS embeddings in small memory-safe batches
    LOGGER.info("Pre-extracting CLS embeddings from frozen backbone (batch_size=8)...")
    t0 = time.perf_counter()

    def get_all_embeddings(texts: list[str]) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(texts), 8):
            chunk = texts[i : i + 8]
            embs = model.extract_cls(chunk)
            all_embs.append(embs.cpu())
            if device.type == "mps":
                torch.mps.empty_cache()
        return torch.cat(all_embs, dim=0)

    train_x = get_all_embeddings(train_texts).to(device)
    train_y = torch.tensor(train_labels, dtype=torch.long, device=device)
    val_x = get_all_embeddings(val_texts).to(device)
    val_y = torch.tensor(val_labels, dtype=torch.long, device=device)

    LOGGER.info("Extracted embeddings in %.1fs (Train: %s, Val: %s)", time.perf_counter() - t0, train_x.shape, val_x.shape)

    # Train head
    head = model.sentiment_head.to(device)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_state = None

    dataset_size = len(train_x)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(dataset_size)
        total_loss = 0.0
        batches = 0

        for b_start in range(0, dataset_size, batch_size):
            indices = perm[b_start : b_start + batch_size]
            b_x, b_y = train_x[indices], train_y[indices]

            optimizer.zero_grad()
            logits = head(b_x)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        # Validation
        head.eval()
        with torch.no_grad():
            val_logits = head(val_x)
            val_preds = torch.argmax(val_logits, dim=-1).cpu().numpy()
            acc = accuracy_score(val_y.cpu().numpy(), val_preds)
            f1 = f1_score(val_y.cpu().numpy(), val_preds, average="macro", zero_division=0)

        LOGGER.info("Epoch %d/%d - Loss: %.4f | Val Acc: %.2f%% | Val Macro-F1: %.4f", epoch, epochs, total_loss / max(batches, 1), acc * 100, f1)

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        head.train()

    # Load best weights
    if best_state is not None:
        head.load_state_dict(best_state)

    # Temperature Scaling Calibration on Validation Set
    head.eval()
    with torch.no_grad():
        unscaled_logits = head(val_x) * head.temperature

    temp = nn.Parameter(torch.ones(1, device=device))
    temp_opt = torch.optim.LBFGS([temp], lr=0.01, max_iter=50)

    def eval_temp():
        temp_opt.zero_grad()
        loss = criterion(unscaled_logits / torch.clamp(temp, min=0.01), val_y)
        loss.backward()
        return loss

    temp_opt.step(eval_temp)
    best_temp = float(temp.data.item())
    head.temperature.data = temp.data
    LOGGER.info("Learned optimal temperature scaling T: %.4f", best_temp)

    # Save trained head weights
    model.save_head_weights(output_path)
    LOGGER.info("Successfully saved calibrated sentiment head to %s", output_path)

    return {"best_val_f1": best_val_f1, "calibrated_temperature": best_temp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lightweight sentiment head on XLM-RoBERTa encoder")
    parser.add_argument("--train_samples", type=int, default=1500)
    parser.add_argument("--val_samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-4)
    args = parser.parse_args()

    train_sentiment_head(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

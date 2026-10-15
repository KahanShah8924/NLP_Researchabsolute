"""Plain-PyTorch XLM-RoBERTa-base training and robustness evaluation for Colab/Kaggle GPU.

Evaluates transformer fine-tuning under:
- baseline (clean training)
- norm_light (clean training with light rule-based normalization)
- aug_A (train-time Family A spelling augmentation)

Evaluates on Clean, Family A (0.1, 0.2, 0.3), and Held-Out Family B (0.1, 0.2, 0.3).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from robust_kit.textvar import (
    augment_dataset,
    normalize_light,
    perturb_sentence,
)

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SentimentDataset(Dataset):
    """PyTorch Dataset for text classification."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_epoch(model: Any, dataloader: DataLoader, optimizer: Any, scheduler: Any, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(model: Any, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Evaluate accuracy and macro-F1."""
    model.eval()
    all_preds = []
    all_labels = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return acc, f1


def run_training_and_eval(
    method: str,
    seed: int,
    data_dir: Path,
    out_dir: Path,
    model_name: str = "xlm-roberta-base",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_len: int = 128,
) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    if torch.cuda.is_available():
        LOGGER.info("GPU: %s (Memory: %.2f GB)", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9)
        
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    
    # 1. Prepare Training Data based on method
    if method == "norm_light":
        LOGGER.info("Applying norm_light to training data...")
        train_texts = [normalize_light(t) for t in train_df["text"]]
        train_labels = [label_map[l] for l in train_df["label"]]
    elif method == "aug_A":
        LOGGER.info("Applying Family A augmentation to training data...")
        aug_texts, aug_labels = augment_dataset(
            train_df["text"].tolist(), train_df["label"].tolist(), p=0.3, seed=seed
        )
        train_texts = aug_texts
        train_labels = [label_map[l] for l in aug_labels]
    else:  # baseline
        LOGGER.info("Using clean training data for baseline...")
        train_texts = train_df["text"].tolist()
        train_labels = [label_map[l] for l in train_df["label"]]
        
    # Validation data is always clean
    val_texts = val_df["text"].tolist()
    if method == "norm_light":
        val_texts = [normalize_light(t) for t in val_texts]
    val_labels = [label_map[l] for l in val_df["label"]]
    
    # 2. Tokenizer and DataLoader
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = SentimentDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    val_ds = SentimentDataset(val_texts, val_labels, tokenizer, max_len=max_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # 3. Model, Optimizer, Scheduler
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    # 4. Training Loop with best checkpoint tracking on CLEAN validation
    best_val_f1 = -1.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        LOGGER.info("Epoch %d/%d - Loss: %.4f - Val Acc: %.4f - Val Macro-F1: %.4f", epoch, epochs, loss, val_acc, val_f1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    LOGGER.info("Best Validation Macro-F1: %.4f", best_val_f1)
    
    # 5. Evaluate on all test conditions
    TEST_SEED = 2026
    test_conditions = {
        "Clean": list(test_df["text"]),
        "Family_A@0.1": [perturb_sentence(t, "A", p=0.1, seed=TEST_SEED + i) for i, t in enumerate(test_df["text"])],
        "Family_A@0.2": [perturb_sentence(t, "A", p=0.2, seed=TEST_SEED + i) for i, t in enumerate(test_df["text"])],
        "Family_A@0.3": [perturb_sentence(t, "A", p=0.3, seed=TEST_SEED + i) for i, t in enumerate(test_df["text"])],
        "Family_B@0.1": [perturb_sentence(t, "B", p=0.1, seed=TEST_SEED + i) for i, t in enumerate(test_df["text"])],
        "Family_B@0.2": [perturb_sentence(t, "B", p=0.2, seed=TEST_SEED + i) for i, t in enumerate(test_df["text"])],
        "Family_B@0.3": [perturb_sentence(t, "B", p=0.3, seed=TEST_SEED + i) for i, t in enumerate(test_df["text"])],
    }
    test_labels = [label_map[l] for l in test_df["label"]]
    
    results = []
    for cond_name, raw_texts in test_conditions.items():
        eval_texts = [normalize_light(t) for t in raw_texts] if method == "norm_light" else raw_texts
        test_ds = SentimentDataset(eval_texts, test_labels, tokenizer, max_len=max_len)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)
        test_acc, test_f1 = evaluate(model, test_loader, device)
        results.append({
            "model": model_name,
            "method": method,
            "seed": seed,
            "test_condition": cond_name,
            "accuracy": test_acc,
            "macro_f1": test_f1,
            "val_f1_clean": best_val_f1,
        })
        LOGGER.info("[%s | %s | Seed %d] %s: Acc=%.4f Macro-F1=%.4f", model_name, method, seed, cond_name, test_acc, test_f1)
        
    out_df = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"robust_{model_name}_{method}_s{seed}.csv"
    out_df.to_csv(out_file, index=False)
    LOGGER.info("Saved results to %s", out_file)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["baseline", "norm_light", "aug_A"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_dir", default="data/processed", type=Path)
    parser.add_argument("--out_dir", default="results", type=Path)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    args = parser.parse_args()
    run_training_and_eval(
        method=args.method,
        seed=args.seed,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()

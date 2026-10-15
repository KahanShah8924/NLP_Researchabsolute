"""Evaluate trained classical models on the exploratory Gujlish probe set."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score


def evaluate_probe(probe_path: Path, model_path: Path) -> None:
    df = pd.read_csv(probe_path)
    if "label_annotator1" not in df.columns or "label_annotator2" not in df.columns:
        raise ValueError("Probe CSV must have label_annotator1 and label_annotator2.")
        
    valid_mask = df["label_annotator1"].notna() & df["label_annotator2"].notna()
    df_valid = df[valid_mask].copy()
    
    # Inter-annotator agreement
    kappa = cohen_kappa_score(df_valid["label_annotator1"], df_valid["label_annotator2"])
    print(f"Inter-annotator agreement (Cohen's Kappa): {kappa:.4f}")
    
    # Agreed subset
    agreed = df_valid[df_valid["label_annotator1"] == df_valid["label_annotator2"]].copy()
    print(f"Agreed items: {len(agreed)} / {len(df_valid)} ({len(agreed)/len(df_valid)*100:.1f}%)")
    
    if len(agreed) == 0:
        print("No agreed items to evaluate.")
        return
        
    if model_path.exists():
        with model_path.open("rb") as f:
            checkpoint = pickle.load(f)
        features, model = checkpoint["features"], checkpoint["model"]
        X = features.transform(agreed["text"])
        preds = model.predict(X)
        acc = accuracy_score(agreed["label_annotator1"], preds)
        f1 = f1_score(agreed["label_annotator1"], preds, average="macro", zero_division=0)
        print(f"Classical Baseline on Agreed Gujlish Probe: Accuracy={acc:.4f}, Macro-F1={f1:.4f}")
    else:
        print(f"Model file {model_path} not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default=Path("results/gujlish_probe_template.csv"), type=Path)
    parser.add_argument("--model", default=Path("models/logistic_regression.pkl"), type=Path)
    args = parser.parse_args()
    evaluate_probe(args.probe, args.model)

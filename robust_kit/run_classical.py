"""Run classical TF-IDF Logistic Regression robustness experiments with paired-bootstrap CIs.

Methods compared:
1. baseline: clean train, standard TF-IDF + Logistic Regression
2. norm_light: rule-based light normalization applied to train and test
3. norm_skeleton: consonant skeleton reduction (lossy ablation)
4. aug_A: train-time spelling augmentation using Family A perturbations
5. norm_light+aug_A: Family A augmentation + light normalization
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion

from robust_kit.textvar import (
    augment_dataset,
    normalize_light,
    normalize_skeleton,
    perturb_sentence,
)

LOGGER = logging.getLogger(__name__)


def build_tfidf_features(max_features: int = 50000) -> FeatureUnion:
    """Build word (1-2) and character-wb (3-5) TF-IDF feature union."""
    return FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=max_features,
                sublinear_tf=True
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=max_features,
                sublinear_tf=True
            ),
        ),
    ])


def build_classifier(seed: int = 42) -> LogisticRegression:
    """Return reproducible Logistic Regression classifier matching repo baseline."""
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="liblinear"
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy and macro-F1 score."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    }


def paired_bootstrap_ci(
    y_true: np.ndarray,
    pred_method: np.ndarray,
    pred_baseline: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> tuple[float, float, float]:
    """Compute 95% paired-bootstrap confidence interval for (macro_f1_method - macro_f1_baseline)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = np.empty(n_bootstraps, dtype=float)
    
    for i in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        f1_m = f1_score(y_true[idx], pred_method[idx], average="macro", zero_division=0)
        f1_b = f1_score(y_true[idx], pred_baseline[idx], average="macro", zero_division=0)
        diffs[i] = f1_m - f1_b
        
    lower = float(np.percentile(diffs, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0)))
    mean_diff = float(np.mean(diffs))
    return mean_diff, lower, upper


def run_experiments(data_dir: Path, out_path: Path, seed: int = 42) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # 1. Load data
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    LOGGER.info("Loaded splits: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))
    
    # 2. Hard Rule 4 & Leakage Check
    train_clean_set = set(train_df["text"].str.strip())
    test_clean_set = set(test_df["text"].str.strip())
    exact_duplicates = len(train_clean_set.intersection(test_clean_set))
    print(f"LEAKAGE_CHECK: exact train/test duplicates = {exact_duplicates} (out of {len(test_df)} test rows)")
    
    # 3. Prepare Test Sets (Fixed Test Seed = 2026; NEVER reused for training)
    TEST_SEED = 2026
    test_conditions: dict[str, list[str]] = {
        "Clean": list(test_df["text"]),
        "Family_A@0.1": [perturb_sentence(t, "A", p=0.1, seed=TEST_SEED + i) for i, t in enumerate(test_df["text"])],
        "Family_A@0.2": [perturb_sentence(t, "A", p=0.2, seed=TEST_SEED + i) for i, t in enumerate(test_df["text"])],
        "Family_A@0.3": [perturb_sentence(t, "A", p=0.3, seed=TEST_SEED + i) for i, t in enumerate(test_df["text"])],
        "Family_B@0.1": [perturb_sentence(t, "B", p=0.1, seed=TEST_SEED + i) for i, t in enumerate(test_df["text"])],
        "Family_B@0.2": [perturb_sentence(t, "B", p=0.2, seed=TEST_SEED + i) for i, t in enumerate(test_df["text"])],
        "Family_B@0.3": [perturb_sentence(t, "B", p=0.3, seed=TEST_SEED + i) for i, t in enumerate(test_df["text"])],
    }
    
    y_test = np.array(test_df["label"].tolist())
    
    # 4. Prepare Training Data for Each Method
    methods = ["baseline", "norm_light", "norm_skeleton", "aug_A", "norm_light+aug_A"]
    trained_pipelines: dict[str, tuple[FeatureUnion, LogisticRegression]] = {}
    
    LOGGER.info("Training models across 5 methods...")
    
    # Method 1: baseline
    LOGGER.info("Training Method: baseline")
    feat_base = build_tfidf_features()
    clf_base = build_classifier(seed=seed)
    X_train_base = feat_base.fit_transform(train_df["text"])
    clf_base.fit(X_train_base, train_df["label"])
    trained_pipelines["baseline"] = (feat_base, clf_base)
    
    # Method 2: norm_light
    LOGGER.info("Training Method: norm_light")
    train_norm_light = [normalize_light(t) for t in train_df["text"]]
    feat_norm_light = build_tfidf_features()
    clf_norm_light = build_classifier(seed=seed)
    X_train_nl = feat_norm_light.fit_transform(train_norm_light)
    clf_norm_light.fit(X_train_nl, train_df["label"])
    trained_pipelines["norm_light"] = (feat_norm_light, clf_norm_light)
    
    # Method 3: norm_skeleton (ablation)
    LOGGER.info("Training Method: norm_skeleton")
    train_norm_skel = [normalize_skeleton(t) for t in train_df["text"]]
    feat_norm_skel = build_tfidf_features()
    clf_norm_skel = build_classifier(seed=seed)
    X_train_skel = feat_norm_skel.fit_transform(train_norm_skel)
    clf_norm_skel.fit(X_train_skel, train_df["label"])
    trained_pipelines["norm_skeleton"] = (feat_norm_skel, clf_norm_skel)
    
    # Method 4: aug_A (Family A training augmentation, seed=42)
    LOGGER.info("Training Method: aug_A")
    aug_train_texts, aug_train_labels = augment_dataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        p=0.3,
        seed=seed
    )
    LOGGER.info("Augmented training set from %d to %d rows", len(train_df), len(aug_train_texts))
    feat_aug = build_tfidf_features()
    clf_aug = build_classifier(seed=seed)
    X_train_aug = feat_aug.fit_transform(aug_train_texts)
    clf_aug.fit(X_train_aug, aug_train_labels)
    trained_pipelines["aug_A"] = (feat_aug, clf_aug)
    
    # Method 5: norm_light+aug_A
    LOGGER.info("Training Method: norm_light+aug_A")
    aug_nl_texts = [normalize_light(t) for t in aug_train_texts]
    feat_aug_nl = build_tfidf_features()
    clf_aug_nl = build_classifier(seed=seed)
    X_train_aug_nl = feat_aug_nl.fit_transform(aug_nl_texts)
    clf_aug_nl.fit(X_train_aug_nl, aug_train_labels)
    trained_pipelines["norm_light+aug_A"] = (feat_aug_nl, clf_aug_nl)
    
    # 5. Evaluate on All Test Conditions and Record Predictions
    all_preds: dict[tuple[str, str], np.ndarray] = {}
    results_records: list[dict[str, Any]] = []
    
    for cond_name, raw_test_texts in test_conditions.items():
        # Preprocess test texts according to method
        for m_name in methods:
            feat, clf = trained_pipelines[m_name]
            
            if m_name in ("norm_light", "norm_light+aug_A"):
                eval_texts = [normalize_light(t) for t in raw_test_texts]
            elif m_name == "norm_skeleton":
                eval_texts = [normalize_skeleton(t) for t in raw_test_texts]
            else:
                eval_texts = raw_test_texts
                
            X_eval = feat.transform(eval_texts)
            preds = clf.predict(X_eval)
            all_preds[(m_name, cond_name)] = preds
            
            metrics = compute_metrics(y_test, preds)
            results_records.append({
                "method": m_name,
                "test_condition": cond_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"]
            })
            
    res_df = pd.DataFrame(results_records)
    
    # 6. Compute Paired-Bootstrap CIs vs baseline for each condition
    ci_records: list[dict[str, Any]] = []
    for cond_name in test_conditions:
        pred_base = all_preds[("baseline", cond_name)]
        for m_name in methods:
            if m_name == "baseline":
                ci_records.append({
                    "method": m_name,
                    "test_condition": cond_name,
                    "diff_mean": 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "significant": False
                })
            else:
                pred_m = all_preds[(m_name, cond_name)]
                diff_mean, ci_low, ci_high = paired_bootstrap_ci(
                    y_test, pred_m, pred_base, n_bootstraps=1000, seed=seed
                )
                is_sig = not (ci_low <= 0.0 <= ci_high)
                ci_records.append({
                    "method": m_name,
                    "test_condition": cond_name,
                    "diff_mean": diff_mean,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "significant": is_sig
                })
                
    ci_df = pd.DataFrame(ci_records)
    merged_df = pd.merge(res_df, ci_df, on=["method", "test_condition"])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_path, index=False)
    LOGGER.info("Saved full results to %s", out_path)
    
    # 7. Produce Required Tables
    results_dir = out_path.parent
    
    # (a) Pivot table of macro-F1
    pivot_f1 = merged_df.pivot(index="method", columns="test_condition", values="macro_f1")
    # Reorder columns logically
    ordered_cols = ["Clean", "Family_A@0.1", "Family_A@0.2", "Family_A@0.3", "Family_B@0.1", "Family_B@0.2", "Family_B@0.3"]
    pivot_f1 = pivot_f1[ordered_cols].loc[methods]
    pivot_f1.to_csv(results_dir / "pivot_macro_f1.csv")
    LOGGER.info("Saved pivot_macro_f1.csv:\n%s", pivot_f1.round(4).to_string())
    
    # (b) Drop-from-clean table (Clean - Condition)
    drop_df = pd.DataFrame(index=methods)
    for col in ordered_cols:
        drop_df[col + "_drop"] = pivot_f1["Clean"] - pivot_f1[col]
    drop_df.to_csv(results_dir / "drop_from_clean.csv")
    LOGGER.info("Saved drop_from_clean.csv:\n%s", drop_df.round(4).to_string())
    
    # (c) Bootstrap CI table for each method vs baseline
    ci_pivot = merged_df[merged_df["method"] != "baseline"].pivot(
        index="method", columns="test_condition", values="diff_mean"
    )[ordered_cols]
    ci_df.to_csv(results_dir / "bootstrap_ci.csv", index=False)
    LOGGER.info("Saved bootstrap_ci.csv")
    
    # 8. Produce Figures
    sns.set_theme(style="whitegrid", font="sans-serif")
    
    # (d) Grouped bar chart of macro-F1 per test condition
    plt.figure(figsize=(12, 6))
    bar_data = merged_df.copy()
    bar_data["macro_f1_pct"] = bar_data["macro_f1"] * 100
    ax = sns.barplot(
        data=bar_data,
        x="test_condition",
        y="macro_f1_pct",
        hue="method",
        palette="viridis",
        order=ordered_cols
    )
    plt.title("Hinglish Sentiment Robustness: Macro-F1 across Perturbation Conditions", fontsize=14, weight="bold")
    plt.xlabel("Test Condition", fontsize=12)
    plt.ylabel("Macro-F1 (%)", fontsize=12)
    plt.ylim(40, 75)
    plt.xticks(rotation=25)
    plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    bar_chart_path = results_dir / "grouped_bar_macro_f1.png"
    plt.savefig(bar_chart_path, dpi=200)
    plt.close()
    LOGGER.info("Saved grouped bar chart to %s", bar_chart_path)
    
    # (e) Degradation line chart (macro-F1 vs perturbation level 0.0, 0.1, 0.2, 0.3)
    plt.figure(figsize=(10, 5))
    levels = [0.0, 0.1, 0.2, 0.3]
    palette = sns.color_palette("tab10", len(methods))
    
    # Subplot 1: Family A
    plt.subplot(1, 2, 1)
    for m_idx, m_name in enumerate(methods):
        f1_vals = [
            pivot_f1.loc[m_name, "Clean"] * 100,
            pivot_f1.loc[m_name, "Family_A@0.1"] * 100,
            pivot_f1.loc[m_name, "Family_A@0.2"] * 100,
            pivot_f1.loc[m_name, "Family_A@0.3"] * 100,
        ]
        plt.plot(levels, f1_vals, marker="o", label=m_name, color=palette[m_idx], linewidth=2)
    plt.title("Family A: Respelling (Train-seen)", fontsize=11, weight="bold")
    plt.xlabel("Perturbation Level p", fontsize=10)
    plt.ylabel("Macro-F1 (%)", fontsize=10)
    plt.ylim(45, 72)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Subplot 2: Family B (Held Out)
    plt.subplot(1, 2, 2)
    for m_idx, m_name in enumerate(methods):
        f1_vals = [
            pivot_f1.loc[m_name, "Clean"] * 100,
            pivot_f1.loc[m_name, "Family_B@0.1"] * 100,
            pivot_f1.loc[m_name, "Family_B@0.2"] * 100,
            pivot_f1.loc[m_name, "Family_B@0.3"] * 100,
        ]
        plt.plot(levels, f1_vals, marker="s", label=m_name, color=palette[m_idx], linewidth=2)
    plt.title("Family B: Shorthand / Typos (Held Out)", fontsize=11, weight="bold")
    plt.xlabel("Perturbation Level p", fontsize=10)
    plt.ylabel("Macro-F1 (%)", fontsize=10)
    plt.ylim(45, 72)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    line_chart_path = results_dir / "degradation_curves.png"
    plt.savefig(line_chart_path, dpi=200)
    plt.close()
    LOGGER.info("Saved degradation curves to %s", line_chart_path)
    print("Classical experiments completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/processed", type=Path)
    parser.add_argument("--out", default=Path("results/robustness_classical.csv"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    run_experiments(args.data_dir, args.out, seed=args.seed)


if __name__ == "__main__":
    main()

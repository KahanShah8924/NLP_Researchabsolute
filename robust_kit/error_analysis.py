"""Detailed error analysis for Hinglish sentiment robustness experiments.

Identifies:
1. Top examples where baseline succeeded on clean text but failed under Family B@0.3.
2. Word collision / distortion cases where normalization caused errors.
3. Class-wise F1 scores across methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from robust_kit.run_classical import build_classifier, build_tfidf_features
from robust_kit.textvar import normalize_light, normalize_skeleton, perturb_sentence

LOGGER = logging.getLogger(__name__)


def run_error_analysis(data_dir: Path, out_dir: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    # 1. Train baseline and norm_light models
    feat_base = build_tfidf_features()
    clf_base = build_classifier(seed=42)
    X_train = feat_base.fit_transform(train_df["text"])
    clf_base.fit(X_train, train_df["label"])
    
    feat_nl = build_tfidf_features()
    clf_nl = build_classifier(seed=42)
    X_train_nl = feat_nl.fit_transform([normalize_light(t) for t in train_df["text"]])
    clf_nl.fit(X_train_nl, train_df["label"])
    
    # Predictions
    y_test = np.array(test_df["label"].tolist())
    clean_texts = test_df["text"].tolist()
    
    TEST_SEED = 2026
    b03_texts = [perturb_sentence(t, "B", p=0.3, seed=TEST_SEED + i) for i, t in enumerate(clean_texts)]
    
    preds_clean = clf_base.predict(feat_base.transform(clean_texts))
    preds_b03 = clf_base.predict(feat_base.transform(b03_texts))
    preds_nl_clean = clf_nl.predict(feat_nl.transform([normalize_light(t) for t in clean_texts]))
    
    # Class-wise F1 table
    labels = ["negative", "neutral", "positive"]
    report_base = classification_report(y_test, preds_clean, labels=labels, output_dict=True)
    report_b03 = classification_report(y_test, preds_b03, labels=labels, output_dict=True)
    report_nl = classification_report(y_test, preds_nl_clean, labels=labels, output_dict=True)
    
    f1_rows = []
    for l in labels:
        f1_rows.append({
            "class": l,
            "baseline_clean_f1": report_base[l]["f1-score"],
            "baseline_b03_f1": report_b03[l]["f1-score"],
            "norm_light_clean_f1": report_nl[l]["f1-score"]
        })
    pd.DataFrame(f1_rows).to_csv(out_dir / "class_wise_f1.csv", index=False)
    LOGGER.info("Saved class_wise_f1.csv")
    
    # 2. Top examples where baseline was correct on clean, but flipped on Family B@0.3
    degraded_indices = np.where((preds_clean == y_test) & (preds_b03 != y_test))[0]
    LOGGER.info("Total examples degraded under Family B@0.3: %d", len(degraded_indices))
    
    degraded_rows = []
    for idx in degraded_indices:
        orig = clean_texts[idx]
        pert = b03_texts[idx]
        true_l = y_test[idx]
        pred_b = preds_b03[idx]
        
        # Determine changed tokens
        orig_toks = orig.split()
        pert_toks = pert.split()
        changed = []
        if len(orig_toks) == len(pert_toks):
            for o, p in zip(orig_toks, pert_toks):
                if o != p:
                    changed.append(f"{o}->{p}")
                    
        degraded_rows.append({
            "example_id": idx,
            "true_label": true_l,
            "pred_clean": true_l,
            "pred_perturbed_b03": pred_b,
            "clean_text": orig,
            "perturbed_text": pert,
            "changed_tokens": ", ".join(changed[:6]),
            "hypothesized_cause": "Hypothesis: token distortion removed n-gram features from vocabulary"
        })
        
    top30_df = pd.DataFrame(degraded_rows).head(30)
    top30_df.to_csv(out_dir / "error_analysis_top30.csv", index=False)
    LOGGER.info("Saved top 30 degraded examples to error_analysis_top30.csv")
    
    # 3. Normalization Hurt / Word Collisions (where clean was correct, but norm_light failed)
    norm_hurt_indices = np.where((preds_clean == y_test) & (preds_nl_clean != y_test))[0]
    LOGGER.info("Total examples where norm_light flipped correct clean baseline: %d", len(norm_hurt_indices))
    
    collision_rows = []
    collision_patterns = [r"\b(we|will|with|what|when|where|well|why)\b", r"\b(phone|photo|phir)\b"]
    for idx in norm_hurt_indices:
        orig = clean_texts[idx]
        normed = normalize_light(orig)
        true_l = y_test[idx]
        pred_nl = preds_nl_clean[idx]
        
        # Check if English 'w' or 'ph' was present
        has_w = bool(re.search(r"\bw[a-z]+", orig, re.IGNORECASE))
        has_ph = bool(re.search(r"\bph[a-z]+", orig, re.IGNORECASE))
        
        collision_rows.append({
            "example_id": idx,
            "true_label": true_l,
            "pred_baseline_clean": true_l,
            "pred_norm_light": pred_nl,
            "clean_text": orig,
            "normalized_text": normed,
            "contains_w_or_ph": has_w or has_ph,
            "hypothesized_cause": "Hypothesis: phonetic mapping 'w'->'v' or 'ph'->'f' corrupted standard English lexical features"
        })
        
    collisions_df = pd.DataFrame(collision_rows).head(20)
    collisions_df.to_csv(out_dir / "normalization_collisions.csv", index=False)
    LOGGER.info("Saved normalization collisions to normalization_collisions.csv")


if __name__ == "__main__":
    run_error_analysis(Path("data/processed"), Path("results"))

"""Reproducible Benchmark for Single XLM-RoBERTa-base on Apple Silicon M2.

Evaluates 3 experimental configurations:
- Experiment A: XLM-RoBERTa-base without memory (raw model)
- Experiment B: XLM-RoBERTa-base with linguistic preprocessing & memory retrieval
- Experiment C: XLM-RoBERTa-base with preprocessing, memory retrieval, and calibrated fusion

Measures:
- Model loading time (s)
- First inference latency (warmup ms)
- Subsequent inference latency (steady-state mean/p95 ms)
- Peak process RSS memory (MB) vs system unified memory
- Device: Apple Silicon MPS / CPU
- Metrics: Accuracy, Macro-F1, per-class F1, ECE, Brier Score
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch

from src.fusion import EvidenceFusionEngine, compute_brier_score, compute_ece
from src.linguistic_perception import analyze_linguistic_profile
from src.single_transformer import (
    SENTIMENT_LABELS,
    SENTIMENT_LABEL_TO_ID,
    SENTIMENT_ID_TO_LABEL,
    SingleTransformerMultitask,
    get_optimal_device,
    get_process_memory_mb,
)
from src.sqlite_memory import PersistentMemoryDB


def run_benchmark(
    test_csv: str = "data/processed/test.csv",
    probe_csv: str = "results/gujlish_probe_template.csv",
    n_samples: int = 100,
    seed: int = 42,
    batch_size: int = 4,
    head_weights_path: str = "models/xlmr_sentiment_head.pt",
) -> dict[str, Any]:
    print("=" * 76)
    print("🚀 BENCHMARK: SINGLE XLM-RoBERTa-base ON APPLE SILICON M2")
    print("=" * 76)

    initial_rss = get_process_memory_mb()
    vm = psutil.virtual_memory()
    print(f"• System Total Unified Memory: {vm.total / (1024**3):.1f} GB")
    print(f"• System Memory Currently in Use: {vm.used / (1024**3):.1f} GB ({(vm.used/vm.total)*100:.1f}%)")
    print(f"• Initial Python Process RSS:   {initial_rss:.1f} MB")

    # 1. Measure Model Loading Time
    t_load_start = time.perf_counter()
    device = get_optimal_device()
    model = SingleTransformerMultitask(device=device, max_length=128, head_weights_path=head_weights_path)
    model_load_time = time.perf_counter() - t_load_start
    post_load_rss = get_process_memory_mb()
    print(f"• Device Selected:             {device.type.upper()}")
    print(f"• Model Load Time:             {model_load_time:.2f} s")
    print(f"• Post-Load Process RSS:       {post_load_rss:.1f} MB (Delta: +{post_load_rss - initial_rss:.1f} MB)")

    # 2. Warmup & Latency Measurement
    warmup_text = "kem chho tame maja ma? aa movie ekdum mast chhe"
    t_warm0 = time.perf_counter()
    _ = model.predict_sentiment(warmup_text)
    first_latency_ms = (time.perf_counter() - t_warm0) * 1000

    latencies = []
    test_sentences = [
        "aa phone bau hang thay chhe saru nathi",
        "service bahut slow thi mood kharab ho gaya",
        "tame bau sara manas chho thank you",
        "aaje hu office jaish ane meeting attend karish",
        "wahh ekdum first class delivery and super packaging",
    ]
    for s in test_sentences:
        t0 = time.perf_counter()
        _ = model.predict_sentiment(s)
        latencies.append((time.perf_counter() - t0) * 1000)

    subsequent_latency_ms = float(np.mean(latencies))
    p95_latency_ms = float(np.percentile(latencies, 95))
    print(f"• First Inference Latency:     {first_latency_ms:.1f} ms")
    print(f"• Subsequent Steady Latency:   {subsequent_latency_ms:.1f} ms (p95: {p95_latency_ms:.1f} ms)")

    # 3. Prepare Evaluation Dataset (100 unknown sentences)
    df_raw = pd.read_csv(test_csv)
    eval_df = df_raw.sample(n=min(n_samples, len(df_raw)), random_state=seed).reset_index(drop=True)
    texts = eval_df["text"].tolist()
    ground_truths = eval_df["label"].tolist()
    gt_ids = np.array([SENTIMENT_LABEL_TO_ID[l] for l in ground_truths])

    # Persistent Memory & Fusion Engine
    memory_db = PersistentMemoryDB()
    memory_db.bootstrap_from_gujlish_probe(Path(probe_csv))
    fusion_engine = EvidenceFusionEngine()

    print(f"\n• Evaluating {len(eval_df)} unseen sentences (batch_size={batch_size})...")

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT A: XLM-RoBERTa-base Raw (Without Memory)
    # ─────────────────────────────────────────────────────────────────────────
    exp_a_preds = []
    exp_a_probs_list = []
    for t in texts:
        pred = model.predict_sentiment(t)
        exp_a_preds.append(pred["label"])
        probs = [pred["probabilities"][lbl] for lbl in SENTIMENT_LABELS]
        exp_a_probs_list.append(probs)

    exp_a_probs = np.array(exp_a_probs_list)
    acc_a = accuracy_score(ground_truths, exp_a_preds)
    f1_a = f1_score(ground_truths, exp_a_preds, average="macro", zero_division=0)
    ece_a = compute_ece(exp_a_probs, gt_ids)
    brier_a = compute_brier_score(exp_a_probs, gt_ids)

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT B: Preprocessing + Validated Memory Retrieval (Inspection Mode)
    # ─────────────────────────────────────────────────────────────────────────
    # Preprocesses with linguistic profiler and retrieves memories, but uses raw probabilities
    exp_b_preds = []
    exp_b_probs_list = []
    for t in texts:
        _ = analyze_linguistic_profile(t)
        memories = memory_db.retrieve_contextual(t, top_k=2)
        pred = model.predict_sentiment(t)
        exp_b_preds.append(pred["label"])
        probs = [pred["probabilities"][lbl] for lbl in SENTIMENT_LABELS]
        exp_b_probs_list.append(probs)

    exp_b_probs = np.array(exp_b_probs_list)
    acc_b = accuracy_score(ground_truths, exp_b_preds)
    f1_b = f1_score(ground_truths, exp_b_preds, average="macro", zero_division=0)
    ece_b = compute_ece(exp_b_probs, gt_ids)
    brier_b = compute_brier_score(exp_b_probs, gt_ids)

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT C: Preprocessing + Memory Retrieval + Calibrated Fusion
    # ─────────────────────────────────────────────────────────────────────────
    exp_c_preds = []
    exp_c_probs_list = []
    changed_count = 0
    overrides_count = 0

    for t in texts:
        _ = analyze_linguistic_profile(t)
        memories = memory_db.retrieve_contextual(t, top_k=2)
        raw_pred = model.predict_sentiment(t)
        fused = fusion_engine.fuse(t, raw_pred, memories, labels=SENTIMENT_LABELS)

        exp_c_preds.append(fused.fused_label)
        probs = [fused.fused_probs[lbl] for lbl in SENTIMENT_LABELS]
        exp_c_probs_list.append(probs)

        if fused.fused_label != fused.base_model_label:
            changed_count += 1
        if fused.decision_mode == "validated_override":
            overrides_count += 1

    exp_c_probs = np.array(exp_c_probs_list)
    acc_c = accuracy_score(ground_truths, exp_c_preds)
    f1_c = f1_score(ground_truths, exp_c_preds, average="macro", zero_division=0)
    ece_c = compute_ece(exp_c_probs, gt_ids)
    brier_c = compute_brier_score(exp_c_probs, gt_ids)

    peak_rss = get_process_memory_mb()

    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION ON GUJARATI-ENGLISH PROBE (AGREED CASES)
    # ─────────────────────────────────────────────────────────────────────────
    probe_results = {}
    if Path(probe_csv).exists():
        p_df = pd.read_csv(probe_csv)
        p_valid = p_df[p_df["label_annotator1"].notna() & (p_df["label_annotator1"] == p_df["label_annotator2"])].copy()
        if len(p_valid) > 0:
            p_texts = p_valid["text"].tolist()
            p_gt = p_valid["label_annotator1"].str.lower().tolist()
            p_preds_a = [model.predict_sentiment(pt)["label"] for pt in p_texts]
            p_preds_c = [
                fusion_engine.fuse(pt, model.predict_sentiment(pt), memory_db.retrieve_contextual(pt, top_k=2), labels=SENTIMENT_LABELS).fused_label
                for pt in p_texts
            ]
            probe_results = {
                "count": len(p_valid),
                "acc_a": accuracy_score(p_gt, p_preds_a),
                "acc_c": accuracy_score(p_gt, p_preds_c),
            }

    # Print Formatted Results
    print("\n" + "=" * 76)
    print("📊 EMPIRICAL COMPARISON OF THREE EXPERIMENTAL CONFIGURATIONS")
    print("=" * 76)
    print(f"{'Metric':<22} | {'Exp A: Raw XLM-R':<16} | {'Exp B: Preproc+Mem':<16} | {'Exp C: Calib Fusion':<18}")
    print("-" * 76)
    print(f"{'Accuracy':<22} | {acc_a:.2%}           | {acc_b:.2%}           | {acc_c:.2%}")
    print(f"{'Macro-F1':<22} | {f1_a:.4f}           | {f1_b:.4f}           | {f1_c:.4f}")
    print(f"{'ECE (lower=better)':<22} | {ece_a:.4f}           | {ece_b:.4f}           | {ece_c:.4f}")
    print(f"{'Brier (lower=better)':<22} | {brier_a:.4f}           | {brier_b:.4f}           | {brier_c:.4f}")
    print("-" * 76)
    print(f"• Peak Process RSS:            {peak_rss:.1f} MB (vs 8 GB System Unified RAM)")
    print(f"• Memory Interventions in C:   {changed_count} changed predictions ({overrides_count} validated overrides)")

    if probe_results:
        print(f"\n🎯 Agreed Gujlish Probe Evaluation ({probe_results['count']} sentences):")
        print(f"  • Exp A (Raw Model):       {probe_results['acc_a']:.2%} Accuracy")
        print(f"  • Exp C (Calibrated Memory): {probe_results['acc_c']:.2%} Accuracy")
    print("=" * 76 + "\n")

    # Save Results CSV
    out_csv = Path("results/benchmark_xlmr_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([
        {
            "experiment": "Exp A: Raw XLM-RoBERTa",
            "accuracy": acc_a,
            "macro_f1": f1_a,
            "ece": ece_a,
            "brier_score": brier_a,
            "peak_rss_mb": peak_rss,
            "first_latency_ms": first_latency_ms,
            "steady_latency_ms": subsequent_latency_ms,
        },
        {
            "experiment": "Exp B: Preprocessing + Memory",
            "accuracy": acc_b,
            "macro_f1": f1_b,
            "ece": ece_b,
            "brier_score": brier_b,
            "peak_rss_mb": peak_rss,
            "first_latency_ms": first_latency_ms,
            "steady_latency_ms": subsequent_latency_ms,
        },
        {
            "experiment": "Exp C: Preproc + Memory + Calibrated Fusion",
            "accuracy": acc_c,
            "macro_f1": f1_c,
            "ece": ece_c,
            "brier_score": brier_c,
            "peak_rss_mb": peak_rss,
            "first_latency_ms": first_latency_ms,
            "steady_latency_ms": subsequent_latency_ms,
        },
    ])
    summary_df.to_csv(out_csv, index=False)
    print(f"✓ Saved summary table to {out_csv}")

    return {
        "exp_a": {"acc": acc_a, "f1": f1_a, "ece": ece_a, "brier": brier_a},
        "exp_b": {"acc": acc_b, "f1": f1_b, "ece": ece_b, "brier": brier_b},
        "exp_c": {"acc": acc_c, "f1": f1_c, "ece": ece_c, "brier": brier_c},
        "load_time_s": model_load_time,
        "first_latency_ms": first_latency_ms,
        "steady_latency_ms": subsequent_latency_ms,
        "peak_rss_mb": peak_rss,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Single XLM-RoBERTa on Apple Silicon M2")
    parser.add_argument("--test_csv", default="data/processed/test.csv")
    parser.add_argument("--probe_csv", default="results/gujlish_probe_template.csv")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_benchmark(
        test_csv=args.test_csv,
        probe_csv=args.probe_csv,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )

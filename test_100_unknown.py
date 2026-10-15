"""Benchmark 100 Unknown Sentences from Held-Out Test Data.

Evaluates Base Model vs Agentic AI (Knowledge Fusion & Memory) on 100 unseen sentences.
Computes accuracy, macro-F1, per-class metrics, and analyzes where memory helped.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.agent_memory import AgentMemoryEngine
from src.predict import available_models, load_predictor


def run_benchmark(
    test_csv_path: str = "data/processed/test.csv",
    n_samples: int = 100,
    seed: int = 42,
    model_name: str = "logistic_regression",
    output_csv: str = "results/unknown_100_benchmark.csv",
) -> dict:
    df = pd.read_csv(test_csv_path)
    # Stratified or deterministic sample of 100 unknown sentences
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)

    engine = AgentMemoryEngine()
    predictor = load_predictor(model_name)

    records = []
    base_preds = []
    agent_preds = []
    ground_truths = sample_df["label"].tolist()

    for idx, row in sample_df.iterrows():
        text = str(row["text"])
        true_lbl = str(row["label"])

        # Agent inference (which includes base model + memory fusion)
        trace = engine.analyze_and_reason(text, predictor)

        base_lbl = trace.base_model_prediction["label"]
        agent_lbl = trace.final_label

        base_preds.append(base_lbl)
        agent_preds.append(agent_lbl)

        records.append({
            "id": idx + 1,
            "text": text,
            "ground_truth": true_lbl,
            "base_prediction": base_lbl,
            "base_confidence": trace.base_model_prediction["confidence"],
            "agent_prediction": agent_lbl,
            "agent_confidence": trace.final_confidence,
            "decision_source": trace.decision_source,
            "recalled_slang_count": len(trace.recalled_slang),
            "recalled_episodes_count": len(trace.recalled_episodes),
            "memory_changed_decision": (base_lbl != agent_lbl),
            "agent_correct": (agent_lbl == true_lbl),
            "base_correct": (base_lbl == true_lbl),
            "agent_explanation": trace.explanation,
        })

    results_df = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    # Compute Metrics
    labels = ["negative", "neutral", "positive"]
    base_acc = accuracy_score(ground_truths, base_preds)
    agent_acc = accuracy_score(ground_truths, agent_preds)
    base_f1 = f1_score(ground_truths, base_preds, average="macro", labels=labels)
    agent_f1 = f1_score(ground_truths, agent_preds, average="macro", labels=labels)

    base_report = classification_report(ground_truths, base_preds, labels=labels, output_dict=True, zero_division=0)
    agent_report = classification_report(ground_truths, agent_preds, labels=labels, output_dict=True, zero_division=0)

    cm_base = confusion_matrix(ground_truths, base_preds, labels=labels)
    cm_agent = confusion_matrix(ground_truths, agent_preds, labels=labels)

    changed_cases = results_df[results_df["memory_changed_decision"]]
    improved = len(results_df[(~results_df["base_correct"]) & results_df["agent_correct"]])
    hurt = len(results_df[results_df["base_correct"] & (~results_df["agent_correct"])])

    return {
        "n_samples": len(sample_df),
        "base_accuracy": base_acc,
        "agent_accuracy": agent_acc,
        "base_macro_f1": base_f1,
        "agent_macro_f1": agent_f1,
        "base_report": base_report,
        "agent_report": agent_report,
        "confusion_matrix_base": cm_base,
        "confusion_matrix_agent": cm_agent,
        "changed_count": len(changed_cases),
        "improved_count": improved,
        "hurt_count": hurt,
        "output_csv": output_csv,
        "results_df": results_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Test 100 unknown sentences on Base vs Agentic AI")
    parser.add_argument("--test_csv", default="data/processed/test.csv")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="logistic_regression")
    parser.add_argument("--out", default="results/unknown_100_benchmark.csv")
    args = parser.parse_args()

    print(f"\n🚀 Running benchmark on {args.n_samples} unseen sentences from '{args.test_csv}'...")
    res = run_benchmark(args.test_csv, args.n_samples, args.seed, args.model, args.out)

    print("\n" + "=" * 70)
    print(f"📊 BENCHMARK RESULTS: {res['n_samples']} UNKNOWN / UNSEEN SENTENCES")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Base Model':<15} | {'Agentic AI (Memory)':<20}")
    print("-" * 70)
    print(f"{'Accuracy':<25} | {res['base_accuracy']:.2%}          | {res['agent_accuracy']:.2%}")
    print(f"{'Macro-F1':<25} | {res['base_macro_f1']:.4f}          | {res['agent_macro_f1']:.4f}")
    print("-" * 70)

    print("\n🎯 Class-wise F1-Scores:")
    for lbl in ["negative", "neutral", "positive"]:
        b_f1 = res["base_report"][lbl]["f1-score"]
        a_f1 = res["agent_report"][lbl]["f1-score"]
        print(f"  • {lbl:<10} Base F1: {b_f1:.4f}  -->  Agent F1: {a_f1:.4f}")

    print(f"\n🔍 Memory Intervention Analysis:")
    print(f"  • Sentences where memory influenced/changed prediction: {res['changed_count']}")
    print(f"  • Cases directly corrected by memory: {res['improved_count']}")
    print(f"  • Full detailed results saved to: {res['output_csv']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

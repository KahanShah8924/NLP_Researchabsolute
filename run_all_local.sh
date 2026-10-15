#!/usr/bin/env bash
# End-to-end local reproduction pipeline for Hinglish sentiment robustness experiments
# Safe for macOS Apple Silicon (8 GB RAM)

set -euo pipefail

echo "=========================================================="
echo "Hinglish Sentiment Robustness: Local Reproduction Pipeline"
echo "Date: $(date)"
echo "=========================================================="

# 1. Environment activation
if [ -d ".venv" ]; then
    echo "[1/6] Activating virtual environment..."
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Run Task 1 setup first."
    exit 1
fi

# 2. Check data availability and convert
echo "[2/6] Verifying SentiMix data and generating splits..."
if [ ! -f "data/processed/train.csv" ]; then
    echo "Converting CoNLL files..."
    python -m src.convert_sentimix --train data/raw/Hinglish_train_14k_split_conll.txt --dev data/raw/Hinglish_dev_3k_split_conll.txt --output data/raw/hinglish_sentiment.csv --overwrite
    python -m src.data_loader --input data/raw/hinglish_sentiment.csv
else
    echo "Processed data splits already exist in data/processed/."
fi

# 3. Reproduce repo baseline
echo "[3/6] Running repo TF-IDF baseline..."
python -m src.baseline

# 4. Perturbation quality and sanity check
echo "[4/6] Running perturbation sanity check and generating sample pairs..."
python -c '
import pandas as pd
from robust_kit.textvar import perturb_sentence, NEGATION_WORDS

# No-op test
assert perturb_sentence("test text", "A", p=0.0) == "test text"
assert perturb_sentence("test text", "B", p=0.0) == "test text"
print("Sanity check p=0: PASSED")

# Sample generation
test_df = pd.read_csv("data/processed/test.csv")
sample_100 = test_df.sample(n=100, random_state=42).reset_index(drop=True)
rows = []
for idx, row in sample_100.iterrows():
    rows.append({
        "pair_id": f"A_{idx:03d}",
        "family": "Family_A@0.3",
        "clean_text": row["text"],
        "perturbed_text": perturb_sentence(row["text"], "A", p=0.3, seed=1000 + idx),
        "label": row["label"],
        "still_readable_and_same_polarity_y_n": ""
    })
    rows.append({
        "pair_id": f"B_{idx:03d}",
        "family": "Family_B@0.3",
        "clean_text": row["text"],
        "perturbed_text": perturb_sentence(row["text"], "B", p=0.3, seed=2000 + idx),
        "label": row["label"],
        "still_readable_and_same_polarity_y_n": ""
    })
pd.DataFrame(rows).to_csv("results/perturbation_samples.csv", index=False)
print("Saved 200 pairs to results/perturbation_samples.csv")
'

# 5. Run Classical Robustness Grid with Paired Bootstrap CIs
echo "[5/6] Running full classical robustness grid (5 methods x 7 test conditions)..."
caffeinate -i python run_classical.py --data_dir data/processed --out results/robustness_classical.csv --seed 42

# 6. Run Error Analysis
echo "[6/6] Running diagnostic error analysis..."
python -m robust_kit.error_analysis

echo "=========================================================="
echo "Reproduction complete! All tables and plots saved in results/"
echo "=========================================================="

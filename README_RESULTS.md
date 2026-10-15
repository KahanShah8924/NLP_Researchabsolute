# Reproducibility Guide: Hinglish Sentiment Robustness Experiments

**Date**: October 11, 2026  
**Hardware Profile**: Apple MacBook Air M2, 8 GB RAM, macOS Sonoma/Sequoia (arm64, no NVIDIA GPU)  
**Execution Environment**: Python 3.13.9 (`arm64`), `.venv`

---

## 1. Verified Software Environment

Exact installed library versions in `.venv`:
```text
numpy==2.1.3
pandas==2.2.3
scikit-learn==1.5.2
scipy==1.18.1
matplotlib==3.9.2
seaborn==0.13.2
torch==2.14.1 (relaxed from 2.5.1 for Python 3.13 macOS arm64 wheel availability)
transformers==4.46.3
datasets==3.1.0
accelerate==1.1.1
emoji==2.14.0
PyYAML==6.0.2
pytest==8.3.3
```

---

## 2. Deterministic Seeds & Splits

- **Repository Data Stratification Seed**: `42` (`config.yaml: seed: 42`)
- **Training Augmentation Seed**: `42`
- **Model Random State**: `42` (`LogisticRegression(random_state=42)`)
- **Evaluation Test Perturbation Seed**: `2026` (Generated once; strictly never reused for training)
- **Bootstrap Resampling Seed**: `42` (1,000 paired resamples per comparison)

---

## 3. End-to-End Local Execution

To run the complete local evaluation pipeline from scratch:

```bash
# Option A: Execute master reproduction script
./run_all_local.sh

# Option B: Run individual steps
source .venv/bin/activate

# Step 1: Data conversion and splitting
python -m src.convert_sentimix --train data/raw/Hinglish_train_14k_split_conll.txt --dev data/raw/Hinglish_dev_3k_split_conll.txt --output data/raw/hinglish_sentiment.csv --overwrite
python -m src.data_loader --input data/raw/hinglish_sentiment.csv

# Step 2: Repo baseline reproduction
python -m src.baseline

# Step 3: Classical robustness experiments & bootstrap CIs
caffeinate -i python run_classical.py --data_dir data/processed --out results/robustness_classical.csv --seed 42

# Step 4: Diagnostic error analysis
python -m robust_kit.error_analysis
```

---

## 4. Mapping of Artifacts to Paper Tables and Figures

Every metric and figure reported in `paper/paper.md` maps directly to verified files in `results/`:

| Paper Element | Source Artifact in `results/` | Description |
| :--- | :--- | :--- |
| **Table 1: Main Macro-F1 Pivot** | `results/pivot_macro_f1.csv` | 5 methods $\times$ 7 test conditions Macro-F1 matrix |
| **Table 2: Drop-from-Clean** | `results/drop_from_clean.csv` | Performance degradation relative to Clean test set |
| **Table 3: Bootstrap CIs** | `results/bootstrap_ci.csv` | Effect sizes, 95% CIs, and statistical significance flags |
| **Figure 1: Grouped Bar Chart** | `results/grouped_bar_macro_f1.png` | Grouped bar chart comparing Macro-F1 per condition |
| **Figure 2: Degradation Curves** | `results/degradation_curves.png` | Line curves over perturbation levels $p \in \{0.0, 0.1, 0.2, 0.3\}$ |
| **Section 7: Degraded Cases** | `results/error_analysis_top30.csv` | Top 30 cases where clean was correct but B@0.3 flipped |
| **Section 7: Word Collisions** | `results/normalization_collisions.csv` | Examples where $w \rightarrow v$ or $ph \rightarrow f$ corrupted words |
| **Section 7: Class-Wise F1** | `results/class_wise_f1.csv` | Class-wise breakdown for negative, neutral, and positive |
| **Task 4: Quality Check** | `results/perturbation_samples.csv` | 200 clean/perturbed pairs for readability auditing |

---

## 5. Google Colab Transformer Instructions (Task 6)

The notebook `notebooks/colab_xlmr_robustness.ipynb` is ready to run on a free GPU (Google Colab T4):
1. Open Google Colab and upload `notebooks/colab_xlmr_robustness.ipynb`.
2. Connect to a GPU runtime (**Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ **T4 GPU**).
3. Run **Cell 1** to verify GPU allocation (`nvidia-smi`).
4. Run **Cell 2** to mount Google Drive (`drive.mount('/content/drive')`).
5. Run **Cell 3** and **Cell 4** to setup code and install dependencies.
6. Run **Cell 5** (1-Epoch Smoke Test) to verify model loading and forward pass without errors.
7. Run **Cell 6** (Full 9 Runs: `baseline`, `norm_light`, `aug_A` $\times$ seeds `1, 2, 3`). Output CSVs are saved directly to Google Drive.
8. Once finished, download the CSV files from Drive into `results/` locally to merge via Task 7.

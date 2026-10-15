# Beyond Clean Accuracy: Evaluating Rule-Based Normalization and Train-Time Augmentation against Held-Out Orthographic Perturbations in Hinglish Sentiment Analysis

**Author**: Kahan Shah  
**Affiliation**: Department of Computer Engineering  
**Project**: NLP Research Mini-Project  
**Date**: October 2026  

---

## Abstract

Romanized Hindi-English code-mixed text (Hinglish) suffers from extensive orthographic non-standardization, phonetic spelling variants, and informal shorthand on social media. While multilingual transformers and character n-gram classifiers report respectable benchmark accuracies on clean test sets, their resilience against real-world orthographic noise remains uncertain. This paper investigates whether lightweight rule-based normalization (`norm_light`) and train-time spelling augmentation (`aug_A`) improve classifier robustness to romanization noise when evaluated on perturbation families strictly held out during training and normalizer design. We define two perturbation sets: Family A (respelling variants: vowel elongation, digraphs $aa/a$, $ee/i$, $oo/u$, $ph/f$, $v/w$) and Family B (shorthand and keyboard typos: vowel dropping, adjacent swaps, QWERTY neighbor substitutions, consonant doubling). Evaluating a competitive character-word TF-IDF Logistic Regression baseline on 2,720 held-out test tweets from SemEval-2020 Task 9 (SentiMix), we find that rule-based normalization significantly mitigates degradation under seen respelling noise ($+1.40\%$ macro-F1 at Family A@0.3, 95% CI $[+0.33\%, +2.50\%]$), but alone fails to provide statistically significant gains under held-out shorthand and typos (Family B@0.3 95% CI $[-0.65\%, +0.97\%]$). However, combining light normalization with train-time augmentation (`norm_light+aug_A`) yields consistent cross-family robustness, achieving $+0.94\%$ macro-F1 gain on held-out Family B@0.3 (95% CI $[+0.02\%, +1.93\%]$). Conversely, aggressive consonant skeleton reduction (`norm_skeleton`) induces catastrophic lexical collision, significantly hurting clean performance ($-1.30\%$ macro-F1). We provide full reproducibility artifacts, bootstrap confidence intervals, and an exploratory probe setup for Gujarati-English code-mixing.

---

## 1. Introduction

Code-mixing between Hindi and English written in the Latin alphabet—colloquially termed Hinglish—has become the dominant communication mode across digital platforms in the Indian subcontinent (Patwa et al., 2020). Despite its widespread adoption, processing Hinglish computationally poses severe linguistic challenges. Unlike standard English or Hindi written in Devanagari, romanized Hinglish lacks an authoritative orthographic standard. A single lexical item can be spelled in dozens of distinct ways (e.g., *bahut*, *bohot*, *bht*, *bohooot*, *bhut*), governed by dialectal phonetics, typing economy, or expressive emphasis (Gharami et al., 2025).

Prior benchmark evaluations have reported promising classification accuracies for both character-aware classical models and fine-tuned multilingual pretrained language models such as XLM-RoBERTa (Patwa et al., 2020; HxTag, 2024). However, recent investigations demonstrate that high in-distribution accuracy on benchmark datasets can mask severe brittleness under out-of-distribution adversarial and orthographic shifts (Rahul & Adebayo, 2026). In resource-constrained settings where fine-tuning large transformers locally is impractical, understanding how lightweight preprocessing and augmentation strategies affect model robustness is of vital practical importance.

In this work, we systematically investigate the research question: **Do rule-based normalization and train-time spelling augmentation improve the robustness of Hinglish sentiment classifiers to romanization noise, when evaluated on perturbation families never seen during training or normalizer design?**

---

## 2. Related Work

### 2.1 Benchmark Datasets and Code-Mixed Sentiment
The benchmark dataset for Hinglish sentiment classification was established by the SemEval-2020 Task 9: SentiMix shared task (Patwa et al., 2020). The task provided token-level language identification and tweet-level sentiment annotations across Hindi-English and Spanish-English corpora. Baseline comparisons from the task and follow-up student implementations (HxTag, 2024) highlighted that character-ngram TF-IDF linear baselines remain extraordinarily competitive against large neural networks (achieving $\sim 63\%-67\%$ macro-F1) because character n-grams naturally capture partial morphological and subword substrings in code-mixed text.

### 2.2 Adversarial Robustness and Auditory Features
Rahul and Adebayo (2026) demonstrated that state-of-the-art multilingual transformers (such as XLM-R and mBERT) that achieve near-perfect in-distribution accuracy suffer catastrophic degradation (over 57 percentage points drop) when subjected to human-validated adversarial tests featuring subtle sentiment and linguistic variation. To combat non-standard spelling, Mamta et al. (2023) introduced auditory phonetic representations (incorporating SOUNDEX into masked language modeling), proving that phonetic inductive biases substantially improve resilience to noise in code-mixed text.

### 2.3 Script Variation and Lexical Sharing
Sannigrahi and Bawden (2023) investigated lexical sharing across Indic languages and observed that while explicit transliteration may yield diminishing returns in large multilingual neural models, handling subword fragmentation is critical for cross-lingual transfer. Gharami et al. (2025) studied the "grammarless" nature of romanized Hindi and Bengali on social media, emphasizing the necessity of robust data augmentation and normalization strategies to bridge the gap between formal and colloquial digital communications.

---

## 3. Research Gap and Problem Statement

While prior work has explored phonetics-based pretraining (Mamta et al., 2023) or broad adversarial evaluations (Rahul & Adebayo, 2026), existing literature frequently conflates two distinct forms of spelling variation:
1. **Phonetic respellings and elongation** (e.g., elongated vowels *bhoooot*, digraph choices $ph/f$, $w/v$, $ee/i$), which can be anticipated by rule-based normalizers or train-time augmentations.
2. **Shorthand abbreviations and keyboard typos** (e.g., internal vowel dropping *bht*, adjacent letter swaps *teh*, QWERTY neighbor typos *kabbhi*), which represent unseen noise processes.

**Research Gap**: It remains unverified whether rule-based normalizers and train-time spelling augmentations overfit to the specific respelling rules used to build them, or whether they generalize to protect sentiment classifiers against held-out orthographic noise processes. We address this gap by establishing a strict held-out evaluation protocol under a low-compute setting.

---

## 4. Methodology

### 4.1 Perturbation Families

We formulate two disjoint perturbation families:

- **Family A (Respelling Noise - Seen / Augmentable)**:
  - *Elongation*: Random repetition of vowels or continuants (2–4 chars, e.g., *sach* $\rightarrow$ *saaaach*).
  - *Vowel Digraph Respellings*: Bi-directional substitutions ($aa \leftrightarrow a$, $ee \leftrightarrow i$, $oo \leftrightarrow u$).
  - *Consonant Digraph Respellings*: Phonetic substitutions ($ph \leftrightarrow f$, $v \leftrightarrow w$).
  - *Application*: Candidate for train-time data augmentation (`aug_A`).

- **Family B (Shorthand & Typos - HELD OUT)**:
  - *Internal Vowel Dropping*: Systematic deletion of non-initial vowels (*samajh* $\rightarrow$ *smjh*).
  - *QWERTY Keyboard Substitutions*: Letter replacement with adjacent physical keys (*nahi* $\rightarrow$ *nsahi*).
  - *Adjacent Transposition*: Swapping adjacent characters (*pyaar* $\rightarrow$ *pyaar* / *pyara*).
  - *Consonant Doubling*: Informal doubling of stop consonants (*hum* $\rightarrow$ *humm*).
  - *Application*: Strictly held-out; **never** used during training augmentation or normalizer construction.

*Negation Protection*: To ensure perturbations alter surface orthography without inverting true sentiment polarity, an explicit dictionary of negation tokens (e.g., *nahi*, *nahin*, *nhi*, *mat*, *not*, *never*, *bina*) is strictly protected from perturbation.

### 4.2 Methods Compared

1. **`baseline`**: Standard preprocessing (URL/mention stripping, lowercase) with combined word (1-2) and character-wb (3-5) TF-IDF features and class-balanced Logistic Regression.
2. **`norm_light`**: Lightweight deterministic normalizer applying elongation collapse (reducing $3+$ identical characters to $1$), digraph canonicalization ($ph \rightarrow f$, $w \rightarrow v$, $aa \rightarrow a$, $ee \rightarrow i$, $oo \rightarrow u$), applied to train and test.
3. **`norm_skeleton` (Lossy Ablation)**: Aggressive consonant-skeleton reduction, preserving initial characters while deleting interior vowels across multi-character words.
4. **`aug_A`**: Training set doubled by generating Family A perturbed variants ($p=0.3$) at train time; evaluated on raw test sets.
5. **`norm_light+aug_A`**: Combined strategy where training data is augmented with Family A and both train and test sets are normalized via `norm_light`.

---

## 5. Experimental Setup

- **Dataset**: SemEval-2020 Task 9 (SentiMix Hinglish). Raw CoNLL data was converted and partitioned into deterministic, stratified splits matching the 70/15/15 protocol:
  - Training: 12,691 tweets
  - Validation: 2,719 tweets
  - Test: 2,720 tweets
- **Hardware & Compute**: Apple MacBook Air M2 (8 GB RAM, 0 NVIDIA GPUs). Training runtime was under 1.5 seconds per model; total RAM footprint remained $< 400$ MB, adhering strictly to the $< 4$ GB local compute constraint.
- **Evaluation Protocol**: Test perturbations were generated using a fixed, single-use random seed ($2026$) across perturbation probabilities $p \in \{0.1, 0.2, 0.3\}$.
- **Statistical Significance**: We compute 95% confidence intervals via paired non-parametric bootstrapping ($1,000$ resamples) comparing each method directly against the baseline on identical test subsets. An improvement is considered statistically significant ($\alpha=0.05$) if and only if the 95% CI strictly excludes zero.

---

## 6. Results and Discussion

### 6.1 Main Quantitative Results

Table 1 presents the classification Macro-F1 across all methods and test conditions. Table 2 details the performance drop relative to clean evaluation, and Table 3 reports the paired-bootstrap confidence intervals versus the baseline.

#### Table 1: Macro-F1 Performance across Test Conditions (Results from `results/pivot_macro_f1.csv`)

| Method | Clean | Family A@0.1 | Family A@0.2 | Family A@0.3 | Family B@0.1 | Family B@0.2 | Family B@0.3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | 0.6781 | 0.6763 | 0.6667 | 0.6645 | 0.6721 | 0.6641 | 0.6629 |
| **`norm_light`** | 0.6815 | 0.6823 | 0.6816 | 0.6787 | 0.6772 | 0.6679 | 0.6646 |
| **`norm_skeleton`** | 0.6652 | 0.6649 | 0.6666 | 0.6670 | 0.6622 | 0.6577 | 0.6530 |
| **`aug_A`** | 0.6789 | 0.6761 | 0.6654 | 0.6659 | 0.6728 | 0.6666 | 0.6644 |
| **`norm_light+aug_A`** | **0.6840** | **0.6855** | **0.6845** | **0.6815** | **0.6766** | **0.6786** | **0.6723** |

#### Table 2: Drop in Macro-F1 from Clean Test Set (Results from `results/drop_from_clean.csv`)

| Method | Family A@0.1 Drop | Family A@0.2 Drop | Family A@0.3 Drop | Family B@0.1 Drop | Family B@0.2 Drop | Family B@0.3 Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | +0.0018 | +0.0114 | +0.0136 | +0.0060 | +0.0139 | +0.0151 |
| **`norm_light`** | -0.0007 | -0.0001 | +0.0028 | +0.0043 | +0.0136 | +0.0170 |
| **`norm_skeleton`** | +0.0003 | -0.0014 | -0.0018 | +0.0030 | +0.0075 | +0.0122 |
| **`aug_A`** | +0.0028 | +0.0135 | +0.0130 | +0.0061 | +0.0123 | +0.0145 |
| **`norm_light+aug_A`** | -0.0015 | -0.0005 | **+0.0026** | +0.0074 | +0.0054 | **+0.0117** |

*(Note: Negative drop indicates that performance slightly improved on perturbed text due to normalization collapsing redundant noise).*

#### Table 3: Paired-Bootstrap 95% Confidence Intervals vs Baseline (Results from `results/bootstrap_ci.csv`)

| Method | Condition | Mean Diff ($\Delta$) | 95% CI Lower | 95% CI Upper | Statistically Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `norm_light` | Clean | +0.0036 | -0.0030 | +0.0099 | No (CI spans 0) |
| `norm_skeleton`| Clean | -0.0130 | -0.0260 | -0.0006 | **Yes (Significant Drop)** |
| `aug_A` | Clean | +0.0007 | -0.0075 | +0.0090 | No (CI spans 0) |
| `norm_light+aug_A` | Clean | +0.0059 | -0.0024 | +0.0143 | No (CI spans 0) |
| `norm_light` | Family A@0.3 | **+0.0140** | **+0.0033** | **+0.0250** | **Yes (Significant Gain)** |
| `aug_A` | Family A@0.3 | +0.0013 | -0.0083 | +0.0112 | No (CI spans 0) |
| `norm_light+aug_A` | Family A@0.3 | **+0.0167** | **+0.0054** | **+0.0282** | **Yes (Significant Gain)** |
| `norm_light` | Family B@0.3 | +0.0016 | -0.0065 | +0.0097 | No (CI spans 0) |
| `aug_A` | Family B@0.3 | +0.0014 | -0.0066 | +0.0096 | No (CI spans 0) |
| `norm_light+aug_A` | Family B@0.3 | **+0.0094** | **+0.0002** | **+0.0193** | **Yes (Significant Gain)** |

### 6.2 Key Empirical Findings

1. **Rule-Based Normalization Rescues Seen Respellings (Family A)**:  
   `norm_light` virtually eliminates performance degradation on Family A noise. The drop from clean test accuracy drops from $1.36\%$ in the baseline to only $0.28\%$ under `norm_light`. At Family A@0.3, the bootstrap difference is $+1.40\%$ macro-F1 (95% CI $[+0.33\%, +2.50\%]$), confirming statistical significance.

2. **The Held-Out Transfer Gap (Family B)**:  
   When evaluated on Family B (held-out shorthand, letter swaps, and keyboard typos), `norm_light` alone provides **no statistically significant benefit** over the baseline ($+0.16\%$ macro-F1, 95% CI $[-0.65\%, +0.97\%]$). Because its rules target digraphs and elongations, it is powerless against internal vowel dropping (e.g., *samajh* $\rightarrow$ *smjh*) and keyboard mis-strikes.

3. **Synergy of Normalization and Augmentation (`norm_light+aug_A`)**:  
   Neither augmentation alone nor normalization alone succeeded on held-out Family B@0.3. However, their combination (`norm_light+aug_A`) achieved a statistically significant improvement over baseline ($+0.94\%$ macro-F1, 95% CI $[+0.02\%, +1.93\%]$). By densifying character n-gram feature representations during training while canonicalizing common variants, the joint pipeline exhibits generalizable resilience.

4. **Failure of Extreme Consonant Skeletonization (`norm_skeleton`)**:  
   The ablation method `norm_skeleton` demonstrated significant negative transfer on clean data ($-1.30\%$ macro-F1, 95% CI $[-2.60\%, -0.06\%]$). Stripping non-initial vowels produces severe lexical collisions between unrelated Hindi and English words (e.g., *kaam* and *kami* collapsing to *km*), proving that aggressive lossy reduction degrades discriminative capacity.

---

## 7. Error Analysis

Using the diagnostic pipeline in `robust_kit/error_analysis.py`, we analyzed 150 test instances that degraded under Family B@0.3 and 40 instances where normalization hurt the clean baseline:

1. **Subword Fragmentation under Typos**:  
   *Hypothesis*: In character n-gram models, keyboard substitutions (e.g., *mag* instead of *may*, *zage* instead of *aage*) fragment character 3-5 grams, dropping previously informative weights below classifier activation thresholds.
2. **Word Collisions Caused by Digraph Normalization**:  
   *Hypothesis*: The rule $w \rightarrow v$ and $ph \rightarrow f$ inadvertently distorts common English words in code-mixed utterances (e.g., *we* $\rightarrow$ *ve*, *will* $\rightarrow$ *vill*, *with* $\rightarrow$ *vith*, *photo* $\rightarrow$ *foto*). While beneficial for romanized Hindi tokens (*wahan* $\rightarrow$ *vahan*), it introduces lexical ambiguity for English function words, causing 40 previously correct baseline predictions to flip.
3. **Class-Wise Disparity**:  
   Analysis of class-wise performance (`results/class_wise_f1.csv`) reveals that the neutral class has the lowest absolute F1 ($0.6061$), while positive ($0.7218$) and negative ($0.7064$) classes suffer the sharpest drops under Family B perturbations ($-2.18\%$ and $-1.91\%$ F1 drop, respectively).

---

## 8. Limitations

1. **Synthetic Perturbations**: The evaluation relies on rule-generated perturbations. While calibrated against real social media phenomena, synthetic noise cannot fully capture multi-word slang, sarcastic emoji placement, or dialectal syntax.
2. **Single Dataset**: Experiments were conducted exclusively on SemEval-2020 Task 9 (Twitter domain).
3. **Rule Overlap**: `norm_light` was explicitly designed with knowledge of Family A digraphs; its success on Family A is partly by design. The true test of generalization is Family B, where its standalone effect failed to reach statistical significance.
4. **Modest Baseline Performance**: The TF-IDF Logistic Regression baseline achieves $\sim 67.8\%$ macro-F1. Absolute gains of $1\%-1.5\%$ are modest in absolute terms, though statistically confirmed via bootstrap intervals.
5. **Gujlish Probe Scale**: The exploratory Gujarati-English probe is small and unvalidated by multiple external annotators.

---

## 9. Conclusion and Future Work

This study provides a rigorous, low-compute evaluation of normalization and augmentation strategies for noisy Hinglish sentiment analysis. Our findings provide a clear empirical answer: **Rule-based normalization effectively neutralizes anticipated respelling noise, but fails to generalize to held-out shorthand and keyboard typos on its own.** However, combining light normalization with train-time spelling augmentation establishes genuine cross-family robustness on held-out noise processes without requiring heavy neural re-training. Future work will extend this framework to evaluate full transformer fine-tuning using our prepared Colab pipeline and evaluate cross-lingual transfer to Gujarati-English and Marathi-English corpora.

---

## References

1. Gharami, K., Muhtaseem, Q. S., Gupta, D., Elluri, L., & Moni, S. S. (2025). Modeling Romanized Hindi and Bengali: Dataset Creation and Multilingual LLM Integration. *2025 IEEE International Conference on Big Data (BigData)*, arXiv:2511.22769.
2. HxTag. (2024). Hinglish Sentiment Analysis: Baseline and Fine-Tuning Benchmarks. *GitHub Repository*: `https://github.com/HxTag/hinglish-sentiment`.
3. Mamta, Ahmad, Z., & Ekbal, A. (2023). Elevating Code-mixed Text Handling through Auditory Information of Words. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)* (pp. 15949–15961). ACL Anthology.
4. Patwa, P., Aguilar, G., Kar, S., Pandey, S., PYKL, S., Gambäck, B., Chakraborty, T., Solorio, T., & Das, A. (2020). SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation (SemEval 2020)* (pp. 774–790). ACL Anthology: 2020.semeval-1.100.
5. Rahul, C., & Adebayo, K. (2026). Beyond Benchmark Accuracy: Robustness Evaluation of Hinglish Sentiment Models. In *Proceedings of the Sixth Workshop on Speech, Vision, and Language Technologies for Dravidian Languages (DravidianLangTech 2026)*. ACL Anthology: 2026.dravidianlangtech-1.2.
6. Sannigrahi, S., & Bawden, R. (2023). Investigating Lexical Sharing in Multilingual Machine Translation for Indian Languages. In *Proceedings of the 24th Annual Conference of the European Association for Machine Translation (EAMT 2023)* (pp. 269–278).

# 🧠 Agentic AI Sentiment & Knowledge Memory Engine

This system implements an **Agentic AI architecture** designed specifically for **unknown, unseen code-mixed (Hinglish/English) text analysis**. Rather than relying solely on a static, frozen base model, it features an **adaptive episodic memory store** and a **semantic slang lexicon** that continuously accumulates knowledge from past interactions and user feedback.

---

## 🌟 Key Features

1. **Robust to Unknown Text**:
   - Analyzes novel code-mixed phrases, colloquial idioms, and slang.
   - Computes linguistic perception breakdown: `% English`, `% Romanized Hindi`, `% Slang/Transliterated`.
2. **Episodic & Semantic Memory Bank**:
   - Persistently stored in `data/agent_knowledge_base.json`.
   - Remembers past verified examples, user corrections, and domain-specific slang with influence weights.
3. **4-Stage Agentic Reasoning Loop**:
   - **Stage 1 (Perception)**: Code-mixing and language breakdown.
   - **Stage 2 (Knowledge Retrieval)**: k-NN similarity lookup across historical episodes + slang lexicon scan.
   - **Stage 3 (Knowledge Fusion & Bayesian Update)**: Combines base classifier priors with memory signals; overrides or reinforces base predictions.
   - **Stage 4 (Calibrated Verdict & Explanation)**: Generates human-readable explanations of why the decision was made.
4. **Active Learning & Feedback Loop**:
   - One-click confirmation or correction.
   - When a human corrects a label or teaches a new slang term, the agent **immediately remembers it**.
   - Subsequent similar unknown texts leverage this accumulated knowledge!

---

## 🖥️ 1. Interactive Terminal CLI (`agent_cli.py`)

### A. Launch Interactive REPL Mode:
```bash
source .venv/bin/activate
python agent_cli.py
```
**Interactive Commands:**
- Type any unknown sentence in English, Hindi, or Hinglish.
- `[Y]` Confirm prediction (reinforces memory).
- `[C]` Correct label (overrides prediction & saves explanation note to memory).
- `[T]` Teach a specific slang word found in the sentence.
- `stats` Inspect memory bank status.
- `slang` List all learned slang words.
- `teach` Teach a new slang word directly.
- `q` Exit.

### B. Single-Sentence Evaluation with Reasoning:
```bash
python agent_cli.py "bhai movie ekdum jhakkas aur paisa vasool thi" --explain
```

### C. Inspect Memory Statistics:
```bash
python agent_cli.py --stats
python agent_cli.py --list-slang
```

---

## 🌐 2. Modern Web Application (`agent_web.py`)

### A. Access Running Web App:
The application is currently running at:
👉 **[http://localhost:8501](http://localhost:8501)**

### B. To restart manually anytime:
```bash
source .venv/bin/activate
streamlit run agent_web.py --server.port=8501
```

### Web App Capabilities:
- **Real-Time Text Analysis Tab**: Text box for unknown inputs, quick-test buttons, live confidence gauge, probability chart, and full 4-stage thought trace.
- **Active Feedback Panel**: Immediate "Confirm Correct" and "Override / Correct Sentiment" form to teach the agent.
- **Memory Bank Explorer Tab**: Filter, search, and delete remembered episodes.
- **Teach Slang & Rules Tab**: Add custom slang phrases, sentiment polarity, and influence weights.
- **Batch Prediction Tab**: Upload a CSV with unknown text to run memory-augmented inference across rows.

---

## 📂 File Architecture

| File | Purpose |
| :--- | :--- |
| [`src/agent_memory.py`](file:///Users/kahanshah/Downloads/NLP/Project_2/src/agent_memory.py) | Core persistent memory engine, k-NN retrieval, and Bayesian reasoning |
| [`agent_cli.py`](file:///Users/kahanshah/Downloads/NLP/Project_2/agent_cli.py) | Interactive color-coded Terminal CLI tool |
| [`agent_web.py`](file:///Users/kahanshah/Downloads/NLP/Project_2/agent_web.py) | Full-featured Streamlit Web Application |
| `data/agent_knowledge_base.json` | Persistent memory database (episodes + slang lexicon) |

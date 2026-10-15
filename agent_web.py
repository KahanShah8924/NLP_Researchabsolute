"""Streamlit Web Application: Agentic AI Code-Mixed Sentiment Engine with Memory.

Features:
- Real-time unknown text sentiment analysis (Positive / Negative / Neutral)
- Agentic 4-stage thought trace (Perception -> Memory Retrieval -> Inference -> Synthesis)
- Interactive Active Learning feedback loop (Confirm, Correct, Teach new slang)
- Memory Bank Explorer (Inspect, filter, search, and manage remembered knowledge)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_memory import AgentMemoryEngine, AgentThoughtTrace
from src.predict import available_models, load_predictor


# Page configuration
st.set_page_config(
    page_title="Agentic AI - Sentiment & Memory Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .agent-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-positive {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
    }
    .badge-negative {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
    }
    .badge-neutral {
        background: rgba(234, 179, 8, 0.2);
        color: #facc15;
        border: 1px solid #eab308;
    }
    .thought-box {
        background: #1e222d;
        border-left: 4px solid #38bdf8;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 10px;
        margin-bottom: 12px;
    }
    .memory-card {
        background: #181c24;
        border: 1px solid #2b3240;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_memory_engine() -> AgentMemoryEngine:
    return AgentMemoryEngine()


@st.cache_resource(show_spinner="Loading trained ML model…")
def get_predictor(model_name: str):
    return load_predictor(model_name)


# Initialize Engine
memory_engine = get_memory_engine()
models_available = available_models()

# Sidebar
with st.sidebar:
    st.title("🧠 Agent Controls")
    st.markdown("Persistent knowledge & adaptive reasoning settings.")

    if not models_available:
        st.error("No model checkpoints found in `models/`.")
        st.stop()

    selected_model = st.selectbox(
        "Base Classifier",
        list(models_available.keys()),
        index=0,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    predictor = get_predictor(selected_model)

    st.divider()
    st.subheader("📊 Memory Bank Status")
    stats = memory_engine.get_stats()
    c1, c2 = st.columns(2)
    c1.metric("Episodes", stats["total_episodes"])
    c2.metric("Slang Words", stats["total_slang_entries"])
    st.metric("Human Corrections", stats["user_corrected"])
    st.caption(f"💾 Storage: `{stats['storage_file']}`")

    st.divider()
    if st.button("🔄 Reload Memory from Disk", use_container_width=True):
        memory_engine.load()
        st.success("Memory refreshed from disk!")
        st.rerun()

    if st.button("⚡ Reset to Factory Defaults", use_container_width=True):
        memory_engine.episodes = []
        memory_engine.slang_lexicon = {}
        memory_engine.bootstrap_default_knowledge()
        memory_engine.save()
        st.warning("Reset to initial factory knowledge.")
        st.rerun()


# Main View Header
st.markdown(
    """
    <div class="agent-header">
        <h1 style="margin:0; font-size: 2rem; color: #f8fafc;">🧠 Agentic AI Sentiment Engine</h1>
        <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
            Continuous Learning & Memory for Unknown Hinglish / Code-Mixed Text. Recalls past user corrections & vernacular slang.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab_analyze, tab_memory, tab_teach, tab_batch = st.tabs([
    "🔍 Real-Time Text Analysis",
    "📚 Memory Bank Explorer",
    "📖 Teach Slang & Rules",
    "⚡ Batch Prediction with Memory",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Real-Time Text Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab_analyze:
    col_input, col_examples = st.columns([3, 2])

    with col_examples:
        st.markdown("**Try Unknown / Slang Examples:**")
        sample_prompts = [
            "Movie ekdum mast thi bhai, paisa vasool!",
            "Delivery bahut slow thi, mood kharab ho gaya.",
            "Bhai camera quality zabardast hai but battery normal hai.",
            "Ye update bilkul ghanta kaam nahi karta, chindi app.",
            "Faadu performance tha actor ka, loved it ❤️",
            "Product theek thaak hai, kaam chalau experience.",
        ]
        chosen_sample = None
        for s in sample_prompts:
            if st.button(f"👉 {s}", key=f"btn_{s[:20]}", use_container_width=True):
                chosen_sample = s

    with col_input:
        default_val = chosen_sample or st.session_state.get("current_text", "Movie ekdum mast thi bhai, paisa vasool!")
        user_text = st.text_area(
            "Enter any unseen/unknown sentence (Hinglish, Hindi, English, Slang):",
            value=default_val,
            height=120,
            placeholder="e.g. Phone ka look toh jhakkas hai but sound bilkul bekar hai...",
        )
        analyze_clicked = st.button("🚀 Analyze with Agentic AI", type="primary", use_container_width=True)

    if user_text:
        # Run inference
        trace = memory_engine.analyze_and_reason(user_text, predictor)

        st.markdown("---")
        st.subheader("🎯 Agent Verdict & Reasoning")

        # Top verdict cards
        v_col1, v_col2, v_col3 = st.columns([2, 2, 3])

        badge_class = f"badge-{trace.final_label}"
        with v_col1:
            st.markdown(
                f"""
                <div style="background: #1e222d; padding: 18px; border-radius: 10px; border: 1px solid #334155; text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 6px;">FINAL PREDICTION</div>
                    <span class="metric-badge {badge_class}" style="font-size: 1.4rem; padding: 6px 18px;">
                        {trace.final_label.upper()}
                    </span>
                    <div style="margin-top: 12px; font-weight: 600; color: #38bdf8; font-size: 1.1rem;">
                        {trace.final_confidence:.1%} Confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with v_col2:
            st.markdown(
                f"""
                <div style="background: #1e222d; padding: 18px; border-radius: 10px; border: 1px solid #334155; text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 6px;">BASE MODEL VS AGENT</div>
                    <div style="font-size: 1.1rem; color: #f1f5f9; margin-top: 4px;">
                        Base: <b>{trace.base_model_prediction['label'].title()}</b> ({trace.base_model_prediction['confidence']:.1%})
                    </div>
                    <div style="font-size: 0.95rem; color: #a78bfa; margin-top: 8px;">
                        Signal: <b>{trace.decision_source.replace('_', ' ').title()}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with v_col3:
            # Probability chart
            probs_df = pd.DataFrame(
                {
                    "Sentiment": list(trace.knowledge_adjusted_probabilities.keys()),
                    "Adjusted Probability": list(trace.knowledge_adjusted_probabilities.values()),
                }
            ).set_index("Sentiment")
            st.bar_chart(probs_df, height=140)

        # 4-Stage Agent Thought Trace
        with st.expander("🧠 Inspect Agentic 4-Stage Thought Process", expanded=True):
            st.markdown(f"**Explanation:** {trace.explanation}")

            t_col1, t_col2 = st.columns(2)

            with t_col1:
                st.markdown("#### 🔍 1. Linguistic Perception")
                perc = trace.perception
                st.write(
                    f"- **Total Words:** {perc.get('total_words', 0)}\n"
                    f"- **English Ratio:** `{perc.get('english_ratio', 0.0):.0%}`\n"
                    f"- **Romanized Hindi Ratio:** `{perc.get('hindi_ratio', 0.0):.0%}`\n"
                    f"- **Slang / Transliterated / Other:** `{perc.get('mixed_or_slang_ratio', 0.0):.0%}`"
                )

                st.markdown("#### 📚 2. Past Knowledge Recall")
                if trace.recalled_slang:
                    st.write("**Recognized Slang Terms:**")
                    for sl in trace.recalled_slang:
                        st.markdown(
                            f"- **`{sl['phrase']}`** → `{sl['label'].upper()}` (weight: {sl['weight']}) *({sl['notes']})*"
                        )
                else:
                    st.caption("No matching words in the semantic slang lexicon.")

                if trace.recalled_episodes:
                    st.write("**Top Matching Past Episodes in Memory:**")
                    for ep in trace.recalled_episodes:
                        src_icon = "🏷️ (User Corrected)" if ep["source"] == "user_corrected" else "📌 (Verified)"
                        st.markdown(
                            f"- {src_icon} **`{ep['raw_text']}`** → **{ep['label'].upper()}** *(Sim: {ep['similarity']:.0%})*"
                        )
                        if ep.get("notes"):
                            st.caption(f"  Note: {ep['notes']}")
                else:
                    st.caption("No strongly similar past sentences. Processing as novel unknown phrasing.")

            with t_col2:
                st.markdown("#### ⚖️ 3. Knowledge Fusion & Bayesian Update")
                st.write(
                    f"- **Base Model Raw Probs:** `{trace.base_model_prediction['probabilities']}`\n"
                    f"- **Knowledge Adjusted Probs:** `{trace.knowledge_adjusted_probabilities}`\n"
                    f"- **Decision Logic Mode:** `{trace.decision_source}`"
                )

                st.markdown("#### 📝 4. Text Normalization")
                st.write(
                    f"- **Raw Input:** `{trace.input_text}`\n"
                    f"- **Cleaned Representation:** `{trace.cleaned_text}`"
                )

        # Active Feedback / Continuous Learning Box
        st.markdown("### 🔄 Active Feedback Loop (Teach the Agent)")
        st.info(
            "Is the agent's prediction right or wrong? By giving feedback below, you store this unknown text in the agent's persistent memory so future predictions remember your guidance!"
        )

        fb_col1, fb_col2 = st.columns([1, 2])
        with fb_col1:
            if st.button("✅ Confirm Correct", type="secondary", use_container_width=True):
                memory_engine.remember_feedback(
                    user_text,
                    trace.final_label,
                    source="user_confirmed",
                    notes="Verified by human via Web UI",
                )
                st.success(f"✓ Saved! Confirmed as '{trace.final_label}'. Memory updated!")
                st.rerun()

        with fb_col2:
            with st.form("correction_form"):
                st.markdown("**Override / Correct Sentiment:**")
                correct_label = st.selectbox(
                    "Select correct label:",
                    ["positive", "negative", "neutral"],
                    index=["positive", "negative", "neutral"].index(
                        trace.final_label if trace.final_label in ["positive", "negative", "neutral"] else "neutral"
                    ),
                )
                reason_note = st.text_input(
                    "Why? (e.g. 'sarcasm', 'colloquial slang meaning')",
                    placeholder="Enter reason or context note",
                )
                submit_correction = st.form_submit_button("💾 Teach Agent & Update Memory", use_container_width=True)

                if submit_correction:
                    memory_engine.remember_feedback(
                        user_text,
                        correct_label,
                        source="user_corrected",
                        notes=reason_note or f"Corrected from {trace.final_label} to {correct_label}",
                    )
                    st.success(f"✓ Memory updated! Future unknown texts similar to this will recall '{correct_label}'.")
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Memory Bank Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab_memory:
    st.subheader("📚 Persistent Memory Bank")
    st.caption("Inspect all learned episodes and verified cases saved in the agent's persistent database.")

    mem_filter = st.radio(
        "Filter Episodes by Source:",
        ["All", "user_corrected", "user_confirmed", "bootstrap"],
        horizontal=True,
    )

    filtered_episodes = memory_engine.episodes
    if mem_filter != "All":
        filtered_episodes = [ep for ep in filtered_episodes if ep.source == mem_filter]

    st.markdown(f"**Showing {len(filtered_episodes)} memory episode(s):**")

    if filtered_episodes:
        data_table = []
        for ep in reversed(filtered_episodes):
            data_table.append(
                {
                    "ID": ep.id,
                    "Raw Text": ep.raw_text,
                    "Sentiment": ep.label.upper(),
                    "Source": ep.source,
                    "Notes": ep.notes,
                    "Timestamp": ep.timestamp[:19].replace("T", " "),
                }
            )
        st.dataframe(pd.DataFrame(data_table), use_container_width=True, height=350)

        # Episode deletion
        with st.expander("🗑️ Delete an Episode from Memory"):
            ep_to_delete = st.selectbox(
                "Select Episode ID to remove:",
                [ep.id for ep in filtered_episodes],
                format_func=lambda i: f"ID: {i} - {[e.raw_text for e in filtered_episodes if e.id == i][0][:40]}...",
            )
            if st.button("Delete Episode", type="primary"):
                if memory_engine.delete_episode(ep_to_delete):
                    st.success(f"Deleted episode {ep_to_delete}")
                    st.rerun()
    else:
        st.info("No episodes found under this filter.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Teach Slang & Rules
# ─────────────────────────────────────────────────────────────────────────────
with tab_teach:
    st.subheader("📖 Semantic Slang & Vernacular Lexicon")
    st.caption("Teach new Hinglish/English slang words, idioms, or custom vocabulary to the agent.")

    col_add, col_list = st.columns([2, 3])

    with col_add:
        with st.form("add_slang_form"):
            st.markdown("#### ➕ Add / Update Slang Term")
            slang_phrase = st.text_input("Phrase / Word", placeholder="e.g. bawaal, faadu, dhansu, bogus")
            slang_label = st.selectbox("Sentiment Polarity", ["positive", "negative", "neutral"])
            slang_weight = st.slider("Influence Weight", min_value=0.1, max_value=1.0, value=0.85, step=0.05)
            slang_notes = st.text_input("Context Notes", placeholder="e.g. colloquial for mind-blowing")
            add_slang_submitted = st.form_submit_button("Teach Word", use_container_width=True)

            if add_slang_submitted and slang_phrase:
                memory_engine.teach_slang(slang_phrase, slang_label, slang_weight, slang_notes)
                st.success(f"✓ Learned slang '{slang_phrase}' as {slang_label}!")
                st.rerun()

    with col_list:
        st.markdown(f"#### 📖 Current Lexicon ({len(memory_engine.slang_lexicon)} words)")
        slang_df = pd.DataFrame(
            [
                {
                    "Phrase": entry.phrase,
                    "Sentiment": entry.label.upper(),
                    "Weight": entry.weight,
                    "Notes": entry.notes,
                }
                for entry in sorted(memory_engine.slang_lexicon.values(), key=lambda x: x.phrase)
            ]
        )
        st.dataframe(slang_df, use_container_width=True, height=380)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Batch Prediction with Memory
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("⚡ Batch Unknown Text Evaluation with Memory")
    st.caption("Upload a CSV with unknown text rows to see how the agent reasons with past knowledge on each row.")

    uploaded_file = st.file_uploader("Upload CSV file (must have a 'text' column)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("Uploaded CSV must have a 'text' column.")
            else:
                st.write(f"Loaded {len(df)} rows. Processing first 50 rows...")
                sample_df = df.head(50).copy()

                results = []
                for _, row in sample_df.iterrows():
                    txt = str(row["text"])
                    tr = memory_engine.analyze_and_reason(txt, predictor)
                    results.append(
                        {
                            "text": txt,
                            "predicted_label": tr.final_label,
                            "confidence": tr.final_confidence,
                            "decision_source": tr.decision_source,
                            "recalled_slang_count": len(tr.recalled_slang),
                            "recalled_episodes_count": len(tr.recalled_episodes),
                            "explanation": tr.explanation,
                        }
                    )
                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True)

                st.download_button(
                    "📥 Download Results CSV",
                    res_df.to_csv(index=False).encode("utf-8"),
                    "agentic_sentiment_results.csv",
                    "text/csv",
                )
        except Exception as err:
            st.error(f"Error processing CSV: {err}")

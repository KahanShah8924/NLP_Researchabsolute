#!/usr/bin/env python3
"""Interactive Terminal CLI for Agentic AI Sentiment Analysis with Past Knowledge Memory.

Usage:
    Interactive mode:
        python agent_cli.py

    Direct evaluation:
        python agent_cli.py "bhai movie ekdum mast thi, paisa vasool"
        python agent_cli.py --text "bilkul bekar update" --explain

    Memory management:
        python agent_cli.py --stats
        python agent_cli.py --list-slang
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_memory import AgentMemoryEngine, AgentThoughtTrace
from src.predict import available_models, load_predictor


# ANSI Terminal Colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
WHITE = "\033[97m"


def color_label(label: str) -> str:
    lbl = label.lower()
    if lbl == "positive":
        return f"{GREEN}{BOLD}POSITIVE{RESET}"
    if lbl == "negative":
        return f"{RED}{BOLD}NEGATIVE{RESET}"
    return f"{YELLOW}{BOLD}NEUTRAL{RESET}"


def print_banner() -> None:
    print(f"""{CYAN}{BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║             🧠 AGENTIC AI CODE-MIXED SENTIMENT REASONING ENGINE              ║
║         Zero-Shot & Unknown Text Inference with Continuous Memory Bank       ║
╚══════════════════════════════════════════════════════════════════════════════╝{RESET}
{DIM}• Handles unseen/unknown Hinglish & English text
• Recalls past episodes & semantic slang lexicon
• Learns continuously from user corrections & feedback{RESET}
""")


def display_trace(trace: AgentThoughtTrace, verbose: bool = True) -> None:
    print("\n" + f"{BLUE}{'─' * 76}{RESET}")
    print(f"{BOLD}INPUT TEXT:{RESET} {WHITE}\"{trace.input_text}\"{RESET}")
    print(f"{DIM}Cleaned:   \"{trace.cleaned_text}\"{RESET}")

    # 1. Perception
    p = trace.perception
    print(f"\n{CYAN}{BOLD}🔍 1. Linguistic Perception:{RESET}")
    print(
        f"   Words: {p.get('total_words', 0)} | "
        f"English: {p.get('english_ratio', 0.0):.0%} | "
        f"Romanized Hindi: {p.get('hindi_ratio', 0.0):.0%} | "
        f"Slang/Other: {p.get('mixed_or_slang_ratio', 0.0):.0%}"
    )

    # 2. Knowledge Retrieval
    print(f"\n{MAGENTA}{BOLD}📚 2. Past Knowledge Recall:{RESET}")
    if trace.recalled_slang:
        print(f"   Recognized Slang/Lexicon:")
        for s in trace.recalled_slang:
            print(f"   • {BOLD}'{s['phrase']}'{RESET} -> {color_label(s['label'])} (weight: {s['weight']}) [{s['notes']}]")
    else:
        print(f"   • No matching slang terms in lexicon.")

    if trace.recalled_episodes:
        print(f"   Recalled Past Episodes:")
        for ep in trace.recalled_episodes:
            src_tag = f"{RED}[User Corrected]{RESET}" if ep["source"] == "user_corrected" else f"{GREEN}[Verified]{RESET}"
            print(f"   • {src_tag} (sim: {ep['similarity']:.0%}) \"{ep['raw_text']}\" -> {color_label(ep['label'])}")
            if ep.get("notes"):
                print(f"     {DIM}Note: {ep['notes']}{RESET}")
    else:
        print(f"   • No highly similar past cases found (this appears to be novel unknown phrasing).")

    # 3. Model vs Agent Decision
    print(f"\n{YELLOW}{BOLD}⚖️ 3. Inference & Knowledge Fusion:{RESET}")
    bm = trace.base_model_prediction
    print(f"   Base Classifier:  {color_label(bm['label'])} ({bm['confidence']:.1%}) {DIM}probs: {bm['probabilities']}{RESET}")
    print(f"   Adjusted Probs:   {DIM}{trace.knowledge_adjusted_probabilities}{RESET}")
    print(f"   Decision Logic:   {CYAN}{trace.decision_source}{RESET}")
    print(f"   Agent Reasoning:  {WHITE}{trace.explanation}{RESET}")

    # 4. Final Verdict
    print(f"\n{BOLD}🎯 FINAL VERDICT:{RESET} {color_label(trace.final_label)}  "
          f"{BOLD}Confidence:{RESET} {CYAN}{trace.final_confidence:.1%}{RESET}")
    print(f"{BLUE}{'─' * 76}{RESET}\n")


def interactive_loop(engine: AgentMemoryEngine, predictor: dict, model_name: str) -> None:
    print_banner()
    stats = engine.get_stats()
    print(f"{DIM}Loaded model: {BOLD}{model_name}{RESET}{DIM} | Memory Episodes: {stats['total_episodes']} | Slang Entries: {stats['total_slang_entries']}{RESET}\n")

    while True:
        try:
            prompt_text = input(f"{BOLD}{GREEN}Enter unknown text to analyze (or 'help' / 'q' to exit):{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not prompt_text:
            continue
        if prompt_text.lower() in {"q", "quit", "exit"}:
            print("Exiting Agentic AI. Memory saved. Goodbye!")
            break
        if prompt_text.lower() == "help":
            print(f"""
{CYAN}Commands:{RESET}
  • Type any sentence in English, Hindi, or Hinglish to analyze
  • 'stats'       : Show memory bank statistics
  • 'slang'       : List all remembered slang terms
  • 'teach'       : Add a new slang word/phrase
  • 'q' or 'exit' : Quit
""")
            continue
        if prompt_text.lower() == "stats":
            st = engine.get_stats()
            print(f"\n{CYAN}Memory Bank Stats:{RESET}")
            for k, v in st.items():
                print(f"  {k}: {v}")
            print()
            continue
        if prompt_text.lower() == "slang":
            print(f"\n{CYAN}Remembered Slang Lexicon ({len(engine.slang_lexicon)} entries):{RESET}")
            for phrase, entry in sorted(engine.slang_lexicon.items()):
                print(f"  • {BOLD}{phrase:<18}{RESET} -> {color_label(entry.label)} (weight: {entry.weight}) {DIM}{entry.notes}{RESET}")
            print()
            continue
        if prompt_text.lower() == "teach":
            phrase = input("Enter slang phrase: ").strip()
            if not phrase:
                continue
            lbl = input("Enter sentiment label (positive/negative/neutral): ").strip().lower()
            if lbl not in {"positive", "negative", "neutral"}:
                print("Invalid label. Must be positive, negative, or neutral.")
                continue
            notes = input("Enter notes/context (optional): ").strip()
            engine.teach_slang(phrase, lbl, 0.85, notes)
            print(f"{GREEN}✓ Successfully saved slang '{phrase}' ({lbl}) into persistent memory!{RESET}\n")
            continue

        # Run Agent Analysis
        try:
            trace = engine.analyze_and_reason(prompt_text, predictor)
            display_trace(trace)
        except Exception as e:
            print(f"{RED}Error analyzing text: {e}{RESET}")
            continue

        # Active Feedback Loop
        while True:
            feedback = input(
                f"{YELLOW}[Feedback Loop]{RESET} "
                f"[Y] Confirm | [C] Correct Label | [T] Teach Slang | [Enter] Next: "
            ).strip().lower()

            if feedback in {"y", "yes"}:
                engine.remember_feedback(prompt_text, trace.final_label, source="user_confirmed", notes="Confirmed by user")
                print(f"{GREEN}✓ Memory updated! Confirmed as {trace.final_label}.{RESET}\n")
                break
            elif feedback in {"c", "correct"}:
                print("\nSelect correct label:")
                print("  1) Positive")
                print("  2) Negative")
                print("  3) Neutral")
                choice = input("Enter choice (1/2/3): ").strip()
                mapping = {"1": "positive", "2": "negative", "3": "neutral", "pos": "positive", "neg": "negative", "neu": "neutral"}
                correct_label = mapping.get(choice)
                if not correct_label:
                    print(f"{RED}Invalid choice.{RESET}")
                    continue
                reason = input("Why was the model incorrect? (e.g. 'sarcastic idiom'): ").strip()
                engine.remember_feedback(
                    prompt_text,
                    correct_label,
                    source="user_corrected",
                    notes=reason or f"User corrected from {trace.final_label} to {correct_label}",
                )
                print(f"{GREEN}✓ Memory updated! Future unknown texts similar to this will recall '{correct_label}'.{RESET}\n")
                break
            elif feedback in {"t", "teach"}:
                phrase = input("Enter specific slang/word in this text: ").strip()
                if phrase:
                    lbl = input("Label for this word (positive/negative/neutral): ").strip().lower()
                    if lbl in {"positive", "negative", "neutral"}:
                        note = input("Context note: ").strip()
                        engine.teach_slang(phrase, lbl, 0.85, note)
                        print(f"{GREEN}✓ Learned slang '{phrase}' -> {lbl}!{RESET}")
                break
            else:
                # Enter or skip
                print()
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic AI Code-Mixed Sentiment Analyzer with Persistent Memory")
    parser.add_argument("text", nargs="?", help="Direct text to evaluate")
    parser.add_argument("--model", default="logistic_regression", help="Trained model name (default: logistic_regression)")
    parser.add_argument("--explain", action="store_true", help="Print full agent reasoning trace")
    parser.add_argument("--stats", action="store_true", help="Print memory bank statistics")
    parser.add_argument("--list-slang", action="store_true", help="List all remembered slang words")
    args = parser.parse_args()

    engine = AgentMemoryEngine()

    if args.stats:
        print_banner()
        print(f"{CYAN}Memory Bank Statistics:{RESET}")
        for k, v in engine.get_stats().items():
            print(f"  {k}: {v}")
        return

    if args.list_slang:
        print_banner()
        print(f"{CYAN}Remembered Slang Lexicon ({len(engine.slang_lexicon)} items):{RESET}")
        for phrase, entry in sorted(engine.slang_lexicon.items()):
            print(f"  • {BOLD}{phrase:<18}{RESET} -> {color_label(entry.label)} (weight: {entry.weight}) {DIM}{entry.notes}{RESET}")
        return

    available = available_models()
    if not available:
        print(f"{RED}Error: No trained models found in 'models/'.{RESET}")
        sys.exit(1)

    model_name = args.model if args.model in available else list(available.keys())[0]
    predictor = load_predictor(model_name)

    if args.text:
        trace = engine.analyze_and_reason(args.text, predictor)
        if args.explain:
            display_trace(trace)
        else:
            print(f"Input:    \"{trace.input_text}\"")
            print(f"Verdict:  {color_label(trace.final_label)} ({trace.final_confidence:.1%})")
            print(f"Reason:   {trace.explanation}")
        return

    # Default to interactive REPL
    interactive_loop(engine, predictor, model_name)


if __name__ == "__main__":
    main()

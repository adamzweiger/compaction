"""
Demo: compact a mock article's KV cache and compare QA accuracy before/after.

This script:
1. Loads a model and encodes a short mock article into its KV cache
2. Asks 3 multiple-choice questions using the full (original) cache
3. Compacts the cache to 10% of its original size using Attention Matching
4. Asks the same 3 questions using the compacted cache
5. Compares the answers

Usage:
    python -m examples.qa_demo
    python -m examples.qa_demo --model Qwen/Qwen3-4B
    python -m examples.qa_demo --target-size 0.05
"""
import argparse
import time

import torch

from evaluation.utils import (
    load_model_and_tokenizer,
    extract_full_kv_cache,
    format_question,
    parse_model_choice,
)
from evaluation.configs.utils import load_query_config
from compaction.compaction_methods.registry import get_compaction_method
from models.generate import generate_with_compacted_cache_batch

# ---------------------------------------------------------------------------
# Mock article (opus-generated) and questions
# ---------------------------------------------------------------------------

ARTICLE = """\
The small island nation of Verandia, located in the southern Pacific Ocean, \
declared independence from colonial rule on March 3, 1987. Its first president, \
Elena Korvath, served two consecutive terms from 1987 to 1999. Under her \
leadership, Verandia transitioned from a plantation economy dependent on \
sugar cane exports to a diversified economy centered on eco-tourism, \
sustainable fisheries, and a growing technology sector.

In 2003, Verandia established the Korvath Marine Reserve, named after the \
former president, which spans 12,000 square kilometers around the island's \
coral reef system. The reserve became a UNESCO World Heritage Site in 2008 \
and attracts roughly 40,000 visitors per year. Revenue from the reserve's \
entrance fees funds the majority of the island's public school system.

Verandia's current population stands at approximately 58,000 people. The \
official languages are English and Verandian Creole. The capital, Port Alani, \
is home to the National University of Verandia, which opened in 1995 and \
is known for its marine biology program. The university collaborates with \
research institutions worldwide and operates a deep-sea research vessel, \
the RV Coral Pioneer, launched in 2019.
"""

QUESTIONS = [
    {
        "question": "When did Verandia declare independence?",
        "options": [
            "June 15, 1972",
            "March 3, 1987",
            "January 1, 2000",
            "August 22, 1965",
        ],
        "gold_label": 2,  # B (1-indexed)
    },
    {
        "question": "What is the Korvath Marine Reserve primarily named after?",
        "options": [
            "A species of coral found near the island",
            "The body of water surrounding Verandia",
            "Verandia's first president, Elena Korvath",
            "A local Verandian Creole word for ocean",
        ],
        "gold_label": 3,  # C
    },
    {
        "question": "What is the name of the deep-sea research vessel operated by the National University of Verandia?",
        "options": [
            "RV Ocean Explorer",
            "RV Deep Current",
            "RV Reef Guardian",
            "RV Coral Pioneer",
        ],
        "gold_label": 4,  # D
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask_questions(
    model,
    tokenizer,
    questions,
    *,
    compacted_cache=None,
    original_seq_len=None,
    formatted_context=None,
    model_name=None,
    max_new_tokens=2048,
):
    """Ask questions using either the compacted cache or the full text context."""
    results = []
    prompts = [
        format_question(tokenizer, q["question"], q["options"], model_name)
        for q in questions
    ]

    if compacted_cache is not None:
        # Cache-based generation
        answers = generate_with_compacted_cache_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            compacted_cache=compacted_cache,
            max_new_tokens=max_new_tokens,
            original_seq_len=original_seq_len,
        )
    else:
        # Text-based generation (full context prepended to each prompt)
        full_prompts = [formatted_context + p for p in prompts]
        device = next(model.parameters()).device
        generated = []
        for fp in full_prompts:
            inputs = tokenizer(fp, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            answer_ids = out[0][inputs["input_ids"].shape[1]:]
            generated.append(tokenizer.decode(answer_ids, skip_special_tokens=True))
        answers = generated

    for q, answer in zip(questions, answers):
        choice = parse_model_choice(answer)
        correct = choice == q["gold_label"] if choice else False
        letter = chr(64 + choice) if choice else "?"
        gold_letter = chr(64 + q["gold_label"])
        results.append({
            "question": q["question"],
            "answer_text": answer.strip()[:200],
            "parsed": letter,
            "gold": gold_letter,
            "correct": correct,
        })
    return results


def print_results(results, label):
    """Pretty-print QA results."""
    n_correct = sum(r["correct"] for r in results)
    print(f"\n{'='*60}")
    print(f"  {label}  —  {n_correct}/{len(results)} correct")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        mark = "CORRECT" if r["correct"] else "WRONG"
        print(f"\n  Q{i}: {r['question']}")
        print(f"  Model: {r['parsed']}  |  Gold: {r['gold']}  [{mark}]")
        print(f"  Answer: {r['answer_text'][:120]}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KV cache compaction QA demo")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HuggingFace model name")
    parser.add_argument("--target-size", type=float, default=0.1,
                        help="Fraction of article tokens to keep (default: 0.1)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. Load model
    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)

    # 2. Extract full KV cache
    print(f"\nEncoding article ({len(ARTICLE)} chars) ...")
    seq_len, past_key_values, article_indices, formatted_context, _ = extract_full_kv_cache(
        model, tokenizer, ARTICLE, args.device, model_name=args.model,
    )
    article_len = len(article_indices)
    print(f"  Total tokens: {seq_len}  |  Article tokens: {article_len}")

    # 3. Ask questions with ORIGINAL (full) cache — use the unmodified KV cache
    #    We build a trivial "compacted" cache from the original past_key_values so
    #    we can reuse the same generate_with_compacted_cache_batch path.
    original_cache = tuple(
        (k, torch.zeros_like(k[:, :, :, 0]), v)
        for k, v in past_key_values
    )
    print("\nAnswering questions with ORIGINAL cache ...")
    original_results = ask_questions(
        model, tokenizer, QUESTIONS,
        compacted_cache=original_cache,
        original_seq_len=seq_len,
        model_name=args.model,
    )
    del original_cache
    print_results(original_results, "Original (full cache)")

    # 4. Compact the KV cache
    algorithm_kwargs = {
        "algorithm": "highest_attention_keys",
        "score_method": "rms",
        "nnls_iters": 2,
        "nnls_lower_bound": 0.05,
        "nnls_upper_bound": 20.0,
        "c2_method": "lsq",
        "precomputed_budget_path": "head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json",
    }
    compaction_method = get_compaction_method(
        "AM-HighestAttnKeys", method_kwargs=algorithm_kwargs,
    )
    query_config = load_query_config("repeat")

    target_article_tokens = max(1, int(article_len * args.target_size))
    non_article_tokens = seq_len - article_len
    actual_target = target_article_tokens + non_article_tokens
    print(f"\nCompacting cache: {article_len} -> {target_article_tokens} article tokens "
          f"({args.target_size:.0%}) ...")

    t0 = time.time()
    compacted_cache, stats = compaction_method.compact_kv_cache(
        past_key_values=past_key_values,
        target_size=actual_target,
        indices=article_indices,
        query_config=query_config,
        model=model,
        tokenizer=tokenizer,
        formatted_context=formatted_context,
        compute_stats=False,
    )
    dt = time.time() - t0
    print(f"  Compaction took {dt:.2f}s")

    # 5. Ask questions with COMPACTED cache
    print("\nAnswering questions with COMPACTED cache ...")
    compacted_results = ask_questions(
        model, tokenizer, QUESTIONS,
        compacted_cache=compacted_cache,
        original_seq_len=seq_len,
        model_name=args.model,
    )
    print_results(compacted_results, f"Compacted ({args.target_size:.0%} of article)")

    # 6. Summary
    orig_acc = sum(r["correct"] for r in original_results)
    comp_acc = sum(r["correct"] for r in compacted_results)
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Article tokens : {article_len}")
    print(f"  Compacted to   : {target_article_tokens} tokens ({args.target_size:.0%})")
    print(f"  Original acc   : {orig_acc}/{len(QUESTIONS)}")
    print(f"  Compacted acc  : {comp_acc}/{len(QUESTIONS)}")
    print(f"  Compaction time: {dt:.2f}s")
    print()


if __name__ == "__main__":
    main()

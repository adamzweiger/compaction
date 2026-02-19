# head_budget_optimization/evaluate.py
"""
Evaluate optimized head budget allocations by comparing predicted vs true perplexity.

This script:
1. Loads optimized proportions and the head curves used to compute them
2. Evaluates baseline (uniform) and optimized allocations on articles
3. Computes true perplexity delta and predicted delta (assuming separability)
4. Saves results to a JSON analysis file

Example usage:
    python head_budget_optimization/evaluate.py \
        --per-article-curves-dir logs/budget_optimization/Qwen3-4B/optimized \
        --optimized-proportions head_budget_optimization/head_budgets/Qwen3-4B/optimized_omp/optimized_t0.05.json \
        --target-ratio 0.05 \
        --n-articles 5
"""
import sys
from pathlib import Path

# Add project root to path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import json
import math
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from evaluation.utils import (
    load_model_and_tokenizer,
    extract_full_kv_cache,
    format_question,
    get_or_generate_reference_answers,
    compute_perplexity_on_compacted_cache,
    initialize_vllm,
)
from evaluation.datasets import load_dataset
from evaluation.configs.utils import load_algorithm_config, load_query_config
from compaction.query_generation import QueryGenerator
from models.generate import get_sliding_layer_info

from .influence import load_and_aggregate_article_curves
from .solver import HeadBudgetSolver


def compute_predicted_delta(
    head_curves: Dict[str, List[Tuple[float, float]]],
    baseline_ratios: Dict[str, float],
    optimized_ratios: Dict[str, float],
    solver: HeadBudgetSolver,
) -> float:
    """
    Compute predicted delta log perplexity assuming separability.

    The predicted delta is the sum of individual head deltas:
    predicted_delta = sum(delta(optimized_ratio[h]) - delta(baseline_ratio[h])) for all heads h

    Parameters
    ----------
    head_curves : dict
        Per-head influence curves
    baseline_ratios : dict
        Baseline (uniform) ratios per head
    optimized_ratios : dict
        Optimized ratios per head
    solver : HeadBudgetSolver
        Solver instance for interpolation

    Returns
    -------
    predicted_delta : float
        Predicted change in log perplexity (negative = improvement)
    """
    total_delta = 0.0
    for head_key in baseline_ratios:
        baseline_ratio = baseline_ratios[head_key]
        optimized_ratio = optimized_ratios[head_key]

        # Get delta at each ratio
        baseline_delta = solver.interpolate_delta(head_key, baseline_ratio)
        optimized_delta = solver.interpolate_delta(head_key, optimized_ratio)

        # Contribution from this head
        total_delta += optimized_delta - baseline_delta

    return total_delta


def proportions_to_ratios(
    proportions: Dict[str, float],
    target_ratio: float,
    num_layers: int,
    num_heads: int,
) -> Dict[str, float]:
    """
    Convert proportions to per-head ratios.

    Proportions sum to 1.0 and represent fraction of total allocated budget.
    Ratios represent fraction of original keys for each head.

    Note: For models with sliding window layers, num_layers should be the
    number of global (non-sliding) layers only, since proportions only
    include global layer heads.

    Parameters
    ----------
    proportions : dict
        Per-head proportions (sum to 1.0)
    target_ratio : float
        Overall target ratio (e.g., 0.05)
    num_layers : int
        Number of global (non-sliding) layers
    num_heads : int
        Number of heads per layer

    Returns
    -------
    ratios : dict
        Per-head ratios (each in range [0, 1])
    """
    total_heads = num_layers * num_heads
    # Total budget = target_ratio * total_heads (in units of "article_len per head")
    # Each head's ratio = proportion * total_budget = proportion * target_ratio * total_heads
    ratios = {}
    for head_key, proportion in proportions.items():
        ratios[head_key] = proportion * target_ratio * total_heads
    return ratios


def evaluate_allocation_on_article(
    model,
    tokenizer,
    article_data: Dict,
    proportions: Dict[str, float],
    target_ratio: float,
    query_config,
    algorithm_name: str,
    algorithm_kwargs: Dict,
    device: str,
    max_new_tokens: int,
    vllm_model=None,
) -> Tuple[float, float]:
    """
    Evaluate a head budget allocation on a single article.

    Parameters
    ----------
    model : PreTrainedModel
        Language model
    tokenizer : PreTrainedTokenizer
        Tokenizer
    article_data : dict
        Article data with questions
    proportions : dict
        Per-head proportions
    target_ratio : float
        Overall target ratio
    query_config : QueryConfig
        Query generation config
    algorithm_name : str
        Name of compaction algorithm
    algorithm_kwargs : dict
        Algorithm keyword arguments
    device : str
        Device to use
    max_new_tokens : int
        Max tokens for generation
    vllm_model : optional
        vLLM model for self-study queries

    Returns
    -------
    avg_perplexity : float
        Average perplexity across questions
    avg_log_perplexity : float
        Average log perplexity across questions
    """
    from compaction.algorithms.highest_attention_keys import HighestAttentionKeysCompaction
    from compaction.algorithms.omp import OMPCompaction

    # Extract KV cache
    seq_len, past_key_values, article_indices, formatted_context, _ = extract_full_kv_cache(
        model=model,
        tokenizer=tokenizer,
        article_text=article_data['article'],
        device=device,
    )

    article_len = len(article_indices)
    num_layers = len(past_key_values)

    # Get sliding layer info
    sliding_layer_indices, sliding_window = get_sliding_layer_info(model)
    global_layer_indices = [i for i in range(num_layers) if i not in sliding_layer_indices]
    num_global_layers = len(global_layer_indices)

    # Get num_heads from a global (non-sliding) layer
    ref_layer_idx = global_layer_indices[0] if global_layer_indices else 0
    num_heads = past_key_values[ref_layer_idx][0].shape[1]

    # Get reference answers
    questions = article_data['questions']
    reference_answers = get_or_generate_reference_answers(
        article_id=article_data['article_id'],
        model=model,
        tokenizer=tokenizer,
        formatted_context=formatted_context,
        questions=questions,
        model_name=model.config._name_or_path,
        max_new_tokens=max_new_tokens,
        device=device,
        vllm_model=vllm_model,
    )

    # Generate queries
    generator = QueryGenerator(
        model=model,
        tokenizer=tokenizer,
        config=query_config,
        device=device,
        dtype=past_key_values[0][0].dtype,
        vllm_model=vllm_model,
    )

    queries, query_stats, _ = generator.generate_queries(
        formatted_context=formatted_context,
        past_key_values=past_key_values,
        indices=article_indices,
    )

    # Create algorithm instance
    if algorithm_name == 'omp':
        algorithm = OMPCompaction(**algorithm_kwargs)
    else:
        algorithm = HighestAttentionKeysCompaction(**algorithm_kwargs)

    # Compute per-head budgets from proportions (global layers only)
    total_budget = int(target_ratio * article_len * num_global_layers * num_heads)
    head_budgets = {}
    for head_key, proportion in proportions.items():
        head_budgets[head_key] = max(1, int(proportion * total_budget))

    # Compact each global layer head
    head_results = {}
    for layer_idx in global_layer_indices:
        keys_layer = past_key_values[layer_idx][0][0]
        values_layer = past_key_values[layer_idx][1][0]

        for head_idx in range(num_heads):
            head_key = f"L{layer_idx}H{head_idx}"
            budget = head_budgets.get(head_key, int(target_ratio * article_len))

            K = keys_layer[head_idx, list(article_indices), :]
            V = values_layer[head_idx, list(article_indices), :]
            queries_head = queries[layer_idx, head_idx, :, :]

            C1, beta, C2, selected_indices = algorithm.compute_compacted_cache(
                K, V, queries_head, budget
            )

            head_results[head_key] = {
                'C1': C1,
                'beta': beta,
                'C2': C2,
            }

    # Build compacted cache
    max_compacted_size = max(hr['C1'].shape[0] for hr in head_results.values())

    # Get keep indices (for global layers)
    global_seq_len = past_key_values[ref_layer_idx][0].shape[2]
    all_indices = set(range(global_seq_len))
    article_set = set(article_indices)
    keep_indices = sorted(all_indices - article_set)
    article_start = article_indices.start
    keep_before = [i for i in keep_indices if i < article_start]
    keep_after = [i for i in keep_indices if i >= article_indices.stop]

    compacted_layers = []
    head_dim = past_key_values[ref_layer_idx][0].shape[-1]
    device = past_key_values[ref_layer_idx][0].device
    dtype = past_key_values[ref_layer_idx][0].dtype

    for layer_idx in range(num_layers):
        keys_layer = past_key_values[layer_idx][0][0]
        values_layer = past_key_values[layer_idx][1][0]

        # Handle sliding window layers differently
        if layer_idx in sliding_layer_indices:
            # For sliding layers, store as (keys, beta=0, values)
            sliding_seq_len = keys_layer.shape[1]
            C1_layer = keys_layer.unsqueeze(0)  # (1, num_heads, sliding_seq_len, head_dim)
            beta_layer = torch.zeros(1, num_heads, sliding_seq_len, device=device, dtype=dtype)
            C2_layer = values_layer.unsqueeze(0)
            compacted_layers.append((C1_layer, beta_layer, C2_layer))
            continue

        C1_heads = []
        beta_heads = []
        C2_heads = []

        for head_idx in range(num_heads):
            head_key = f"L{layer_idx}H{head_idx}"
            hr = head_results[head_key]

            # Get kept portions
            K_keep_before = keys_layer[head_idx, keep_before, :] if keep_before else keys_layer.new_zeros(0, head_dim)
            V_keep_before = values_layer[head_idx, keep_before, :] if keep_before else values_layer.new_zeros(0, head_dim)
            K_keep_after = keys_layer[head_idx, keep_after, :] if keep_after else keys_layer.new_zeros(0, head_dim)
            V_keep_after = values_layer[head_idx, keep_after, :] if keep_after else values_layer.new_zeros(0, head_dim)

            beta_keep_before = keys_layer.new_zeros(len(keep_before))
            beta_keep_after = keys_layer.new_zeros(len(keep_after))

            # Pad compacted portion
            C1_compact = hr['C1']
            beta_compact = hr['beta']
            C2_compact = hr['C2']

            if C1_compact.shape[0] < max_compacted_size:
                pad_len = max_compacted_size - C1_compact.shape[0]
                C1_compact = torch.cat([C1_compact, C1_compact.new_zeros(pad_len, head_dim)], dim=0)
                C2_compact = torch.cat([C2_compact, C2_compact.new_zeros(pad_len, head_dim)], dim=0)
                beta_compact = torch.cat([beta_compact, beta_compact.new_full((pad_len,), float('-inf'))], dim=0)

            # Concatenate
            C1 = torch.cat([K_keep_before, C1_compact, K_keep_after], dim=0)
            beta = torch.cat([beta_keep_before, beta_compact, beta_keep_after], dim=0)
            C2 = torch.cat([V_keep_before, C2_compact, V_keep_after], dim=0)

            C1_heads.append(C1.unsqueeze(0).unsqueeze(0))
            beta_heads.append(beta.unsqueeze(0).unsqueeze(0))
            C2_heads.append(C2.unsqueeze(0).unsqueeze(0))

        C1_layer = torch.cat(C1_heads, dim=1)
        beta_layer = torch.cat(beta_heads, dim=1)
        C2_layer = torch.cat(C2_heads, dim=1)

        compacted_layers.append((C1_layer, beta_layer, C2_layer))

    compacted_cache = tuple(compacted_layers)

    # Compute perplexity
    perplexities = []
    for i, (question_id, gen_token_ids, gen_text) in enumerate(reference_answers):
        q = questions[i]
        question_formatted = format_question(
            tokenizer, q['question'], q.get('options', None),
            model.config._name_or_path
        )

        ppl, log_ppl = compute_perplexity_on_compacted_cache(
            model=model,
            tokenizer=tokenizer,
            compacted_cache=compacted_cache,
            generated_token_ids=gen_token_ids,
            question_prompt=question_formatted,
            device=device,
            original_seq_len=seq_len,
        )
        perplexities.append((ppl, log_ppl))

    avg_ppl = sum(p for p, _ in perplexities) / len(perplexities)
    avg_log_ppl = sum(lp for _, lp in perplexities) / len(perplexities)

    # Clean up
    del past_key_values, compacted_cache
    torch.cuda.empty_cache()

    return avg_ppl, avg_log_ppl


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate optimized head budget allocations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required inputs
    parser.add_argument(
        '--per-article-curves-dir',
        type=str,
        required=True,
        help='Directory containing per-article curve files (article_*.json)'
    )
    parser.add_argument(
        '--optimized-proportions',
        type=str,
        required=True,
        help='Path to optimized proportions JSON file'
    )

    # Evaluation settings
    parser.add_argument(
        '--target-ratio',
        type=float,
        required=True,
        help='Target compaction ratio (e.g., 0.05)'
    )
    parser.add_argument(
        '--baseline-proportions',
        type=str,
        default='head_budget_optimization/head_budgets/Qwen3-4B/uniform.json',
        help='Path to baseline (uniform) proportions JSON'
    )

    # Data arguments
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='quality',
        help='Dataset name (default: quality)'
    )
    parser.add_argument(
        '--n-articles',
        type=int,
        default=1,
        help='Number of articles to evaluate (default: 1)'
    )
    parser.add_argument(
        '--start-article',
        type=int,
        default=0,
        help='Starting article index (default: 0)'
    )

    # Model arguments
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen3-4B',
        help='HuggingFace model name (default: Qwen/Qwen3-4B)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: cuda if available)'
    )

    # Algorithm config
    parser.add_argument(
        '--algorithm-config',
        type=str,
        default='best',
        help='Name of algorithm config file (default: best)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='highest_attn_keys_rms_nnls2_-3_3_lsq',
        help='Method name within the algorithm config'
    )
    parser.add_argument(
        '--query-config',
        type=str,
        default='repeat',
        help='Name of query generation config (default: repeat)'
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for evaluation results (default: <per-article-curves-dir>/evaluation.json)'
    )

    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Max tokens for reference answer generation (default: 2048)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Require CUDA
    if args.device != 'cuda':
        raise RuntimeError(
            "CUDA is currently required for evaluation. "
            "Please run on a machine with a GPU."
        )

    # Default output path
    if args.output is None:
        args.output = str(Path(args.per_article_curves_dir) / 'evaluation.json')

    print(f"\n{'='*60}")
    print("HEAD BUDGET ALLOCATION EVALUATION")
    print(f"{'='*60}")
    print(f"Per-article curves: {args.per_article_curves_dir}")
    print(f"Optimized proportions: {args.optimized_proportions}")
    print(f"Baseline proportions: {args.baseline_proportions}")
    print(f"Target ratio: {args.target_ratio}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    # Load head curves
    print("Loading and aggregating head curves...")
    head_curves, article_metadata = load_and_aggregate_article_curves(args.per_article_curves_dir)

    # Infer model dimensions from head curves (global layers only)
    head_keys = list(head_curves.keys())
    global_layers = set()
    heads = set()
    for key in head_keys:
        layer_idx = int(key[1:].split('H')[0])
        head_idx = int(key.split('H')[1])
        global_layers.add(layer_idx)
        heads.add(head_idx)
    num_global_layers = len(global_layers)
    num_heads = max(heads) + 1
    print(f"Model dimensions from curves: {num_global_layers} global layers x {num_heads} heads")

    # Load proportions
    print(f"Loading baseline proportions from {args.baseline_proportions}")
    with open(args.baseline_proportions) as f:
        baseline_proportions = json.load(f)

    print(f"Loading optimized proportions from {args.optimized_proportions}")
    with open(args.optimized_proportions) as f:
        optimized_proportions = json.load(f)

    # Convert proportions to ratios (using global layer count)
    baseline_ratios = proportions_to_ratios(baseline_proportions, args.target_ratio, num_global_layers, num_heads)
    optimized_ratios = proportions_to_ratios(optimized_proportions, args.target_ratio, num_global_layers, num_heads)

    # Create solver for interpolation
    # Note: num_global_layers is passed as num_layers, but the solver infers global layers from curve keys
    solver = HeadBudgetSolver(head_curves, num_global_layers, num_heads)

    # Compute predicted delta from curves (assuming separability)
    predicted_delta = compute_predicted_delta(
        head_curves, baseline_ratios, optimized_ratios, solver
    )
    print(f"\nPredicted delta log perplexity (from curves): {predicted_delta:.6f}")

    # Load model
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)

    # Load configs
    query_config = load_query_config(args.query_config)
    algorithm_config = load_algorithm_config(args.algorithm_config, target_size=args.target_ratio)
    method_config = algorithm_config[args.method]
    algorithm_name = method_config.get('algorithm', 'highest_attention_keys')
    algorithm_kwargs = {k: v for k, v in method_config.items() if k != 'algorithm'}

    # Initialize vLLM if needed
    vllm_model = None
    self_study_config = query_config.get_method_config('self_study')
    if self_study_config is not None:
        print("Initializing vLLM for self-study query generation...")
        vllm_model = initialize_vllm(args.model_name)

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)

    # Determine articles
    if args.n_articles == -1:
        article_indices = list(range(args.start_article, len(dataset)))
    else:
        end_article = min(args.start_article + args.n_articles, len(dataset))
        article_indices = list(range(args.start_article, end_article))

    print(f"Evaluating on {len(article_indices)} articles: {article_indices}")

    # Evaluate on each article
    results_per_article = []

    for i, article_idx in enumerate(article_indices):
        article_data = dataset[article_idx]

        print(f"\n{'='*60}")
        print(f"Article {i+1}/{len(article_indices)}: {article_data['title']}")
        print(f"{'='*60}")

        # Evaluate baseline
        print("Evaluating baseline allocation...")
        baseline_ppl, baseline_log_ppl = evaluate_allocation_on_article(
            model=model,
            tokenizer=tokenizer,
            article_data=article_data,
            proportions=baseline_proportions,
            target_ratio=args.target_ratio,
            query_config=query_config,
            algorithm_name=algorithm_name,
            algorithm_kwargs=algorithm_kwargs,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            vllm_model=vllm_model,
        )
        print(f"  Baseline perplexity: {baseline_ppl:.4f} (log: {baseline_log_ppl:.4f})")

        # Evaluate optimized
        print("Evaluating optimized allocation...")
        optimized_ppl, optimized_log_ppl = evaluate_allocation_on_article(
            model=model,
            tokenizer=tokenizer,
            article_data=article_data,
            proportions=optimized_proportions,
            target_ratio=args.target_ratio,
            query_config=query_config,
            algorithm_name=algorithm_name,
            algorithm_kwargs=algorithm_kwargs,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            vllm_model=vllm_model,
        )
        print(f"  Optimized perplexity: {optimized_ppl:.4f} (log: {optimized_log_ppl:.4f})")

        # Compute true delta
        true_delta = optimized_log_ppl - baseline_log_ppl
        print(f"  True delta log perplexity: {true_delta:.6f}")

        results_per_article.append({
            'article_idx': article_idx,
            'article_title': article_data['title'],
            'article_id': article_data['article_id'],
            'baseline_perplexity': baseline_ppl,
            'baseline_log_perplexity': baseline_log_ppl,
            'optimized_perplexity': optimized_ppl,
            'optimized_log_perplexity': optimized_log_ppl,
            'true_delta_log_perplexity': true_delta,
        })

    # Aggregate results
    avg_baseline_ppl = np.mean([r['baseline_perplexity'] for r in results_per_article])
    avg_baseline_log_ppl = np.mean([r['baseline_log_perplexity'] for r in results_per_article])
    avg_optimized_ppl = np.mean([r['optimized_perplexity'] for r in results_per_article])
    avg_optimized_log_ppl = np.mean([r['optimized_log_perplexity'] for r in results_per_article])
    avg_true_delta = np.mean([r['true_delta_log_perplexity'] for r in results_per_article])

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Average baseline perplexity: {avg_baseline_ppl:.4f} (log: {avg_baseline_log_ppl:.4f})")
    print(f"Average optimized perplexity: {avg_optimized_ppl:.4f} (log: {avg_optimized_log_ppl:.4f})")
    print(f"Average true delta log perplexity: {avg_true_delta:.6f}")
    print(f"Predicted delta log perplexity (from curves): {predicted_delta:.6f}")
    print(f"Prediction error: {abs(avg_true_delta - predicted_delta):.6f}")

    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'per_article_curves_dir': args.per_article_curves_dir,
            'optimized_proportions': args.optimized_proportions,
            'baseline_proportions': args.baseline_proportions,
            'target_ratio': args.target_ratio,
            'model_name': args.model_name,
            'dataset_name': args.dataset_name,
            'algorithm_config': args.algorithm_config,
            'method': args.method,
            'query_config': args.query_config,
            'n_articles': len(article_indices),
            'article_indices': article_indices,
        },
        'predicted_delta_log_perplexity': predicted_delta,
        'results_per_article': results_per_article,
        'summary': {
            'avg_baseline_perplexity': float(avg_baseline_ppl),
            'avg_baseline_log_perplexity': float(avg_baseline_log_ppl),
            'avg_optimized_perplexity': float(avg_optimized_ppl),
            'avg_optimized_log_perplexity': float(avg_optimized_log_ppl),
            'avg_true_delta_log_perplexity': float(avg_true_delta),
            'predicted_delta_log_perplexity': predicted_delta,
            'prediction_error': float(abs(avg_true_delta - predicted_delta)),
        },
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

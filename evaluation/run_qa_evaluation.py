# evaluation/run_qa_evaluation.py
"""
Run QA evaluation for KV cache compaction methods.

This script evaluates compaction methods on a QA dataset by:
1. Extracting KV cache from articles
2. Compacting the cache using specified methods
3. Generating answers to questions using original vs compacted cache
4. Evaluating QA accuracy and logging results

Example usage:
    # Evaluate OMP on first 3 articles with stats
    python -m evaluation.run_qa_evaluation --algorithm-config default --methods AM-HighestAttnKeys --n-articles 3 --compute-stats 1

    # Compare multiple methods
    python -m evaluation.run_qa_evaluation --algorithm-config --methods original AM-HighestAttnKeys AM-OMP

    # Evaluate with custom target size
    python -m evaluation.run_qa_evaluation --target-size 0.05

    # Evaluate with chunking on a long context dataset with nonuniform budgets
    python -m evaluation.run_qa_evaluation --dataset-name longhealth5 --chunking longhealth --model-name google/gemma-3-4b-it --query-config repeat --precomputed-budget-path head_budget_optimization/head_budgets/gemma-3-4b-it/optimized_agnostic.json
"""
import argparse
import torch

from .qa_evaluator import QAEvaluator
from .configs.utils import load_algorithm_config, load_query_config


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate KV cache compaction on QA tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Method selection
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['original', 'all'],
        help='Compaction methods to evaluate. Options: original, omp, random_subset_keys, '
             'optim_joint, highest_attention_keys, etc. (default: original all)'
    )

    # Data arguments
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='quality',
        help='Dataset name. Options: quality, longhealth, longhealthX (e.g., longhealth10), lqa32k (default: quality)'
    )
    parser.add_argument(
        '--n-articles',
        type=int,
        default=1,
        help='Number of articles to evaluate. Use -1 for all articles (default: 1)'
    )
    parser.add_argument(
        '--start-article',
        type=int,
        default=0,
        help='Starting article index. Evaluates articles from start_article to start_article + n_articles (default: 0)'
    )
    parser.add_argument(
        '--n-questions-per-article',
        type=int,
        default=None,
        help='Number of questions per article. If specified, uses a random shuffled set of n questions per article (or all questions if less than n) (default: None, uses all questions)'
    )

    # Compaction arguments
    parser.add_argument(
        '--target-size',
        type=float,
        default=0.1,
        help='Target compacted article sequence length. If between 0 and 1, treated as fraction of original size (default: 0.1 = 10%%). Note that this specifies only the target compaction size of the article portion, keeping the system prompt and chat template intact (unless --ignore-article-indices is set).'
    )
    parser.add_argument(
        '--ignore-article-indices',
        action='store_true',
        help='Ablation: ignore article boundaries and compact the entire sequence. target_size then refers to the whole sequence length rather than just the article portion.'
    )

    # Chunked compaction arguments
    parser.add_argument(
        '--chunking',
        type=str,
        default=None,
        choices=['none', 'fixed', 'longhealth', 'longhealth_fine', 'lqa'],
        help='Chunking strategy for compaction. Options: none (default), fixed (fixed-size chunks), '
             'longhealth (split on <text_0>, grouping note chains), longhealth_fine (split on each <text_X> tag), '
             'lqa (split on [start of ...] markers)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=4096,
        help='Chunk size in tokens for fixed chunking (default: 4096)'
    )
    parser.add_argument(
        '--use-kv-based',
        type=int,
        default=1,
        help='Use KV-based chunked compaction: extract full cache once, then slice chunks from it (default: 1). '
             'Set to 1 to enable. This avoids RoPE corrections by using pre-extracted cache with correct positions.'
    )

    # Evaluation arguments
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens to generate for each answer (default: 2048)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Batch size for generation. If specified and nonzero, processes questions in batches for faster generation (default: 20)'
    )
    parser.add_argument(
        '--compute-stats',
        type=int,
        default=0,
        help='Compute detailed statistics (cosine similarity, MSE, etc.). Warning: may cause OOM if context is too long. 1 to enable, 0 to disable (default: 0)'
    )
    parser.add_argument(
        '--verbose-logging',
        type=int,
        default=0,
        help='Enable verbose logging for compaction and query generation (saves detailed stats like beta_stats, selected_indices). 1 to enable, 0 to disable (default: 0)'
    )
    parser.add_argument(
        '--compute-perplexity',
        type=int,
        default=1,
        help='Compute perplexity of original generation under compacted cache. 1 to enable, 0 to disable (default: 1)'
    )
    parser.add_argument(
        '--perplexity-only',
        type=int,
        default=0,
        help='Only compute perplexity without generating new answers (implies --compute-perplexity). 1 to enable, 0 to disable (default: 0)'
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
        help='Device to use (cpu or cuda). If not specified, uses cuda if available, else cpu'
    )
    parser.add_argument(
        '--long-context',
        action='store_true',
        help='Enable long context support with YaRN rope scaling. '
             'Sets max_model_len=131072 for vLLM and enables YaRN for Qwen3-4B.'
    )

    # Configuration files
    parser.add_argument(
        '--algorithm-config',
        type=str,
        default='fast',
        help='Name of algorithm hyperparameter config (e.g., "fast") - automatically uses configs/algorithms/ directory'
    )
    parser.add_argument(
        '--query-config',
        type=str,
        default='repeat',
        help='Name of query generation config (e.g., "repeat", "self-study") - automatically uses configs/query_generation/ directory'
    )
    parser.add_argument(
        '--precomputed-budget-path',
        type=str,
        default=None,
        help='Path to precomputed head budget proportions JSON file (for nonuniform head budgets)'
    )
    parser.add_argument(
        '--max-ratio-per-head',
        type=float,
        default=1.0,
        help='Maximum ratio per head when using precomputed budgets (default: 1.0). '
             'If budgets would assign a higher ratio, proportions are blended towards uniform.'
    )

    # Logging arguments
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/qa_evaluation',
        help='Directory for logs (default: logs/qa_evaluation)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name for logging (default: None)'
    )

    args = parser.parse_args()

    # Load algorithm hyperparameter config (pass target_size for configs that use it)
    method_config = load_algorithm_config(args.algorithm_config, target_size=args.target_size)
    print(f"Loaded algorithm config from: {args.algorithm_config}")
    print(f"Available configs: {list(method_config.keys())}")

    methods_to_run = []
    if 'all' in args.methods:
        # Run all configurations in the config file, including 'original' if requested
        if 'original' in args.methods:
            methods_to_run.append('original')
        methods_to_run.extend(list(method_config.keys()))
    else:
        methods_to_run = args.methods
        # Validate that all requested configs exist
        invalid_configs = [m for m in methods_to_run if m != 'original' and m not in method_config]
        if invalid_configs:
            raise ValueError(
                f"Unknown configurations: {invalid_configs}. "
                f"Available in {args.algorithm_config}: {list(method_config.keys())}"
            )
        
    method_kwargs = {}
    for method_name in methods_to_run:
        if method_name == 'original':
            continue

        if method_name in method_config:
            algo_config = method_config[method_name]
            method_kwargs[method_name] = {k: v for k, v in algo_config.items()}

            # Inject chunking parameters if specified via CLI
            if args.chunking is not None and args.chunking != 'none':
                method_kwargs[method_name]['chunking'] = args.chunking
                method_kwargs[method_name]['chunk_size'] = args.chunk_size
                method_kwargs[method_name]['use_kv_based'] = bool(args.use_kv_based)

            # Inject precomputed budget path if specified via CLI
            if args.precomputed_budget_path is not None:
                method_kwargs[method_name]['precomputed_budget_path'] = args.precomputed_budget_path
                method_kwargs[method_name]['max_ratio_per_head'] = args.max_ratio_per_head

    # Load query generation config
    query_config = load_query_config(args.query_config)
    # Set verbose flag from CLI
    if args.verbose_logging:
        query_config.verbose = True
    print(f"Loaded query config from: {args.query_config}")
    print(f"  - methods: {[mc.method for mc in query_config.method_configs]}")
    print(f"  - fractions: {[mc.fraction for mc in query_config.method_configs]}")
    print(f"  - max_query_vectors_per_kv_head: {query_config.max_query_vectors_per_kv_head}")

    # Auto-detect device if not specified
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Require CUDA
    if args.device != 'cuda':
        raise RuntimeError(
            "CUDA is currently required for evaluation. "
            "Please run on a machine with a GPU."
        )

    # Generate experiment name based on target size if not provided
    if args.name is None:
        if args.chunking and args.chunking != 'none':
            args.name = f"target{args.target_size}_chunked_{args.chunking}"
        else:
            args.name = f"target{args.target_size}"

    # Print chunking configuration
    if args.chunking and args.chunking != 'none':
        print(f"\nChunked compaction enabled:")
        print(f"  - strategy: {args.chunking}")
        if args.chunking == 'fixed':
            print(f"  - chunk_size: {args.chunk_size}")
        print(f"  - use_kv_based: {bool(args.use_kv_based)}")

    # Initialize evaluator
    # Convert --long-context flag to max_model_len value
    max_model_len = 131072 if args.long_context else None
    evaluator = QAEvaluator(
        model_name=args.model_name,
        device=args.device,
        max_model_len=max_model_len,
    )

    # Run evaluation
    evaluator.run_evaluation(
        dataset_name=args.dataset_name,
        compaction_methods=methods_to_run,
        target_size=args.target_size,
        n_articles=args.n_articles,
        start_article=args.start_article,
        n_questions_per_article=args.n_questions_per_article,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        query_config=query_config,
        compute_stats=bool(args.compute_stats),
        verbose_logging=bool(args.verbose_logging),
        compute_perplexity=bool(args.compute_perplexity or args.perplexity_only),
        perplexity_only=bool(args.perplexity_only),
        method_kwargs=method_kwargs,
        log_dir=args.log_dir,
        experiment_name=args.name,
        algorithm_config_file=args.algorithm_config,
        query_config_file=args.query_config,
        ignore_article_indices=args.ignore_article_indices
    )

if __name__ == "__main__":
    main()

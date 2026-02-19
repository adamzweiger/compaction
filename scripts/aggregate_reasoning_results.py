# scripts/aggregate_reasoning_results.py
"""
Aggregate reasoning evaluation results from multiple JSON files.

Groups results by experiment_name (which encodes the method configuration)
and aggregates statistics across multiple runs of the same experiment.

Usage:
    python scripts/aggregate_reasoning_results.py [--eval-dir DIR]

    --eval-dir: Directory containing reasoning evaluation JSON files
                (default: logs/reasoning_evaluation/qwen-reasoning)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


# Default directory containing reasoning evaluation results
DEFAULT_EVAL_DIR = "logs/reasoning_evaluation/qwen-reasoning"


def aggregate_reasoning_stats(stats_list):
    """
    Aggregate statistics for a single method across multiple runs.

    For reasoning evaluations, we aggregate:
    - Total problems and correct counts (sum)
    - Accuracy (recomputed from totals)
    - Token counts (weighted average by num problems)
    - Times (weighted average by num problems)
    - Compaction stats (weighted average by num problems)

    Args:
        stats_list: List of (overall_stats, config, n_problems) tuples

    Returns:
        Aggregated statistics dictionary
    """
    if not stats_list:
        return {}

    aggregated = {}

    # Sum total problems and correct
    total_problems = sum(s['total_problems'] for s, _, _ in stats_list)
    total_correct = sum(s['correct'] for s, _, _ in stats_list)

    aggregated['total_problems'] = total_problems
    aggregated['correct'] = total_correct
    aggregated['accuracy'] = total_correct / total_problems if total_problems > 0 else 0.0
    aggregated['num_runs'] = len(stats_list)

    # Weighted averages (by number of problems in each run)
    if total_problems > 0:
        # Token statistics
        aggregated['avg_reasoning_tokens'] = sum(
            s['avg_reasoning_tokens'] * s['total_problems']
            for s, _, _ in stats_list
        ) / total_problems

        # Time statistics
        aggregated['avg_generation_time'] = sum(
            s['avg_generation_time'] * s['total_problems']
            for s, _, _ in stats_list
        ) / total_problems

        # Compaction-specific stats (only for compaction mode)
        compaction_stats = [(s, c, n) for s, c, n in stats_list if 'avg_compaction_time' in s]
        if compaction_stats:
            compaction_total = sum(s['total_problems'] for s, _, _ in compaction_stats)
            if compaction_total > 0:
                aggregated['avg_compaction_time'] = sum(
                    s['avg_compaction_time'] * s['total_problems']
                    for s, _, _ in compaction_stats
                ) / compaction_total

                aggregated['avg_compaction_ratio'] = sum(
                    s['avg_compaction_ratio'] * s['total_problems']
                    for s, _, _ in compaction_stats
                ) / compaction_total

    # Include config info from first run (should be same across runs)
    _, first_config, _ = stats_list[0]
    aggregated['mode'] = first_config.get('mode')
    aggregated['method'] = first_config.get('method')
    aggregated['target_size'] = first_config.get('target_size')
    aggregated['max_reasoning_tokens'] = first_config.get('max_reasoning_tokens')
    aggregated['first_phase_tokens'] = first_config.get('first_phase_tokens')
    aggregated['second_phase_tokens'] = first_config.get('second_phase_tokens')

    return aggregated


def aggregate_per_problem_results(all_results_lists):
    """
    Aggregate per-problem results across multiple runs.

    For each problem, computes statistics across all runs.

    Args:
        all_results_lists: List of results lists from each run

    Returns:
        Dict mapping problem_id to aggregated per-problem stats
    """
    # Group results by problem_id
    problem_results = defaultdict(list)
    for results_list in all_results_lists:
        for result in results_list:
            problem_id = result['problem_id']
            problem_results[problem_id].append(result)

    aggregated = {}
    for problem_id, results in problem_results.items():
        n_runs = len(results)
        n_correct = sum(1 for r in results if r.get('is_correct', False))

        aggregated[problem_id] = {
            'problem_title': results[0].get('problem_title'),
            'ground_truth': results[0].get('ground_truth'),
            'n_runs': n_runs,
            'n_correct': n_correct,
            'accuracy': n_correct / n_runs if n_runs > 0 else 0.0,
            'model_answers': [r.get('model_answer') for r in results],
            'avg_reasoning_tokens': sum(
                r.get('total_reasoning_tokens', r.get('reasoning_tokens', 0))
                for r in results
            ) / n_runs if n_runs > 0 else 0,
            'avg_generation_time': sum(
                r.get('generation_time', 0) for r in results
            ) / n_runs if n_runs > 0 else 0,
        }

        # Add compaction-specific stats if present
        compaction_results = [r for r in results if 'compaction_time' in r]
        if compaction_results:
            aggregated[problem_id]['avg_compaction_time'] = sum(
                r['compaction_time'] for r in compaction_results
            ) / len(compaction_results)
            aggregated[problem_id]['avg_compaction_ratio'] = sum(
                r.get('compaction_ratio', 0) for r in compaction_results
            ) / len(compaction_results)

    return aggregated


def main(eval_dir=None):
    """
    Main function to aggregate reasoning evaluation results.

    Args:
        eval_dir: Directory containing JSON result files
    """
    eval_path = Path(eval_dir or DEFAULT_EVAL_DIR)

    if not eval_path.exists():
        print(f"Error: Directory {eval_path} does not exist")
        return

    # Find all JSON files (excluding aggregated results)
    json_files = [f for f in eval_path.glob("*.json") if not f.name.startswith("aggregated")]
    print(f"Found {len(json_files)} JSON files in {eval_path}")

    if not json_files:
        print("No JSON files found")
        return

    # Group files by experiment_name
    experiment_data = defaultdict(list)

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            experiment_name = data.get('experiment_name', json_file.stem)
            config = data.get('config', {})
            overall_stats = data.get('overall_stats', {})
            results = data.get('results', [])

            n_problems = overall_stats.get('total_problems', len(results))

            experiment_data[experiment_name].append({
                'file': json_file.name,
                'overall_stats': overall_stats,
                'config': config,
                'results': results,
                'n_problems': n_problems,
            })

            print(f"  {json_file.name}: {experiment_name} ({n_problems} problems)")

        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            continue

    print(f"\nFound {len(experiment_data)} unique experiments")

    # Aggregate statistics for each experiment
    aggregated_results = {}
    per_problem_results = {}

    for experiment_name, runs in sorted(experiment_data.items()):
        print(f"\nAggregating {len(runs)} run(s) for: {experiment_name}")

        # Prepare stats list for aggregation
        stats_list = [
            (run['overall_stats'], run['config'], run['n_problems'])
            for run in runs
        ]

        # Aggregate overall stats
        aggregated_results[experiment_name] = aggregate_reasoning_stats(stats_list)

        # Aggregate per-problem results
        all_results_lists = [run['results'] for run in runs]
        per_problem_results[experiment_name] = aggregate_per_problem_results(all_results_lists)

    # Write aggregated results
    output_file = eval_path / "aggregated_results.json"
    with open(output_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    print(f"\nAggregated results written to {output_file}")

    # Write per-problem results
    per_problem_file = eval_path / "aggregated_per_problem.json"
    with open(per_problem_file, 'w') as f:
        json.dump(per_problem_results, f, indent=2)
    print(f"Per-problem results written to {per_problem_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<35} {'Runs':>5} {'Acc':>8} {'Correct':>10} {'Avg Tokens':>12}")
    print("-" * 70)

    for name, stats in sorted(aggregated_results.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True):
        accuracy = stats.get('accuracy', 0)
        n_runs = stats.get('num_runs', 1)
        correct = stats.get('correct', 0)
        total = stats.get('total_problems', 0)
        avg_tokens = stats.get('avg_reasoning_tokens', 0)

        print(f"{name:<35} {n_runs:>5} {accuracy:>7.1%} {correct:>4}/{total:<5} {avg_tokens:>12.0f}")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate reasoning evaluation results from multiple JSON files."
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=DEFAULT_EVAL_DIR,
        help=f"Directory containing reasoning evaluation JSON files (default: {DEFAULT_EVAL_DIR})"
    )
    args = parser.parse_args()
    main(eval_dir=args.eval_dir)

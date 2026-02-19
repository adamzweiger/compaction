# scripts/aggregate_qa_results.py
"""
Aggregate QA evaluation results from multiple JSON files.

Usage:
    python scripts/aggregate_qa_results.py [--max_articles N] [--group-by-target-size]
    
    --max_articles: Maximum number of articles to include (0-indexed, so --max_articles=10 means articles 0-9).
                    If not specified, processes all articles.
    --group-by-target-size: If specified, groups results by target_size in addition to query_config and method.
                            This keeps different target_size experiments separate instead of aggregating them together.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Put directory containing qa_evaluation results. This script will aggregate them into 1 json file.
EVAL_DIR = "logs/qa_evaluation/gemma12b-lh5"


def extract_budget_path_id(hyperparameters):
    """
    Extract a unique identifier from precomputed_budget_path in hyperparameters.

    e.g., from "logs/budget_optimization/Qwen3-4B/optimized_20251223_234504/optimized_t0.02.json"
    extract "optimized_20251223_234504/optimized_t0.02"

    Args:
        hyperparameters: Dict of method hyperparameters from the JSON file

    Returns:
        Budget path identifier string, or None if not found
    """
    if not hyperparameters:
        return None

    for method_name, kwargs in hyperparameters.items():
        if isinstance(kwargs, dict) and 'precomputed_budget_path' in kwargs:
            budget_path = kwargs['precomputed_budget_path']
            # Extract part after model name (e.g., "Qwen3-4B/") and before ".json"
            # Match pattern: .../ModelName/rest_of_path.json -> extract rest_of_path (without .json)
            match = re.search(r'/[^/]+/([^/]+/[^/]+)\.json$', budget_path)
            if match:
                return match.group(1).replace('/', '_')

    return None


def clean_method_name(method_name, query_config):
    """
    Clean up method name by removing 'self_study' prefix for certain method types.

    If the method contains "summarize", "original", or "no_context",
    remove the query_config prefix (typically "self_study") from the final name.

    Args:
        method_name: The raw method name
        query_config: The query config prefix

    Returns:
        Cleaned method name
    """
    # Check if method contains any of the special keywords
    special_keywords = ["summarize", "original", "no_context"]
    contains_special = any(keyword in method_name.lower() for keyword in special_keywords)

    if contains_special:
        # Remove the query_config prefix (e.g., "self_study_")
        if method_name.startswith(f"{query_config}_"):
            return method_name[len(query_config) + 1:]  # +1 for the underscore

    return method_name


def get_aggregation_type(stat_name):
    """Determine how to aggregate a statistic based on its name."""
    if stat_name.startswith("mean_"):
        return "mean"
    elif stat_name.startswith("max_"):
        return "max"
    elif stat_name.startswith("min_"):
        return "min"
    elif stat_name.startswith("rms_"):
        return "rms"
    else:
        # Default to mean for other stats
        return "mean"


def aggregate_nested_stats(all_stats, weight_key="num_articles"):
    """
    Aggregate nested statistics dictionaries (like train_stats and test_stats).

    Args:
        all_stats: List of tuples (stats_dict, weight)
        weight_key: Key to use for weighting

    Returns:
        Aggregated statistics dictionary
    """
    if not all_stats:
        return {}

    # Get all possible keys from all stats dicts
    all_keys = set()
    for stats_dict, _ in all_stats:
        if stats_dict:
            all_keys.update(stats_dict.keys())

    aggregated = {}
    for key in all_keys:
        # Collect values and weights for this key
        values_and_weights = []
        for stats_dict, weight in all_stats:
            if stats_dict and key in stats_dict:
                values_and_weights.append((stats_dict[key], weight))

        if not values_and_weights:
            continue

        agg_type = get_aggregation_type(key)

        if agg_type == "mean":
            # Weighted mean
            total_weight = sum(w for _, w in values_and_weights)
            if total_weight > 0:
                aggregated[key] = sum(v * w for v, w in values_and_weights) / total_weight
        elif agg_type == "max":
            # Maximum value
            aggregated[key] = max(v for v, _ in values_and_weights)
        elif agg_type == "min":
            # Minimum value
            aggregated[key] = min(v for v, _ in values_and_weights)
        elif agg_type == "rms":
            # Root mean square (weighted)
            total_weight = sum(w for _, w in values_and_weights)
            if total_weight > 0:
                aggregated[key] = (sum(v**2 * w for v, w in values_and_weights) / total_weight) ** 0.5

    return aggregated


def aggregate_method_stats(method_stats_list):
    """
    Aggregate statistics for a single method across multiple files.

    Args:
        method_stats_list: List of stats dictionaries for the same method

    Returns:
        Aggregated statistics dictionary
    """
    if not method_stats_list:
        return {}

    # Initialize aggregated stats
    aggregated = {}

    # Sum total_correct and total_questions
    aggregated["total_correct"] = sum(s.get("total_correct", 0) for s in method_stats_list)
    aggregated["total_questions"] = sum(s.get("total_questions", 0) for s in method_stats_list)
    aggregated["num_articles"] = sum(s.get("num_articles", 0) for s in method_stats_list)

    # Calculate overall accuracy
    if aggregated["total_questions"] > 0:
        aggregated["overall_accuracy"] = aggregated["total_correct"] / aggregated["total_questions"]
    else:
        aggregated["overall_accuracy"] = 0.0

    # Weighted averages
    total_articles = aggregated["num_articles"]

    if total_articles > 0:
        # Average perplexity (weighted by num_articles with perplexity)
        # Support both old field name (overall_avg_perplexity) and
        # new field name (overall_avg_perplexity) for backwards compatibility
        perplexity_stats = []
        for s in method_stats_list:
            num_perplexity_articles = s.get("num_articles_with_perplexity", 0)
            if num_perplexity_articles > 0:
                # Try new name first, then fall back to old name
                perplexity = s.get("overall_avg_perplexity") or s.get("overall_avg_perplexity", 0)
                perplexity_stats.append((perplexity, num_perplexity_articles))
        if perplexity_stats:
            total_perplexity_articles = sum(w for _, w in perplexity_stats)
            if total_perplexity_articles > 0:
                aggregated["overall_avg_perplexity"] = sum(
                    p * w for p, w in perplexity_stats
                ) / total_perplexity_articles
                aggregated["num_articles_with_perplexity"] = total_perplexity_articles

        # Average log perplexity (weighted by num_articles with perplexity)
        log_perplexity_stats = []
        for s in method_stats_list:
            num_perplexity_articles = s.get("num_articles_with_perplexity", 0)
            if num_perplexity_articles > 0:
                log_perplexity = s.get("overall_avg_log_perplexity")
                if log_perplexity is not None:
                    log_perplexity_stats.append((log_perplexity, num_perplexity_articles))
        if log_perplexity_stats:
            total_log_perplexity_articles = sum(w for _, w in log_perplexity_stats)
            if total_log_perplexity_articles > 0:
                aggregated["overall_avg_log_perplexity"] = sum(
                    p * w for p, w in log_perplexity_stats
                ) / total_log_perplexity_articles

        # Average target_size_param (weighted by num_articles)
        if all("avg_target_size_param" in s for s in method_stats_list):
            aggregated["avg_target_size_param"] = sum(
                s["avg_target_size_param"] * s.get("num_articles", 0) for s in method_stats_list
            ) / total_articles

        # Aggregate timing statistics (weighted by num_articles)
        timing_keys = [
            "avg_extraction_time_per_article",
            "avg_compaction_time_per_article",
            "avg_query_generation_time_per_article",
            "avg_train_stats_time_per_article",
            "avg_test_stats_time_per_article"
        ]
        for key in timing_keys:
            if all(key in s for s in method_stats_list):
                aggregated[key] = sum(
                    s[key] * s.get("num_articles", 0) for s in method_stats_list
                ) / total_articles

    # Sum total times
    total_time_keys = [
        "total_extraction_time",
        "total_compaction_time",
        "total_query_generation_time",
        "total_generation_time",
        "total_train_stats_time",
        "total_test_stats_time",
        "total_generated_tokens"
    ]
    for key in total_time_keys:
        if any(key in s for s in method_stats_list):
            aggregated[key] = sum(s.get(key, 0) for s in method_stats_list)

    # Average per-question/per-token times (weighted by total_questions or total_generated_tokens)
    total_questions = aggregated["total_questions"]
    if total_questions > 0:
        if all("avg_generation_time_per_question" in s for s in method_stats_list):
            aggregated["avg_generation_time_per_question"] = sum(
                s.get("avg_generation_time_per_question", 0) * s.get("total_questions", 0)
                for s in method_stats_list
            ) / total_questions

    total_tokens = aggregated.get("total_generated_tokens", 0)
    if total_tokens > 0:
        if all("avg_time_per_token" in s for s in method_stats_list):
            aggregated["avg_time_per_token"] = sum(
                s.get("avg_time_per_token", 0) * s.get("total_generated_tokens", 0)
                for s in method_stats_list
            ) / total_tokens
            # Calculate avg_tokens_per_second as inverse of avg_time_per_token
            if aggregated["avg_time_per_token"] > 0:
                aggregated["avg_tokens_per_second"] = 1.0 / aggregated["avg_time_per_token"]

    # Aggregate parse rate (weighted by total_questions)
    if total_questions > 0:
        aggregated["total_parseable"] = sum(s.get("total_parseable", 0) for s in method_stats_list)
        aggregated["overall_parse_rate"] = aggregated["total_parseable"] / total_questions

    # Aggregate train stats
    train_stats_with_weights = [
        (s.get("overall_all_head_train_stats"), s.get("num_articles", 0))
        for s in method_stats_list
        if "overall_all_head_train_stats" in s
    ]
    if train_stats_with_weights:
        aggregated["overall_all_head_train_stats"] = aggregate_nested_stats(train_stats_with_weights)

    # Aggregate test stats
    test_stats_with_weights = [
        (s.get("overall_all_head_test_stats"), s.get("num_articles", 0))
        for s in method_stats_list
        if "overall_all_head_test_stats" in s
    ]
    if test_stats_with_weights:
        aggregated["overall_all_head_test_stats"] = aggregate_nested_stats(test_stats_with_weights)

    # Aggregate memory stats (weighted by num_articles)
    if total_articles > 0:
        memory_stats_list = [(s.get("memory_stats"), s.get("num_articles", 0))
                             for s in method_stats_list if "memory_stats" in s]
        if memory_stats_list:
            aggregated["memory_stats"] = {}
            # Get all keys from all memory_stats dicts
            all_memory_keys = set()
            for ms, _ in memory_stats_list:
                if ms:
                    all_memory_keys.update(ms.keys())

            for key in all_memory_keys:
                values_weights = [(ms.get(key), w) for ms, w in memory_stats_list if ms and key in ms]
                if values_weights:
                    total_weight = sum(w for _, w in values_weights)
                    if total_weight > 0:
                        aggregated["memory_stats"][key] = sum(
                            v * w for v, w in values_weights
                        ) / total_weight

    # Aggregate article compaction stats (weighted by num_articles)
    if total_articles > 0:
        article_comp_stats_list = [(s.get("article_compaction_stats"), s.get("num_articles", 0))
                                   for s in method_stats_list if "article_compaction_stats" in s]
        if article_comp_stats_list:
            aggregated["article_compaction_stats"] = {}
            # Get all keys from all article_compaction_stats dicts
            all_comp_keys = set()
            for acs, _ in article_comp_stats_list:
                if acs:
                    all_comp_keys.update(acs.keys())

            for key in all_comp_keys:
                values_weights = [(acs.get(key), w) for acs, w in article_comp_stats_list if acs and key in acs]
                if values_weights:
                    total_weight = sum(w for _, w in values_weights)
                    if total_weight > 0:
                        aggregated["article_compaction_stats"][key] = sum(
                            v * w for v, w in values_weights
                        ) / total_weight

    return aggregated


def main(max_articles=None, group_by_target_size=False):
    """
    Main function to aggregate QA evaluation results.

    Args:
        max_articles: Maximum number of articles to include (0-indexed, so max_articles=10 means articles 0-9).
                     If None, processes all articles.
        group_by_target_size: If True, groups results by target_size in addition to query_config and method.
                              This keeps different target_size experiments separate.
    """
    eval_path = Path(EVAL_DIR)

    if not eval_path.exists():
        print(f"Error: Directory {EVAL_DIR} does not exist")
        return

    # Find all JSON files
    json_files = list(eval_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {EVAL_DIR}")

    if max_articles is not None:
        print(f"Limiting to first {max_articles} articles (indices 0-{max_articles-1})")
    else:
        print("Processing all articles")

    if not json_files:
        print("No JSON files found")
        return

    # First pass: collect all target sizes to determine if we need to auto-group
    all_target_sizes = set()
    skipped_files = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check article indices if max_articles is specified
            if max_articles is not None:
                config = data.get("config", {})
                article_indices = config.get("article_indices", [])

                # Skip files that contain articles outside the specified range
                if article_indices and max(article_indices) >= max_articles:
                    skipped_files += 1
                    del data
                    continue

            # Collect avg_target_size_param from stats
            overall_stats = data.get("overall_stats", {})
            for method_name, stats in overall_stats.items():
                avg_target_size_param = stats.get("avg_target_size_param")
                if avg_target_size_param is not None:
                    all_target_sizes.add(avg_target_size_param)

            del data

        except Exception as e:
            print(f"Error scanning {json_file.name}: {e}")
            continue

    # Determine if we should group by target size
    auto_group_by_target_size = len(all_target_sizes) > 1
    should_group_by_target_size = group_by_target_size or auto_group_by_target_size

    if auto_group_by_target_size and not group_by_target_size:
        print(f"\nDetected {len(all_target_sizes)} different target sizes: {sorted(all_target_sizes)}")
        print("Automatically grouping by target size and creating separate output files\n")

    # Second pass: collect stats
    method_query_stats = defaultdict(list)
    all_query_configs = set()
    all_ignore_article_indices = set()
    skipped_files = 0

    for json_file in json_files:
        try:
            # Extract only what we need to minimize memory usage
            # The 'results' key is very large (~191MB) but we don't need it
            with open(json_file, 'r') as f:
                data = json.load(f)

            config = data.get("config", {})

            # Check article indices if max_articles is specified
            if max_articles is not None:
                article_indices = config.get("article_indices", [])

                # Skip files that contain articles outside the specified range
                if article_indices and max(article_indices) >= max_articles:
                    skipped_files += 1
                    print(f"Skipping {json_file.name} (contains articles outside range 0-{max_articles-1})")
                    del data
                    continue

            print(f"Processing {json_file.name}...")
            query_config_file = data.get("query_config_file", "unknown")
            all_query_configs.add(query_config_file)
            overall_stats = data.get("overall_stats", {})

            # Extract budget_path_id for uniqueness
            hyperparameters = data.get("hyperparameters", {})
            budget_path_id = extract_budget_path_id(hyperparameters)

            # Extract ignore_article_indices for grouping (default False if not present)
            ignore_article_indices = config.get("ignore_article_indices", False)
            all_ignore_article_indices.add(ignore_article_indices)

            # Extract stats and create new dicts to avoid holding references to original data
            # This allows the large 'results' key to be garbage collected
            for method_name, stats in overall_stats.items():
                # Determine target_size for grouping
                # Priority: 1) avg_target_size_param from stats, 2) target_size from config
                avg_target_size_param = stats.get("avg_target_size_param")
                if avg_target_size_param is not None:
                    # Round to avoid floating point precision issues
                    target_size = round(avg_target_size_param, 6)
                elif should_group_by_target_size:
                    target_size = config.get("target_size", "unknown")
                else:
                    target_size = None

                # Determine key based on whether we need to group by target size
                # Include budget_path_id for uniqueness when precomputed budgets are used
                # Include ignore_article_indices for distinguishing methods
                if should_group_by_target_size and target_size is not None:
                    key = (query_config_file, method_name, target_size, budget_path_id, ignore_article_indices)
                else:
                    key = (query_config_file, method_name, budget_path_id, ignore_article_indices)

                # Create a new dict from the stats to break reference to original data
                # Use dict() constructor which creates a shallow copy of the top-level dict
                # For nested structures, we need to recursively copy
                stats_copy = json.loads(json.dumps(stats))  # Serialize/deserialize to break all references
                method_query_stats[key].append(stats_copy)

            # Explicitly delete the large data structure to free memory immediately
            del data
            del overall_stats

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    if skipped_files > 0:
        print(f"\nSkipped {skipped_files} file(s) that contained articles outside the specified range")

    # Determine if we need to include query_config or target_size in the output names
    # Check if all methods appear with only one query_config or if there are multiple query_configs
    method_to_query_configs = defaultdict(set)
    method_to_target_sizes = defaultdict(set)
    method_to_budget_paths = defaultdict(set)
    method_to_ignore_article_indices = defaultdict(set)

    if should_group_by_target_size:
        for (query_config, method, target_size, budget_path_id, ignore_article_indices), _ in method_query_stats.items():
            method_to_query_configs[method].add(query_config)
            method_to_target_sizes[method].add(target_size)
            method_to_budget_paths[method].add(budget_path_id)
            method_to_ignore_article_indices[method].add(ignore_article_indices)
    else:
        for (query_config, method, budget_path_id, ignore_article_indices), _ in method_query_stats.items():
            method_to_query_configs[method].add(query_config)
            method_to_budget_paths[method].add(budget_path_id)
            method_to_ignore_article_indices[method].add(ignore_article_indices)

    needs_query_prefix = True # any(len(configs) > 1 for configs in method_to_query_configs.values())
    needs_budget_suffix = any(len(paths) > 1 for paths in method_to_budget_paths.values())
    needs_ignore_suffix = any(len(vals) > 1 for vals in method_to_ignore_article_indices.values())

    # Group results by target size if needed
    if should_group_by_target_size:
        # Group aggregated results by target size
        results_by_target_size = defaultdict(dict)

        for (query_config, method_name, target_size, budget_path_id, ignore_article_indices), stats_list in method_query_stats.items():
            # Determine the output key name (without target size suffix since it's in the filename)
            if needs_query_prefix:
                combined_name = f"{query_config}_{method_name}"
                output_key = clean_method_name(combined_name, query_config)
            else:
                output_key = clean_method_name(method_name, query_config)

            # Add budget_path_id suffix if there are multiple budget paths for this method
            if needs_budget_suffix and budget_path_id:
                output_key = f"{output_key}_{budget_path_id}"

            # Add ignore_article_indices suffix if there are multiple values for this method
            if needs_ignore_suffix and ignore_article_indices:
                output_key = f"{output_key}_ignore-article-idx"

            print(f"Aggregating {len(stats_list)} results for: {output_key} (target_size={target_size})")
            results_by_target_size[target_size][output_key] = aggregate_method_stats(stats_list)

        # Write separate files for each target size
        output_files = []
        for target_size, aggregated_results in sorted(results_by_target_size.items()):
            output_file = eval_path / f"aggregated_results_t{target_size}.json"
            with open(output_file, 'w') as f:
                json.dump(aggregated_results, f, indent=2)
            output_files.append(output_file)
            print(f"\nAggregated results for target_size={target_size} written to {output_file}")

        # Print summary for all target sizes
        print("\n=== Summary ===")
        for target_size, aggregated_results in sorted(results_by_target_size.items()):
            print(f"\nTarget Size: {target_size}")
            for method_name, stats in sorted(aggregated_results.items()):
                accuracy = stats.get("overall_accuracy", 0)
                num_articles = stats.get("num_articles", 0)
                print(f"  {method_name}:")
                print(f"    Articles: {num_articles}")
                print(f"    Accuracy: {accuracy:.4f} ({stats.get('total_correct', 0)}/{stats.get('total_questions', 0)})")
    else:
        # Original behavior: single output file
        aggregated_results = {}
        for (query_config, method_name, budget_path_id, ignore_article_indices), stats_list in method_query_stats.items():
            # Determine the output key name
            if needs_query_prefix:
                combined_name = f"{query_config}_{method_name}"
                output_key = clean_method_name(combined_name, query_config)
            else:
                output_key = clean_method_name(method_name, query_config)

            # Add budget_path_id suffix if there are multiple budget paths for this method
            if needs_budget_suffix and budget_path_id:
                output_key = f"{output_key}_{budget_path_id}"

            # Add ignore_article_indices suffix if there are multiple values for this method
            if needs_ignore_suffix and ignore_article_indices:
                output_key = f"{output_key}_ignore-article-idx"

            print(f"Aggregating {len(stats_list)} results for: {output_key}")
            aggregated_results[output_key] = aggregate_method_stats(stats_list)

        # Write aggregated results
        output_file = eval_path / "aggregated_results.json"
        with open(output_file, 'w') as f:
            json.dump(aggregated_results, f, indent=2)

        print(f"\nAggregated results written to {output_file}")

        # Print summary
        print("\n=== Summary ===")
        for method_name, stats in sorted(aggregated_results.items()):
            accuracy = stats.get("overall_accuracy", 0)
            num_articles = stats.get("num_articles", 0)
            print(f"{method_name}:")
            print(f"  Articles: {num_articles}")
            print(f"  Accuracy: {accuracy:.4f} ({stats.get('total_correct', 0)}/{stats.get('total_questions', 0)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate QA evaluation results from multiple JSON files."
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=None,
        help="Maximum number of articles to include (0-indexed, so --max_articles=10 means articles 0-9). "
             "If not specified, processes all articles."
    )
    parser.add_argument(
        "--group-by-target-size",
        action="store_true",
        help="If specified, groups results by target_size in addition to query_config and method. "
             "This keeps different target_size experiments separate instead of aggregating them together."
    )
    args = parser.parse_args()
    main(max_articles=args.max_articles, group_by_target_size=args.group_by_target_size)

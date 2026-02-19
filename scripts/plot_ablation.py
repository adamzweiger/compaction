# scripts/plot_metric_vs_target_size.py
"""
Plot metrics vs target size from multiple aggregated_results_t{x}.json files.
Creates publication-quality plots using seaborn/matplotlib showing how different
algorithms perform across different target sizes.

Usage:
    python scripts/plot_metric_vs_target_size.py

    Easily modify METRIC, LOG_SCALE, and OUTPUT_FORMAT below to customize the plot.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory containing aggregated_results_t*.json files
EVAL_DIR = "logs/qa_evaluation"

# Directory to save output plots
OUTPUT_DIR = "logs/qa_evaluation"

# Metric to plot on y-axis
# Choose from common metrics or nested metrics (see lists below)
# METRIC = "overall_accuracy"
METRIC = "overall_avg_perplexity"

# Use log scale for y-axis
LOG_SCALE = False

# Output format: "png", "pdf", or "both"
OUTPUT_FORMAT = "png"

# Set to True to create a multi-panel plot with multiple metrics
MULTI_METRIC_PLOT = False

# Figure size (width, height) in inches
FIGURE_SIZE = (12, 7)

# DPI for output images
DPI = 300

# Add jitter to points to make overlapping points visible
# Set to 0 to disable jitter
JITTER_AMOUNT = 0  # Fraction of data range to jitter

# X-axis tick positions to show (set to None to show all ticks)
X_TICKS = [0.01, 0.02, 0.05]

# Target sizes to load from aggregated_results_t{x}.json files
# Set to None to load all available files, or specify a list like [0.01, 0.02, 0.05, 0.1, 0.2, 0.99]
TARGET_SIZES_TO_LOAD = [0.01, 0.02, 0.05]

# Baseline methods to plot as horizontal reference lines (not as points)
BASELINE_METHODS = ["original", "no_context"]
BASELINE_LABELS = {
    "original": "Original Cache",
    "no_context": "No Context"
}

# Method configuration: (method_name, display_name, color)
# Methods are plotted in this order, and the legend follows this order.
# Set to None to include all methods (with auto-generated colors).
# Methods with the same display name will share the same color (first one wins).
METHOD_CONFIG = [
    # Ablation study methods
    # Base method (full system)
    ("ss-plus-repeat_omp_nnls0_-inf_7_drop-7_lsq_progressive_on-policy_Qwen3-4B_optimized_agnostic", "Full Method", "#D4A017"),
    # Ablations
    # ("ss-plus-repeat_omp_nnls0_-inf_7_drop-7_lsq_progressive_on-policy_Qwen3-4B_optimized_agnostic_ignore-article-idx", "- Article Indices", "#A23B72"),  # magenta
    ("repeat_omp_nnls0_-inf_7_drop-7_lsq_progressive_on-policy_Qwen3-4B_optimized_agnostic", "No Self-Study (repeat-prefill)", "#52b788"),  # orange
    ("ss-plus-repeat_omp_nnls0_-inf_7_drop-7_lsq_progressive_on-policy", "No Head Budget (uniform)", "#C73E1D"),  # red
    ("ss-plus-repeat_omp_nnls0_-inf_7_drop-7_lsq_progressive_Qwen3-4B_optimized_agnostic", "No On-Policy", "#710627"),  # dark brown
    ("ss-plus-repeat_omp_nnls0_-inf_7_drop-7_direct_progressive_on-policy_Qwen3-4B_optimized_agnostic", "No learned values", "#0A2239"),
    ("ss-plus-repeat_omp_nnls0_-inf_7_drop-7_zerobeta_lsq_progressive_on-policy_Qwen3-4B_optimized_agnostic", "No biases", "#2E86AB"),
    ("ss-plus-repeat_omp_nnls0_-inf_7_drop-7_zerobeta_direct_progressive_on-policy_Qwen3-4B_optimized_agnostic", "No learned values/biases", "#3B1F2B"),
]

# Special items that appear at the end of the legend (with their colors)
SPECIAL_LEGEND_ITEMS = [
    ("Original Cache", "black"),
    ("No Context", "gray"),
]

# Build helper dicts from METHOD_CONFIG
def _build_method_lookups():
    """Build lookup dicts from METHOD_CONFIG."""
    if METHOD_CONFIG is None:
        return None, {}, {}
    methods_to_include = [m[0] for m in METHOD_CONFIG]
    display_names = {m[0]: m[1] for m in METHOD_CONFIG}
    # For colors, use the first color defined for each display name
    colors = {}
    for _, display, color in METHOD_CONFIG:
        if display not in colors:
            colors[display] = color
    # Also map method names to their display name's color
    method_colors = {m[0]: colors[m[1]] for m in METHOD_CONFIG}
    return methods_to_include, display_names, method_colors

METHODS_TO_INCLUDE, METHOD_DISPLAY_NAMES, METHOD_COLORS = _build_method_lookups()

# ============================================================================
# AVAILABLE METRICS (for reference)
# ============================================================================

# Common metrics to plot
COMMON_METRICS = {
    "overall_accuracy": "Accuracy",
    "avg_compaction_ratio": "Average compaction ratio",
    "overall_avg_perplexity": "log(Perplexity)",
    "avg_compaction_time_per_article": "Avg Compaction Time per Article (s)",
    "avg_generation_time_per_question": "Avg Generation Time per Question (s)",
    "avg_tokens_per_second": "Avg Tokens per Second",
    "overall_parse_rate": "Overall Parse Rate",
}

# Nested metrics
NESTED_METRICS = {
    "train:mean_mean_output_cosine_sim": "Train: Mean Output Cosine Similarity",
    "test:mean_mean_output_cosine_sim": "Test: Mean Output Cosine Similarity",
    "train:mean_mean_output_mse": "Train: Mean Output MSE",
    "test:mean_mean_output_mse": "Test: Mean Output MSE",
    "train:mean_rms_output_relative_l2_error": "Train: RMS Output Relative L2 Error",
    "test:mean_rms_output_relative_l2_error": "Test: RMS Output Relative L2 Error",
    "train:mean_mean_sumexp_relative_error": "Train: Mean SumExp Relative Error",
    "test:mean_mean_sumexp_relative_error": "Test: Mean SumExp Relative Error",
}

ALL_METRICS = {**COMMON_METRICS, **NESTED_METRICS}


def get_method_prefix(method_name):
    """Extract the prefix (first word) from a method name."""
    if "_" in method_name:
        return method_name.split("_")[0]
    return method_name


def apply_jitter(x_values, y_values, jitter_amount, seed=None):
    """
    Apply jitter to data points to make overlapping points visible.

    Args:
        x_values: List of x coordinates
        y_values: List of y coordinates
        jitter_amount: Fraction of data range to use for jitter
        seed: Random seed for reproducibility (optional)

    Returns:
        Tuple of (jittered_x, jittered_y) as numpy arrays
    """
    if jitter_amount == 0 or len(x_values) == 0:
        return np.array(x_values), np.array(y_values)

    if seed is not None:
        np.random.seed(seed)

    x_arr = np.array(x_values)
    y_arr = np.array(y_values)

    # Calculate jitter magnitude based on data range
    x_range = x_arr.max() - x_arr.min() if len(x_arr) > 1 else 1.0
    y_range = y_arr.max() - y_arr.min() if len(y_arr) > 1 else 1.0

    # Add random jitter
    x_jitter = np.random.uniform(-jitter_amount * x_range, jitter_amount * x_range, len(x_arr))
    y_jitter = np.random.uniform(-jitter_amount * y_range, jitter_amount * y_range, len(y_arr))

    return x_arr + x_jitter, y_arr + y_jitter


def extract_metric(stats, metric_name):
    """
    Extract a metric value from stats dictionary.
    Supports nested metrics with prefix "train:" or "test:".
    For perplexity metrics, returns log(perplexity).

    Args:
        stats: Dictionary of statistics for a method
        metric_name: Name of metric to extract

    Returns:
        Metric value or None if not found
    """
    if metric_name.startswith("train:"):
        nested_metric = metric_name[6:]
        train_stats = stats.get("overall_all_head_train_stats", {})
        value = train_stats.get(nested_metric)
    elif metric_name.startswith("test:"):
        nested_metric = metric_name[5:]
        test_stats = stats.get("overall_all_head_test_stats", {})
        value = test_stats.get(nested_metric)
    else:
        value = stats.get(metric_name)

    # Apply log transform for perplexity metrics
    if value is not None and "perplexity" in metric_name.lower():
        value = np.log(value)

    return value


def load_aggregated_results(eval_dir):
    """
    Load aggregated_results_t{x}.json files from the evaluation directory.

    Args:
        eval_dir: Directory containing aggregated results files

    Returns:
        Dictionary mapping target_size -> {method_name: stats}
    """
    eval_path = Path(eval_dir)

    if not eval_path.exists():
        raise FileNotFoundError(f"Directory {eval_dir} does not exist")

    # Find all aggregated_results_t*.json files
    result_files = list(eval_path.glob("aggregated_results_t*.json"))

    if not result_files:
        raise FileNotFoundError(f"No aggregated_results_t*.json files found in {eval_dir}")

    results_by_target_size = {}

    for file_path in sorted(result_files):
        # Extract target size from filename: aggregated_results_t{size}.json
        filename = file_path.name
        if filename.startswith("aggregated_results_t") and filename.endswith(".json"):
            target_size_str = filename[len("aggregated_results_t"):-len(".json")]
            try:
                target_size = float(target_size_str)
            except ValueError:
                print(f"Warning: Could not parse target size from {filename}, skipping")
                continue

            # Filter by TARGET_SIZES_TO_LOAD if specified
            if TARGET_SIZES_TO_LOAD is not None and target_size not in TARGET_SIZES_TO_LOAD:
                continue

            with open(file_path, 'r') as f:
                data = json.load(f)

            results_by_target_size[target_size] = data
            print(f"Loaded {len(data)} methods from {filename} (target_size={target_size})")

    return results_by_target_size


def prepare_plot_data(results_by_target_size, metric):
    """
    Prepare data for plotting: organize by method name across target sizes.

    Args:
        results_by_target_size: Dict of target_size -> {method_name: stats}
        metric: Metric to extract

    Returns:
        Tuple of (data_by_method, baselines)
        - data_by_method: Dict mapping method_name -> {target_sizes: [...], values: [...]}
        - baselines: Dict mapping baseline_name -> metric_value
        Note: target_sizes are actually 1/avg_compaction_ratio values
    """
    data_by_method = defaultdict(lambda: {"target_sizes": [], "values": []})
    baselines = {}

    for target_size in sorted(results_by_target_size.keys()):
        methods_data = results_by_target_size[target_size]

        for method_name, stats in methods_data.items():
            value = extract_metric(stats, metric)

            # Check if this is a baseline method
            if method_name in BASELINE_METHODS:
                # Store baseline value (should be consistent across target sizes)
                if method_name not in baselines and value is not None:
                    baselines[method_name] = value
                continue  # Don't plot baselines as regular points

            # Filter methods based on METHODS_TO_INCLUDE
            if METHODS_TO_INCLUDE is not None and method_name not in METHODS_TO_INCLUDE:
                continue

            article_stats = stats.get("article_compaction_stats", {})
            compaction_ratio = article_stats.get("avg_article_compaction_ratio")

            if value is not None and compaction_ratio is not None and compaction_ratio != 0:
                # Use 1/avg_compaction_ratio as the x-axis value
                x_value = 1.0 / compaction_ratio
                data_by_method[method_name]["target_sizes"].append(x_value)
                data_by_method[method_name]["values"].append(value)

    return dict(data_by_method), baselines


def create_plot(data_by_method, metric, baselines=None, log_scale=False):
    """
    Create a seaborn/matplotlib plot.

    Args:
        data_by_method: Dict of method_name -> {target_sizes: [...], values: [...]}
        metric: Metric being plotted
        baselines: Dict of baseline_name -> metric_value (optional)
        log_scale: Whether to use log scale for y-axis

    Returns:
        Matplotlib figure and axes objects
    """
    if baselines is None:
        baselines = {}
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Get method names in the order defined by METHOD_CONFIG (or sorted if no config)
    if METHOD_CONFIG is not None:
        # Use the order from METHOD_CONFIG, filtering to methods that have data
        config_order = [m[0] for m in METHOD_CONFIG]
        method_names = [m for m in config_order if m in data_by_method]
        # Add any methods not in config at the end
        for m in sorted(data_by_method.keys()):
            if m not in method_names:
                method_names.append(m)
    else:
        method_names = sorted(data_by_method.keys())

    # Combine all summarization data points into one series
    summarize_points = {"target_sizes": [], "values": []}
    non_summarize_methods = []

    for method_name in method_names:
        if "summarize" in method_name.lower():
            method_data = data_by_method[method_name]
            summarize_points["target_sizes"].extend(method_data["target_sizes"])
            summarize_points["values"].extend(method_data["values"])
        else:
            non_summarize_methods.append(method_name)

    # Sort summarization points by target_size for proper line plotting
    if summarize_points["target_sizes"]:
        sorted_indices = sorted(range(len(summarize_points["target_sizes"])),
                                key=lambda i: summarize_points["target_sizes"][i])
        summarize_points["target_sizes"] = [summarize_points["target_sizes"][i] for i in sorted_indices]
        summarize_points["values"] = [summarize_points["values"][i] for i in sorted_indices]

    # Plot each method
    added_labels = set()  # Track which labels have been added to avoid duplicates

    # Default color for methods not in METHOD_COLORS
    default_color = '#761922'

    # Auto-generate colors for methods when METHOD_CONFIG is None
    if METHOD_CONFIG is None:
        # Use a colorblind-friendly palette with enough distinct colors
        palette = sns.color_palette("husl", n_colors=len(method_names))
        auto_colors = {method: palette[i] for i, method in enumerate(method_names)}
    else:
        auto_colors = {}

    # Get summarization color (use first summarize method's color, or default)
    summarize_color = None
    for method_name in method_names:
        if "summarize" in method_name.lower():
            if method_name in METHOD_COLORS:
                summarize_color = METHOD_COLORS[method_name]
                break
            elif method_name in auto_colors:
                summarize_color = auto_colors[method_name]
                break
    if summarize_color is None:
        summarize_color = default_color

    # Track rightmost points for each method (for adding text labels)
    rightmost_points = {}  # display_name -> (x, y, color)

    # Plot non-summarization methods
    for method_idx, method_name in enumerate(non_summarize_methods):
        method_data = data_by_method[method_name]

        # Get display name and color
        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        color = METHOD_COLORS.get(method_name, auto_colors.get(method_name, default_color))

        # Only add label if we haven't added it yet (for duplicate labels)
        if display_name in added_labels:
            label = None  # Don't add duplicate to legend
        else:
            label = display_name
            added_labels.add(display_name)

        # Apply jitter to make overlapping points visible
        x_plot, y_plot = apply_jitter(
            method_data["target_sizes"],
            method_data["values"],
            JITTER_AMOUNT,
            seed=method_idx  # Different seed per method for reproducibility
        )

        ax.plot(
            x_plot,
            y_plot,
            marker='o',
            linewidth=2,
            markersize=10,
            label=label,
            color=color,
            alpha=0.9,
            markeredgecolor='black',
            markeredgewidth=1.0
        )

        # Track rightmost point for this method
        if len(x_plot) > 0:
            max_x_idx = np.argmax(x_plot)
            rightmost_x, rightmost_y = x_plot[max_x_idx], y_plot[max_x_idx]
            # Only update if this is the rightmost we've seen for this display name
            if display_name not in rightmost_points or rightmost_x > rightmost_points[display_name][0]:
                rightmost_points[display_name] = (rightmost_x, rightmost_y, color)

    # Plot all summarization points as a single connected line
    if summarize_points["target_sizes"]:
        # Apply jitter to make overlapping points visible
        x_plot, y_plot = apply_jitter(
            summarize_points["target_sizes"],
            summarize_points["values"],
            JITTER_AMOUNT,
            seed=len(non_summarize_methods)  # Unique seed for summarization
        )

        ax.plot(
            x_plot,
            y_plot,
            marker='o',
            linewidth=2,
            markersize=10,
            label="Summarization (various prompts)",
            color=summarize_color,
            alpha=0.9,
            markeredgecolor='black',
            markeredgewidth=1.0
        )

        # Track rightmost point for summarization
        if len(x_plot) > 0:
            max_x_idx = np.argmax(x_plot)
            rightmost_points["Summarization"] = (x_plot[max_x_idx], y_plot[max_x_idx], summarize_color)

    # Add text labels to the right of each line's last point
    for display_name, (x, y, color) in rightmost_points.items():
        # Drop "Full Method" label slightly to avoid overlap
        y_offset = -3 if display_name == "Full Method" else 1 if display_name == "No On-Policy" else 0
        ax.annotate(
            display_name,
            xy=(x, y),
            xytext=(8, y_offset),  # 8 points to the right, with optional y offset
            textcoords='offset points',
            fontsize=12,
            color=color,
            fontweight='bold',
            va='center',
            ha='left'
        )

    # Get metric display name
    metric_display = ALL_METRICS.get(metric, metric)

    # Set labels
    ax.set_xlabel("Compacted Size", fontsize=18, fontweight='bold')
    ax.set_ylabel(metric_display, fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)

    # Set log scale if requested
    if log_scale:
        ax.set_yscale('log')

    # Add baseline reference lines
    baseline_colors = {'original': 'black', 'no_context': 'gray'}
    for baseline_name, baseline_value in baselines.items():
        label = BASELINE_LABELS.get(baseline_name, baseline_name)
        color = baseline_colors.get(baseline_name, 'black')
        ax.axhline(
            y=baseline_value,
            linestyle='--',
            linewidth=2,
            color=color,
            alpha=0.7,
            label=label,
            zorder=1  # Draw below the data points
        )

    # Legend is disabled since we have text labels on the graph
    # Uncomment below to re-enable the legend
    # # Add legend inside the plot area with custom ordering
    # # Get current handles and labels
    # handles, labels = ax.get_legend_handles_labels()
    #
    # # Define the desired order based on METHOD_CONFIG, then special items at the end
    # # Build order: unique display names from METHOD_CONFIG (in order), then special items
    # if METHOD_CONFIG is not None:
    #     display_name_order = list(dict.fromkeys(m[1] for m in METHOD_CONFIG))
    #     # Replace "Summarization" with "Summarization (various prompts)" in the order
    #     if "Summarization" in display_name_order:
    #         idx = display_name_order.index("Summarization")
    #         display_name_order[idx] = "Summarization (various prompts)"
    # else:
    #     display_name_order = list(dict.fromkeys(METHOD_DISPLAY_NAMES.values()))
    #
    # # Special items order from SPECIAL_LEGEND_ITEMS (baselines at the end)
    # special_items_order = [item[0] for item in SPECIAL_LEGEND_ITEMS]
    #
    # # Create a mapping from label to its sort key
    # def get_sort_key(label):
    #     # Check if it's in the display name order
    #     if label in display_name_order:
    #         return (0, display_name_order.index(label))
    #     # Check if it's a special item at the end
    #     if label in special_items_order:
    #         return (1, special_items_order.index(label))
    #     # Unknown items go between display names and special items
    #     return (0, len(display_name_order))
    #
    # # Sort handles and labels together
    # sorted_pairs = sorted(zip(handles, labels), key=lambda x: get_sort_key(x[1]))
    # if sorted_pairs:
    #     handles, labels = zip(*sorted_pairs)
    #
    # ax.legend(
    #     handles, labels,
    #     title="Method",
    #     loc='best',  # Auto-placement, or use 'upper right', 'upper left', etc.
    #     frameon=True,
    #     fancybox=True,
    #     shadow=True,
    #     framealpha=0.9  # Slight transparency so data behind is visible
    # )

    # Set custom x-ticks if specified
    if X_TICKS is not None:
        ax.set_xticks(X_TICKS)

    # Rotate x-axis labels to prevent overlap
    ax.tick_params(axis='x', rotation=45)

    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add extra right margin for text labels (asymmetric: less on left, more on right)
    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]
    ax.set_xlim(xlim[0] - 0.02 * x_range, xlim[1] + 0.32 * x_range)  # 2% left, 45% right

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    return fig, ax


def create_multi_metric_plot(results_by_target_size, metrics, log_scales=None):
    """
    Create a subplot with multiple metrics.

    Args:
        results_by_target_size: Dict of target_size -> {method_name: stats}
        metrics: List of metrics to plot
        log_scales: Dict of metric -> bool for log scale (optional)

    Returns:
        Matplotlib figure and axes objects
    """
    if log_scales is None:
        log_scales = {}

    n_metrics = len(metrics)
    rows = (n_metrics + 1) // 2  # 2 columns
    cols = 2 if n_metrics > 1 else 1

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Get all unique methods across all metrics
    all_methods = set()
    for methods_data in results_by_target_size.values():
        all_methods.update(methods_data.keys())

    # Order methods based on METHOD_CONFIG if available
    if METHOD_CONFIG is not None:
        config_order = [m[0] for m in METHOD_CONFIG]
        method_names = [m for m in config_order if m in all_methods]
        for m in sorted(all_methods):
            if m not in method_names:
                method_names.append(m)
    else:
        method_names = sorted(all_methods)

    # Default color for methods not in METHOD_COLORS
    default_color = '#666666'

    # Create method to color mapping using METHOD_COLORS or auto-generated colors
    method_to_color = {}
    if METHOD_CONFIG is None:
        # Auto-generate colors when no config
        palette = sns.color_palette("husl", n_colors=len(method_names))
        for i, method_name in enumerate(method_names):
            method_to_color[method_name] = palette[i]
    else:
        for method_name in method_names:
            method_to_color[method_name] = METHOD_COLORS.get(method_name, default_color)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Prepare data for this metric
        data_by_method, baselines = prepare_plot_data(results_by_target_size, metric)

        # Combine all summarization data points into one series
        summarize_points = {"target_sizes": [], "values": []}
        non_summarize_methods = []

        for method_name in sorted(data_by_method.keys()):
            if "summarize" in method_name.lower():
                method_data = data_by_method[method_name]
                summarize_points["target_sizes"].extend(method_data["target_sizes"])
                summarize_points["values"].extend(method_data["values"])
            else:
                non_summarize_methods.append(method_name)

        # Sort summarization points by target_size for proper line plotting
        if summarize_points["target_sizes"]:
            sorted_indices = sorted(range(len(summarize_points["target_sizes"])),
                                    key=lambda i: summarize_points["target_sizes"][i])
            summarize_points["target_sizes"] = [summarize_points["target_sizes"][i] for i in sorted_indices]
            summarize_points["values"] = [summarize_points["values"][i] for i in sorted_indices]

        # Plot each method
        added_labels = set()  # Track which labels have been added to avoid duplicates

        # Plot non-summarization methods
        for method_name in non_summarize_methods:
            method_data = data_by_method[method_name]
            # Get display name
            display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)

            # Only add label if we haven't added it yet AND it's the first subplot
            if idx == 0:
                if display_name in added_labels:
                    label = None  # Don't add duplicate to legend
                else:
                    label = display_name
                    added_labels.add(display_name)
            else:
                label = None  # No labels for subplots after the first

            ax.plot(
                method_data["target_sizes"],
                method_data["values"],
                marker='o',
                linewidth=1.5,
                markersize=10,
                label=label,
                color=method_to_color[method_name],
                alpha=0.9,
                markeredgecolor='black',
                markeredgewidth=1.0
            )

        # Plot all summarization points as a single connected line
        if summarize_points["target_sizes"]:
            # Get color for summarization
            summarize_color = method_to_color[next(m for m in data_by_method.keys() if "summarize" in m.lower())]
            ax.plot(
                summarize_points["target_sizes"],
                summarize_points["values"],
                marker='o',
                linewidth=1.5,
                markersize=10,
                label="Summarization (various prompts)" if idx == 0 else None,
                color=summarize_color,
                alpha=0.9,
                markeredgecolor='black',
                markeredgewidth=1.0
            )

        # Get metric display name
        metric_display = ALL_METRICS.get(metric, metric)

        # Set labels and title
        ax.set_xlabel("Target Size", fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_display, fontsize=11, fontweight='bold')
        ax.set_title(metric_display, fontsize=12, fontweight='bold', pad=10)

        # Set log scale if requested
        if log_scales.get(metric, False):
            ax.set_yscale('log')

        # Add baseline reference lines
        baseline_colors = {'original': 'black', 'no_context': 'gray'}
        for baseline_name, baseline_value in baselines.items():
            label = BASELINE_LABELS.get(baseline_name, baseline_name)
            color = baseline_colors.get(baseline_name, 'black')
            ax.axhline(
                y=baseline_value,
                linestyle='--',
                linewidth=2,
                color=color,
                alpha=0.7,
                label=label if idx == 0 else None,  # Only label in first subplot
                zorder=1
            )

        # Only add legend to first subplot
        if idx == 0:
            ax.legend(
                title="Compaction Method",
                loc='best',  # Auto-placement inside the plot
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=8,
                framealpha=0.9  # Slight transparency
            )

        # Set custom x-ticks if specified
        if X_TICKS is not None:
            ax.set_xticks(X_TICKS)

        # Rotate x-axis labels to prevent overlap
        ax.tick_params(axis='x', rotation=45)

        # Grid styling
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title
    fig.suptitle("Metrics vs Target Size", fontsize=16, fontweight='bold', y=0.995)

    # Tight layout
    plt.tight_layout()

    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description="Plot metrics vs target size from aggregated results files. "
                    "Edit configuration variables at the top of this script to customize."
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available metrics and exit"
    )

    args = parser.parse_args()

    # List available metrics if requested
    if args.list_metrics:
        print("\nCommon Metrics:")
        for metric, display in COMMON_METRICS.items():
            print(f"  {metric:50s} - {display}")
        print("\nNested Metrics:")
        for metric, display in NESTED_METRICS.items():
            print(f"  {metric:50s} - {display}")
        return

    # Use global configuration variables
    print(f"Configuration:")
    print(f"  EVAL_DIR: {EVAL_DIR}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  METRIC: {METRIC}")
    print(f"  TARGET_SIZES_TO_LOAD: {TARGET_SIZES_TO_LOAD}")
    print(f"  LOG_SCALE: {LOG_SCALE}")
    print(f"  OUTPUT_FORMAT: {OUTPUT_FORMAT}")
    print(f"  MULTI_METRIC_PLOT: {MULTI_METRIC_PLOT}")
    print()

    # Load data
    print(f"Loading aggregated results from {EVAL_DIR}...\n")
    try:
        results_by_target_size = load_aggregated_results(EVAL_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not results_by_target_size:
        print("No data found")
        return

    print(f"\nFound {len(results_by_target_size)} different target sizes: {sorted(results_by_target_size.keys())}\n")

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create plot
    if MULTI_METRIC_PLOT:
        # Multi-metric plot with common metrics
        common_metric_keys = [
            "overall_accuracy",
            "overall_avg_perplexity",
            "avg_compaction_time_per_article",
            "test:mean_mean_output_cosine_sim",
            "test:mean_rms_output_relative_l2_error",
        ]
        log_scales = {
            "avg_compaction_time_per_article": True,
            "test:mean_rms_output_relative_l2_error": True,
        }

        print("Creating multi-metric plot...")
        fig, axes = create_multi_metric_plot(results_by_target_size, common_metric_keys, log_scales)

        # Save plot
        base_name = "metrics_vs_target_size_multi"
    else:
        # Single metric plot
        print(f"Preparing data for metric: {METRIC}...")
        data_by_method, baselines = prepare_plot_data(results_by_target_size, METRIC)

        if not data_by_method:
            print(f"\nError: No data found for metric '{METRIC}'")
            print("Use --list-metrics to see available metrics")
            return

        print(f"Found {len(data_by_method)} methods with data for {METRIC}\n")
        if baselines:
            print(f"Found {len(baselines)} baseline references\n")

        print("Creating plot...")
        fig, ax = create_plot(data_by_method, METRIC, baselines=baselines, log_scale=LOG_SCALE)

        # Save plot
        safe_metric = METRIC.replace(':', '_').replace('/', '_')
        base_name = f"{safe_metric}_vs_target_size"

    if OUTPUT_FORMAT in ["png", "both"]:
        output_file = output_dir / f"{base_name}.png"
        fig.savefig(str(output_file), dpi=DPI, bbox_inches='tight')
        print(f"Saved PNG plot to: {output_file}")

    if OUTPUT_FORMAT in ["pdf", "both"]:
        output_file = output_dir / f"{base_name}.pdf"
        fig.savefig(str(output_file), format='pdf', bbox_inches='tight')
        print(f"Saved PDF plot to: {output_file}")

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()

# scripts/plot_metric_vs_compaction_time.py
"""
Plot metrics vs compaction time from multiple aggregated_results_t{x}.json files.
Creates publication-quality plots using seaborn/matplotlib showing how different
algorithms perform at different compaction times.

Usage:
    python scripts/plot_metric_vs_compaction_time.py

    Modify METRIC, TARGET_SIZES_TO_INCLUDE, MIN_COMPACTION_RATIO, and other settings
    below to customize the plot.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patheffects
import seaborn as sns
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory containing aggregated_results_t*.json files
EVAL_DIR = "logs/qa_evaluation/qwen-quality"

# Directory to save output plots
OUTPUT_DIR = "logs/qa_evaluation/qwen-quality"

# Metric to plot on y-axis
METRIC = "overall_accuracy"
# METRIC = "overall_avg_log_perplexity"

# compaction ratios to generate plots for
# For each ratio, TARGET_SIZES_TO_INCLUDE will be set to [1/ratio, 0.99]
COMPACTION_RATIOS_TO_PLOT = [50]

# Use log scale for y-axis
LOG_SCALE = False

# Use log scale for x-axis (compaction time)
X_LOG_SCALE = True

# Output format: "png", "pdf", or "both"
OUTPUT_FORMAT = "pdf"

# Figure size (width, height) in inches
FIGURE_SIZE = (12, 8)

# DPI for output images
DPI = 300

# Scaling controls for overall readability
FONT_SCALE = 1.4
TITLE_FONT_SIZE = 33
LABEL_FONT_SIZE = 26
Y_LABEL_PAD = 2
TICK_FONT_SIZE = 20
ANNOTATION_FONT_SIZE = 20
MARKER_SIZE = 15
SCATTER_SIZE = 180  # matplotlib scatter marker area
LINE_WIDTH = 2.5
MARKER_EDGE_WIDTH = 1.4
LEFT_MARGIN = 0.02

# Manual title override (set to None to use auto-generated title)
# Example: PLOT_TITLE = "My Custom Title"
PLOT_TITLE = "Qwen3 (50× Compaction)"

# Add manual Cartridges data points
ADD_CARTRIDGES_DATA = True

# Green box dimensions (data units)
GREEN_BOX_WIDTH_SECONDS = 60  # horizontal span
GREEN_BOX_HEIGHT_ACCURACY = 0.1  # accuracy delta (10 percentage points)
GREEN_BOX_HEIGHT_PERPLEXITY = 0.5  # perplexity delta

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
    # (method_name, display_name, color)
    #
    # OMP methods
    ("ss-plus-repeat_omp_nnls0_-inf_7_drop-7_lsq_on-policy", "AM-OMP", "#5B2C83"),
    # ("ss-plus-repeat_omp_nnls0_-inf_7_drop-7_lsq_progressive_on-policy", "AM-OMP", "#FAA307"),
    ("ss-plus-repeat_omp_nnls0_-inf_7_drop-7_lsq_k4int2_on-policy", "AM-OMP-fast", "#5B2C83"),
    #
    # Highest Attention Keys methods
    ("ss-plus-repeat_highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy", "AM-HighestAttnKeys", "#C44E52"), # E85D04
    # ("context-prefill_highest_attn_keys_rms_nnls2_-3_3_lsq", "AM-HighestAttnKeys (context-prefill)", "#264653"),
    ("repeat_highest_attn_keys_rms_nnls2_-3_3_lsq", "AM-HighestAttnKeys-fast", "#C44E52"),
    #
    # Cartridges (manually added data)
    ("Cartridges", "Cartridges", "#008080"),
    #
    # Summarization methods
    ("summarize", "Summarization", "#28649D"),
    ("summarize_concise", "Summarization", "#28649D"),
    ("summarize_keypoints_questions", "Summarization", "#28649D"),
    ("summarize_very_concise", "Summarization", "#28649D"),
    ("summarize_indepth", "Summarization", "#28649D"),
    #
    # Baseline methods - 4 different shades (ordered: H2O, SnapKV, PyramidKV, KVzip)
    # H2O - darkest gray
    ("context-prefill_highest_attn_keys_mean_nobeta_direct", "H2O+", "#6B5A4A"),
    # ("context-prefill_highest_attn_keys_max_nobeta_direct", "H2O+ (max)", "#6B5A4A"),
    # ("context-prefill_highest_attn_keys_rms_nobeta_direct", "H2O+ (rms)", "#6B5A4A"),
    # KVzip / KVzip-uniform - dark-medium gray
    ("repeat_highest_attn_keys_mean_nobeta_direct", "KVzip-uniform", "#4A5568"),
    # ("repeat_highest_attn_keys_max_nobeta_direct", "KVzip-uniform (max)", "#4A5568"),
    # ("repeat_highest_attn_keys_rms_nobeta_direct", "KVzip-uniform (rms)", "#4A5568"),
    # ("repeat_global_highest_attn_keys_mean_nobeta_direct", "KVzip", "#4A5568"),
    # ("repeat_global_highest_attn_keys_max_nobeta_direct", "KVzip (max)", "#4A5568"),
    ("repeat_global_highest_attn_keys_rms_nobeta_direct", "KVzip", "#4A5568"),
    # SnapKV - lightest gray
    ("ss-one-question_snapkv_mean", "SnapKV", "#A0AEC0"),
    # ("ss-one-question_snapkv_max", "SnapKV (max)", "#718096"),
    # ("ss-one-question_snapkv_rms", "SnapKV (rms)", "#718096"),
    # PyramidKV - medium-light gray
    ("ss-one-question_pyramidkv_mean", "PyramidKV", "#CBD5E0"),
    # ("ss-one-question_pyramidkv_max", "PyramidKV (max)", "#A0AEC0"),
    # ("ss-one-question_pyramidkv_rms", "PyramidKV (rms)", "#A0AEC0"),

    # DuoAttention
    ("ss-one-question_duo_attention", "DuoAttention", "#E2E8F0")
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
    "overall_avg_log_perplexity": "log(Perplexity)",
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


def extract_metric(stats, metric_name):
    """
    Extract a metric value from stats dictionary.
    Supports nested metrics with prefix "train:" or "test:".
    For perplexity metrics, returns log(perplexity).
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

    return value


def load_aggregated_results(eval_dir):
    """
    Load all aggregated_results_t{x}.json files from the evaluation directory.

    Returns:
        Dictionary mapping target_size -> {method_name: stats}
    """
    eval_path = Path(eval_dir)

    if not eval_path.exists():
        raise FileNotFoundError(f"Directory {eval_dir} does not exist")

    result_files = list(eval_path.glob("aggregated_results_t*.json"))

    if not result_files:
        raise FileNotFoundError(f"No aggregated_results_t*.json files found in {eval_dir}")

    results_by_target_size = {}

    for file_path in sorted(result_files):
        filename = file_path.name
        if filename.startswith("aggregated_results_t") and filename.endswith(".json"):
            target_size_str = filename[len("aggregated_results_t"):-len(".json")]
            try:
                target_size = float(target_size_str)
            except ValueError:
                print(f"Warning: Could not parse target size from {filename}, skipping")
                continue

            with open(file_path, 'r') as f:
                data = json.load(f)

            results_by_target_size[target_size] = data
            print(f"Loaded {len(data)} methods from {filename} (target_size={target_size})")

    return results_by_target_size


def prepare_plot_data(results_by_target_size, metric, target_sizes_filter=None, min_compaction_ratio=None):
    """
    Prepare data for plotting: organize by method name across compaction times.

    Note: "compaction_times" is actually the total time = extraction + compaction + query generation

    Args:
        results_by_target_size: Dict of target_size -> {method_name: stats}
        metric: Metric to extract
        target_sizes_filter: List of target sizes to include, or None to include all
        min_compaction_ratio: Minimum compaction ratio to include, or None for no filter

    Returns:
        Tuple of (data_by_method, baselines)
        - data_by_method: Dict mapping method_name -> {compaction_times: [...], values: [...]}
        - baselines: Dict mapping baseline_name -> metric_value
    """
    data_by_method = defaultdict(lambda: {"compaction_times": [], "values": []})
    baselines = {}

    for target_size in sorted(results_by_target_size.keys()):
        # Filter by target size if specified
        if target_sizes_filter is not None and target_size not in target_sizes_filter:
            continue

        methods_data = results_by_target_size[target_size]

        for method_name, stats in methods_data.items():
            value = extract_metric(stats, metric)

            # Check if this is a baseline method
            if method_name in BASELINE_METHODS:
                if method_name not in baselines and value is not None:
                    baselines[method_name] = value
                continue  # Don't plot baselines as regular points

            # Filter methods based on METHODS_TO_INCLUDE
            if METHODS_TO_INCLUDE is not None and method_name not in METHODS_TO_INCLUDE:
                continue

            # Get total time (extraction + compaction + query generation)
            extraction_time = stats.get("avg_extraction_time_per_article", 0)
            compaction_time = stats.get("avg_compaction_time_per_article", 0)
            query_gen_time = stats.get("avg_query_generation_time_per_article", 0)

            # Calculate total time
            total_time = extraction_time + compaction_time + query_gen_time

            # Get compaction ratio from article_compaction_stats
            article_stats = stats.get("article_compaction_stats", {})
            compaction_ratio = article_stats.get("avg_article_compaction_ratio")

            # Apply compaction ratio filter (with 10% tolerance to include methods that are close)
            if min_compaction_ratio is not None:
                tolerance = 0.10  # 10% tolerance
                threshold = min_compaction_ratio * (1 - tolerance)
                if compaction_ratio is None or compaction_ratio < threshold:
                    continue

            if value is not None and total_time is not None and total_time > 0:
                data_by_method[method_name]["compaction_times"].append(total_time)
                data_by_method[method_name]["values"].append(value)

    # Add manual Cartridges data if enabled
    if ADD_CARTRIDGES_DATA:
        # Each target size can have multiple data points (list of dicts)
        cartridges_data = {
            0.01: [{"accuracy": 0.5994, "log-perplexity": 0.656441, "compaction_time": 4.84 * 60 * 60}],
            0.02: [
                {"accuracy": 0.5994, "log-perplexity": 0.617770, "compaction_time": 4.84 * 60 * 60},
                {"accuracy": 0.5882, "compaction_time": 3.23 * 60 * 60},
                {"accuracy": 0.5714, "compaction_time": 1.61 * 60 * 60},
                {"accuracy": 0.501, "compaction_time": 0.436 * 60 * 60},
            ],
            0.05: [{"accuracy": 0.6190, "log-perplexity": 0.572436, "compaction_time": 4.84 * 60 * 60}],
            0.1: [{"accuracy": 0.6162, "log-perplexity": 0.543985, "compaction_time": 4.84 * 60 * 60}],
        }

        for target_size, metrics_list in cartridges_data.items():
            # Filter by target size if specified
            if target_sizes_filter is not None and target_size not in target_sizes_filter:
                continue

            for metrics in metrics_list:
                if metric == "overall_accuracy":
                    data_by_method["Cartridges"]["compaction_times"].append(metrics["compaction_time"])
                    data_by_method["Cartridges"]["values"].append(metrics["accuracy"])
                elif metric == "overall_avg_log_perplexity":
                    data_by_method["Cartridges"]["compaction_times"].append(metrics["compaction_time"])
                    data_by_method["Cartridges"]["values"].append(metrics["log-perplexity"])

    return dict(data_by_method), baselines


def create_plot(data_by_method, metric, baselines=None, log_scale=False, x_log_scale=False, min_compaction_ratio=None):
    """
    Create a seaborn/matplotlib plot.

    Args:
        data_by_method: Dict of method_name -> {compaction_times: [...], values: [...]}
        metric: Metric being plotted
        baselines: Dict of baseline_name -> metric_value (optional)
        log_scale: Whether to use log scale for y-axis
        x_log_scale: Whether to use log scale for x-axis
        min_compaction_ratio: compaction ratio for title (optional)

    Returns:
        Matplotlib figure and axes objects
    """
    if baselines is None:
        baselines = {}

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=FONT_SCALE)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Get method names sorted
    method_names = sorted(data_by_method.keys())

    # Combine all summarization data points into one series
    summarize_points = {"compaction_times": [], "values": []}
    non_summarize_methods = []
    label_annotations = []
    first_summarize_label_pos = None  # Track position of first summarization method for labeling

    def get_label_position(times, values):
        """Return the (x, y) pair at the max x-value for labeling."""
        if not times:
            return None
        max_idx = max(range(len(times)), key=lambda i: times[i])
        return times[max_idx], values[max_idx]

    # Get summarization methods in METHOD_CONFIG order
    if METHOD_CONFIG is not None:
        summarize_methods_ordered = [m[0] for m in METHOD_CONFIG if "summarize" in m[0].lower()]
    else:
        summarize_methods_ordered = [m for m in method_names if "summarize" in m.lower()]

    for method_name in method_names:
        if "summarize" in method_name.lower():
            method_data = data_by_method[method_name]
            # Capture the first summarization method's position for labeling
            if first_summarize_label_pos is None and method_name in summarize_methods_ordered:
                # Find first one in METHOD_CONFIG order that has data
                for first_method in summarize_methods_ordered:
                    if first_method in data_by_method and data_by_method[first_method]["compaction_times"]:
                        first_data = data_by_method[first_method]
                        # Use the point with max x-value from the first method
                        first_summarize_label_pos = get_label_position(
                            first_data["compaction_times"], first_data["values"]
                        )
                        break
            summarize_points["compaction_times"].extend(method_data["compaction_times"])
            summarize_points["values"].extend(method_data["values"])
        else:
            non_summarize_methods.append(method_name)

    # Sort summarization points by compaction_time for proper line plotting
    if summarize_points["compaction_times"]:
        sorted_indices = sorted(range(len(summarize_points["compaction_times"])),
                                key=lambda i: summarize_points["compaction_times"][i])
        summarize_points["compaction_times"] = [summarize_points["compaction_times"][i] for i in sorted_indices]
        summarize_points["values"] = [summarize_points["values"][i] for i in sorted_indices]

    # Plot each method
    added_labels = set()

    # Plot non-summarization methods
    for method_name in non_summarize_methods:
        method_data = data_by_method[method_name]

        # Get display name and color
        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        method_color = METHOD_COLORS.get(method_name, '#005f73')

        # Only add label if we haven't added it yet
        if display_name in added_labels:
            label = None
        else:
            label = display_name
            added_labels.add(display_name)

        # Use scatter for Cartridges with dashed line connecting points
        if method_name == "Cartridges":
            # Sort points by compaction time for proper line connection
            sorted_indices = sorted(range(len(method_data["compaction_times"])),
                                    key=lambda i: method_data["compaction_times"][i])
            sorted_times = [method_data["compaction_times"][i] for i in sorted_indices]
            sorted_values = [method_data["values"][i] for i in sorted_indices]
            # Draw dashed line connecting points
            ax.plot(
                sorted_times,
                sorted_values,
                linestyle='--',
                linewidth=LINE_WIDTH,
                color=method_color,
                alpha=0.7,
                zorder=4
            )
            # Draw scatter points on top
            ax.scatter(
                method_data["compaction_times"],
                method_data["values"],
                s=SCATTER_SIZE,  # marker size (area)
                color=method_color,
                alpha=0.9,
                edgecolors='black',
                linewidths=MARKER_EDGE_WIDTH,
                zorder=5
            )
        else:
            ax.plot(
                method_data["compaction_times"],
                method_data["values"],
                marker='o',
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                color=method_color,
                alpha=0.9,
                markeredgecolor='black',
                markeredgewidth=MARKER_EDGE_WIDTH
            )

        if label:
            label_pos = get_label_position(method_data["compaction_times"], method_data["values"])
            if label_pos is not None:
                # Set label positions for specific methods
                if method_name == "Cartridges":
                    # Use the second point (index 1) for label, or first if only one point
                    idx = 2 if len(method_data["values"]) > 1 else 0
                    label_pos = (method_data["compaction_times"][idx], method_data["values"][idx])
                    label_annotations.append((*label_pos, label, method_color, "cartridges"))
                elif label == "AM-OMP":
                    label_annotations.append((*label_pos, label, method_color, "am-omp"))
                elif label == "AM-HighestAttnKeys":
                    label_annotations.append((*label_pos, label, method_color, "am-highestattnkeys"))
                elif label == "AM-OMP-fast":
                    label_annotations.append((*label_pos, label, method_color, "am-omp-fast"))
                elif min_compaction_ratio == 20 and ("h2o" in method_name.lower() or "H2O" in label):
                    label_annotations.append((*label_pos, label, method_color, "above_left"))
                elif "kvzip" in label.lower():
                    label_annotations.append((*label_pos, label, method_color, "below_left"))
                elif "perplexity" in metric.lower():
                    label_annotations.append((*label_pos, label, method_color, "above_right"))
                else:
                    label_annotations.append((*label_pos, label, method_color, "below_right"))

    # Plot all summarization points as individual markers (no connecting line)
    if summarize_points["compaction_times"]:
        # Get color for summarization from METHOD_COLORS (use "summarize" key)
        summarize_color = METHOD_COLORS.get("summarize", '#3F88C5')
        ax.scatter(
            summarize_points["compaction_times"],
            summarize_points["values"],
            s=SCATTER_SIZE,  # marker size (area)
            color=summarize_color,
            alpha=0.9,
            edgecolors='black',
            linewidths=MARKER_EDGE_WIDTH,
            zorder=5
        )

        # Use the first summarization method's position for the label (with below offset)
        if first_summarize_label_pos is not None:
            label_annotations.append((*first_summarize_label_pos, "Summarization", summarize_color, "below_right"))

    # Add text labels next to the last point of each labeled series
    for item in label_annotations:
        if len(item) == 5:
            x_val, y_val, label_text, color, position = item
        else:
            x_val, y_val, label_text, color = item
            position = "below_right"

        if position == "below":
            xytext = (0, -8)
            ha = 'center'
            va = 'top'
        elif position == "below_right":
            xytext = (4, -4)
            ha = 'left'
            va = 'top'
        elif position == "below_left":
            xytext = (-4, -4)
            ha = 'right'
            va = 'top'
        elif position == "above":
            xytext = (0, 8)
            ha = 'center'
            va = 'bottom'
        elif position == "above_right":
            xytext = (4, 4)
            ha = 'left'
            va = 'bottom'
        elif position == "above_left":
            xytext = (-4, 4)
            ha = 'right'
            va = 'bottom'
        elif position == "left":
            xytext = (-8, 0)
            ha = 'right'
            va = 'center'
        elif position == "right":
            xytext = (8, 0)
            ha = 'left'
            va = 'center'
        elif position == "am-omp":
            xytext = (9.5, 7.5)
            ha = 'left'
            va = 'top'
        elif position == "am-omp-fast":
            xytext = (2, -5)
            ha = 'left'
            va = 'top'
        elif position == "am-highestattnkeys":
            xytext = (-1, -9)
            ha = 'left'
            va = 'top'
        elif position == "cartridges":
            xytext = (4, -4)
            ha = 'left'
            va = 'top'
        else:
            # Default: below_right
            xytext = (4, -4)
            ha = 'left'
            va = 'top'

        annotation = ax.annotate(
            label_text,
            (x_val, y_val),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            va=va,
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight='medium',
            color=color,
            alpha=0.9,
            zorder=200
        )
        annotation.set_path_effects([
            patheffects.Stroke(linewidth=3, foreground='white'),
            patheffects.Normal()
        ])

    # Collect all points for Pareto frontier calculation
    all_points = []
    for method_name, method_data in data_by_method.items():
        for time, value in zip(method_data["compaction_times"], method_data["values"]):
            all_points.append((time, value))

    # Calculate and plot Pareto frontier
    # For perplexity, lower is better, so we need to negate for frontier calculation
    is_perplexity = "perplexity" in metric.lower()

    if all_points:
        # Sort by x (time), then by y
        if is_perplexity:
            # For perplexity: minimize time, minimize value (lower is better)
            sorted_points = sorted(all_points, key=lambda p: (p[0], p[1]))
        else:
            # For accuracy: minimize time, maximize value (higher is better)
            sorted_points = sorted(all_points, key=lambda p: (p[0], -p[1]))

        # Build Pareto frontier
        frontier_points = []
        if is_perplexity:
            # For perplexity: point is on frontier if no other point has both lower time AND lower value
            current_best_value = float('inf')
            for point in sorted_points:
                if point[1] < current_best_value:
                    frontier_points.append(point)
                    current_best_value = point[1]
        else:
            # For accuracy: point is on frontier if no other point has both lower time AND higher value
            current_best_value = float('-inf')
            for point in sorted_points:
                if point[1] > current_best_value:
                    frontier_points.append(point)
                    current_best_value = point[1]

        # Plot frontier
        if frontier_points:
            frontier_x = [p[0] for p in frontier_points]
            frontier_y = [p[1] for p in frontier_points]
            ax.plot(
                frontier_x,
                frontier_y,
                linestyle='-',
                linewidth=LINE_WIDTH,
                color='#434343',
                alpha=0.9,
                zorder=100
            )

    # Get metric display name
    metric_display = ALL_METRICS.get(metric, metric)

    # Set labels and title
    ax.set_xlabel("Avg Compaction Time per Article (s)", fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.set_ylabel(metric_display, fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=Y_LABEL_PAD)
    if PLOT_TITLE is not None:
        title_text = PLOT_TITLE
    elif min_compaction_ratio is not None:
        title_text = f"{metric_display} vs Compaction Time ({min_compaction_ratio}x Compaction)"
    else:
        title_text = f"{metric_display} vs Compaction Time"
    ax.set_title(title_text, fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=20)

    # Set log scale if requested
    if log_scale:
        ax.set_yscale('log')
    if x_log_scale:
        ax.set_xscale('log')
        # Add more x-axis tick labels for log scale
        from matplotlib.ticker import LogLocator, LogFormatterSciNotation
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', length=4)

    # Note: For perplexity, lower values are better and appear at the bottom (normal orientation)

    # Add baseline reference lines
    baseline_colors = {'original': 'black', 'no_context': 'gray'}
    for baseline_name, baseline_value in baselines.items():
        label = BASELINE_LABELS.get(baseline_name, baseline_name)
        color = baseline_colors.get(baseline_name, 'black')
        ax.axhline(
            y=baseline_value,
            linestyle='--',
            linewidth=LINE_WIDTH,
            color=color,
            alpha=0.7,
            zorder=1
        )

    # Rotate x-axis labels to prevent overlap and increase tick label size
    ax.tick_params(axis='x', rotation=45, labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    # Grid styling - major grid for both axes, minor grid for x-axis only
    ax.grid(True, which='major', alpha=0.4, linestyle='--', linewidth=0.6)
    if x_log_scale:
        ax.grid(True, which='minor', axis='x', alpha=0.2, linestyle='-', linewidth=0.4)

    # Add green box in top-left corner to indicate "better" region
    # Get axis limits and enforce x-axis starts at 10 seconds or earlier
    xlim = ax.get_xlim()
    if xlim[0] > 10:
        ax.set_xlim(left=10)
        xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Label baselines directly on the axes (legend removed)
    if baselines:
        for baseline_name, baseline_value in baselines.items():
            label = BASELINE_LABELS.get(baseline_name, baseline_name)
            color = baseline_colors.get(baseline_name, 'black')
            baseline_annotation = ax.annotate(
                label,
                (xlim[1], baseline_value),
                textcoords="offset points",
                xytext=(-8, 0),
                ha='right',
                va='center',
                fontsize=ANNOTATION_FONT_SIZE,
                color=color,
                alpha=0.85
            )
            baseline_annotation.set_path_effects([
                patheffects.Stroke(linewidth=3, foreground='white'),
                patheffects.Normal()
            ])

    # Normalize axis bounds (y can be inverted when plotting perplexity)
    y_axis_min = min(ylim[0], ylim[1])
    y_axis_max = max(ylim[0], ylim[1])
    x_axis_min = min(xlim[0], xlim[1])
    x_axis_max = max(xlim[0], xlim[1])

    # Get original cache baseline value to cap the box
    original_baseline = baselines.get('original')

    # Calculate box width (from left edge to GREEN_BOX_WIDTH_SECONDS)
    box_x_start = x_axis_min
    box_x_end = min(x_axis_max, GREEN_BOX_WIDTH_SECONDS)

    # Determine where the green box should be
    # Accuracy: extend 5 percentage points below the baseline (since higher is better)
    # Perplexity: extend 0.5 perplexity above the baseline (since lower is better)

    if original_baseline is not None:
        if is_perplexity:
            expanded_value = original_baseline + GREEN_BOX_HEIGHT_PERPLEXITY
            box_y_end = min(y_axis_max, expanded_value)
            box_y_start = original_baseline
        else:
            box_y_end = original_baseline
            reduced_value = original_baseline - GREEN_BOX_HEIGHT_ACCURACY
            box_y_start = max(y_axis_min, reduced_value)
    else:
        # If no baseline, anchor to the top (accuracy) or bottom (perplexity) of the axis
        if is_perplexity:
            box_y_end = y_axis_min
            box_y_start = min(y_axis_max, y_axis_min + GREEN_BOX_HEIGHT_PERPLEXITY)
        else:
            box_y_end = y_axis_max
            box_y_start = max(y_axis_min, y_axis_max - GREEN_BOX_HEIGHT_ACCURACY)

    # Normalize vertical bounds so the rectangle height is always positive.
    # Without this, inverted axes (perplexity plots) would draw the box above the baseline.
    box_y_lower = min(box_y_start, box_y_end)
    box_y_upper = max(box_y_start, box_y_end)

    # Draw the green box (from box_y_lower to box_y_upper on y-axis)
    rect = Rectangle(
        (xlim[0], box_y_lower),
        box_x_end - xlim[0],
        box_y_upper - box_y_lower,
        linewidth=LINE_WIDTH,
        edgecolor='green',
        facecolor='green',
        alpha=0.15,
        zorder=0
    )
    ax.add_patch(rect)

    # Add label inside the green box
    # For log scale x-axis, compute center in log space
    if x_log_scale:
        box_center_x = np.sqrt(xlim[0] * box_x_end)  # geometric mean for log scale
    else:
        box_center_x = (xlim[0] + box_x_end) / 2
    box_center_y = (box_y_lower + box_y_upper) / 2
    ax.text(
        box_center_x, box_center_y,
        "high quality compaction\n in < 1 minute", # high quality compaction\n in < 1 minute
        ha='center', va='center',
        fontsize=ANNOTATION_FONT_SIZE, fontweight='bold',
        color='green',
        alpha=0.8,
        zorder=1
    )

    # Tight layout to prevent label cutoff, then pull axes left for a flush y-label
    plt.tight_layout()
    fig.subplots_adjust(left=LEFT_MARGIN)

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Plot metrics vs compaction time from aggregated results files. "
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
    print(f"  COMPACTION_RATIOS_TO_PLOT: {COMPACTION_RATIOS_TO_PLOT}")
    print(f"  LOG_SCALE: {LOG_SCALE}")
    print(f"  X_LOG_SCALE: {X_LOG_SCALE}")
    print(f"  OUTPUT_FORMAT: {OUTPUT_FORMAT}")
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

    # Loop through each compaction ratio and create a plot
    for compaction_ratio in COMPACTION_RATIOS_TO_PLOT:
        print(f"\n{'='*60}")
        print(f"Creating plot for {compaction_ratio}x compaction ratio")
        print(f"{'='*60}")

        # Compute target sizes for this compaction ratio
        target_sizes = [1.0 / compaction_ratio, 0.99]
        min_ratio = compaction_ratio

        print(f"Preparing data for metric: {METRIC}...")
        print(f"Filtering by target sizes: {target_sizes}")
        print(f"Filtering by min compaction ratio: {min_ratio}")

        data_by_method, baselines = prepare_plot_data(
            results_by_target_size,
            METRIC,
            target_sizes_filter=target_sizes,
            min_compaction_ratio=min_ratio
        )

        if not data_by_method:
            print(f"\nWarning: No data found for metric '{METRIC}' at {compaction_ratio}x compaction")
            print(f"Skipping this compaction ratio...")
            continue

        print(f"Found {len(data_by_method)} methods with data for {METRIC}")
        if baselines:
            print(f"Found {len(baselines)} baseline references")

        # Create plot
        print("Creating plot...")
        fig, ax = create_plot(
            data_by_method,
            METRIC,
            baselines=baselines,
            log_scale=LOG_SCALE,
            x_log_scale=X_LOG_SCALE,
            min_compaction_ratio=min_ratio
        )

        # Save plot
        safe_metric = METRIC.replace(':', '_').replace('/', '_')
        base_name = f"{safe_metric}_vs_compaction_time_{compaction_ratio}x"

        if OUTPUT_FORMAT in ["png", "both"]:
            output_file = output_dir / f"{base_name}.png"
            fig.savefig(str(output_file), dpi=DPI, bbox_inches='tight')
            print(f"Saved PNG plot to: {output_file}")

        if OUTPUT_FORMAT in ["pdf", "both"]:
            output_file = output_dir / f"{base_name}.pdf"
            fig.savefig(str(output_file), format='pdf', bbox_inches='tight')
            print(f"Saved PDF plot to: {output_file}")

        plt.close(fig)

    print("\nDone! Created plots for all compaction ratios.")


if __name__ == "__main__":
    main()

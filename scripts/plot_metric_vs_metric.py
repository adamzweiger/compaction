# scripts/plot_metric_vs_metric.py
"""
Plot one metric against another metric from aggregated results.
Easily modify X_METRIC and Y_METRIC to visualize different relationships.
Supports top-level metrics and nested metrics from train/test stats.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from adjustText import adjust_text

# ============================================================================
# CONFIGURATION
# ============================================================================

# List of aggregated results files to load (will merge data from all files)
AGGREGATED_RESULTS_FILES = [
    "logs/qa_evaluation/qwen-quality/aggregated_results_t0.01.json",
    "logs/qa_evaluation/qwen-quality/aggregated_results_t0.02.json",
    "logs/qa_evaluation/qwen-quality/aggregated_results_t0.05.json",
    "logs/qa_evaluation/qwen-quality/aggregated_results_t0.1.json",
    "logs/qa_evaluation/qwen-quality/aggregated_results_t0.99.json",
]
OUTPUT_DIR = "logs/qa_evaluation/qwen-quality"

# Add manual Cartridges data points
ADD_CARTRIDGES_DATA = True

# Cartridges data: target_size -> {metrics}
CARTRIDGES_DATA = {
    0.01: {"overall_accuracy": 0.5994, "overall_avg_log_perplexity": 0.656441},
    0.02: {"overall_accuracy": 0.5994, "overall_avg_log_perplexity": 0.617770},
    0.05: {"overall_accuracy": 0.6190, "overall_avg_log_perplexity": 0.572436},
    0.1: {"overall_accuracy": 0.6162, "overall_avg_log_perplexity": 0.543985},
}

# Choose which metrics to plot
# X_METRIC is plotted on the x-axis, Y_METRIC on the y-axis
#
# Top-level metrics:
#   - "overall_accuracy"
#   - "avg_compaction_ratio"
#   - "overall_avg_perplexity"
#   - "avg_compaction_time_per_article"
#   - etc.
#
# Nested metrics (prefix with "train:" or "test:"):
#   - "train:mean_mean_output_mse"
#   - "test:mean_mean_output_mse"
#   - "train:mean_mean_output_cosine_sim"
#   - "test:mean_mean_output_cosine_sim"
#   - "train:mean_rms_output_relative_l2_error"
#   - "test:mean_rms_output_relative_l2_error"
#   - "train:mean_mean_sumexp_relative_error"
#   - "test:mean_mean_sumexp_relative_error"
#   - etc.

Y_METRIC = "overall_accuracy"
X_METRIC = "overall_avg_log_perplexity"

# Custom axis labels (set to None to use metric name)
X_LABEL = "Log(Perplexity of Original Cache Generation)"
Y_LABEL = "Accuracy"

# Custom plot title (set to None for auto-generated title)
PLOT_TITLE = "Accuracy vs Log(Perplexity)"

# Output format: "png", "pdf", or "both"
OUTPUT_FORMAT = "pdf"

# Optional: Set to True to use log scale for x-axis
X_LOG_SCALE = False
# Optional: Set to True to use log scale for y-axis
Y_LOG_SCALE = False
# Optional: Set to True to invert x-axis (useful for error metrics)
X_INVERT = False
# Optional: Set to True to invert y-axis
Y_INVERT = False

# ============================================================================
# COLOR CONFIGURATION
# ============================================================================

# Category colors for the plot
CATEGORY_COLORS = {
    "original": "#000000",      # black
    "no_context": "#7f7f7f",    # gray
    "summarization": "#28649D", # blue
    "Cartridges": "#008080",    # teal
    "attention_matching": "#C44E52",
    "token_pruning": "#6B5A4A", # brown
}


def get_method_category(method_name):
    """
    Determine the category of a method for coloring purposes.

    Categories:
    - original: the original method
    - no_context: the no_context baseline
    - summarization: any method containing 'summarize'
    - Cartridges: Cartridges methods
    - attention_matching: methods containing 'lsq'
    - token_pruning: everything else
    """
    if method_name == "original":
        return "original"
    elif method_name == "no_context":
        return "no_context"
    elif "summarize" in method_name.lower():
        return "summarization"
    elif method_name.startswith("Cartridges"):
        return "Cartridges"
    elif "lsq" in method_name.lower():
        return "attention_matching"
    else:
        return "token_pruning"


# ============================================================================
# SCRIPT
# ============================================================================

def get_method_prefix(method_name):
    """Extract the prefix (first word) from a method name."""
    if "_" in method_name:
        return method_name.split("_")[0]
    return method_name


def extract_metric(stats, metric_name):
    """
    Extract a metric value from stats dictionary.
    Supports nested metrics with prefix "train:" or "test:".

    Args:
        stats: Dictionary of statistics for a method
        metric_name: Name of metric to extract (e.g., "overall_accuracy" or "test:mean_mean_output_mse")

    Returns:
        Metric value or None if not found
    """
    if metric_name.startswith("train:"):
        # Extract from overall_all_head_train_stats
        nested_metric = metric_name[6:]  # Remove "train:" prefix
        train_stats = stats.get("overall_all_head_train_stats", {})
        return train_stats.get(nested_metric)
    elif metric_name.startswith("test:"):
        # Extract from overall_all_head_test_stats
        nested_metric = metric_name[5:]  # Remove "test:" prefix
        test_stats = stats.get("overall_all_head_test_stats", {})
        return test_stats.get(nested_metric)
    else:
        # Top-level metric
        return stats.get(metric_name)


# Methods that are baselines (same across all target sizes, don't need _t{x} suffix)
BASELINE_METHODS = {"original", "no_context"}


def extract_target_size_from_filename(file_path):
    """Extract target size from filename like aggregated_results_t0.02.json."""
    filename = Path(file_path).name
    if filename.startswith("aggregated_results_t") and filename.endswith(".json"):
        target_size_str = filename[len("aggregated_results_t"):-len(".json")]
        try:
            return float(target_size_str)
        except ValueError:
            return None
    return None


def load_data(file_paths, x_metric, y_metric):
    """
    Load data from multiple aggregated results files and extract metrics.
    Appends _t{x} suffix to method names to distinguish across target sizes.
    Baseline methods (original, no_context) are only included once.

    Args:
        file_paths: List of paths to aggregated results JSON files
        x_metric: Metric to plot on x-axis
        y_metric: Metric to plot on y-axis

    Returns:
        Tuple of (methods, x_values, y_values, prefixes)
    """
    methods = []
    x_values = []
    y_values = []
    prefixes = []
    seen_baselines = set()

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: {file_path} does not exist, skipping")
            continue

        target_size = extract_target_size_from_filename(file_path)
        print(f"Loading from: {file_path} (target_size={target_size})")
        with open(path, 'r') as f:
            data = json.load(f)

        for method_name, stats in data.items():
            # Handle baseline methods - only include once
            if method_name in BASELINE_METHODS:
                if method_name in seen_baselines:
                    continue
                seen_baselines.add(method_name)
                display_name = method_name
            else:
                # Add target size suffix for non-baseline methods
                if target_size is not None:
                    display_name = f"{method_name}_t{target_size}"
                else:
                    display_name = method_name

            x_value = extract_metric(stats, x_metric)
            y_value = extract_metric(stats, y_metric)

            if x_value is not None and y_value is not None:
                methods.append(display_name)
                x_values.append(x_value)
                y_values.append(y_value)
                prefixes.append(get_method_prefix(method_name))
                print(f"  {display_name}: {x_metric}={x_value:.6f}, {y_metric}={y_value:.6f}")

    # Add Cartridges data if enabled
    if ADD_CARTRIDGES_DATA:
        print("\nAdding Cartridges data points:")
        for target_size, metrics in CARTRIDGES_DATA.items():
            x_value = metrics.get(x_metric)
            y_value = metrics.get(y_metric)

            if x_value is not None and y_value is not None:
                method_name = f"Cartridges_t{target_size}"
                methods.append(method_name)
                x_values.append(x_value)
                y_values.append(y_value)
                prefixes.append("Cartridges")
                print(f"  {method_name}: {x_metric}={x_value:.6f}, {y_metric}={y_value:.6f}")

    return methods, x_values, y_values, prefixes


def plot_data(methods, x_values, y_values, prefixes, x_metric, y_metric,
              x_log=False, y_log=False, x_invert=False, y_invert=False):
    """
    Create scatter plot of x_metric vs y_metric.

    Args:
        methods: List of method names
        x_values: List of x-axis values
        y_values: List of y-axis values
        prefixes: List of method prefixes (unused, kept for compatibility)
        x_metric: Name of x-axis metric
        y_metric: Name of y-axis metric
        x_log: Whether to use log scale for x-axis
        y_log: Whether to use log scale for y-axis
        x_invert: Whether to invert x-axis
        y_invert: Whether to invert y-axis
    """
    # Assign colors based on method category
    categories = [get_method_category(method) for method in methods]
    colors = [CATEGORY_COLORS[cat] for cat in categories]

    # Create the plot
    plt.figure(figsize=(14, 10))

    # Plot points
    plt.scatter(x_values, y_values, c=colors, s=100, alpha=0.9,
               edgecolors='black', linewidth=1)

    # Add labels for each point using adjust_text to avoid overlaps
    texts = []
    for i, method in enumerate(methods):
        text = plt.text(x_values[i], y_values[i], method,
                       fontsize=8, alpha=0.8)
        texts.append(text)

    # Adjust text positions to avoid overlaps
    adjust_text(texts,
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5),
                expand_points=(1.2, 1.2),
                force_text=(0.3, 0.3),
                force_points=(0.1, 0.1))

    # Create legend with only the categories present in the data
    # Define display names for legend
    category_display_names = {
        "original": "Original",
        "no_context": "No Context",
        "summarization": "Summarization",
        "Cartridges": "Cartridges",
        "attention_matching": "Attention Matching",
        "token_pruning": "Token Pruning",
    }
    # Define the desired legend order
    legend_order = ["original", "attention_matching", "Cartridges", "summarization", "token_pruning", "no_context"]
    # Filter to only categories present in the data, maintaining the desired order
    unique_categories = [cat for cat in legend_order if cat in categories]
    legend_patches = [mpatches.Patch(color=CATEGORY_COLORS[cat], label=category_display_names[cat])
                     for cat in unique_categories]
    plt.legend(handles=legend_patches, loc='best', title='Method Type', fontsize=16, title_fontsize=18)

    # Configure axes
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    if x_invert:
        plt.gca().invert_xaxis()
    if y_invert:
        plt.gca().invert_yaxis()

    # Labels and title
    x_label = X_LABEL if X_LABEL is not None else x_metric
    y_label = Y_LABEL if Y_LABEL is not None else y_metric
    if x_log and X_LABEL is None:
        x_label += " (log scale)"
    if y_log and Y_LABEL is None:
        y_label += " (log scale)"

    plt.xlabel(x_label, fontsize=22, fontweight='bold')
    plt.ylabel(y_label, fontsize=22, fontweight='bold')

    title = PLOT_TITLE if PLOT_TITLE is not None else f'{y_metric} vs {x_metric}'
    plt.title(title, fontsize=26, fontweight='bold')

    # Tick label sizes
    plt.gca().tick_params(axis='x', rotation=45, labelsize=16)
    plt.gca().tick_params(axis='y', labelsize=16)

    # Grid
    plt.grid(True, alpha=0.3)

    # Save the plot
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create safe filename
    safe_x = x_metric.replace(':', '_').replace('/', '_')
    safe_y = y_metric.replace(':', '_').replace('/', '_')
    base_name = f"{safe_y}_vs_{safe_x}"

    plt.tight_layout()

    if OUTPUT_FORMAT in ["png", "both"]:
        output_file = output_dir / f"{base_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_file}")

    if OUTPUT_FORMAT in ["pdf", "both"]:
        output_file = output_dir / f"{base_name}.pdf"
        plt.savefig(output_file, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {output_file}")

    # Display statistics
    print(f"\nPlotted {len(methods)} methods")
    print(f"Categories: {', '.join(unique_categories)}")
    print(f"\n{x_metric} range: {min(x_values):.6f} - {max(x_values):.6f}")
    print(f"{y_metric} range: {min(y_values):.6f} - {max(y_values):.6f}")


def main():
    # Check that at least one file exists
    existing_files = [f for f in AGGREGATED_RESULTS_FILES if Path(f).exists()]
    if not existing_files:
        print(f"Error: None of the specified files exist:")
        for f in AGGREGATED_RESULTS_FILES:
            print(f"  - {f}")
        print("Please run aggregate_qa_results.py first")
        return

    print(f"Plotting: {Y_METRIC} vs {X_METRIC}\n")
    print(f"Files to load: {len(AGGREGATED_RESULTS_FILES)}\n")

    # Load data from all files
    methods, x_values, y_values, prefixes = load_data(
        AGGREGATED_RESULTS_FILES, X_METRIC, Y_METRIC
    )

    if not methods:
        print("\nError: No valid data found.")
        print(f"Make sure '{X_METRIC}' and '{Y_METRIC}' exist in the aggregated results.")
        return

    # Create plot
    plot_data(methods, x_values, y_values, prefixes, X_METRIC, Y_METRIC,
             x_log=X_LOG_SCALE, y_log=Y_LOG_SCALE,
             x_invert=X_INVERT, y_invert=Y_INVERT)

    print("\nDone!")


if __name__ == "__main__":
    main()

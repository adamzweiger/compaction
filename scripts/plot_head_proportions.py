# scripts/plot_head_proportions.py
"""
Plot head proportions from global RMS selection across different target ratios.
Creates a bar/line chart showing the proportion of each of 288 attention heads
selected at different target ratios.

Usage:
    python scripts/plot_head_proportions.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set to "target_ratio", "article", or "optimized" to switch between options
PLOT_MODE = "target_ratio"

# Option 1: Plot by target ratio (overall aggregated files)
TARGET_RATIO_FILES = {
    "t=0.01 (average of 10 articles)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/global_rms_t0.01.json",
    "t=0.02 (average of 10 articles)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/global_rms_t0.02.json",
    "t=0.05 (average of 10 articles)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/global_rms_t0.05.json",
    "t=0.1 (average of 10 articles)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/global_rms_t0.1.json",
}

# Option 2: Plot by article (individual article files at same target ratio)
ARTICLE_FILES = {
    "article_1 (t=0.05)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/rms_t0.05/head_proportions_t0.05_global_rms_nnls0_-inf_inf_lsq_20251217_011342.json",
    "article_2 (t=0.05)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/rms_t0.05/head_proportions_t0.05_global_rms_nnls0_-inf_inf_lsq_20251217_011409.json",
    "article_3 (t=0.05)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/rms_t0.05/head_proportions_t0.05_global_rms_nnls0_-inf_inf_lsq_20251217_011443.json",
    "article_4 (t=0.05)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/rms_t0.05/head_proportions_t0.05_global_rms_nnls0_-inf_inf_lsq_20251217_011518.json",
    # "article_5 (t=0.05)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/rms_t0.05/head_proportions_t0.05_global_rms_nnls0_-inf_inf_lsq_20251217_011546.json",
    # "article_6 (t=0.05)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/rms_t0.05/head_proportions_t0.05_global_rms_nnls0_-inf_inf_lsq_20251217_011608.json",
    # "article_7 (t=0.05)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/rms_t0.05/head_proportions_t0.05_global_rms_nnls0_-inf_inf_lsq_20251217_011628.json",
    # "article_8 (t=0.05)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/rms_t0.05/head_proportions_t0.05_global_rms_nnls0_-inf_inf_lsq_20251217_011648.json",
    # "article_9 (t=0.05)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/rms_t0.05/head_proportions_t0.05_global_rms_nnls0_-inf_inf_lsq_20251217_011716.json",
    # "article_10 (t=0.05)": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/rms_t0.05/head_proportions_t0.05_global_rms_nnls0_-inf_inf_lsq_20251217_011737.json",
}

# Option 3: Plot optimized vs global RMS comparison
OPTIMIZED_FILES = {
    "Optimized": "head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json",
    "Global RMS t=0.05": "head_budget_optimization/head_budgets/Qwen3-4B/global_rms/global_rms_t0.05.json",
}

# Baseline file (uniform distribution) - plotted in gray if enabled
SHOW_BASELINE = True
BASELINE_FILE = "head_budget_optimization/head_budgets/Qwen3-4B/uniform.json"
BASELINE_LABEL = "Uniform"

# Plot settings (auto-configured based on PLOT_MODE)
PLOT_TITLE = "Head Proportions through Global RMS Selection By Target Ratio"

# Output settings
OUTPUT_DIR = "logs/budget_optimization"
OUTPUT_FORMAT = "pdf"  # "png", "pdf", or "both"
FIGURE_SIZE = (16, 6)
DPI = 300


def parse_head_key(key):
    """
    Parse head key like 'L0H0' into (layer, head) tuple.
    Returns (layer_num, head_num) for sorting.
    """
    # Extract layer and head numbers
    key = key.upper()
    l_idx = key.index('L')
    h_idx = key.index('H')
    layer = int(key[l_idx + 1:h_idx])
    head = int(key[h_idx + 1:])
    return (layer, head)


def sort_heads(head_keys):
    """
    Sort head keys in order: L0H0, L0H1, ..., L0H7, L1H0, L1H1, ..., L35H7
    """
    return sorted(head_keys, key=parse_head_key)


def load_head_proportions(file_path):
    """Load head proportions from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def main():
    # Select files and settings based on PLOT_MODE
    if PLOT_MODE == "target_ratio":
        head_proportion_files = TARGET_RATIO_FILES
        legend_title = "Target Ratio"
        output_filename = "head_proportions_by_target_ratio"
    elif PLOT_MODE == "article":
        head_proportion_files = ARTICLE_FILES
        legend_title = "Article"
        output_filename = "head_proportions_by_article"
    elif PLOT_MODE == "optimized":
        head_proportion_files = OPTIMIZED_FILES
        legend_title = "Method"
        output_filename = "head_proportions_optimized"
    else:
        raise ValueError(f"Unknown PLOT_MODE: {PLOT_MODE}. Use 'target_ratio', 'article', or 'optimized'.")

    # Load all data
    all_data = {}
    sorted_heads = None
    baseline_data = None

    for label, file_path in head_proportion_files.items():
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue

        data = load_head_proportions(path)
        all_data[label] = data

        # Get sorted head order from first file
        if sorted_heads is None:
            sorted_heads = sort_heads(data.keys())

    # Load baseline if enabled
    if SHOW_BASELINE:
        baseline_path = Path(BASELINE_FILE)
        if baseline_path.exists():
            baseline_data = load_head_proportions(baseline_path)
            print(f"Loaded baseline from {BASELINE_FILE}")
        else:
            print(f"Warning: Baseline file {BASELINE_FILE} not found")

    if not all_data:
        print("Error: No data files found")
        return

    print(f"Loaded {len(all_data)} files with {len(sorted_heads)} heads each")

    # Create x-axis positions
    x = np.arange(len(sorted_heads))

    # Set up plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Colors - use custom color palette
    color_palette = ["#7EBDC2", "#BB4430", "#004F2D", "#4C257E", "#E3D081", "#3D0814"]
    colors = [color_palette[i % len(color_palette)] for i in range(len(all_data))]

    # Sort labels: try numeric sort (for t=X format), fall back to natural sort
    def sort_key(label):
        if '=' in label:
            try:
                return (0, float(label.split('=')[1].split()[0]))
            except ValueError:
                pass
        # Try to extract number from label (e.g., "article_1" -> 1)
        import re
        match = re.search(r'(\d+)', label)
        if match:
            return (0, int(match.group(1)))
        return (1, label)

    sorted_labels = sorted(all_data.keys(), key=sort_key)

    # Plot baseline first (in gray, behind other lines)
    if baseline_data is not None:
        baseline_values = [baseline_data[head] for head in sorted_heads]
        ax.plot(x, baseline_values, label=BASELINE_LABEL, color='gray',
                linewidth=1.5, alpha=0.6, linestyle='--')

    # Plot each line
    for idx, label in enumerate(sorted_labels):
        data = all_data[label]
        values = [data[head] for head in sorted_heads]
        ax.plot(x, values, label=label, color=colors[idx % len(colors)],
                linewidth=1.0, alpha=0.8)

    # Customize plot
    ax.set_xlabel("KV-Head", fontsize=14, fontweight='bold')
    ax.set_ylabel("Selection Proportion", fontsize=14, fontweight='bold')
    ax.set_title(PLOT_TITLE, fontsize=16, fontweight='bold')

    # Set x-ticks at layer boundaries (every 8 heads)
    layer_boundaries = list(range(0, len(sorted_heads), 8))
    layer_labels = [f"L{i}" for i in range(len(layer_boundaries))]
    ax.set_xticks(layer_boundaries)
    ax.set_xticklabels(layer_labels, fontsize=10, rotation=45)

    # Add minor ticks for individual heads
    ax.set_xlim(-0.5, len(sorted_heads) - 0.5)

    # Add vertical lines at layer boundaries for clarity
    for boundary in layer_boundaries[1:]:
        ax.axvline(x=boundary - 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    # Add legend
    ax.legend(title=legend_title, loc='upper right', frameon=True,
              fancybox=True, shadow=True, fontsize=14, title_fontsize=15)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

    # Tight layout
    plt.tight_layout()

    # Save plot
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = output_filename

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

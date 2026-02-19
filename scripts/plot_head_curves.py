# scripts/plot_head_curve.py
"""
Plot per-head influence curves showing delta log perplexity vs ratio.

Usage:
    python scripts/plot_head_curves.py \
        --per-article-curves-dir logs/budget_optimization/Qwen3-4B/optimized \
        --heads L0H0,L7H1,L14H4,L15H2,L21H2 \
        --output head_curves.png
"""
import argparse
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple


def load_and_aggregate_curves(
    curves_dir: str,
    error_bars: str = None,
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Load per-article curves and aggregate by averaging.

    Parameters
    ----------
    curves_dir : str
        Directory containing article_*.json files
    error_bars : str
        None (no error bars), or "sem" / "std" for shading width

    Returns
    -------
    aggregated : dict
        If error_bars is None: {head_key: [(ratio, mean_delta), ...]}
        If error_bars is "sem" or "std": {head_key: [(ratio, mean_delta, err), ...]}
    """
    dir_path = Path(curves_dir)
    pattern = str(dir_path / 'article_*.json')
    files = sorted(glob.glob(pattern))

    if not files:
        raise ValueError(f"No article curve files found in {curves_dir}")

    all_curves = []
    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)
        curves = {}
        for head_key, curve in data['head_curves'].items():
            curves[head_key] = [(p[0], p[1]) for p in curve]
        all_curves.append(curves)

    print(f"Loaded curves from {len(files)} articles")

    # Aggregate by averaging
    head_keys = list(all_curves[0].keys())
    aggregated = {}

    for head_key in head_keys:
        curves = [article_curves[head_key] for article_curves in all_curves]
        ratios = [point[0] for point in curves[0]]

        averaged_curve = []
        for i, ratio in enumerate(ratios):
            deltas = [curve[i][1] for curve in curves]
            mean_delta = np.mean(deltas)
            if error_bars is not None:
                std_delta = np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0
                if error_bars == "sem":
                    err = std_delta / np.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
                elif error_bars == "std":
                    err = std_delta
                else:
                    raise ValueError(f"Unknown error_bars mode: {error_bars} (expected None/'sem'/'std')")
                averaged_curve.append((ratio, mean_delta, err))
            else:
                averaged_curve.append((ratio, mean_delta))

        aggregated[head_key] = averaged_curve

    return aggregated


def plot_head_curves(
    head_curves: Dict[str, List[Tuple[float, float]]],
    heads_to_plot: List[str],
    output_path: str,
    title: str = "Head Influence Curves",
    show_error_bars: bool = False,
):
    """Plot influence curves for specified heads."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Custom color palette
    color_palette = ["#004F2D", "#7EBDC2", "#BB4430", "#4C257E", "#E3D081", "#3D0814", "#FF6666", "#CCFF66", "#5D2E8C", "#2EC4B6", "#F1E8B8", "#FFA400", "#009FFD", "#2A2A72", "#232528", "#EAF6FF", "#183A37", "#C44900", "#432534", "#EFD6AC"]
    colors = [color_palette[i % len(color_palette)] for i in range(len(heads_to_plot))]

    for i, head_key in enumerate(heads_to_plot):
        if head_key not in head_curves:
            print(f"Warning: {head_key} not found in curves, skipping")
            continue

        curve = head_curves[head_key]
        ratios = [p[0] for p in curve]
        deltas = [p[1] for p in curve]

        # Check if we have std data (tuple of 3 elements)
        has_std = len(curve[0]) >= 3 if curve else False

        # Plot line
        ax.plot(ratios, deltas, 'o-', label=head_key, color=colors[i],
                markersize=6, linewidth=2)

        # Add shaded error region if requested
        if show_error_bars and has_std:
            stds = [p[2] for p in curve]
            lower = [d - s for d, s in zip(deltas, stds)]
            upper = [d + s for d, s in zip(deltas, stds)]
            ax.fill_between(ratios, lower, upper, color=colors[i], alpha=0.2)

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
    ax.axvline(x=0.05, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlabel('Ratio (fraction of original keys)', fontsize=16)
    ax.set_ylabel('Δ log(perplexity) from baseline', fontsize=16)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=15)
    ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.8, axis='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot per-head influence curves'
    )
    parser.add_argument(
        '--per-article-curves-dir',
        type=str,
        required=True,
        help='Directory containing per-article curve files (article_*.json)'
    )
    parser.add_argument(
        '--heads',
        type=str,
        default='L0H0,L7H1,L15H2,L21H2',  # 'L0H0,L7H1,L15H2,L21H2,L24H5,L26H6'
        help='Comma-separated list of heads to plot (default: L0H0,L7H1,L14H4,L15H2,L21H2)'
    ) 
    # Llama: --heads L0H0,L10H7,L13H0,L16H5
    # Gemma: --heads L5H0,L5H1,L5H2,L5H3,L11H0,L11H1,L11H2,L11H3,L17H0,L17H1,L17H2,L17H3,L23H0,L23H1,L23H2,L23H3
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the plot (default: <per-article-curves-dir>/head_curves.<format>)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['png', 'pdf'],
        default='png',
        help='Output format: png or pdf (default: png)'
    )
    parser.add_argument(
        '--title',
        type=str,
        default='Head Influence Curves',
        help='Plot title'
    )
    parser.add_argument(
        '--error-bars',
        type=str,
        choices=['sem', 'std'],
        default=None,
        help='Show error shading: "sem" (standard error) or "std" (standard deviation) across articles'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    heads_to_plot = [h.strip() for h in args.heads.split(',')]
    print(f"Plotting heads: {heads_to_plot}")

    # Default output path to per-article-curves-dir
    if args.output is None:
        args.output = str(Path(args.per_article_curves_dir) / f'head_curves.{args.format}')

    head_curves = load_and_aggregate_curves(
        args.per_article_curves_dir,
        error_bars=args.error_bars,
    )
    print(f"Loaded {len(head_curves)} head curves")

    plot_head_curves(
        head_curves=head_curves,
        heads_to_plot=heads_to_plot,
        output_path=args.output,
        title=args.title,
        show_error_bars=(args.error_bars is not None),
    )


if __name__ == '__main__':
    main()

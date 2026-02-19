# head_budget_optimization/generate_pyramidkv_schedule.py
"""
Generate PyramidKV-style head budget schedules.

PyramidKV uses layer-wise budgets where early layers get more tokens and later layers get fewer.
Within each layer, all heads get the same budget (uniform allocation per layer).

The schedule follows a linear interpolation from max_budget (layer 0) to min_budget (last layer),
where the average across all layers equals the target budget.

Usage:
    # Generate schedule for Qwen3-4B (36 layers, 8 heads)
    python -m head_budget_optimization.generate_pyramidkv_schedule \
        --num-layers 36 \
        --num-heads 8 \
        --beta 20 \
        --output head_budget_optimization/head_budgets/Qwen3-4B/pyramidkv_beta20.json

    # Generate schedule for Llama-3.1-8B-Instruct (32 layers, 8 heads)
    python -m head_budget_optimization.generate_pyramidkv_schedule \
        --num-layers 32 \
        --num-heads 8 \
        --beta 20 \
        --output head_budget_optimization/head_budgets/Llama-3.1-8B-Instruct/pyramidkv_beta20.json
"""
import argparse
import json
from pathlib import Path


def compute_pyramidkv_layer_budgets(
    num_layers: int,
    beta: int = 20,
) -> list:
    """
    Compute PyramidKV-style layer-wise budgets as proportions.

    The pyramid schedule gives early layers more budget and later layers less.
    The formula follows the original PyramidKV paper:
        min_num = base_budget / beta
        max_num = base_budget * 2 - min_num
        budget[layer] = max_num - layer * step

    We compute this in terms of proportions that sum to 1.0 across all layer-heads.

    Parameters
    ----------
    num_layers : int
        Number of transformer layers
    beta : int
        PyramidKV beta parameter controlling the pyramid steepness.
        Higher beta = more uniform distribution.
        Default is 20 (from original PyramidKV paper).

    Returns
    -------
    layer_budgets : list of float
        Budget proportion for each layer. All heads in a layer share the same budget.
        Sum of (layer_budget * num_heads) across all layers equals 1.0.
    """
    # Work in arbitrary units where baseline per-layer budget = 1.0
    # Then normalize at the end
    base_budget = 1.0

    min_budget = base_budget / beta
    max_budget = base_budget * 2 - min_budget

    # Compute layer budgets using floating-point interpolation
    if num_layers == 1:
        return [1.0]

    # Linear interpolation from max_budget to min_budget
    float_budgets = [
        max_budget - i * (max_budget - min_budget) / (num_layers - 1)
        for i in range(num_layers)
    ]

    # Normalize so they sum to num_layers (average = 1.0 per layer)
    total = sum(float_budgets)
    layer_budgets = [b * num_layers / total for b in float_budgets]

    return layer_budgets


def generate_pyramidkv_schedule(
    num_layers: int,
    num_heads: int,
    beta: int = 20,
) -> dict:
    """
    Generate a PyramidKV head budget schedule.

    Parameters
    ----------
    num_layers : int
        Number of transformer layers
    num_heads : int
        Number of KV heads per layer
    beta : int
        PyramidKV beta parameter (default: 20)

    Returns
    -------
    schedule : dict
        Dictionary mapping "L{layer}H{head}" to budget proportion.
        All proportions sum to 1.0.
    """
    layer_budgets = compute_pyramidkv_layer_budgets(num_layers, beta)

    # Each head in a layer gets layer_budget / num_heads of the total
    # The total budget is num_layers * num_heads units, so we divide by that
    total_heads = num_layers * num_heads

    schedule = {}
    for layer_idx, layer_budget in enumerate(layer_budgets):
        # layer_budget is normalized so sum across layers = num_layers
        # Each head in this layer gets layer_budget / num_heads proportion
        # of the total layer budget
        head_proportion = layer_budget / total_heads

        for head_idx in range(num_heads):
            key = f"L{layer_idx}H{head_idx}"
            schedule[key] = head_proportion

    # Verify proportions sum to 1.0
    total = sum(schedule.values())
    assert abs(total - 1.0) < 1e-10, f"Proportions sum to {total}, expected 1.0"

    return schedule


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyramidKV-style head budget schedule",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        required=True,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        required=True,
        help="Number of KV heads per layer",
    )
    parser.add_argument(
        "--beta",
        type=int,
        default=20,
        help="PyramidKV beta parameter (default: 20). Higher = more uniform.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the JSON schedule file",
    )

    args = parser.parse_args()

    # Generate schedule
    schedule = generate_pyramidkv_schedule(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        beta=args.beta,
    )

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save schedule
    with open(output_path, "w") as f:
        json.dump(schedule, f, indent=2)

    print(f"Generated PyramidKV schedule:")
    print(f"  Layers: {args.num_layers}")
    print(f"  Heads per layer: {args.num_heads}")
    print(f"  Beta: {args.beta}")
    print(f"  Total heads: {args.num_layers * args.num_heads}")
    print(f"  Output: {output_path}")

    # Print layer budget summary
    layer_budgets = compute_pyramidkv_layer_budgets(args.num_layers, args.beta)
    print(f"\nLayer budget proportions (relative to uniform):")
    print(f"  Layer 0 (first):  {layer_budgets[0]:.4f}x")
    print(f"  Layer {args.num_layers//2} (middle): {layer_budgets[args.num_layers//2]:.4f}x")
    print(f"  Layer {args.num_layers-1} (last):   {layer_budgets[-1]:.4f}x")


if __name__ == "__main__":
    main()

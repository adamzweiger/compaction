# head_budget_optimization/
"""
Head Budget Optimization for Nonuniform KV Cache Compaction.

This module solves for optimal per-head compaction ratios by:
1. Computing per-head influence curves (how perplexity changes as each head's budget varies)
2. Aggregating curves across multiple articles
3. Using a greedy solver to find optimal allocations for target compaction ratios

Usage:
    python -m head_budget_optimization.run \
        --baseline-schedule head_budget_optimization/head_budgets/Qwen3-4B/uniform.json \
        --target-ratio 0.05 \
        --n-articles 10 \
        --solve-ratios 0.01,0.02,0.05,0.1
"""

from .influence import (
    HeadInfluenceComputer,
    aggregate_head_curves,
    save_head_curves,
    load_head_curves,
    load_and_aggregate_article_curves,
)
from .solver import HeadBudgetSolver, analyze_head_curves

__all__ = [
    'HeadInfluenceComputer',
    'HeadBudgetSolver',
    'aggregate_head_curves',
    'save_head_curves',
    'load_head_curves',
    'load_and_aggregate_article_curves',
    'analyze_head_curves',
]

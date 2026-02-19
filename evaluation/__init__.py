# evaluation/__init__.py
"""
Evaluation framework for KV cache compaction on downstream tasks.

This package provides tools to evaluate compaction methods on:
- Question answering tasks
- Text generation quality
- Other language modeling benchmarks
"""
from .qa_evaluator import QAEvaluator

__all__ = [
    'QAEvaluator',
]

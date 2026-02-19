# compaction/algorithms/__init__.py
"""
KV Cache Compaction Algorithms

This package contains different algorithms for compacting KV caches
in attention mechanisms. All algorithms implement the same interface
defined in base.CompactionAlgorithm.
"""
from .base import CompactionAlgorithm, evaluate_compaction
from .omp import OMPCompaction
from .omp_full import OMPFullCompaction
from .omp_batched import BatchedOMPCompaction
from .optim import OptimC1BetaCompaction, OptimJointCompaction
from .optim_batched import BatchedOptimJointCompaction
from .random_subset_keys import RandomSubsetKeysCompaction
from .random_vector_keys import RandomVectorKeysCompaction
from .truncate import TruncationCompaction
from .highest_attention_keys import HighestAttentionKeysCompaction

__all__ = [
    'CompactionAlgorithm',
    'evaluate_compaction',
    'OMPCompaction',
    'BatchedOMPCompaction',
    'OMPFullCompaction',
    'OptimC1BetaCompaction',
    'OptimJointCompaction',
    'BatchedOptimJointCompaction',
    'RandomSubsetKeysCompaction',
    'RandomVectorKeysCompaction',
    'TruncationCompaction',
    'HighestAttentionKeysCompaction',
]

# Registry of per-layer-head algorithms with their string names
# Used by compaction_methods/registry.py
ALGORITHM_REGISTRY = {
    'omp': OMPCompaction,
    'batched_omp': BatchedOMPCompaction,
    'omp_full': OMPFullCompaction,
    'random_subset_keys': RandomSubsetKeysCompaction,
    'random_vector_keys': RandomVectorKeysCompaction,
    'truncate': TruncationCompaction,
    'optim_c1beta': OptimC1BetaCompaction,
    'optim_joint': OptimJointCompaction,
    'batched_optim_joint': BatchedOptimJointCompaction,
    'highest_attention_keys': HighestAttentionKeysCompaction,
}

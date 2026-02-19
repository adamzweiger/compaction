# compaction/compaction_methods/__init__.py
"""
Compaction Methods

This package contains the full KV cache compaction framework:
- Base classes for full-cache compaction algorithms
- Wrappers that apply per-layer-head algorithms to the full cache
- Registry and factory for instantiating compaction methods by name
"""
from .base import FullCacheCompactionAlgorithm
from .per_layer_head import PerLayerHeadCompaction
from .per_layer_head_on_policy import PerLayerHeadOnPolicyCompaction
from .global_highest_attention_keys import GlobalHighestAttentionKeysCompaction
from .global_omp import GlobalOMPCompaction
from .summarize import SummarizeCompaction
from .summarize_then_compact import SummarizeThenCompact
from .duo_attention import DuoAttentionCompaction
from .no_context import NoContextCompaction
from .registry import get_compaction_method, OriginalCacheMethod


# Lazy import for ChunkedCompaction to avoid circular import
def __getattr__(name):
    if name == 'ChunkedCompaction':
        from .chunked import ChunkedCompaction
        return ChunkedCompaction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'FullCacheCompactionAlgorithm',
    'PerLayerHeadCompaction',
    'PerLayerHeadOnPolicyCompaction',
    'GlobalHighestAttentionKeysCompaction',
    'GlobalOMPCompaction',
    'SummarizeCompaction',
    'SummarizeThenCompact',
    'DuoAttentionCompaction',
    'NoContextCompaction',
    'ChunkedCompaction',
    'get_compaction_method',
    'OriginalCacheMethod',
]

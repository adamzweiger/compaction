# compaction/compaction_methods/registry.py
"""
Registry and factory for compaction methods.

This module provides a unified interface to get compaction algorithms
for evaluation purposes.
"""
from typing import Dict, Any, Optional

from .base import FullCacheCompactionAlgorithm
from .per_layer_head import PerLayerHeadCompaction
from .per_layer_head_on_policy import PerLayerHeadOnPolicyCompaction
from .global_highest_attention_keys import GlobalHighestAttentionKeysCompaction
from .global_omp import GlobalOMPCompaction
from .summarize import SummarizeCompaction
from .summarize_then_compact import SummarizeThenCompact
from .duo_attention import DuoAttentionCompaction
from .no_context import NoContextCompaction
from ..algorithms import ALGORITHM_REGISTRY


def get_compaction_method(
    method_name: str,
    method_kwargs: Optional[Dict[str, Any]] = None
) -> FullCacheCompactionAlgorithm:
    """
    Get a full-cache compaction method by name.

    Parameters
    ----------
    method_name : str
        Name of the compaction method (config name). Options:
        - 'original': No compaction (returns original cache)
        - 'global_highest_attention_keys': Non-uniform compaction with global key selection
        - Per-layer-head methods: see algorithms.ALGORITHM_REGISTRY for full list
        - Future: custom full-cache methods can be added here
    method_kwargs : dict, optional
        Keyword arguments to pass to the algorithm constructor.
        If 'algorithm' key is present, it specifies the base algorithm to use.

    Returns
    -------
    compaction_method : FullCacheCompactionAlgorithm
        The compaction algorithm instance
    """
    method_kwargs = method_kwargs or {}

    if method_name == 'original':
        # Special case: return a no-op compaction method
        return OriginalCacheMethod()

    # Check if chunked compaction is enabled FIRST
    # This allows any compaction method to be wrapped with chunking
    # EXCEPT for methods that don't produce a KV cache (like no_context)
    chunking = method_kwargs.get('chunking', None)
    base_algorithm = method_kwargs.get('algorithm', method_name)

    # no_context removes all content and cannot be used with chunking
    # (it returns a string, not a KV cache that can be concatenated)
    if base_algorithm == 'no_context' and chunking is not None and chunking.lower() != 'none':
        print(f"Warning: 'no_context' method ignores chunking (chunking={chunking}) - "
              "removing all context doesn't require chunked processing")
        chunking = None

    if chunking is not None and chunking.lower() != 'none':
        # Lazy imports to avoid circular dependency
        from .chunked import ChunkedCompaction
        from ..chunking import get_chunking_strategy

        # Chunked compaction mode
        chunking_strategy = get_chunking_strategy(
            chunking,
            chunk_size=method_kwargs.get('chunk_size', 4096)
        )

        # Build kwargs for the inner compaction method (exclude chunking-specific fields)
        excluded_keys = ('chunking', 'chunk_size', 'use_kv_based')
        inner_kwargs = {k: v for k, v in method_kwargs.items() if k not in excluded_keys}

        # Create a factory function that creates the inner compaction method
        # This allows any FullCacheCompactionAlgorithm to be used, not just per-layer-head
        def inner_method_factory():
            return get_compaction_method(method_name, inner_kwargs)

        return ChunkedCompaction(
            inner_compaction_method=inner_method_factory,
            chunking_strategy=chunking_strategy,
            config_name=method_name,
            use_kv_based=method_kwargs.get('use_kv_based', True),
        )

    # Extract base algorithm name (defaults to method_name if not specified)
    base_algorithm = method_kwargs.get('algorithm', method_name)

    if base_algorithm == 'global_highest_attention_keys':
        algorithm_kwargs = {k: v for k, v in method_kwargs.items() if k != 'algorithm'}
        return GlobalHighestAttentionKeysCompaction(config_name=method_name, **algorithm_kwargs)

    if base_algorithm == 'duo_attention':
        algorithm_kwargs = {k: v for k, v in method_kwargs.items() if k != 'algorithm'}
        return DuoAttentionCompaction(config_name=method_name, **algorithm_kwargs)

    if base_algorithm == 'global_omp':
        algorithm_kwargs = {k: v for k, v in method_kwargs.items() if k != 'algorithm'}
        return GlobalOMPCompaction(config_name=method_name, **algorithm_kwargs)

    if base_algorithm == 'summarize':
        algorithm_kwargs = {k: v for k, v in method_kwargs.items() if k != 'algorithm'}
        return SummarizeCompaction(config_name=method_name, **algorithm_kwargs)

    if base_algorithm == 'no_context':
        # Filter out chunking-related kwargs that no_context doesn't accept
        excluded_keys = ('algorithm', 'chunking', 'chunk_size', 'use_kv_based')
        algorithm_kwargs = {k: v for k, v in method_kwargs.items() if k not in excluded_keys}
        return NoContextCompaction(config_name=method_name, **algorithm_kwargs)

    if base_algorithm == 'summarize_then_compact':
        # Summarize-then-compact: first summarize, then apply inner compaction method
        # inner_algorithm specifies which compaction method to use after summarization
        inner_algorithm = method_kwargs.get('inner_algorithm', 'omp')
        summarize_prompt = method_kwargs.get(
            'prompt',
            "Summarize the following text:\n\n{article_text}\n\nSummary:"
        )

        # Build kwargs for the inner compaction method (exclude summarize-specific fields)
        excluded_keys = ('algorithm', 'inner_algorithm', 'prompt')
        inner_kwargs = {k: v for k, v in method_kwargs.items() if k not in excluded_keys}

        # Create a factory function for the inner method
        def inner_method_factory():
            return get_compaction_method(inner_algorithm, inner_kwargs)

        return SummarizeThenCompact(
            inner_compaction_method=inner_method_factory,
            summarize_prompt=summarize_prompt,
            config_name=method_name,
        )

    # Get algorithm kwargs (everything except wrapper-level fields)
    # These wrapper-level fields are handled by PerLayerHead(OnPolicy)Compaction, not the inner algorithm
    wrapper_fields = ('algorithm', 'on_policy', 'use_batched', 'precomputed_budget_path', 'max_ratio_per_head')
    algorithm_kwargs = {k: v for k, v in method_kwargs.items() if k not in wrapper_fields}

    # Check if error propagation is enabled
    use_on_policy = method_kwargs.get('on_policy', False)

    # Check if batched processing is enabled
    use_batched = method_kwargs.get('use_batched', False)

    # Get precomputed budget path if specified
    precomputed_budget_path = method_kwargs.get('precomputed_budget_path', None)

    # Get max ratio per head (for temperature blending of precomputed budgets)
    max_ratio_per_head = method_kwargs.get('max_ratio_per_head', 1.0)

    if base_algorithm in ALGORITHM_REGISTRY:
        algorithm_class = ALGORITHM_REGISTRY[base_algorithm]
        if use_on_policy:
            return PerLayerHeadOnPolicyCompaction(
                config_name=method_name,
                algorithm_class=algorithm_class,
                algorithm_kwargs=algorithm_kwargs,
                precomputed_budget_path=precomputed_budget_path,
                max_ratio_per_head=max_ratio_per_head,
            )
        else:
            return PerLayerHeadCompaction(
                config_name=method_name,
                algorithm_class=algorithm_class,
                algorithm_kwargs=algorithm_kwargs,
                use_batched=use_batched,
                precomputed_budget_path=precomputed_budget_path,
                max_ratio_per_head=max_ratio_per_head,
            )
    else:
        raise ValueError(
            f"Unknown base algorithm: {base_algorithm}. "
            f"Available algorithms: 'original', 'global_highest_attention_keys' (or 'nonuniform'), "
            f"'global_omp' (or 'nonuniform_omp'), 'summarize', 'no_context', {', '.join(ALGORITHM_REGISTRY.keys())}"
        )


class OriginalCacheMethod(FullCacheCompactionAlgorithm):
    """
    No-op compaction method that returns the original context text.

    This is used as a baseline to compare against. Generation uses vLLM
    with the full context text rather than a compacted KV cache.
    """

    def name(self) -> str:
        return "original"

    def returns_cache(self) -> bool:
        """Original method returns context text, not a cache."""
        return False

    def requires_preextracted_cache(self) -> bool:
        """Original method doesn't need a pre-extracted cache."""
        return False

    def compact_kv_cache(
        self,
        past_key_values,
        target_size,
        indices,
        query_config,
        model=None,
        tokenizer=None,
        formatted_context=None,
        compute_stats=False,
        vllm_model=None,
        verbose_logging=False,
        sliding_layer_indices=None,
    ):
        """
        Return the original context text without compaction.

        Returns
        -------
        context_text : str
            The original formatted context (unchanged)
        stats : dict
            Statistics about the "compaction" (ratio=1.0)
        """
        # Get seq_len from past_key_values if available, otherwise estimate from context
        if past_key_values is not None:
            # Find a non-sliding layer to get the full sequence length
            # (sliding layers may have shorter seq_len due to window size)
            sliding_layer_indices = sliding_layer_indices or set()
            ref_layer_idx = 0
            for i in range(len(past_key_values)):
                if i not in sliding_layer_indices:
                    ref_layer_idx = i
                    break
            seq_len = past_key_values[ref_layer_idx][0].shape[2]
        else:
            # Estimate from tokenizer if no cache provided
            seq_len = len(tokenizer.encode(formatted_context, add_special_tokens=False)) if tokenizer and formatted_context else 0

        stats = {
            'method': 'original',
            'compaction_ratio': 1.0,
            'tensor_compacted_seq_len': seq_len,
            'effective_compacted_seq_len': seq_len,
        }

        return formatted_context, stats

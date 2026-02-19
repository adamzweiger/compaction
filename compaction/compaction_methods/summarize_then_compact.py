# compaction/compaction_methods/summarize_then_compact.py
"""
Summarize-then-compact compaction method.

This method first summarizes the article using vLLM, then applies a cache-based
compaction method (like OMP) to the summarized context's KV cache.
"""
import torch
from typing import Tuple, Dict, Optional, Any, Callable, Union

from .base import FullCacheCompactionAlgorithm
from .summarize import SummarizeCompaction
from ..query_generation import QueryConfig


class SummarizeThenCompact(FullCacheCompactionAlgorithm):
    """
    Two-stage compaction: summarize first, then apply cache compaction.

    This method:
    1. Uses vLLM to generate a summary of the article (via SummarizeCompaction)
    2. Extracts KV cache from the summarized context
    3. Applies an inner compaction method (e.g., PerLayerHeadCompaction with OMP)
    4. Returns the compacted cache

    This combines the benefits of semantic compression (summarization) with
    representation compression (cache compaction).
    """

    def __init__(
        self,
        inner_compaction_method: Union[FullCacheCompactionAlgorithm, Callable[[], FullCacheCompactionAlgorithm]],
        summarize_prompt: str = "Summarize the following text:\n\n{article_text}\n\nSummary:",
        config_name: Optional[str] = None,
    ):
        """
        Initialize the summarize-then-compact method.

        Parameters
        ----------
        inner_compaction_method : FullCacheCompactionAlgorithm or callable
            The compaction method to apply after summarization. Can be:
            - A FullCacheCompactionAlgorithm instance
            - A callable that returns a new FullCacheCompactionAlgorithm
        summarize_prompt : str, optional
            Prompt template for summarization. Must contain '{article_text}' placeholder.
        config_name : str, optional
            Name of the configuration (used for logging).
        """
        self.inner_compaction_method = inner_compaction_method
        self.summarize_prompt = summarize_prompt
        self.config_name = config_name

        # Create summarizer (with return_cache=False since we handle cache extraction ourselves)
        self.summarizer = SummarizeCompaction(
            prompt=summarize_prompt,
            config_name=f"{config_name}_summarize" if config_name else "summarize_stage",
            return_cache=False,
        )

        # Determine if we have a factory or an instance
        self._is_factory = callable(inner_compaction_method) and not isinstance(
            inner_compaction_method, FullCacheCompactionAlgorithm
        )

    def _get_inner_method(self) -> FullCacheCompactionAlgorithm:
        """Get the inner compaction method, creating a new instance if factory."""
        if self._is_factory:
            return self.inner_compaction_method()
        return self.inner_compaction_method

    def name(self) -> str:
        """Return the config name if provided, otherwise 'summarize_then_compact'."""
        if self.config_name:
            return self.config_name
        return "summarize_then_compact"

    def returns_cache(self) -> bool:
        """Return True since this method returns a compacted cache."""
        return True

    def requires_preextracted_cache(self) -> bool:
        """This method handles its own cache extraction."""
        return False

    def compact_kv_cache(
        self,
        past_key_values,
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        compute_stats: bool = False,
        vllm_model: Optional[Any] = None,
        verbose_logging: bool = False,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        Summarize the article, then compact the summarized cache.

        Parameters
        ----------
        past_key_values : ignored
            Not used - we extract our own cache from the summarized context
        target_size : int
            Target compacted sequence length for the final output
        indices : range
            Indices of the article portion in formatted_context
        query_config : QueryConfig
            Configuration for query generation (used by inner compaction method)
        model : Any
            HuggingFace model instance
        tokenizer : Any
            Tokenizer
        formatted_context : str
            Formatted context string containing the article
        compute_stats : bool
            If True, compute detailed statistics
        vllm_model : Any
            Pre-initialized vLLM model for summary generation
        verbose_logging : bool
            Whether to save selected indices in stats
        sliding_layer_indices : set, optional
            Set of layer indices that use sliding window attention

        Returns
        -------
        compacted_cache : tuple of tuples
            ((C1_layer0, beta_layer0, C2_layer0), ...)
        stats : dict
            Combined statistics from both stages
        """
        if indices is None:
            raise ValueError(
                "indices must be provided for summarize-then-compact. "
                "This should specify which portion of the context to summarize."
            )

        print(f"\n{'='*60}")
        print(f"Summarize-then-compact: {self.name()}")
        print(f"{'='*60}")

        # Get original sequence info
        tokenized_full = tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False)
        full_token_ids = tokenized_full.input_ids[0]
        original_seq_len = len(full_token_ids)
        original_article_len = len(indices)
        non_article_tokens = original_seq_len - original_article_len

        print(f"Original context: {original_seq_len} tokens (article: {original_article_len}, other: {non_article_tokens})")

        # Stage 1: Summarize
        # Pass target_size to summarizer - it will compute max_summary_tokens from this
        print(f"\n--- Stage 1: Summarization ---")
        summarized_context, summarize_stats = self.summarizer.compact_kv_cache(
            past_key_values=None,
            target_size=target_size,  # Summarizer uses this to compute max summary length
            indices=indices,
            query_config=query_config,
            model=model,
            tokenizer=tokenizer,
            formatted_context=formatted_context,
            compute_stats=False,  # Don't compute stats for summarization stage
            vllm_model=vllm_model,
            sliding_layer_indices=sliding_layer_indices,
        )

        # Get summarized context info
        summarized_token_ids = tokenizer.encode(summarized_context, add_special_tokens=False)
        summarized_seq_len = len(summarized_token_ids)
        summarized_article_len = summarize_stats.get('summary_tokens', summarized_seq_len - non_article_tokens)

        print(f"Summarized context: {summarized_seq_len} tokens (summary: {summarized_article_len}, other: {non_article_tokens})")

        # Stage 2: Extract KV cache from summarized context
        print(f"\n--- Stage 2: Cache extraction ---")
        device = next(model.parameters()).device

        summarized_inputs = tokenizer(
            summarized_context, return_tensors="pt", add_special_tokens=False
        ).to(device)

        with torch.no_grad():
            outputs = model(summarized_inputs.input_ids, use_cache=True, return_dict=True)
        summarized_past_key_values = outputs.past_key_values

        print(f"Extracted KV cache: {len(summarized_past_key_values)} layers, {summarized_seq_len} tokens")

        # Compute new article indices in the summarized context
        # The article portion is now the summary, which is in the same position
        # (between text_before and text_after)
        new_article_start = indices.start  # Same start position
        new_article_end = new_article_start + summarized_article_len
        new_article_indices = range(new_article_start, new_article_end)

        print(f"New article indices: {new_article_start}-{new_article_end} ({len(new_article_indices)} tokens)")

        # Stage 3: Apply inner compaction method
        # Compute new target size as a fraction of the summarized article portion
        # target_size can be a fraction (0 < x < 1) or absolute token count
        if 0 < target_size < 1:
            # Fraction: apply to summarized article length, then add non-article tokens
            compacted_article_target = max(1, int(target_size * summarized_article_len))
            compaction_target_size = compacted_article_target + non_article_tokens
        else:
            # Absolute: compute what fraction of original article this represents, apply to summarized article
            original_fraction = (target_size - non_article_tokens) / original_article_len
            compacted_article_target = max(1, int(original_fraction * summarized_article_len))
            compaction_target_size = compacted_article_target + non_article_tokens
        print(f"\n--- Stage 3: Cache compaction ---")
        if 0 < target_size < 1:
            print(f"Compaction target: {target_size:.1%} of summary article ({summarized_article_len}) = {compacted_article_target} + {non_article_tokens} non-article = {compaction_target_size} tokens")
        else:
            print(f"Compaction target: {original_fraction:.1%} of summary article ({summarized_article_len}) = {compacted_article_target} + {non_article_tokens} non-article = {compaction_target_size} tokens")
        inner_method = self._get_inner_method()
        print(f"Using inner method: {inner_method.name()}")

        compacted_cache, compact_stats = inner_method.compact_kv_cache(
            past_key_values=summarized_past_key_values,
            target_size=compaction_target_size,
            indices=new_article_indices,
            query_config=query_config,
            model=model,
            tokenizer=tokenizer,
            formatted_context=summarized_context,
            compute_stats=compute_stats,
            vllm_model=vllm_model,
            verbose_logging=verbose_logging,
            sliding_layer_indices=sliding_layer_indices,
        )

        # Combine stats from both stages
        stats = {
            'method': 'summarize_then_compact',
            'original_seq_len': original_seq_len,
            'original_article_tokens': original_article_len,
            # Summarization stage
            'summarize_stats': summarize_stats,
            'summarized_seq_len': summarized_seq_len,
            'summarized_article_tokens': summarized_article_len,
            'summarization_ratio': original_article_len / summarized_article_len if summarized_article_len > 0 else float('inf'),
            # Compaction stage
            'compact_stats': compact_stats,
            'tensor_compacted_seq_len': compact_stats.get('tensor_compacted_seq_len'),
            'effective_compacted_seq_len': compact_stats.get('effective_compacted_seq_len'),
            'effective_article_tokens': compact_stats.get('effective_article_tokens'),
            'tensor_article_tokens': compact_stats.get('tensor_article_tokens'),
            # Overall
            'compaction_ratio': original_seq_len / compact_stats.get('effective_compacted_seq_len', 1),
            'per_layer_head_metrics': compact_stats.get('per_layer_head_metrics', {}),
            'train_stats_time': compact_stats.get('train_stats_time', 0.0),
            'query_generation': compact_stats.get('query_generation', {}),
        }

        if 'is_partial_compaction' in compact_stats:
            stats['is_partial_compaction'] = compact_stats['is_partial_compaction']
        if 'compaction_indices' in compact_stats:
            stats['compaction_indices'] = compact_stats['compaction_indices']

        effective_len = stats.get('effective_compacted_seq_len', stats.get('tensor_compacted_seq_len', target_size))
        print(f"\n{'='*60}")
        print(f"Summarize-then-compact complete!")
        print(f"  Original: {original_seq_len} tokens (article: {original_article_len})")
        print(f"  After summarization: {summarized_seq_len} tokens (summary: {summarized_article_len})")
        print(f"  After compaction: {effective_len} tokens")
        print(f"  Overall compaction: {stats['compaction_ratio']:.2f}x")
        print(f"{'='*60}\n")

        return compacted_cache, stats

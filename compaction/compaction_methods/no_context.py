# compaction/compaction_methods/no_context.py
"""
No-context compaction method.

This method removes the article completely, replacing it with an empty string.
This serves as a baseline to evaluate QA performance with no article context.
"""
from typing import Tuple, Dict, Optional, Any

from .base import FullCacheCompactionAlgorithm


class NoContextCompaction(FullCacheCompactionAlgorithm):
    """
    Compaction by removing all article context.

    This is a text-based method that:
    1. Removes the article text completely from the context
    2. Returns the modified context text for vLLM generation

    This serves as a baseline to measure QA performance when no article context is provided.
    """

    def __init__(self, config_name: Optional[str] = None):
        """
        Initialize the no-context compaction method.

        Parameters
        ----------
        config_name : str, optional
            Name of the configuration (used for logging). If not provided, uses "no_context".
        """
        self.config_name = config_name

    def name(self) -> str:
        """Return the config name if provided, otherwise 'no_context'."""
        if self.config_name:
            return self.config_name
        return "no_context"

    def returns_cache(self) -> bool:
        """Return False since this method returns context text, not a cache."""
        return False

    def requires_preextracted_cache(self) -> bool:
        """No-context method doesn't need a pre-extracted cache."""
        return False

    def compact_kv_cache(
        self,
        past_key_values,
        target_size: int,
        indices: Optional[range],
        query_config,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        compute_stats: bool = False,
        vllm_model: Optional[Any] = None,
        verbose_logging: bool = False,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple[str, Dict]:
        """
        Remove all article context and return the modified context text.

        Parameters
        ----------
        past_key_values : ignored
            Not used for text-based methods
        target_size : int
            Target compacted sequence length (ignored - we just remove the article)
        indices : range, optional
            Indices of sequence positions to remove (the article portion).
            These correspond to token positions in formatted_context.
        query_config : ignored
            Not used for no_context
        model : Any
            Model instance (used for config info)
        tokenizer : Any
            Tokenizer
        formatted_context : str
            Formatted context string containing the article
        compute_stats : bool
            If True, compute stats (not implemented for no_context)
        vllm_model : optional
            Pre-initialized vLLM model (not used)
        verbose_logging : bool
            Not used for no_context
        sliding_layer_indices : set, optional
            Not used for no_context

        Returns
        -------
        no_context : str
            The context text with the article removed
        stats : dict
            Statistics about the compaction process
        """
        if indices is None:
            raise ValueError(
                "indices must be provided for no-context compaction. "
                "This should specify which portion of the context to remove (typically the article)."
            )

        print(f"\n{'='*60}")
        print("No-context compaction (removing article completely)")
        print(f"{'='*60}")

        # Tokenize to get token counts and extract article
        tokenized_full = tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False)
        full_token_ids = tokenized_full.input_ids[0]  # (seq_len,)
        seq_len = len(full_token_ids)

        # Extract article token indices
        article_token_ids = full_token_ids[indices.start:indices.stop]

        print(f"Original article length: {len(article_token_ids)} tokens")
        print(f"Removing article completely (replacing with empty string)")

        # Reconstruct formatted_context with empty string replacing the article
        tokens_before_article = full_token_ids[:indices.start]
        tokens_after_article = full_token_ids[indices.stop:]

        text_before = tokenizer.decode(tokens_before_article, skip_special_tokens=False)
        text_after = tokenizer.decode(tokens_after_article, skip_special_tokens=False)

        # Reconstruct with empty article (just concatenate before and after)
        no_context = text_before + text_after

        # Get new sequence length
        no_context_token_ids = tokenizer.encode(no_context, add_special_tokens=False)
        new_seq_len = len(no_context_token_ids)

        print(f"Original context length: {seq_len} tokens")
        print(f"No-context length: {new_seq_len} tokens")
        print(f"Compaction ratio: {seq_len / new_seq_len:.2f}x")

        # Compile statistics
        stats = {
            'method': 'no_context',
            'original_seq_len': seq_len,
            'tensor_compacted_seq_len': new_seq_len,
            'effective_compacted_seq_len': new_seq_len,
            'compaction_ratio': seq_len / new_seq_len,
            'original_article_tokens': len(article_token_ids),
            'removed_tokens': len(article_token_ids),
            'effective_article_tokens': 0,
            'tensor_article_tokens': 0,
            'per_layer_head_metrics': {},  # Required for test stats computation
            'train_stats_time': 0.0,  # No train stats for no_context
        }

        stats['is_partial_compaction'] = True
        stats['compaction_indices'] = {
            'start': indices.start,
            'end': indices.stop,
            'num_positions': len(indices),
        }
        stats['sub_target_size'] = 0

        print(f"\n{'='*60}")
        print(f"No-context compaction complete!")
        print(f"  Original: {seq_len} tokens")
        print(f"  No-context: {new_seq_len} tokens")
        print(f"  Compaction: {stats['compaction_ratio']:.2f}x")
        print(f"  Article: {len(article_token_ids)} tokens removed completely")
        print(f"{'='*60}\n")

        return no_context, stats

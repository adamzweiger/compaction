# compaction/compaction_methods/base.py
"""
Base class for full KV cache compaction algorithms.

These algorithms operate on the entire KV cache across all layers and heads,
unlike the per-layer-head algorithms in the algorithms/ package.
"""
import json
import torch
from typing import Tuple, Dict, Optional
from abc import ABC, abstractmethod

from ..query_generation import QueryConfig

class FullCacheCompactionAlgorithm(ABC):
    """
    Base class for algorithms that compact the entire KV cache.

    The input is the full KV cache from a model forward pass,
    and the output is a compacted version that can be used for generation.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name for logging."""
        pass

    def returns_cache(self) -> bool:
        """
        Return whether this method returns a compacted KV cache.

        If True (default), compact_kv_cache() returns (compacted_cache, stats) where
        compacted_cache is a tuple of (C1, beta, C2) tensors for each layer.
        Generation uses generate_with_compacted_cache_batch().

        If False, compact_kv_cache() returns (context_text, stats) where context_text
        is a string. Generation uses vLLM with the full context text.
        This is used for methods like 'original' (no compaction) and 'summarize'.

        Returns
        -------
        bool
            True if returning a cache, False if returning context text.
        """
        return True

    def requires_preextracted_cache(self) -> bool:
        """
        Return whether this method requires a pre-extracted KV cache.

        Most methods require the caller to first run a forward pass to extract
        the KV cache, then pass it to compact_kv_cache(). However, some methods
        (like ChunkedCompaction) handle their own KV cache extraction internally
        and should NOT have a full prefill run beforehand (to avoid OOM on long
        contexts).

        Returns
        -------
        bool
            True if compact_kv_cache() expects a valid past_key_values tuple.
            False if compact_kv_cache() handles its own cache extraction and
            past_key_values can be None.
        """
        return True

    @abstractmethod
    def compact_kv_cache(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: object,
        tokenizer: object,
        formatted_context: str,
        compute_stats: bool = False,
        vllm_model: Optional[object] = None,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        Compact the entire KV cache across all layers and heads.

        Parameters
        ----------
        past_key_values : tuple of tuples
            The KV cache from model.forward(..., use_cache=True).past_key_values
            Structure: ((keys_layer0, values_layer0), (keys_layer1, values_layer1), ...)
            Each keys/values tensor has shape (batch_size, num_heads, seq_len, head_dim)
        target_size : int
            Target compacted sequence length (t) for the full cache.
            If indices is provided, this refers to the total length after compaction,
            not just the compacted subset.
        indices : range, optional
            Indices of the sequence positions to compact. If None, compact the entire
            sequence. If provided (e.g., range(start, end)), only compact those positions
            and leave the rest unchanged. The algorithm will compute the sub-target size
            needed to achieve the overall target_size.
        query_config : QueryConfig
            Configuration for query generation (required)
        model : object
            Model instance (required)
        tokenizer : object
            Tokenizer (required)
        formatted_context : str
            Formatted context string (required)
        compute_stats : bool
            If True, compute train stats using generated queries (default: False)
        vllm_model : object, optional
            Pre-initialized vLLM model for self-study query generation (default: None)
        sliding_layer_indices : set, optional
            Set of layer indices that use sliding window attention (e.g., for Gemma3).
            These layers should keep their original KV cache instead of being compacted.

        Returns
        -------
        compacted_cache : tuple of tuples
            Compacted KV cache in the format:
            ((C1_layer0, beta_layer0, C2_layer0), (C1_layer1, beta_layer1, C2_layer1), ...)
            where:
            - C1 has shape (batch_size, num_heads, t, head_dim) - compacted keys
            - beta has shape (batch_size, num_heads, t) - bias terms
            - C2 has shape (batch_size, num_heads, t, head_dim) - compacted values
        stats : dict
            Statistics about the compaction process (e.g., per-layer metrics,
            cosine similarities, selected indices, etc.)
        """
        pass

    def apply_compacted_cache(
        self,
        compacted_cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        query_tensor: torch.Tensor,
        layer_idx: int,
        head_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply compacted attention at inference time.

        This is a helper method that can be used during generation to compute
        attention with the compacted cache.

        Parameters
        ----------
        compacted_cache : tuple of tuples
            The compacted cache from compact_kv_cache
        query_tensor : Tensor
            Query tensor at inference time
            Shape: (batch_size, num_heads, seq_len, head_dim) or similar
        layer_idx : int
            Which layer to apply compacted attention
        head_idx : int, optional
            If provided, apply only to this head. Otherwise apply to all heads.

        Returns
        -------
        output : Tensor
            Attention output using compacted cache
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses that support "
            "inference-time application of compacted cache."
        )

# Budget loading and capping utilities

def load_budgets_from_json(
    path: str,
    num_layers: int,
    num_heads: int,
) -> Dict[Tuple[int, int], Optional[float]]:
    """
    Load per-layer/head budget proportions from a JSON file.

    The JSON file should have keys like 'L0H0', 'L0H1', etc. with float values.
    Example: {"L0H0": 0.014, "L0H1": 0.005, ...}

    The values are interpreted as proportions (0-1 range) that should sum to ~1.0 across all heads.

    Parameters
    ----------
    path : str
        Path to the JSON file
    num_layers : int
        Number of layers
    num_heads : int
        Number of heads per layer

    Returns
    -------
    budgets : dict
        Mapping from (layer_idx, head_idx) to proportion (0-1 range), or None if missing
    """
    with open(path, 'r') as f:
        data = json.load(f)

    budgets = {}
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            key_str = f'L{layer_idx}H{head_idx}'
            if key_str in data:
                budgets[(layer_idx, head_idx)] = data[key_str]
            else:
                budgets[(layer_idx, head_idx)] = None

    return budgets


def apply_max_ratio_cap(
    proportions: Dict[Tuple[int, int], Optional[float]],
    max_ratio_per_head: float,
    target_ratio: float,
    total_heads: int,
    verbose: bool = True,
) -> Dict[Tuple[int, int], Optional[float]]:
    """
    Apply max ratio cap using water-filling redistribution.

    Caps heads that exceed max_ratio_per_head and redistributes excess budget
    to other heads according to their relative proportions.

    Parameters
    ----------
    proportions : dict
        Mapping from (layer_idx, head_idx) to proportion (0-1 range), or None
    max_ratio_per_head : float
        Maximum ratio any single head can have (e.g., 1.0 means head can keep all its tokens)
    target_ratio : float
        Overall target compression ratio (actual_target_size / article_len)
    total_heads : int
        Total number of heads being compacted (num_global_layers * num_heads)
    verbose : bool
        Whether to print debug information

    Returns
    -------
    capped_proportions : dict
        New proportions with max ratio cap applied
    """
    if max_ratio_per_head >= 1.0 and target_ratio * total_heads <= 1.0:
        # No capping needed
        return proportions

    # max_proportion = max_ratio_per_head / (target_ratio * total_heads)
    # This is the max proportion a head can have before it exceeds max_ratio_per_head
    if target_ratio * total_heads > 0:
        max_proportion = max_ratio_per_head / (target_ratio * total_heads)
    else:
        return proportions

    # Separate valid and None proportions
    valid_items = [(k, v) for k, v in proportions.items() if v is not None]
    if not valid_items:
        return proportions

    # Check if any head exceeds the cap
    max_opt_proportion = max(v for _, v in valid_items)
    if max_opt_proportion <= max_proportion:
        return proportions

    # Iterative redistribution: cap heads that exceed max_proportion and
    # redistribute their excess to other heads proportionally
    result = {k: v for k, v in proportions.items()}

    for iteration in range(total_heads + 1):  # Max iterations = number of heads
        # Find heads that exceed cap
        excess_total = 0.0
        uncapped_total = 0.0
        capped_keys = set()

        for key, prop in result.items():
            if prop is None:
                continue
            if prop > max_proportion:
                excess_total += prop - max_proportion
                capped_keys.add(key)
            else:
                uncapped_total += prop

        if excess_total == 0:
            break  # No more excess to redistribute

        if uncapped_total == 0:
            # All heads are capped, just cap them all
            for key in result:
                if result[key] is not None:
                    result[key] = min(result[key], max_proportion)
            break

        # Cap the exceeding heads and redistribute to uncapped heads
        for key in result:
            if result[key] is None:
                continue
            if key in capped_keys:
                result[key] = max_proportion
            else:
                # Redistribute proportionally
                result[key] = result[key] + (result[key] / uncapped_total) * excess_total

    if verbose:
        new_max = max((p for p in result.values() if p is not None), default=0)
        original_max = max((p for p in proportions.values() if p is not None), default=0)
        print(f"Applied max ratio cap (max_ratio_per_head={max_ratio_per_head})")
        print(f"  Target ratio: {target_ratio:.4f}")
        print(f"  Max allowed proportion: {max_proportion:.6f}")
        print(f"  Original max proportion: {original_max:.6f}")
        print(f"  New max proportion: {new_max:.6f}")

    return result

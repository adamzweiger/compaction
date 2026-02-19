# models/cache.py
from typing import Optional, Dict, Any, Tuple, List, Union
import torch
from torch import Tensor

from transformers.cache_utils import CacheLayerMixin, Cache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer


class CompactedPrefixLayer(CacheLayerMixin):
    """
    One layer's cache for compacted prefix + appended tokens.

    C1:  (B, KV, t, D)   -> initial 'keys' prefix
    C2:  (B, KV, t, D)   -> initial 'values' prefix
    beta:(B, KV, t)      -> attention biases for prefix
    """

    is_sliding = False
    is_compileable = False  # Important to avoid compiling this dynamic-style cache

    def __init__(self, C1: Tensor, beta: Tensor, C2: Tensor, clone: bool = False):
        super().__init__()

        # Store the compacted prefix as the initial K/V
        # Shapes are assumed (B, KV, t, D) / (B, KV, t)
        if clone:
            self.keys = C1.clone()
            self.values = C2.clone()
            self.beta = beta.clone()
        else:
            self.keys = C1
            self.values = C2
            self.beta = beta

        self.base_len = self.keys.shape[-2]  # t
        self.dtype = self.keys.dtype
        self.device = self.keys.device
        self.is_initialized = True

    def lazy_initialization(self, key_states: Tensor):
        """
        For this cache, we already have a prefix at construction time.
        If someone calls lazy_initialization(), we just ensure dtype/device
        are consistent. You could also assert here if you want.
        """
        if not self.is_initialized:
            self.dtype = key_states.dtype
            self.device = key_states.device
            # Empty prefix in this weird edge case
            self.keys = torch.tensor([], dtype=self.dtype, device=self.device).view(
                key_states.shape[0], key_states.shape[1], 0, key_states.shape[-1]
            )
            self.values = self.keys.clone()
            self.base_len = 0
            self.is_initialized = True

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Append new tokens to the existing (compacted-prefix + previous) K/V.

        key_states, value_states: (B, KV, cur_len, D) for the new tokens.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # Standard dynamic behavior: concat on seq dimension (-2)
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)

        # Return full K/V for this layer (prefix + all appended tokens)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: Tensor) -> Tuple[int, int]:
        """
        HF contract: return (kv_length, kv_offset) where kv_length is the length of
        **past states** plus the current query block length, and kv_offset is the
        starting index (0 for full attention).
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """
        Sequence length of cached states for this layer.
        (prefix + all appended tokens).
        """
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return int(self.keys.shape[-2])

    def get_max_cache_shape(self) -> int:
        """
        Dynamic cache: no static maximum length.
        """
        return -1


class CompactedPrefixCache(Cache):
    """
    Cache container that holds one CompactedPrefixLayer per transformer layer,
    with per-layer rope_base support. Stores pad_counts per sequence in the batch
    to aid in RoPE position calculation.

    `compacted_cache` is a tuple-of-tuples like:
        compacted_cache[layer_idx] = (C1, beta, C2)
        C1, C2: (B, KV, t, D)
        beta:   (B, KV, t)

    For models with sliding window layers, the C1/C2 tensors contain the keys/values
    and beta is ignored (should be zeros). These layers use DynamicSlidingWindowLayer
    which keeps only the last `sliding_window` tokens.

    RoPE Position Handling
    ----------------------
    All layers use the same rope_base offset, computed as:
        rope_base = original_seq_len - max_compacted_prefix_len

    This works because cache_position is based on the maximum cache length, so all
    layers are at the same absolute position in the original sequence. Layers just
    have different amounts of historical context in their KV caches.
    """

    def __init__(
        self,
        compacted_cache: Tuple[Tuple[Tensor, Tensor, Tensor], ...],
        original_seq_len: Optional[int] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
        pad_counts: Optional[Tensor] = None,
        clone: bool = False,
        sliding_layer_indices: Optional[set] = None,
        sliding_window: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        compacted_cache : tuple of tuples
            (C1, beta, C2) for each layer. For sliding layers, C1/C2 are keys/values
            and beta is ignored (should be zeros).
        original_seq_len : int, optional
            Original sequence length before compaction (used for RoPE offset).
        sliding_layer_indices : set, optional
            Set of layer indices that use sliding window attention.
        sliding_window : int, optional
            Sliding window size for sliding layers. Required if sliding_layer_indices is provided.
        """
        layers: List[Union[CompactedPrefixLayer, DynamicSlidingWindowLayer]] = []
        sliding_indices = sliding_layer_indices or set()

        # Track the maximum compacted prefix length to compute rope_base
        max_compacted_len = 0

        for layer_idx, (C1, beta, C2) in enumerate(compacted_cache):
            if layer_idx in sliding_indices:
                # Use DynamicSlidingWindowLayer for sliding layers
                # C1 = keys, C2 = values, beta is ignored (should be zeros)
                if sliding_window is None:
                    raise ValueError("sliding_window must be provided when sliding_layer_indices is set")
                sliding_layer = DynamicSlidingWindowLayer(sliding_window=sliding_window)
                # Initialize with the provided keys/values from C1/C2
                sliding_layer.lazy_initialization(C1)
                if clone:
                    sliding_layer.keys = C1.clone()
                    sliding_layer.values = C2.clone()
                else:
                    sliding_layer.keys = C1
                    sliding_layer.values = C2
                # cumulative_length tracks total tokens seen, not just what's stored.
                sliding_layer.cumulative_length = original_seq_len if original_seq_len is not None else C1.shape[-2]
                layers.append(sliding_layer)
            else:
                # Use CompactedPrefixLayer for global attention layers
                layer = CompactedPrefixLayer(C1, beta, C2, clone=clone)
                layers.append(layer)

                # Track the maximum compacted length across all global layers
                compacted_len = C1.shape[-2]
                max_compacted_len = max(max_compacted_len, compacted_len)

        # Compute single rope_base for all layers: original_seq_len - max_compacted_len
        if original_seq_len is not None and max_compacted_len > 0:
            self._rope_base = int(original_seq_len) - int(max_compacted_len)
        else:
            self._rope_base = 0

        super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)

        self._pad_counts = pad_counts
        self._sliding_layer_indices = sliding_indices

    def rope_base(self) -> int:
        """
        Get the rope_base offset for the cache.

        For nonuniform caches, all layers use the same rope_base because cache_position
        is based on the maximum cache length. This ensures all layers get the correct
        RoPE embeddings for the current absolute position in the sequence.

        Returns
        -------
        int
            The rope_base offset (original_seq_len - max_compacted_prefix_len).
        """
        return self._rope_base

    def beta_for_layer(self, i: int) -> Tensor:
        layer = self.layers[i]
        if isinstance(layer, CompactedPrefixLayer):
            return layer.beta
        else:
            # Sliding layers don't have beta - return zeros
            # This shouldn't be called for sliding layers anyway
            keys = layer.keys
            return torch.zeros(keys.shape[0], keys.shape[1], keys.shape[2],
                             dtype=keys.dtype, device=keys.device)

    def pad_counts(self) -> Optional[Tensor]:
        return self._pad_counts

    def is_sliding_layer(self, i: int) -> bool:
        return i in self._sliding_layer_indices

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """
        Returns the sequence length of the cache.

        For CompactedPrefixCache with variable-length global layers, we return the
        MAXIMUM compacted length across all global layers. This ensures the attention
        mask created by HF's masking utilities is large enough to accommodate all layers.
        Individual layers will slice the mask to their actual KV length.

        Note: cache_position will start at this maximum value, but that's okay because:
        1. Attention masks are sliced per-layer to match actual KV lengths
        2. RoPE positions are corrected with per-layer rope_base offsets
        3. KV cache updates don't use cache_position (they just concatenate)
        """
        if layer_idx >= len(self.layers):
            return 0

        # Find the maximum length across all global (non-sliding) layers
        max_len = 0
        for i, layer in enumerate(self.layers):
            if i not in self._sliding_layer_indices:
                layer_len = layer.get_seq_length()
                if layer_len > max_len:
                    max_len = layer_len

        # Fallback: if somehow all layers are sliding (shouldn't happen), use layer 0
        if max_len == 0:
            return self.layers[layer_idx].get_seq_length()
        return max_len

    def get_mask_sizes(self, cache_position: Tensor, layer_idx: int = 0) -> Tuple[int, int]:
        """
        Return mask sizes for nonuniform caches.

        For sliding layers, return that specific layer's actual KV length.
        For global layers, return the MAXIMUM length across all global layers.
        """
        kv_offset = 0
        query_length = cache_position.shape[0]

        # For sliding layers, return that specific layer's actual length
        if layer_idx < len(self.layers) and layer_idx in self._sliding_layer_indices:
            return self.layers[layer_idx].get_mask_sizes(cache_position)
        else:
            # For global layers, return the maximum across all global layers
            layer_cache_length = self.get_seq_length()

        kv_length = layer_cache_length + query_length
        return kv_length, kv_offset


def clone_dynamic_cache(pkv: DynamicCache) -> DynamicCache:
    """Clone a DynamicCache by creating a new instance with cloned tensors."""
    out = DynamicCache(offloading=getattr(pkv, "offloading", False),
                       offload_only_non_sliding=getattr(pkv, "only_non_sliding", False))
    out.layers = []
    for layer in pkv.layers:
        if isinstance(layer, DynamicSlidingWindowLayer):
            nl = DynamicSlidingWindowLayer(sliding_window=layer.sliding_window)
            nl.cumulative_length = layer.cumulative_length
        else:
            nl = DynamicLayer()

        nl.is_initialized = layer.is_initialized
        if layer.is_initialized:
            nl.dtype, nl.device = layer.keys.dtype, layer.keys.device
            nl.keys = layer.keys.clone()
            nl.values = layer.values.clone()
        out.layers.append(nl)
    return out


def clone_compacted_prefix_cache(pkv: CompactedPrefixCache) -> CompactedPrefixCache:
    """Clone a CompactedPrefixCache by extracting the base prefix from each layer."""
    compacted_cache_tuples = []
    sliding_layer_indices = set()
    sliding_window = None

    for layer_idx, layer in enumerate(pkv.layers):
        if isinstance(layer, DynamicSlidingWindowLayer):
            # For sliding window layers, clone the full keys/values
            # Beta is zeros (not used for sliding layers)
            C1 = layer.keys.clone()
            C2 = layer.values.clone()
            beta = torch.zeros(C1.shape[0], C1.shape[1], C1.shape[2],
                             dtype=C1.dtype, device=C1.device)
            compacted_cache_tuples.append((C1, beta, C2))
            sliding_layer_indices.add(layer_idx)
            sliding_window = layer.sliding_window
        elif isinstance(layer, CompactedPrefixLayer):
            # Clone the cache to prevent mutation during forward pass
            C1 = layer.keys.clone()
            beta = layer.beta.clone()
            C2 = layer.values.clone()
            compacted_cache_tuples.append((C1, beta, C2))
        else:
            raise TypeError(f"Unsupported layer type: {type(layer)}")

    # Find max_compacted_len across all non-sliding layers (matching constructor logic)
    max_compacted_len = 0
    for layer_idx, layer in enumerate(pkv.layers):
        if layer_idx not in sliding_layer_indices and isinstance(layer, CompactedPrefixLayer):
            max_compacted_len = max(max_compacted_len, layer.base_len)

    return CompactedPrefixCache(
        compacted_cache=tuple(compacted_cache_tuples),
        original_seq_len=pkv.rope_base() + max_compacted_len,
        pad_counts=pkv.pad_counts(),
        sliding_layer_indices=sliding_layer_indices if sliding_layer_indices else None,
        sliding_window=sliding_window,
    )

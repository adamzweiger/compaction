# compaction/query_generation/cache_keys.py
"""Cache keys query generation for KV cache compaction."""

import torch
from typing import Optional, Tuple, Dict, Any

from .config import CacheKeysConfig


class CacheKeysQueryGenerator:
    """
    Generate queries from cache keys for KV cache compaction.

    This class uses the key vectors from the KV cache itself as query vectors,
    optionally applying q_norm scaling to match the scale of actual queries.

    All shapes are in KV space:

        (num_layers, num_kv_heads, n_queries_per_kv_head, head_dim)
    """

    def __init__(
        self,
        model,  # The model instance (e.g., Qwen3ForCausalLM)
        config: CacheKeysConfig,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the cache keys query generator.

        Parameters
        ----------
        model : transformers.PreTrainedModel
            The language model (used to extract q_norm weights if scale_by_qnorm=True)
        config : CacheKeysConfig
            Configuration for cache keys query generation
        device : str, optional
            Device to use. If None, uses model's device.
        dtype : torch.dtype, optional
            Data type for queries. If None, uses model's dtype.
        """
        self.model = model
        self.config = config
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

    def generate_queries(
        self,
        n_queries_per_kv_head: int,
        past_key_values: Optional[Tuple] = None,
        indices: Optional[range] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate queries from cache keys, in KV space.

        Parameters
        ----------
        n_queries_per_kv_head : int
            Number of queries to generate per KV head (used for subsampling if needed)
        past_key_values : tuple, optional
            Pre-computed KV cache.
            Expected structure:
                past_key_values[layer_idx][0] = keys for that layer
                keys shape: (batch_size, num_kv_heads, seq_len, head_dim)
        indices : range, optional
            Indices of sequence positions to extract keys from. If None, use all positions.

        Returns
        -------
        queries : torch.Tensor
            Queries extracted from cache keys, in KV space.
            Shape: (num_layers, num_kv_heads, n_queries_per_kv_head, head_dim)
        stats : dict
            Statistics about query generation
        """
        if past_key_values is None:
            raise ValueError("past_key_values is required for cache_keys query generation")

        # Convert range to list if provided
        indices_list = list(indices) if indices is not None else None

        # Collect keys for each layer
        keys_all_layers = []
        for layer_idx in range(len(past_key_values)):
            keys_layer = past_key_values[layer_idx][0]  # (batch_size, num_kv_heads, seq_len, head_dim)
            # Remove batch dimension (assuming batch_size=1)
            keys_layer = keys_layer[0].to(device=self.device, dtype=self.dtype)  # (num_kv_heads, seq_len, head_dim)

            # Slice keys if indices are provided
            if indices_list is not None:
                keys_layer = keys_layer[:, indices_list, :]  # (num_kv_heads, len(indices), head_dim)

            keys_all_layers.append(keys_layer)

        # Stack: (num_layers_cache, num_kv_heads, seq_len, head_dim)
        cache_keys = torch.stack(keys_all_layers, dim=0)

        num_layers, num_kv_heads, seq_len, head_dim = cache_keys.shape

        # Sanity check vs model config; we don't hard-fail if they differ, but record it.
        stats: Dict[str, Any] = {
            "n_cache_keys_available_per_kv_head": seq_len,
            "indices_provided": indices is not None,
            "num_indices": len(indices_list) if indices_list is not None else None,
        }

        queries = cache_keys  # (num_layers, num_kv_heads, seq_len, head_dim)

        # Scale by q_norm weights if requested
        if self.config.scale_by_qnorm:
            queries = self._apply_qnorm(queries)
            stats["scaled_by_qnorm"] = True
        else:
            stats["scaled_by_qnorm"] = False

        # Subsample along the sequence dimension if we have more keys than requested
        n_queries_available = queries.shape[2]
        if n_queries_available > n_queries_per_kv_head:
            # Randomly subsample the same positions for all layers / KV heads
            indices = torch.randperm(n_queries_available, device=queries.device)[:n_queries_per_kv_head]
            indices = indices.sort()[0]
            queries = queries[:, :, indices, :]

        return queries, stats

    def _apply_qnorm(
        self,
        queries: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale random queries by q_norm weights.

        For each layer, scales the random vectors by the
        corresponding q_norm weights. This ensures the random vectors match
        the scale of actual query vectors after q_norm is applied.

        Parameters
        ----------
        queries : torch.Tensor
            Random queries of shape (num_layers, num_attention_heads, n_queries, head_dim)
        head_dim : int
            Dimension of each attention head

        Returns
        -------
        scaled_queries : torch.Tensor
            Queries scaled by q_norm weights, same shape as input
        """
        num_layers, _, _, _ = queries.shape

        # For Qwen3, q_norm is applied per layer

        scaled_queries = queries.clone()

        # Disable gradients since we don't want to train on query vectors
        with torch.no_grad():
            for layer_idx in range(num_layers):
                # Get the q_norm module for this layer
                # For Qwen3: model.model.layers[layer_idx].self_attn.q_norm
                q_norm = self.model.model.layers[layer_idx].self_attn.q_norm

                # Take all heads & queries in this layer: (num_attention_heads, n_queries, head_dim)
                layer_queries = queries[layer_idx]

                # q_norm works on (..., head_dim)
                layer_scaled = q_norm(layer_queries)  # shape stays (H, N, D)

                scaled_queries[layer_idx] = layer_scaled

        return scaled_queries

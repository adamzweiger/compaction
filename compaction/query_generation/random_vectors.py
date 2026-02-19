# compaction/query_generation/random_vectors.py
"""Random vector query generation for KV cache compaction."""

import torch
from typing import Optional, Tuple, Dict, Any

from .config import RandomVectorConfig


class RandomVectorQueryGenerator:
    """
    Generate random vector queries for KV cache compaction.

    This class generates random normal vectors and optionally scales them
    by q_norm weights to match the scale of actual query vectors.
    """

    def __init__(
        self,
        model,  # The model instance (e.g., Qwen3ForCausalLM)
        config: RandomVectorConfig,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the random vector query generator.

        Parameters
        ----------
        model : transformers.PreTrainedModel
            The language model (used to extract q_norm weights if scale_by_qnorm=True)
        config : RandomVectorConfig
            Configuration for random vector query generation
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
        n_queries_per_attention_head: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate random vector queries.

        Parameters
        ----------
        n_queries_per_attention_head : int
            Number of queries to generate per attention head

        Returns
        -------
        queries : torch.Tensor
            Random queries of shape (num_layers, num_attention_heads, n_queries_per_attention_head, head_dim)
        stats : dict
            Statistics about query generation
        """
        # Get model dimensions
        num_layers = self.model.config.num_hidden_layers
        num_attention_heads = self.model.config.num_attention_heads
        # Try to get head_dim from config first (e.g., Qwen3 has this)
        head_dim = getattr(self.model.config, 'head_dim',
                           self.model.config.hidden_size // num_attention_heads)

        stats = {}

        # Generate random queries: (num_layers, num_attention_heads, n_queries_per_attention_head, head_dim)
        queries = torch.randn(
            num_layers, num_attention_heads, n_queries_per_attention_head, head_dim,
            dtype=self.dtype,
            device=self.device
        )

        # Scale by q_norm weights if requested
        if self.config.scale_by_qnorm:
            queries = self._apply_qnorm(queries)
            stats['scaled_by_qnorm'] = True
        else:
            stats['scaled_by_qnorm'] = False

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

                # q_norm_weights = q_norm.weight.data
                # for head_idx in range(num_attention_heads):
                #     # queries for this head: (n_queries, head_dim)
                #     # Scale element-wise by q_norm weights
                #     scaled_queries[layer_idx, head_idx, :, :] = queries[layer_idx, head_idx, :, :] * q_norm_weights

                # Take all heads & queries in this layer: (num_attention_heads, n_queries, head_dim)
                layer_queries = queries[layer_idx]

                # q_norm works on (..., head_dim)
                layer_scaled = q_norm(layer_queries)  # shape stays (H, N, D)

                scaled_queries[layer_idx] = layer_scaled

        return scaled_queries

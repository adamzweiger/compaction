# compaction/query_generation/query_generator.py
"""Unified query generation dispatcher."""

import torch
import time
from typing import Optional, Tuple, Dict, Any
from transformers import AutoTokenizer
from models.cache import CompactedPrefixCache
from transformers.cache_utils import DynamicCache

from .config import QueryConfig


class QueryGenerator:
    """
    Unified query generator that dispatches to different query generation methods.

    This class coordinates multiple query generation methods (e.g., self-study,
    random vectors) based on the QueryConfig specification.
    """

    def __init__(
        self,
        model,  # The model instance (e.g., Qwen3ForCausalLM)
        tokenizer: AutoTokenizer,
        config: QueryConfig,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        vllm_model: Optional[Any] = None,
    ):
        """
        Initialize the query generator.

        Parameters
        ----------
        model : transformers.PreTrainedModel
            The language model
        tokenizer : AutoTokenizer
            Tokenizer for the model
        config : QueryConfig
            Configuration for query generation
        device : str, optional
            Device to use. If None, uses model's device.
        dtype : torch.dtype, optional
            Data type for queries. If None, uses model's dtype.
        vllm_model : optional
            Pre-initialized vLLM model to pass to self-study generator
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype
        self.vllm_model = vllm_model

        # Initialize method-specific generators
        self.method_generators = {}
        self._initialize_generators()

    def _validate_past_key_values_for_queries(self, past_key_values: Optional[Tuple]) -> Optional[Tuple]:
        """
        Ensure past_key_values is in a supported form for query generation.

        Supported:
          - CompactedPrefixCache
          - DynamicCache
          - Tuple/list of per-layer (keys, values)
        Unsupported:
          - Per-layer tuples with attention bias (keys, bias, values)
        """
        if past_key_values is None:
            return None

        if isinstance(past_key_values, (CompactedPrefixCache, DynamicCache)):
            return past_key_values

        if isinstance(past_key_values, (tuple, list)):
            for layer_idx, layer in enumerate(past_key_values):
                if not isinstance(layer, (tuple, list)):
                    raise TypeError(f"past_key_values layer {layer_idx} must be a tuple/list, got {type(layer)}")
                if len(layer) == 2:
                    continue
                if len(layer) == 3:
                    raise ValueError(
                        "past_key_values with attention bias (keys, bias, values) is not supported for query generation; "
                        f"found 3-tuple at layer {layer_idx}. Pass a CompactedPrefixCache or (keys, values) tuples."
                    )
                raise ValueError(
                    f"past_key_values layer {layer_idx} must have length 2 (keys, values); got length {len(layer)}"
                )
            return past_key_values

        raise TypeError(
            f"Unsupported past_key_values type {type(past_key_values)}; expected CompactedPrefixCache, DynamicCache, or tuple-of-(keys, values)."
        )

    def _initialize_generators(self):
        """Initialize method-specific generators based on config."""
        for method_config in self.config.method_configs:
            method = method_config.method

            if method == 'self_study':
                from .self_study import SelfStudyQueryGenerator
                self.method_generators[method] = SelfStudyQueryGenerator(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    config=method_config.config,
                    device=self.device,
                    dtype=self.dtype,
                    verbose=self.config.verbose,
                    vllm_model=self.vllm_model,
                )

            elif method == 'random_vectors':
                from .random_vectors import RandomVectorQueryGenerator
                self.method_generators[method] = RandomVectorQueryGenerator(
                    model=self.model,
                    config=method_config.config,
                    device=self.device,
                    dtype=self.dtype,
                )

            elif method == 'cache_keys':
                from .cache_keys import CacheKeysQueryGenerator
                self.method_generators[method] = CacheKeysQueryGenerator(
                    model=self.model,
                    config=method_config.config,
                    device=self.device,
                    dtype=self.dtype,
                )

            elif method == 'context_prefill':
                from .context_prefill import ContextPrefillQueryGenerator
                self.method_generators[method] = ContextPrefillQueryGenerator(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    config=method_config.config,
                    device=self.device,
                    dtype=self.dtype,
                )

            else:
                raise ValueError(f"Unknown query generation method: {method}")
            
    @staticmethod
    def _attention_to_kv(
        queries_attention_heads: torch.Tensor,
        num_kv_heads: int,
        heads_per_kv_head: int,
    ) -> torch.Tensor:
        """
        Convert queries from attention-head space to KV-head space.

        Parameters
        ----------
        queries_attention_heads : torch.Tensor
            Tensor of shape (num_layers, num_attention_heads, n_queries_per_head, head_dim)
        num_kv_heads : int
            Number of KV heads
        heads_per_kv_head : int
            Number of attention heads that share each KV head (GQA grouping)

        Returns
        -------
        queries_kv : torch.Tensor
            Tensor of shape (num_layers, num_kv_heads, n_queries_per_kv_head, head_dim)

        Notes
        -----
        For GQA (num_kv_heads < num_attention_heads), we group attention heads
        into KV heads and concatenate their queries along the query dimension:

            n_queries_per_kv_head = heads_per_kv_head * n_queries_per_head

        For standard MHA (num_kv_heads == num_attention_heads), this is a no-op.
        """
        num_layers, num_attention_heads, n_queries_per_head, head_dim = queries_attention_heads.shape

        if num_kv_heads == num_attention_heads:
            # MHA: 1:1 mapping – nothing to do
            return queries_attention_heads

        # GQA: multiple attention heads share each KV head
        # Shape: (L, num_kv_heads, heads_per_kv_head, n_q, D)
        queries_reshaped = queries_attention_heads.view(
            num_layers,
            num_kv_heads,
            heads_per_kv_head,
            n_queries_per_head,
            head_dim,
        )
        # Merge (heads_per_kv_head, n_q) into the query dimension
        queries_kv = queries_reshaped.reshape(
            num_layers,
            num_kv_heads,
            heads_per_kv_head * n_queries_per_head,
            head_dim,
        )
        return queries_kv

    def generate_queries(
        self,
        formatted_context: Optional[str] = None,
        past_key_values: Optional[Tuple] = None,
        head_dim: Optional[int] = None,
        indices: Optional[range] = None,
        return_sequences: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Optional[list]]:
        """
        Generate training queries using configured methods.

        Parameters
        ----------
        formatted_context : str, optional
            The formatted context string (required for self-study method)
        past_key_values : tuple, optional
            Pre-computed KV cache for the context. Supported formats:
              - CompactedPrefixCache
              - HuggingFace DynamicCache
              - Tuple/list per layer of (keys, values)
            Per-layer (keys, bias, values) tuples with attention bias are not supported for query generation.
        head_dim : int, optional
            Head dimension to use
        indices : range, optional
            Indices of sequence positions to compact (used by cache_keys method)
        return_sequences : bool, optional
            If True and using self-study method, return sequence information for on-policy re-extraction

        Returns
        -------
        queries : torch.Tensor
            Training queries of shape (num_layers, num_kv_heads, n_queries_per_kv_head, head_dim)
        stats : dict
            Statistics about query generation
        sequences : list or None
            If return_sequences=True and self-study is used, list of sequence info dicts. Otherwise None.
        """
        query_gen_start_time = time.time()

        # Determine head dimension
        if head_dim is None:
            head_dim = getattr(
                self.model.config,
                "head_dim",
                self.model.config.hidden_size // self.model.config.num_attention_heads,
            )

        # Get model dimensions
        num_layers = self.model.config.num_hidden_layers
        num_attention_heads = self.model.config.num_attention_heads
        num_kv_heads = getattr(
            self.model.config,
            "num_key_value_heads",
            num_attention_heads,
        )
        heads_per_kv_head = max(1, num_attention_heads // num_kv_heads)

        # Budget is specified per KV head
        n_train_per_kv_head = self.config.max_query_vectors_per_kv_head

        stats: Dict[str, Any] = {
            "n_total_requested_per_kv_head": n_train_per_kv_head,
            "head_dim": head_dim,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "num_kv_heads": num_kv_heads,
            "heads_per_kv_head": heads_per_kv_head,
            "methods_used": {},
        }
        normalized_past_key_values = self._validate_past_key_values_for_queries(past_key_values)

        # Generate queries from each method in KV space
        all_queries_by_method_kv = []
        sequences = None  # Will be populated if return_sequences=True and self-study is used

        for method_config in self.config.method_configs:
            method = method_config.method
            fraction = method_config.fraction

            if fraction <= 0.0:
                continue

            # Budget for this method per KV head
            n_queries_per_kv_head_for_method = int(n_train_per_kv_head * fraction)
            if n_queries_per_kv_head_for_method <= 0:
                continue

            n_queries_per_attention_head = max(
                1, (n_queries_per_kv_head_for_method + heads_per_kv_head - 1) // heads_per_kv_head
            )

            generator = self.method_generators[method]

            if self.config.verbose:
                print(
                    f"Generating ~{n_queries_per_kv_head_for_method} queries per KV head "
                    f"using '{method}' method..."
                )

            # Dispatch per method
            if method == "self_study":
                # Self-study operates in attention space
                # We choose per-attention-head count so that, after grouping,
                # we match the per-KV-head budget.
                method_queries_attention, method_stats, method_sequences = generator.generate_queries(
                    n_queries_per_attention_head=n_queries_per_attention_head,
                    formatted_context=formatted_context,
                    past_key_values=normalized_past_key_values,
                    return_sequences=return_sequences,
                    indices=indices,
                )
                method_queries_kv = self._attention_to_kv(
                    method_queries_attention,
                    num_kv_heads=num_kv_heads,
                    heads_per_kv_head=heads_per_kv_head,
                )
                # Store sequences if returned
                if return_sequences and method_sequences is not None:
                    sequences = method_sequences

            elif method == "random_vectors":
                # Random vectors also in attention space
                method_queries_attention, method_stats = generator.generate_queries(
                    n_queries_per_attention_head=n_queries_per_attention_head,
                )
                method_queries_kv = self._attention_to_kv(
                    method_queries_attention,
                    num_kv_heads=num_kv_heads,
                    heads_per_kv_head=heads_per_kv_head,
                )

            elif method == "cache_keys":
                # Cache-keys generator works directly in KV space
                method_queries_kv, method_stats = generator.generate_queries(
                    n_queries_per_kv_head=n_queries_per_kv_head_for_method,
                    past_key_values=normalized_past_key_values,
                    indices=indices,
                )

            elif method == "context_prefill":
                # Context prefill generator works in attention space
                method_queries_attention, method_stats = generator.generate_queries(
                    n_queries_per_attention_head=n_queries_per_attention_head,
                    formatted_context=formatted_context,
                    past_key_values=normalized_past_key_values,
                    indices=indices,
                )
                method_queries_kv = self._attention_to_kv(
                    method_queries_attention,
                    num_kv_heads=num_kv_heads,
                    heads_per_kv_head=heads_per_kv_head,
                )

            else:
                raise ValueError(f"Unknown method: {method}")

            # Sanity: ensure KV-space shape
            if method_queries_kv.shape[0] != num_layers or method_queries_kv.shape[1] != num_kv_heads:
                raise ValueError(
                    f"Method '{method}' produced KV queries of shape "
                    f"{tuple(method_queries_kv.shape)}, expected "
                    f"(num_layers={num_layers}, num_kv_heads={num_kv_heads}, *, head_dim={head_dim})"
                )

            all_queries_by_method_kv.append(method_queries_kv)

            stats["methods_used"][method] = {
                "fraction": fraction,
                "n_queries_requested_per_kv_head": n_queries_per_kv_head_for_method,
                "n_queries_actual_per_kv_head": method_queries_kv.shape[2],
                "stats": method_stats,
            }

        # Concatenate queries from all methods along the query dimension (KV space)
        if not all_queries_by_method_kv:
            raise ValueError("No queries generated from any method")

        # Subsample each method's queries to respect fractions even if some methods
        # returned fewer queries than requested

        # Build a list of (method_config, method_queries_kv) tuples for methods that were actually used
        method_data = []
        queries_idx = 0
        for method_config in self.config.method_configs:
            if method_config.fraction <= 0.0:
                continue
            method_data.append((method_config, all_queries_by_method_kv[queries_idx]))
            queries_idx += 1

        # Find the minimum scaling factor needed to respect all fractions
        # For each method: scaling_factor = n_actual / n_requested
        scaling_factors = []
        for method_config, method_queries_kv in method_data:
            n_requested = int(n_train_per_kv_head * method_config.fraction)
            n_actual = method_queries_kv.shape[2]
            if n_requested > 0:
                scaling_factors.append(n_actual / n_requested)

        # The minimum scaling factor determines how much we can sample from all methods
        # while respecting the fractions
        min_scaling = min(scaling_factors) if scaling_factors else 1.0

        # Subsample each method proportionally
        final_queries_list = []
        query_offset = 0  # Track offset in final concatenated queries
        for method_config, method_queries_kv in method_data:
            method = method_config.method
            fraction = method_config.fraction
            n_actual = method_queries_kv.shape[2]

            # Target number of queries for this method after subsampling
            n_target = int(n_train_per_kv_head * fraction * min_scaling)

            if n_target > n_actual:
                # This shouldn't happen if min_scaling is correct, but just in case
                n_target = n_actual

            if n_target < n_actual:
                # Randomly subsample to n_target
                indices = torch.randperm(n_actual, device=method_queries_kv.device)[:n_target]
                indices = indices.sort()[0]
                method_queries_kv = method_queries_kv[:, :, indices, :]

                if method == "self_study" and return_sequences:
                    stats["methods_used"][method]["subsample_indices_kv"] = indices.detach().cpu().tolist()

                if self.config.verbose:
                    print(
                        f"Subsampled '{method}' from {n_actual} to {n_target} queries per KV head "
                        f"to maintain fraction {fraction:.2f}"
                    )
            else:
                if method == "self_study" and return_sequences:
                    stats["methods_used"][method]["subsample_indices_kv"] = None

            final_queries_list.append(method_queries_kv)

            # Update stats with final counts and query range in concatenated tensor
            n_queries_for_method = method_queries_kv.shape[2]
            stats["methods_used"][method]["n_queries_final_per_kv_head"] = n_queries_for_method
            stats["methods_used"][method]["query_range"] = (query_offset, query_offset + n_queries_for_method)
            query_offset += n_queries_for_method

        final_queries = torch.cat(final_queries_list, dim=2)

        stats["final_n_queries_per_kv_head"] = final_queries.shape[2]
        stats["scaling_factor_applied"] = min_scaling

        # Add timing information
        query_gen_end_time = time.time()
        stats["query_generation_time"] = query_gen_end_time - query_gen_start_time

        return final_queries, stats, sequences

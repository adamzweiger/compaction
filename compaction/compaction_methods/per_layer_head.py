# compaction/compaction_methods/per_layer_head.py
"""
Per-layer-head compaction wrapper.

This wraps existing single layer/head compaction algorithms (from ..algorithms/)
and applies them to each layer and head independently.
"""
import torch
import time
from typing import Tuple, Dict, Optional, Type, Any

from .base import FullCacheCompactionAlgorithm, load_budgets_from_json, apply_max_ratio_cap
from ..algorithms.base import CompactionAlgorithm
from ..query_generation import QueryConfig
from models.cache import CompactedPrefixCache, CompactedPrefixLayer, DynamicSlidingWindowLayer


class PerLayerHeadCompaction(FullCacheCompactionAlgorithm):
    """
    Apply a per-layer-head compaction algorithm to the entire KV cache.

    This wrapper takes any algorithm from the algorithms/ package that works
    on a single (layer, head) pair and applies it independently to all
    (layer, head) combinations.
    """

    def __init__(
        self,
        algorithm_class: Type[CompactionAlgorithm],
        algorithm_kwargs: Optional[Dict] = None,
        use_batched: bool = False,
        config_name: Optional[str] = None,
        precomputed_budget_path: Optional[str] = None,
        max_ratio_per_head: float = 1.0,
    ):
        """
        Initialize the per-layer-head compaction wrapper.

        Parameters
        ----------
        algorithm_class : class
            A CompactionAlgorithm class (e.g., OMPCompaction, RandomSubsetKeysCompaction)
        algorithm_kwargs : dict, optional
            Keyword arguments to pass to the algorithm constructor
        use_batched : bool, optional
            If True, use batched processing across all layers and heads (default: False).
            Supported for OMPCompaction and RandomSubsetKeysCompaction. Falls back to sequential for other algorithms.
        config_name : str, optional
            Name of the configuration (used for logging). If not provided, uses the algorithm name.
        precomputed_budget_path : str, optional
            Path to a JSON file containing per-layer/head selection ratios.
            If None (default), use uniform budgets across all heads.
        max_ratio_per_head : float, optional
            Maximum ratio any single head can have (default: 1.0).
            If precomputed budgets would assign a higher ratio, proportions are
            blended towards uniform until the constraint is satisfied.
        """
        self.algorithm_class = algorithm_class
        self.algorithm_kwargs = algorithm_kwargs or {}
        self.use_batched = use_batched
        # Create one instance to get the name
        self._name_instance = algorithm_class(**self.algorithm_kwargs)
        self.config_name = config_name
        self.precomputed_budget_path = precomputed_budget_path
        self.max_ratio_per_head = max_ratio_per_head

    def name(self) -> str:
        """Return the config name if provided, otherwise the algorithm name."""
        if self.config_name:
            return self.config_name
        return f"per_layer_head_{self._name_instance.name()}"

    @staticmethod
    def _split_layer_cache(layer_tuple):
        """
        Normalize a single layer cache tuple to (keys, bias, values).

        Supports (K, V) or (K, bias, V). Returns bias=None if not provided.
        """
        if not isinstance(layer_tuple, (list, tuple)):
            raise TypeError(f"Expected layer cache to be tuple/list, got {type(layer_tuple)}")
        if len(layer_tuple) == 2:
            keys, values = layer_tuple
            bias = None
        elif len(layer_tuple) == 3:
            keys, bias, values = layer_tuple
        else:
            raise ValueError(f"Layer cache must have 2 or 3 elements (K,[bias],V); got {len(layer_tuple)}")
        return keys, bias, values

    @staticmethod
    def _compacted_prefix_to_tuple(cache: CompactedPrefixCache):
        """
        Convert a CompactedPrefixCache into a tuple-of-tuples representation.

        Sliding window layers do not store beta, so we synthesize zeros for beta.
        """
        layers = []
        for layer in cache.layers:
            if isinstance(layer, CompactedPrefixLayer):
                layers.append((layer.keys, layer.beta, layer.values))
            elif isinstance(layer, DynamicSlidingWindowLayer):
                keys = layer.keys
                values = layer.values
                beta = torch.zeros(
                    keys.shape[0],
                    keys.shape[1],
                    keys.shape[2],
                    device=keys.device,
                    dtype=keys.dtype,
                )
                layers.append((keys, beta, values))
            else:
                raise TypeError(f"Unsupported layer type in CompactedPrefixCache: {type(layer)}")
        return tuple(layers)

    def compact_kv_cache(
        self,
        past_key_values: Tuple,
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        compute_stats: bool = False,
        verbose_logging: bool = False,
        vllm_model: Optional[Any] = None,
        sliding_layer_indices: Optional[set] = None,
        past_key_values_for_queries: Optional[Any] = None,
        full_query_extraction: bool = False,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        Compact each (layer, head) pair independently.

        Parameters
        ----------
        past_key_values : tuple of tuples or CompactedPrefixCache
            KV cache structure. Supports:
              - (keys, values) per layer
              - (keys, attention_bias, values) per layer where attention_bias matches keys along
                (batch_size, num_heads, seq_len)
              - CompactedPrefixCache (will be converted internally to tuple-of-tuples)
            keys/values shape: (batch_size, num_heads, seq_len, head_dim)
        target_size : int
            Target compacted sequence length for the full cache.
            If indices is provided, this is the total length after partial compaction.
        indices : range, optional
            Indices of sequence positions to compact. If None, compact entire sequence.
            If provided, only compact these positions and leave others unchanged.
        query_config : QueryConfig
            Configuration for query generation
        model : Any
            Model instance
        tokenizer : Any
            Tokenizer
        formatted_context : str
            Formatted context string
        compute_stats : bool
            If True, compute train stats using generated queries (default: False)
        vllm_model : optional
            Pre-initialized vLLM model to pass to query generator
        sliding_layer_indices : set, optional
            Set of layer indices that use sliding window attention.
            These layers keep their original KV cache instead of being compacted.
        past_key_values_for_queries : optional
            Alternative KV cache to use for query generation only.
            If provided, this cache (which may be a CompactedPrefixCache) is used
            for generating queries, while past_key_values is used for compaction.
            This is useful in KV-based chunked compaction where queries need
            the full context prefix and correct RoPE positions.
        full_query_extraction : bool, optional
            If True, generate queries over the full cache (ignore indices when querying).

        Returns
        -------
        compacted_cache : tuple of tuples
            ((C1_layer0, beta_layer0, C2_layer0), ...)
        stats : dict
            Statistics including per-layer-head metrics and query generation stats
        """
        # Keep a reference for query generation before we normalize the main cache
        kv_for_queries = past_key_values_for_queries if past_key_values_for_queries is not None else past_key_values

        # Normalize CompactedPrefixCache to tuple-of-tuples for downstream logic
        if isinstance(past_key_values, CompactedPrefixCache):
            if sliding_layer_indices is None:
                sliding_layer_indices = getattr(past_key_values, "_sliding_layer_indices", None)
            past_key_values = self._compacted_prefix_to_tuple(past_key_values)

        # Detect whether any layer carries an attention bias
        has_bias = any(isinstance(layer, (tuple, list)) and len(layer) == 3 for layer in past_key_values)

        num_layers = len(past_key_values)

        # Find a non-sliding layer to get the full sequence length
        # (sliding layers may have shorter seq_len due to window size)
        sliding_layer_indices = sliding_layer_indices or set()
        ref_layer_idx = 0
        for i in range(num_layers):
            if i not in sliding_layer_indices:
                ref_layer_idx = i
                break
        batch_size, num_heads, seq_len, head_dim = past_key_values[ref_layer_idx][0].shape

        # For now, we only support batch_size=1
        if batch_size != 1:
            raise NotImplementedError(
                "PerLayerHeadCompaction currently only supports batch_size=1"
            )

        num_global_layers = num_layers - len(sliding_layer_indices)
        if sliding_layer_indices:
            print(f"Using {len(sliding_layer_indices)} sliding window layers (indices: {sorted(sliding_layer_indices)})")
            print(f"Only compacting {num_global_layers} global attention layers")

        from ..query_generation import QueryGenerator

        device = past_key_values[0][0].device
        dtype = past_key_values[0][0].dtype
        generator = QueryGenerator(
            model=model,
            tokenizer=tokenizer,
            config=query_config,
            device=device,
            dtype=dtype,
            vllm_model=vllm_model,
        )

        # Returns: (num_layers, num_kv_heads, n_queries, head_dim)
        query_indices = indices
        if full_query_extraction:
            full_len = tokenizer(
                formatted_context,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.shape[1]
            query_indices = range(0, full_len)
        base_queries, query_stats, _ = generator.generate_queries(
            formatted_context=formatted_context,
            past_key_values=kv_for_queries,
            indices=query_indices,
        )

        # Add batch dimension: (1, num_layers, num_kv_heads, n_queries_per_kv_head, head_dim)
        queries = base_queries.unsqueeze(0)

        # Print query generation info once
        print(f"Generated {query_stats['final_n_queries_per_kv_head']} queries per KV head")
        for method, method_stats in query_stats.get('methods_used', {}).items():
            print(f"  {method}: {method_stats['n_queries_actual_per_kv_head']} queries ({method_stats['fraction']:.1%})")
        print("Train queries shape:", queries.shape)

        # Determine if we're doing partial compaction
        # This must come before per-head budget loading so we use actual_target_size
        if indices is not None:
            # Convert range to list of indices
            indices_list = list(indices)
            num_to_compact = len(indices_list)
            num_to_keep = seq_len - num_to_compact

            # Compute sub-target size for the compacted portion
            # target_size = num_to_keep + sub_target_size
            # sub_target_size = target_size - num_to_keep
            sub_target_size = target_size - num_to_keep

            if sub_target_size <= 0:
                raise ValueError(
                    f"target_size ({target_size}) must be greater than the number of "
                    f"positions to keep ({num_to_keep}). Got sub_target_size = {sub_target_size}"
                )

            # Create mask for positions NOT being compacted
            all_indices = torch.arange(seq_len)
            compact_mask = torch.zeros(seq_len, dtype=torch.bool)
            compact_mask[indices_list] = True
            keep_mask = ~compact_mask
            keep_indices = all_indices[keep_mask].tolist()

            actual_target_size = sub_target_size
            is_partial_compaction = True
        else:
            indices_list = None
            keep_indices = None
            actual_target_size = target_size
            is_partial_compaction = False

        # Load per-head budgets if using precomputed budgets
        # Use actual_target_size (article portion only) not target_size (which includes keep tokens)
        per_head_budgets = None
        if self.precomputed_budget_path is not None:
            print(f"Loading per-head budgets from {self.precomputed_budget_path}...")
            per_head_proportions = load_budgets_from_json(
                self.precomputed_budget_path, num_layers, num_heads
            )

            # Convert proportions to absolute budgets
            # Each head gets: proportion * actual_target_size * total_num_heads
            # This way proportions sum to 1.0 and the total budget equals actual_target_size * num_global_layers * num_heads
            # Note: We use num_global_layers because sliding window layers are not compacted
            article_len = len(indices_list) if is_partial_compaction else seq_len
            target_ratio = actual_target_size / article_len if article_len > 0 else 1.0
            total_heads = num_global_layers * num_heads

            # Apply max ratio cap if needed (redistributes excess to other heads)
            per_head_proportions = apply_max_ratio_cap(
                per_head_proportions,
                self.max_ratio_per_head,
                target_ratio,
                total_heads,
            )

            per_head_budgets = {}
            for (layer_idx, head_idx), proportion in per_head_proportions.items():
                if proportion is not None:
                    per_head_budgets[(layer_idx, head_idx)] = int(proportion * actual_target_size * total_heads)
                else:
                    per_head_budgets[(layer_idx, head_idx)] = actual_target_size

            # Print budget distribution
            total_budget = sum(per_head_budgets.values())
            total_proportions = sum(p for p in per_head_proportions.values() if p is not None)
            max_head_budget = max(per_head_budgets.values()) if per_head_budgets else 0
            print(f"Per-head proportions sum to: {total_proportions:.4f}")
            print(f"Per-head budget distribution: total={total_budget}, "
                  f"min={min(per_head_budgets.values()) if per_head_budgets else 0}, "
                  f"max={max_head_budget}, "
                  f"mean={total_budget / (num_global_layers * num_heads):.1f}")

        # Check if we should use batched processing
        # Batched mode doesn't support per-head budgets
        use_batched_path = self.use_batched and self._supports_batched()
        if has_bias and use_batched_path:
            raise NotImplementedError("Attention bias past_key_values is not yet supported in batched per-layer-head compaction; use sequential mode.")
        if use_batched_path and per_head_budgets is not None:
            print("Per-head budgets (precomputed budget path) not supported in batched mode; using sequential processing")
            use_batched_path = False
        if use_batched_path:
            print("Using batched processing across all layers and heads")
        else:
            if self.use_batched and per_head_budgets is not None:
                print("Per-head budgets (precomputed budget path) require sequential processing")
            elif self.use_batched:
                print(f"Batched processing not supported for {self.algorithm_class.__name__}, using sequential")
            else:
                print("Using sequential processing")

        compacted_layers = []
        all_stats = {
            'per_layer_head_metrics': {},
            'is_partial_compaction': is_partial_compaction,
            'train_stats_time': 0.0,
            'num_sliding_layers': len(sliding_layer_indices),
            'num_global_layers': num_global_layers,
        }

        if is_partial_compaction:
            all_stats['compaction_indices'] = {
                'start': indices_list[0],
                'end': indices_list[-1] + 1,
                'num_positions': len(indices_list),
            }
            all_stats['keep_indices'] = {
                'num_positions': len(keep_indices),
            }

        # Batched path
        if use_batched_path:
            # Check if we should use per-layer batching (batch heads within each layer)
            # or full batching (batch all layers and heads together)
            use_per_layer_batching = self.algorithm_kwargs.get('use_per_layer_batching', False)

            if use_per_layer_batching:
                print("Using per-layer batched processing (batching heads within each layer)")
                compacted_layers, all_stats = self._compact_per_layer_batched(
                    past_key_values, queries, actual_target_size,
                    is_partial_compaction, indices_list, keep_indices,
                    all_stats, query_stats, compute_stats, query_config.eval_queries_per_kv_head,
                    model, verbose_logging, per_head_budgets,
                    sliding_layer_indices
                )
            else:
                print("Using full batched processing (batching all layers and heads together)")
                compacted_layers, all_stats = self._compact_batched(
                    past_key_values, queries, actual_target_size,
                    is_partial_compaction, indices_list, keep_indices,
                    all_stats, query_stats, compute_stats, query_config.eval_queries_per_kv_head,
                    model, verbose_logging, per_head_budgets,
                    sliding_layer_indices
                )
        # Sequential path (original implementation)
        else:
            compacted_layers, all_stats = self._compact_sequential(
                past_key_values, queries, actual_target_size,
                is_partial_compaction, indices_list, keep_indices,
                all_stats, query_stats, compute_stats, query_config.eval_queries_per_kv_head,
                model, verbose_logging, per_head_budgets,
                sliding_layer_indices
            )

        return tuple(compacted_layers), all_stats

    def _get_batched_algorithm_map(self):
        """Get the mapping from algorithm class to its batched version."""
        from ..algorithms.omp import OMPCompaction
        from ..algorithms.omp_batched import BatchedOMPCompaction
        from ..algorithms.optim import OptimJointCompaction
        from ..algorithms.optim_batched import BatchedOptimJointCompaction

        return {
            OMPCompaction: BatchedOMPCompaction,
            OptimJointCompaction: BatchedOptimJointCompaction,
        }

    def _supports_batched(self) -> bool:
        """Check if the algorithm supports batched processing."""
        batched_map = self._get_batched_algorithm_map()
        return self.algorithm_class in batched_map

    def _compact_sequential(
        self,
        past_key_values,
        queries,
        actual_target_size,
        is_partial_compaction,
        indices_list,
        keep_indices,
        all_stats,
        query_stats,
        compute_stats,
        eval_queries_per_kv_head,
        model=None,
        verbose_logging=False,
        per_head_budgets=None,
        sliding_layer_indices=None,
    ):
        """Sequential processing path (original implementation)."""
        num_layers = len(past_key_values)
        sliding_layer_indices = sliding_layer_indices or set()

        # Find a non-sliding layer to get the full sequence length
        ref_layer_idx = 0
        for i in range(num_layers):
            if i not in sliding_layer_indices:
                ref_layer_idx = i
                break
        num_heads = past_key_values[ref_layer_idx][0].shape[1]
        seq_len = past_key_values[ref_layer_idx][0].shape[2]
        head_dim = past_key_values[ref_layer_idx][0].shape[3]
        compacted_layers = []

        # For nonuniform caches, each layer can have different sequence lengths
        # We pad heads within each layer to the layer's max, but don't enforce global uniformity
        # This is handled in the per-layer padding step below

        total_effective_article_tokens = 0

        for layer_idx in range(num_layers):
            keys_layer, bias_layer, values_layer = self._split_layer_cache(past_key_values[layer_idx])
            # Ensure bias has shape (1, num_heads, seq_len) if present
            if bias_layer is not None and bias_layer.dim() == 2:
                bias_layer = bias_layer.unsqueeze(0)

            # Handle sliding window layers: keep original KV, no compaction
            if layer_idx in sliding_layer_indices:
                print(f"Layer {layer_idx+1}/{num_layers}: sliding window (keeping original KV)")
                # Create a placeholder entry for compacted_layers
                # This will be replaced by DynamicSlidingWindowLayer in CompactedPrefixCache
                placeholder_C1 = keys_layer.new_zeros(1, num_heads, 0, head_dim)
                placeholder_beta = keys_layer.new_zeros(1, num_heads, 0)
                placeholder_C2 = values_layer.new_zeros(1, num_heads, 0, head_dim)
                compacted_layers.append((placeholder_C1, placeholder_beta, placeholder_C2))
                continue

            print(f"Compacting layer {layer_idx+1}/{num_layers}")

            # Storage for this layer's compacted cache
            C1_heads = []
            beta_heads = []
            C2_heads = []

            for head_idx in range(num_heads):
                # Extract K, V for this head: (seq_len, head_dim)
                K_full = keys_layer[0, head_idx, :, :]  # (seq_len, head_dim)
                V_full = values_layer[0, head_idx, :, :]  # (seq_len, head_dim)

                if is_partial_compaction:
                    # Extract the subset to compact
                    K = K_full[indices_list, :]  # (num_to_compact, head_dim)
                    V = V_full[indices_list, :]  # (num_to_compact, head_dim)
                    attn_bias_head = bias_layer[0, head_idx, indices_list] if bias_layer is not None else None

                    # Also extract the portions to keep unchanged
                    K_keep = K_full[keep_indices, :]  # (num_to_keep, head_dim)
                    V_keep = V_full[keep_indices, :]  # (num_to_keep, head_dim)
                    bias_keep = bias_layer[0, head_idx, keep_indices] if bias_layer is not None else None
                else:
                    K = K_full
                    V = V_full
                    attn_bias_head = bias_layer[0, head_idx, :] if bias_layer is not None else None
                    bias_keep = None

                # Get queries for this head (KV head)
                # queries shape: (batch_size, num_layers, num_kv_heads, n_queries_per_kv_head, head_dim)
                queries_head = queries[0, layer_idx, head_idx, :, :]  # (n_queries_per_kv_head, head_dim)

                # Determine target size for this head
                if per_head_budgets is not None:
                    head_target_size = per_head_budgets.get((layer_idx, head_idx), actual_target_size)
                else:
                    head_target_size = actual_target_size

                # Handle zero-budget heads: skip compaction and return empty tensors
                if head_target_size == 0:
                    C1_compact = K.new_zeros(0, head_dim)
                    beta_compact = K.new_zeros(0)
                    C2_compact = V.new_zeros(0, head_dim)
                    selected_indices = []
                else:
                    # Create algorithm instance for this head
                    algorithm = self.algorithm_class(**self.algorithm_kwargs)

                    # Compact this head (or the subset)
                    C1_compact, beta_compact, C2_compact, selected_indices = algorithm.compute_compacted_cache(
                        K, V, queries_head, head_target_size, attention_bias=attn_bias_head
                    )

                # Check if we got fewer keys than expected (e.g., from drop_key_beta_cutoff running out of keys)
                actual_returned_size = C1_compact.shape[0]
                # Track effective article tokens for this head (before padding)
                total_effective_article_tokens += actual_returned_size
                # Save beta before padding for stats
                beta_for_stats = beta_compact
                if actual_returned_size < head_target_size:
                    # Pad with dummy keys to maintain consistent shape across heads
                    num_padding = head_target_size - actual_returned_size
                    C1_padding = K.new_zeros(num_padding, head_dim)
                    C2_padding = V.new_zeros(num_padding, head_dim)
                    # Use -inf beta so these keys are ignored in attention
                    beta_padding = K.new_full((num_padding,), float('-inf'))

                    C1_compact = torch.cat([C1_compact, C1_padding], dim=0)
                    beta_compact = torch.cat([beta_compact, beta_padding], dim=0)
                    C2_compact = torch.cat([C2_compact, C2_padding], dim=0)

                if is_partial_compaction:
                    # Reconstruct the full cache by concatenating kept and compacted portions
                    # We don't need to maintain the relative ordering, but we'll
                    # put the compacted portion back roughly where it came from

                    # For simplicity, we'll concatenate: [K_keep_before, C1_compact, K_keep_after]
                    # where keep_before are indices before the compaction range,
                    # and keep_after are indices after the compaction range

                    # Split keep_indices into before and after
                    compact_start = indices_list[0]
                    compact_end = indices_list[-1] + 1

                    keep_before_mask = [idx < compact_start for idx in keep_indices]
                    keep_after_mask = [idx >= compact_end for idx in keep_indices]

                    K_keep_before = K_keep[keep_before_mask, :] if any(keep_before_mask) else K.new_zeros(0, head_dim)
                    V_keep_before = V_keep[keep_before_mask, :] if any(keep_before_mask) else V.new_zeros(0, head_dim)
                    K_keep_after = K_keep[keep_after_mask, :] if any(keep_after_mask) else K.new_zeros(0, head_dim)
                    V_keep_after = V_keep[keep_after_mask, :] if any(keep_after_mask) else V.new_zeros(0, head_dim)

                    # Create zero betas for kept portions
                    if bias_keep is not None:
                        bias_keep_before = bias_keep[keep_before_mask]
                        bias_keep_after = bias_keep[keep_after_mask]
                    else:
                        bias_keep_before = K.new_zeros(K_keep_before.shape[0])
                        bias_keep_after = K.new_zeros(K_keep_after.shape[0])

                    beta_keep_before = bias_keep_before
                    beta_keep_after = bias_keep_after

                    # Concatenate: [before, compacted, after]
                    C1 = torch.cat([K_keep_before, C1_compact, K_keep_after], dim=0)
                    beta = torch.cat([beta_keep_before, beta_compact, beta_keep_after], dim=0)
                    C2 = torch.cat([V_keep_before, C2_compact, V_keep_after], dim=0)
                else:
                    C1 = C1_compact
                    beta = beta_compact
                    C2 = C2_compact

                # Store results
                C1_heads.append(C1.unsqueeze(0).unsqueeze(0))  # (1, 1, t, head_dim)
                beta_heads.append(beta.unsqueeze(0).unsqueeze(0))  # (1, 1, t)
                C2_heads.append(C2.unsqueeze(0).unsqueeze(0))  # (1, 1, t, head_dim)

                # Store stats
                head_stats = {
                    'layer': layer_idx,
                    'head': head_idx,
                    **({'selected_indices': [int(idx) for idx in selected_indices]} if verbose_logging else {}),
                    'selected_indices_stats': {
                        'count': len(selected_indices),
                        'min': int(min(selected_indices)) if len(selected_indices) > 0 else None,
                        'max': int(max(selected_indices)) if len(selected_indices) > 0 else None,
                    },
                    **({'beta_stats': {
                        'min': float(beta_for_stats.min().item()) if len(beta_for_stats) > 0 else None,
                        'max': float(beta_for_stats.max().item()) if len(beta_for_stats) > 0 else None,
                        'mean': float(beta_for_stats.mean().item()) if len(beta_for_stats) > 0 else None,
                        'std': float(beta_for_stats.std().item()) if len(beta_for_stats) > 1 else None,
                        'num_less_than_minus_7': int((beta_for_stats < -7).sum().item()) if len(beta_for_stats) > 0 else 0,
                    }} if verbose_logging else {})
                }

                # Compute train stats if requested
                if compute_stats:
                    start_time = time.time()
                    from ..algorithms.base import evaluate_compaction

                    # Subsample queries to eval_queries_per_kv_head per KV head
                    n_train_queries = queries_head.shape[0]
                    if n_train_queries > eval_queries_per_kv_head:
                        indices = torch.randperm(n_train_queries)[:eval_queries_per_kv_head]
                        queries_subsample = queries_head[indices]
                    else:
                        queries_subsample = queries_head

                    # Evaluate on the portion that was compacted
                    train_metrics = evaluate_compaction(
                        K, V, C1_compact, beta_compact, C2_compact, queries_subsample, attention_bias=attn_bias_head
                    )
                    head_stats['train_stats'] = {k: float(v) for k, v in train_metrics.items()}
                    eval_time = time.time() - start_time
                    all_stats['train_stats_time'] += eval_time

                all_stats['per_layer_head_metrics'][f'L{layer_idx}H{head_idx}'] = head_stats

            # Pad all heads within this layer to the same size before concatenating
            # Each layer can have different sequence lengths (nonuniform caches)
            target_seq_len = max(h.shape[2] for h in C1_heads)
            for i in range(len(C1_heads)):
                curr_len = C1_heads[i].shape[2]
                if curr_len < target_seq_len:
                    pad_len = target_seq_len - curr_len
                    # Pad C1 and C2 with zeros
                    C1_heads[i] = torch.cat([
                        C1_heads[i],
                        C1_heads[i].new_zeros(1, 1, pad_len, head_dim)
                    ], dim=2)
                    C2_heads[i] = torch.cat([
                        C2_heads[i],
                        C2_heads[i].new_zeros(1, 1, pad_len, head_dim)
                    ], dim=2)
                    # Pad beta with -inf so these positions are ignored
                    beta_heads[i] = torch.cat([
                        beta_heads[i],
                        beta_heads[i].new_full((1, 1, pad_len), float('-inf'))
                    ], dim=2)

            # Concatenate all heads for this layer
            C1_layer = torch.cat(C1_heads, dim=1)  # (1, num_heads, t, head_dim)
            beta_layer = torch.cat(beta_heads, dim=1)  # (1, num_heads, t)
            C2_layer = torch.cat(C2_heads, dim=1)  # (1, num_heads, t, head_dim)

            compacted_layers.append((C1_layer, beta_layer, C2_layer))

        # Compute average tensor length across all global (non-sliding) layers
        # For nonuniform caches, different layers can have different sequence lengths
        total_tensor_len = 0
        num_global_layers = num_layers - len(sliding_layer_indices)
        for layer_idx, layer_data in enumerate(compacted_layers):
            if layer_idx not in sliding_layer_indices:
                total_tensor_len += layer_data[0].shape[2]

        if num_global_layers > 0:
            avg_tensor_compacted_len = total_tensor_len / num_global_layers
        else:
            num_kept = len(keep_indices) if keep_indices is not None else 0
            avg_tensor_compacted_len = actual_target_size + num_kept
        all_stats['tensor_compacted_seq_len'] = avg_tensor_compacted_len

        # Compute effective lengths (accounting for padding with -inf beta)
        # effective_article_tokens is already computed as average across all heads
        total_global_heads = num_global_layers * num_heads
        if total_global_heads > 0:
            effective_article_tokens = total_effective_article_tokens / total_global_heads
        else:
            effective_article_tokens = 0
        num_kept = len(keep_indices) if keep_indices is not None else 0
        effective_compacted_seq_len = effective_article_tokens + num_kept

        # Tensor article tokens is the average tensor size of the article portion
        tensor_article_tokens = avg_tensor_compacted_len - num_kept

        all_stats['effective_article_tokens'] = effective_article_tokens
        all_stats['tensor_article_tokens'] = tensor_article_tokens
        all_stats['effective_compacted_seq_len'] = effective_compacted_seq_len
        all_stats['query_generation'] = query_stats

        # Aggregate train stats across all layers/heads
        if compute_stats:
            self._aggregate_train_stats(all_stats, eval_queries_per_kv_head)
            if all_stats['train_stats_time'] > 0:
                print(f"Total train stats computation time: {all_stats['train_stats_time']:.2f}s")

        return compacted_layers, all_stats

    def _compact_per_layer_batched(
        self,
        past_key_values,
        queries,
        actual_target_size,
        is_partial_compaction,
        indices_list,
        keep_indices,
        all_stats,
        query_stats,
        compute_stats,
        eval_queries_per_kv_head,
        model=None,
        verbose_logging=False,
        per_head_budgets=None,
        sliding_layer_indices=None,
    ):
        """
        Per-layer batched processing: batch all heads within each layer, iterate over layers.

        This is a middle ground between fully sequential and fully batched:
        - Batch size = num_heads (e.g., 4) instead of 1 or num_layers × num_heads
        - Lower memory usage than full batching
        - Better GPU utilization than sequential
        """
        # Get the batched algorithm class
        batched_class_map = self._get_batched_algorithm_map()
        batched_class = batched_class_map.get(self.algorithm_class)

        if batched_class is None:
            print(f"Batched processing not supported for {self.algorithm_class.__name__}, using sequential")
            return self._compact_sequential(
                past_key_values, queries, actual_target_size,
                is_partial_compaction, indices_list, keep_indices,
                all_stats, query_stats, compute_stats, eval_queries_per_kv_head,
                model, verbose_logging, per_head_budgets,
                sliding_layer_indices
            )

        num_layers = len(past_key_values)
        sliding_layer_indices = sliding_layer_indices or set()

        # Find a non-sliding layer to get the full sequence length
        ref_layer_idx = 0
        for i in range(num_layers):
            if i not in sliding_layer_indices:
                ref_layer_idx = i
                break
        num_heads = past_key_values[ref_layer_idx][0].shape[1]
        seq_len = past_key_values[ref_layer_idx][0].shape[2]
        head_dim = past_key_values[ref_layer_idx][0].shape[3]
        device = past_key_values[ref_layer_idx][0].device

        compacted_layers = []
        # Track total effective article tokens across all heads for computing effective lengths
        total_effective_article_tokens = 0

        for layer_idx in range(num_layers):
            keys_layer_full = past_key_values[layer_idx][0]  # (1, num_heads, seq_len, head_dim)
            values_layer_full = past_key_values[layer_idx][1]

            # Handle sliding window layers: keep original KV, no compaction
            if layer_idx in sliding_layer_indices:
                print(f"Layer {layer_idx+1}/{num_layers}: sliding window (keeping original KV)")
                placeholder_C1 = keys_layer_full.new_zeros(1, num_heads, 0, head_dim)
                placeholder_beta = keys_layer_full.new_zeros(1, num_heads, 0)
                placeholder_C2 = values_layer_full.new_zeros(1, num_heads, 0, head_dim)
                compacted_layers.append((placeholder_C1, placeholder_beta, placeholder_C2))
                continue

            print(f"Compacting layer {layer_idx+1}/{num_layers} (batched across {num_heads} heads)")

            # Get K, V for this layer: (1, num_heads, seq_len, head_dim)
            keys_layer = past_key_values[layer_idx][0][0]  # (num_heads, seq_len, head_dim)
            values_layer = past_key_values[layer_idx][1][0]  # (num_heads, seq_len, head_dim)

            # Get queries for this layer: (num_heads, n_queries, head_dim)
            queries_layer = queries[0, layer_idx, :, :, :]  # (num_heads, n_queries, head_dim)

            # Handle partial compaction
            if is_partial_compaction:
                K_to_compact = keys_layer[:, indices_list, :]  # (num_heads, num_to_compact, head_dim)
                V_to_compact = values_layer[:, indices_list, :]

                K_keep = keys_layer[:, keep_indices, :]  # (num_heads, num_to_keep, head_dim)
                V_keep = values_layer[:, keep_indices, :]

                # Split into before/after
                compact_start = indices_list[0]
                compact_end = indices_list[-1] + 1
                keep_before_mask = torch.tensor([idx < compact_start for idx in keep_indices], device=device)
                keep_after_mask = torch.tensor([idx >= compact_end for idx in keep_indices], device=device)

                K_keep_before = K_keep[:, keep_before_mask, :]
                V_keep_before = V_keep[:, keep_before_mask, :]
                K_keep_after = K_keep[:, keep_after_mask, :]
                V_keep_after = V_keep[:, keep_after_mask, :]

                K_input = K_to_compact
                V_input = V_to_compact
            else:
                K_input = keys_layer
                V_input = values_layer

            # Create batched algorithm instance
            batched_alg = batched_class(**self.algorithm_kwargs)

            # Compact all heads in this layer at once
            # K_input, V_input, queries_layer all have shape (num_heads, ...)
            C1_compact, beta_compact, C2_compact, indices_batched = batched_alg.compute_compacted_cache_batched(
                K_input, V_input, queries_layer, actual_target_size
            )  # Each: (num_heads, t, d) or (num_heads, t)

            # Check if we got fewer keys than expected for any head
            actual_returned_size = C1_compact.shape[1]  # sequence dimension
            # Track effective article tokens for all heads in this layer (before padding)
            total_effective_article_tokens += actual_returned_size * num_heads
            # Save for stats before padding
            beta_for_stats = beta_compact
            if actual_returned_size < actual_target_size:
                # Pad with dummy keys to reach target size
                num_padding = actual_target_size - actual_returned_size
                C1_padding = torch.zeros(num_heads, num_padding, C1_compact.shape[2], dtype=C1_compact.dtype, device=device)
                C2_padding = torch.zeros(num_heads, num_padding, C2_compact.shape[2], dtype=C2_compact.dtype, device=device)
                beta_padding = torch.full((num_heads, num_padding), float('-inf'), dtype=beta_compact.dtype, device=device)

                C1_compact = torch.cat([C1_compact, C1_padding], dim=1)
                beta_compact = torch.cat([beta_compact, beta_padding], dim=1)
                C2_compact = torch.cat([C2_compact, C2_padding], dim=1)

            # Handle partial compaction reconstruction
            if is_partial_compaction:
                num_before = K_keep_before.shape[1]
                num_after = K_keep_after.shape[1]

                beta_before = torch.zeros(num_heads, num_before, dtype=beta_compact.dtype, device=device)
                beta_after = torch.zeros(num_heads, num_after, dtype=beta_compact.dtype, device=device)

                C1_layer = torch.cat([K_keep_before, C1_compact, K_keep_after], dim=1)
                beta_layer = torch.cat([beta_before, beta_compact, beta_after], dim=1)
                C2_layer = torch.cat([V_keep_before, C2_compact, V_keep_after], dim=1)
            else:
                C1_layer = C1_compact
                beta_layer = beta_compact
                C2_layer = C2_compact

            # Add batch dimension: (1, num_heads, target_size, head_dim)
            C1_layer = C1_layer.unsqueeze(0)
            beta_layer = beta_layer.unsqueeze(0)
            C2_layer = C2_layer.unsqueeze(0)

            compacted_layers.append((C1_layer, beta_layer, C2_layer))

            # Collect stats for all heads in this layer
            for head_idx in range(num_heads):
                selected_indices = indices_batched[head_idx].cpu().tolist()
                beta_head = beta_for_stats[head_idx]  # Use non-padded beta for stats

                head_stats = {
                    'layer': layer_idx,
                    'head': head_idx,
                    **({'selected_indices': [int(idx) for idx in selected_indices]} if verbose_logging else {}),
                    'selected_indices_stats': {
                        'count': len(selected_indices),
                        'min': int(min(selected_indices)) if len(selected_indices) > 0 else None,
                        'max': int(max(selected_indices)) if len(selected_indices) > 0 else None,
                    },
                    **({'beta_stats': {
                        'min': float(beta_head.min().item()) if len(beta_head) > 0 else None,
                        'max': float(beta_head.max().item()) if len(beta_head) > 0 else None,
                        'mean': float(beta_head.mean().item()) if len(beta_head) > 0 else None,
                        'std': float(beta_head.std().item()) if len(beta_head) > 1 else None,
                        'num_less_than_minus_7': int((beta_head < -7).sum().item()) if len(beta_head) > 0 else 0,
                    }} if verbose_logging else {})
                }

                # Compute train stats if requested
                if compute_stats:
                    start_time = time.time()
                    from ..algorithms.base import evaluate_compaction

                    queries_head = queries_layer[head_idx]
                    n_train_queries = queries_head.shape[0]
                    if n_train_queries > eval_queries_per_kv_head:
                        subsample_indices = torch.randperm(n_train_queries)[:eval_queries_per_kv_head]
                        queries_subsample = queries_head[subsample_indices]
                    else:
                        queries_subsample = queries_head

                    K_head = K_input[head_idx]
                    V_head = V_input[head_idx]
                    C1_head = C1_compact[head_idx]
                    beta_head_train = beta_compact[head_idx]
                    C2_head = C2_compact[head_idx]

                    train_metrics = evaluate_compaction(
                        K_head, V_head, C1_head, beta_head_train, C2_head, queries_subsample
                    )
                    head_stats['train_stats'] = {k: float(v) for k, v in train_metrics.items()}
                    eval_time = time.time() - start_time
                    all_stats['train_stats_time'] += eval_time

                all_stats['per_layer_head_metrics'][f'L{layer_idx}H{head_idx}'] = head_stats

        # Compute average tensor length across all global (non-sliding) layers
        # For nonuniform caches, different layers can have different sequence lengths
        total_tensor_len = 0
        num_global_layers = num_layers - len(sliding_layer_indices)
        for layer_idx, layer_data in enumerate(compacted_layers):
            if layer_idx not in sliding_layer_indices:
                total_tensor_len += layer_data[0].shape[2]

        if num_global_layers > 0:
            avg_tensor_compacted_len = total_tensor_len / num_global_layers
        else:
            num_kept = len(keep_indices) if keep_indices is not None else 0
            avg_tensor_compacted_len = actual_target_size + num_kept
        all_stats['tensor_compacted_seq_len'] = avg_tensor_compacted_len

        # Compute effective lengths (accounting for padding with -inf beta)
        # effective_article_tokens is already computed as average across all heads
        total_global_heads = num_global_layers * num_heads
        if total_global_heads > 0:
            effective_article_tokens = total_effective_article_tokens / total_global_heads
        else:
            effective_article_tokens = 0
        num_kept = len(keep_indices) if keep_indices is not None else 0
        effective_compacted_seq_len = effective_article_tokens + num_kept

        # Tensor article tokens is the average tensor size of the article portion
        tensor_article_tokens = avg_tensor_compacted_len - num_kept

        all_stats['effective_article_tokens'] = effective_article_tokens
        all_stats['tensor_article_tokens'] = tensor_article_tokens
        all_stats['effective_compacted_seq_len'] = effective_compacted_seq_len
        all_stats['query_generation'] = query_stats

        if compute_stats:
            self._aggregate_train_stats(all_stats, eval_queries_per_kv_head)
            if all_stats['train_stats_time'] > 0:
                print(f"Total train stats computation time: {all_stats['train_stats_time']:.2f}s")

        return compacted_layers, all_stats

    def _compact_batched(
        self,
        past_key_values,
        queries,
        actual_target_size,
        is_partial_compaction,
        indices_list,
        keep_indices,
        all_stats,
        query_stats,
        compute_stats,
        eval_queries_per_kv_head,
        model=None,
        verbose_logging=False,
        per_head_budgets=None,
        sliding_layer_indices=None,
    ):
        """Batched processing path across all layers and heads."""
        sliding_layer_indices = sliding_layer_indices or set()

        # If there are sliding layers, fall back to per-layer batched processing
        # Full batching doesn't work well with mixed layer types
        if sliding_layer_indices:
            print("Sliding window layers detected, falling back to per-layer batched processing")
            return self._compact_per_layer_batched(
                past_key_values, queries, actual_target_size,
                is_partial_compaction, indices_list, keep_indices,
                all_stats, query_stats, compute_stats, eval_queries_per_kv_head,
                model, verbose_logging, per_head_budgets,
                sliding_layer_indices
            )

        # Get the mapping from algorithm class to its batched version
        batched_class_map = self._get_batched_algorithm_map()

        batched_class = batched_class_map.get(self.algorithm_class)
        if batched_class is None:
            # Gracefully fall back to sequential processing
            print(f"Batched processing not supported for {self.algorithm_class.__name__}, using sequential")
            return self._compact_sequential(
                past_key_values, queries, actual_target_size,
                is_partial_compaction, indices_list, keep_indices,
                all_stats, query_stats, compute_stats, eval_queries_per_kv_head,
                model, verbose_logging, per_head_budgets,
                sliding_layer_indices
            )

        num_layers = len(past_key_values)
        sliding_layer_indices = sliding_layer_indices or set()

        # Find a non-sliding layer to get the full sequence length
        ref_layer_idx = 0
        for i in range(num_layers):
            if i not in sliding_layer_indices:
                ref_layer_idx = i
                break
        num_heads = past_key_values[ref_layer_idx][0].shape[1]
        seq_len = past_key_values[ref_layer_idx][0].shape[2]
        head_dim = past_key_values[ref_layer_idx][0].shape[3]
        device = past_key_values[ref_layer_idx][0].device

        # Stack all layers and heads into batch dimension
        # past_key_values: tuple of (keys, values) where each is (1, num_heads, seq_len, head_dim)
        # Stack to: (num_layers, num_heads, seq_len, head_dim)
        K_all_layers = torch.stack([layer[0][0] for layer in past_key_values])  # (num_layers, num_heads, seq_len, head_dim)
        V_all_layers = torch.stack([layer[1][0] for layer in past_key_values])  # (num_layers, num_heads, seq_len, head_dim)

        # queries: (1, num_layers, num_heads, n_queries, head_dim) -> (num_layers, num_heads, n_queries, head_dim)
        queries_all = queries[0]  # (num_layers, num_heads, n_queries, head_dim)

        # Flatten layer and head dimensions: (num_layers * num_heads, ...)
        B = num_layers * num_heads
        K_flat = K_all_layers.reshape(B, seq_len, head_dim)  # (B, T, d)
        V_flat = V_all_layers.reshape(B, seq_len, head_dim)  # (B, T, d)
        queries_flat = queries_all.reshape(B, queries_all.shape[2], head_dim)  # (B, n, d)

        # Handle partial compaction
        if is_partial_compaction:
            # Extract subset to compact
            K_to_compact = K_flat[:, indices_list, :]  # (B, num_to_compact, d)
            V_to_compact = V_flat[:, indices_list, :]  # (B, num_to_compact, d)

            # Extract portions to keep
            K_keep = K_flat[:, keep_indices, :]  # (B, num_to_keep, d)
            V_keep = V_flat[:, keep_indices, :]  # (B, num_to_keep, d)

            # Split keep_indices into before and after compaction range
            compact_start = indices_list[0]
            compact_end = indices_list[-1] + 1
            keep_before_mask = torch.tensor([idx < compact_start for idx in keep_indices], device=device)
            keep_after_mask = torch.tensor([idx >= compact_end for idx in keep_indices], device=device)

            K_keep_before = K_keep[:, keep_before_mask, :]  # (B, num_before, d)
            V_keep_before = V_keep[:, keep_before_mask, :]  # (B, num_before, d)
            K_keep_after = K_keep[:, keep_after_mask, :]  # (B, num_after, d)
            V_keep_after = V_keep[:, keep_after_mask, :]  # (B, num_after, d)

            K_input = K_to_compact
            V_input = V_to_compact
        else:
            K_input = K_flat
            V_input = V_flat

        # Create batched algorithm instance
        batched_alg = batched_class(**self.algorithm_kwargs)

        # Compact all (layer, head) pairs at once
        C1_compact, beta_compact, C2_compact, indices_batched = batched_alg.compute_compacted_cache_batched(
            K_input, V_input, queries_flat, actual_target_size
        )  # Each: (B, t, d) or (B, t)

        # Handle partial compaction reconstruction
        if is_partial_compaction:
            num_before = K_keep_before.shape[1]
            num_after = K_keep_after.shape[1]

            # Create zero betas for kept portions
            beta_before = torch.zeros(B, num_before, dtype=beta_compact.dtype, device=device)
            beta_after = torch.zeros(B, num_after, dtype=beta_compact.dtype, device=device)

            # Concatenate: [before, compacted, after]
            C1_flat = torch.cat([K_keep_before, C1_compact, K_keep_after], dim=1)  # (B, target_size, d)
            beta_flat = torch.cat([beta_before, beta_compact, beta_after], dim=1)  # (B, target_size)
            C2_flat = torch.cat([V_keep_before, C2_compact, V_keep_after], dim=1)  # (B, target_size, d)
        else:
            C1_flat = C1_compact
            beta_flat = beta_compact
            C2_flat = C2_compact

        # Reshape back to (num_layers, num_heads, target_size, head_dim)
        target_size = C1_flat.shape[1]
        C1_all = C1_flat.reshape(num_layers, num_heads, target_size, head_dim)
        beta_all = beta_flat.reshape(num_layers, num_heads, target_size)
        C2_all = C2_flat.reshape(num_layers, num_heads, target_size, head_dim)

        # Convert to list of tuples with batch dimension
        compacted_layers = []
        for layer_idx in range(num_layers):
            C1_layer = C1_all[layer_idx].unsqueeze(0)  # (1, num_heads, target_size, head_dim)
            beta_layer = beta_all[layer_idx].unsqueeze(0)  # (1, num_heads, target_size)
            C2_layer = C2_all[layer_idx].unsqueeze(0)  # (1, num_heads, target_size, head_dim)
            compacted_layers.append((C1_layer, beta_layer, C2_layer))

            # Collect stats for this layer
            for head_idx in range(num_heads):
                batch_idx = layer_idx * num_heads + head_idx
                selected_indices = indices_batched[batch_idx].cpu().tolist()
                beta_head = beta_compact[batch_idx]  # Only compacted portion (no padding in this code path)

                head_stats = {
                    'layer': layer_idx,
                    'head': head_idx,
                    **({'selected_indices': [int(idx) for idx in selected_indices]} if verbose_logging else {}),
                    'selected_indices_stats': {
                        'count': len(selected_indices),
                        'min': int(min(selected_indices)) if len(selected_indices) > 0 else None,
                        'max': int(max(selected_indices)) if len(selected_indices) > 0 else None,
                    },
                    **({'beta_stats': {
                        'min': float(beta_head.min().item()) if len(beta_head) > 0 else None,
                        'max': float(beta_head.max().item()) if len(beta_head) > 0 else None,
                        'mean': float(beta_head.mean().item()) if len(beta_head) > 0 else None,
                        'std': float(beta_head.std().item()) if len(beta_head) > 1 else None,
                        'num_less_than_minus_7': int((beta_head < -7).sum().item()) if len(beta_head) > 0 else 0,
                    }} if verbose_logging else {})
                }

                # Compute train stats if requested
                if compute_stats:
                    start_time = time.time()
                    from ..algorithms.base import evaluate_compaction

                    # Get queries for this head
                    queries_head = queries_flat[batch_idx]  # (n_queries, head_dim)

                    # Subsample queries to eval_queries_per_kv_head per KV head
                    n_train_queries = queries_head.shape[0]
                    if n_train_queries > eval_queries_per_kv_head:
                        subsample_indices = torch.randperm(n_train_queries)[:eval_queries_per_kv_head]
                        queries_subsample = queries_head[subsample_indices]
                    else:
                        queries_subsample = queries_head

                    # Get K, V for this head (the portion that was compacted)
                    K_head = K_input[batch_idx]  # (num_to_compact, head_dim)
                    V_head = V_input[batch_idx]  # (num_to_compact, head_dim)
                    C1_head = C1_compact[batch_idx]  # (t, head_dim)
                    beta_head_train = beta_compact[batch_idx]  # (t,)
                    C2_head = C2_compact[batch_idx]  # (t, head_dim)

                    # Evaluate on the portion that was compacted
                    train_metrics = evaluate_compaction(
                        K_head, V_head, C1_head, beta_head_train, C2_head, queries_subsample
                    )
                    head_stats['train_stats'] = {k: float(v) for k, v in train_metrics.items()}
                    eval_time = time.time() - start_time
                    all_stats['train_stats_time'] += eval_time

                all_stats['per_layer_head_metrics'][f'L{layer_idx}H{head_idx}'] = head_stats

        all_stats['tensor_compacted_seq_len'] = target_size

        # For fully batched path, all heads get the same size (no per-head padding)
        # The article portion size is actual_target_size
        effective_article_tokens = float(actual_target_size)
        num_kept = len(keep_indices) if keep_indices is not None else 0
        effective_compacted_seq_len = effective_article_tokens + num_kept

        # Tensor article tokens is the actual tensor size of the article portion
        tensor_article_tokens = target_size - num_kept

        all_stats['effective_article_tokens'] = effective_article_tokens
        all_stats['tensor_article_tokens'] = tensor_article_tokens
        all_stats['effective_compacted_seq_len'] = effective_compacted_seq_len
        all_stats['query_generation'] = query_stats

        # Aggregate train stats across all layers/heads
        if compute_stats:
            self._aggregate_train_stats(all_stats, eval_queries_per_kv_head)
            if all_stats['train_stats_time'] > 0:
                print(f"Total train stats computation time: {all_stats['train_stats_time']:.2f}s")

        return compacted_layers, all_stats

    def _aggregate_train_stats(self, all_stats: Dict, eval_queries_per_kv_head: int):
        """Aggregate train stats across all layers and heads into all_head_train_stats."""
        from evaluation.utils import compute_all_head_stats

        # Extract train_stats from per_layer_head_metrics into a flat dict
        train_stats_per_head = {}
        for head_key, head_metrics in all_stats['per_layer_head_metrics'].items():
            if 'train_stats' in head_metrics:
                train_stats_per_head[head_key] = head_metrics['train_stats']

        # Use compute_all_head_stats utility to compute aggregated stats
        # Note: eval_queries_per_kv_head is the number of queries used for evaluation,
        # which is different from max_query_vectors_per_kv_head (used for training)
        all_stats['all_head_train_stats'] = compute_all_head_stats(
            train_stats_per_head,
            eval_queries_per_kv_head
        )

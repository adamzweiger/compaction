# compaction/compaction_methods/global_highest_attention_keys.py
"""
Global highest attention keys compaction method.

Selects the global top r% of attention keys across all layers and heads,
based on attention scores computed from generated queries.
"""
import torch
import time
import json
import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional, Any

from .base import FullCacheCompactionAlgorithm
from ..algorithms.highest_attention_keys import HighestAttentionKeysCompaction
from ..query_generation import QueryConfig


class GlobalHighestAttentionKeysCompaction(FullCacheCompactionAlgorithm):
    """
    Global highest attention keys compaction.

    Computes attention scores globally across all layers and heads,
    then selects the top r% keys. Returns a cache the same size as target_size
    by concatenating kept and compacted portions.
    """

    def __init__(
        self,
        score_method: str = 'max',
        beta_method: str = 'nnls',
        c2_method: str = 'lsq',
        nnls_iters: int = 0,
        nnls_lower_bound: Optional[float] = None,
        nnls_upper_bound: Optional[float] = None,
        c2_ridge_lambda: float = 0,
        c2_solver: str = 'lstsq',
        c2_ridge_scale: str = 'spectral',
        config_name: Optional[str] = None,
        save_head_proportions: bool = False,
    ):
        """
        Initialize global highest attention keys compaction.

        Parameters
        ----------
        score_method : str, optional
            Method to score keys: 'max' for maximum attention (default),
            'mean' for mean attention, or 'rms' for rms attention.
        beta_method : str, optional
            Method to compute beta: 'nnls' to solve via NNLS (default) or 'zero' to set all beta=0.
        c2_method : str, optional
            Method to compute C2: 'lsq' for least squares (default) or 'direct' for nearest neighbor selection.
        nnls_iters : int, optional
            Number of projected gradient descent iterations for NNLS.
            If 0, uses lstsq with clamping (default: 0).
        nnls_lower_bound : float, optional
            Lower bound for NNLS solution (default: None, uses 1e-12).
        nnls_upper_bound : float, optional
            Upper bound for NNLS solution (default: None, no upper bound).
        c2_ridge_lambda : float, optional
            Regularization parameter for C2 ridge regression (default: 0).
        c2_solver : str, optional
            Solver to use for C2: 'pinv', 'cholesky', or 'lstsq' (default: 'lstsq').
        c2_ridge_scale : str, optional
            How to scale ridge_lambda: 'spectral', 'frobenius', or 'fixed' (default: 'spectral').
        config_name : str, optional
            Name of the configuration (used for logging). If not provided, uses a descriptive name.
        save_head_proportions : bool, optional
            If True, save per-head key allocation proportions to JSON (default: False).
        """
        self.score_method = score_method
        self.beta_method = beta_method
        self.c2_method = c2_method
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale
        self.config_name = config_name
        self.save_head_proportions = save_head_proportions

        # Create algorithm instance for use in key selection and C2 computation
        self.algorithm = HighestAttentionKeysCompaction(
            nnls_iters=nnls_iters,
            nnls_lower_bound=nnls_lower_bound,
            nnls_upper_bound=nnls_upper_bound,
            score_method=score_method,
            c2_method=c2_method,
            beta_method=beta_method,
            c2_ridge_lambda=c2_ridge_lambda,
            c2_solver=c2_solver,
            c2_ridge_scale=c2_ridge_scale,
        )

    def name(self) -> str:
        """Return the config name if provided, otherwise a descriptive name."""
        if self.config_name:
            return self.config_name
        return f"nonuniform_score={self.score_method}_beta={self.beta_method}_c2={self.c2_method}"

    def compact_kv_cache(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
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
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        Compact KV cache using global top-k key selection.

        Parameters
        ----------
        past_key_values : tuple of tuples
            KV cache structure: ((keys_layer0, values_layer0), ...)
            keys/values shape: (batch_size, num_heads, seq_len, head_dim)
        target_size : int
            Target compacted sequence length for the full cache.
            If indices is provided, this is the total length after partial compaction.
        indices : range, optional
            Indices of sequence positions to compact. If None, compact entire sequence.
            If provided, only these positions are considered for global selection,
            and the rest remain unchanged.
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

        queries_max_logits : Tensor, optional
            Precomputed max logits per query over the full cache,
            shape (num_layers, num_heads, n_queries, 1)

        Returns
        -------
        compacted_cache : tuple of tuples
            ((C1_layer0, beta_layer0, C2_layer0), ...)
            Each layer has target_size tokens
        stats : dict
            Statistics including global key selection info and per-layer-head metrics
        """
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

        # Only support batch_size=1 for now
        if batch_size != 1:
            raise NotImplementedError(
                "GlobalHighestAttentionKeysCompaction currently only supports batch_size=1"
            )

        from ..query_generation import QueryGenerator

        device = past_key_values[0][0].device
        dtype = past_key_values[0][0].dtype

        # Handle partial compaction (similar to per_layer_head.py)
        is_partial_compaction = indices is not None
        if is_partial_compaction:
            indices_list = list(indices)
            num_to_compact = len(indices_list)
            num_to_keep = seq_len - num_to_compact

            # Compute sub-target size for the compacted portion
            sub_target_size = target_size - num_to_keep

            if sub_target_size <= 0:
                raise ValueError(
                    f"target_size ({target_size}) must be greater than the number of "
                    f"positions to keep ({num_to_keep}). Got sub_target_size = {sub_target_size}"
                )

            actual_target_size = sub_target_size
            print(f"Partial compaction: compacting positions {indices_list[0]}-{indices_list[-1]} "
                  f"({num_to_compact} tokens) to {actual_target_size} tokens")
            print(f"Keeping {num_to_keep} tokens unchanged")

            # Create mask for positions NOT being compacted
            all_indices = torch.arange(seq_len)
            compact_mask = torch.zeros(seq_len, dtype=torch.bool)
            compact_mask[indices_list] = True
            keep_mask = ~compact_mask
            keep_indices = all_indices[keep_mask].tolist()
        else:
            indices_list = None
            keep_indices = None
            actual_target_size = target_size
            is_partial_compaction = False
            print(f"Full compaction: compacting all {seq_len} tokens to {actual_target_size} tokens")

        # Generate queries
        generator = QueryGenerator(
            model=model,
            tokenizer=tokenizer,
            config=query_config,
            device=device,
            dtype=dtype,
            vllm_model=vllm_model,
        )

        # Use past_key_values_for_queries if provided, otherwise use past_key_values
        kv_for_queries = past_key_values_for_queries if past_key_values_for_queries is not None else past_key_values

        # Returns: (num_layers, num_heads, n_queries, head_dim)
        base_queries, query_stats, _ = generator.generate_queries(
            formatted_context=formatted_context,
            past_key_values=kv_for_queries,
        )

        print(f"Generated {query_stats['final_n_queries_per_kv_head']} queries per KV head")
        for method, method_stats in query_stats.get('methods_used', {}).items():
            print(f"  {method}: {method_stats['n_queries_actual_per_kv_head']} queries ({method_stats['fraction']:.1%})")
        print("Train queries shape:", base_queries.shape)

        num_global_layers = num_layers - len(sliding_layer_indices)
        if sliding_layer_indices:
            print(f"Using {len(sliding_layer_indices)} sliding window layers (indices: {sorted(sliding_layer_indices)})")
            print(f"Only compacting {num_global_layers} global attention layers")

        # Extract keys and values to compact (only the article portion if partial compaction)
        # Only extract from global (non-sliding) layers
        K_all_layers = []
        V_all_layers = []
        K_keep_before_all = []
        V_keep_before_all = []
        K_keep_after_all = []
        V_keep_after_all = []
        global_layer_indices = []  # Track which original layer indices are global

        for layer_idx in range(num_layers):
            # Skip sliding window layers
            if layer_idx in sliding_layer_indices:
                continue

            global_layer_indices.append(layer_idx)
            K_layer = []
            V_layer = []
            K_keep_before_layer = []
            V_keep_before_layer = []
            K_keep_after_layer = []
            V_keep_after_layer = []

            for head_idx in range(num_heads):
                K_full = past_key_values[layer_idx][0][0, head_idx, :, :]  # (seq_len, head_dim)
                V_full = past_key_values[layer_idx][1][0, head_idx, :, :]  # (seq_len, head_dim)

                if is_partial_compaction:
                    # Extract the subset to compact
                    K = K_full[indices_list, :]  # (num_to_compact, head_dim)
                    V = V_full[indices_list, :]  # (num_to_compact, head_dim)

                    # Extract portions to keep unchanged
                    # Split keep_indices into before and after
                    compact_start = indices_list[0]
                    compact_end = indices_list[-1] + 1

                    keep_before_mask = [idx < compact_start for idx in keep_indices]
                    keep_after_mask = [idx >= compact_end for idx in keep_indices]

                    K_keep_before = K_full[[keep_indices[i] for i, m in enumerate(keep_before_mask) if m], :] if any(keep_before_mask) else K.new_zeros(0, head_dim)
                    V_keep_before = V_full[[keep_indices[i] for i, m in enumerate(keep_before_mask) if m], :] if any(keep_before_mask) else V.new_zeros(0, head_dim)
                    K_keep_after = K_full[[keep_indices[i] for i, m in enumerate(keep_after_mask) if m], :] if any(keep_after_mask) else K.new_zeros(0, head_dim)
                    V_keep_after = V_full[[keep_indices[i] for i, m in enumerate(keep_after_mask) if m], :] if any(keep_after_mask) else V.new_zeros(0, head_dim)
                else:
                    K = K_full
                    V = V_full
                    K_keep_before = K.new_zeros(0, head_dim)
                    V_keep_before = V.new_zeros(0, head_dim)
                    K_keep_after = K.new_zeros(0, head_dim)
                    V_keep_after = V.new_zeros(0, head_dim)

                K_layer.append(K)
                V_layer.append(V)
                K_keep_before_layer.append(K_keep_before)
                V_keep_before_layer.append(V_keep_before)
                K_keep_after_layer.append(K_keep_after)
                V_keep_after_layer.append(V_keep_after)

            K_all_layers.append(K_layer)
            V_all_layers.append(V_layer)
            K_keep_before_all.append(K_keep_before_layer)
            V_keep_before_all.append(V_keep_before_layer)
            K_keep_after_all.append(K_keep_after_layer)
            V_keep_after_all.append(V_keep_after_layer)

        # Compute global attention scores over the portion to compact
        # Only compute for global (non-sliding) layers
        print("Computing global attention scores...")
        # Extract queries only for global layers
        queries_global_layers = base_queries[global_layer_indices, :, :, :]
        global_scores, layer_head_info, key_scores_dict = self._compute_global_attention_scores(
            K_all_layers, queries_global_layers, global_layer_indices
        )

        # Select global top-k positions
        total_positions = len(global_scores)

        # TRUE global top-k: select across all (layer, head, position) entries
        # Only count global layers, not sliding layers
        num_to_select = actual_target_size * num_global_layers * num_heads

        # Select globally across all layer-head-position tuples
        selected_by_layer_head = self._select_global_top_k(
            global_scores, layer_head_info, num_to_select
        )

        # Save head proportions to JSON if enabled
        if self.save_head_proportions:
            source_size = num_to_compact if is_partial_compaction else seq_len
            compaction_ratio = actual_target_size / source_size
            self._save_head_proportions_to_json(
                selected_by_layer_head, num_global_layers, num_heads, compaction_ratio
            )

        print(f"Selecting top {num_to_select} positions out of {total_positions} total "
              f"({100 * num_to_select / total_positions:.2f}%)")

        print(f"Selected positions across {len(selected_by_layer_head)} layer/head combinations")

        # For nonuniform caches, we'll pad heads within each layer to that layer's maximum
        # Rather than computing a global maximum upfront

        compacted_layers = []
        all_stats = {
            'per_layer_head_metrics': {},
            'is_partial_compaction': is_partial_compaction,
            'train_stats_time': 0.0,
            'num_sliding_layers': len(sliding_layer_indices),
            'num_global_layers': num_global_layers,
            'global_selection_info': {
                'total_positions': total_positions,
                'num_selected': num_to_select,
                'selection_ratio': num_to_select / total_positions,
                'num_layer_head_combinations': len(selected_by_layer_head),
            }
        }

        if is_partial_compaction:
            all_stats['compaction_indices'] = {
                'start': indices_list[0],
                'end': indices_list[-1] + 1,
                'num_positions': len(indices_list),
            }

        # Create a mapping from layer_idx to global_idx for easy lookup
        layer_to_global_idx = {layer_idx: global_idx for global_idx, layer_idx in enumerate(global_layer_indices)}

        # Track total effective article tokens for stats
        total_effective_article_tokens = 0

        for layer_idx in range(num_layers):
            # Handle sliding window layers: keep original KV, no compaction
            if layer_idx in sliding_layer_indices:
                print(f"Layer {layer_idx+1}/{num_layers}: sliding window (keeping original KV)")
                # Create placeholder entry - sliding layers use their own cache
                placeholder_C1 = past_key_values[ref_layer_idx][0].new_zeros(1, num_heads, 0, head_dim)
                placeholder_beta = past_key_values[ref_layer_idx][0].new_zeros(1, num_heads, 0)
                placeholder_C2 = past_key_values[ref_layer_idx][1].new_zeros(1, num_heads, 0, head_dim)
                compacted_layers.append((placeholder_C1, placeholder_beta, placeholder_C2))
                continue

            print(f"Compacting layer {layer_idx+1}/{num_layers}")

            global_idx = layer_to_global_idx[layer_idx]
            C1_heads = []
            beta_heads = []
            C2_heads = []

            for head_idx in range(num_heads):
                key = (layer_idx, head_idx)

                # Get K, V for this head (only the portion to compact)
                # Use global_idx to access the correct position in K_all_layers
                K = K_all_layers[global_idx][head_idx]
                V = V_all_layers[global_idx][head_idx]
                K_keep_before = K_keep_before_all[global_idx][head_idx]
                V_keep_before = V_keep_before_all[global_idx][head_idx]
                K_keep_after = K_keep_after_all[global_idx][head_idx]
                V_keep_after = V_keep_after_all[global_idx][head_idx]

                queries_head = base_queries[layer_idx, head_idx, :, :]  # (n_queries, head_dim)

                if key in selected_by_layer_head:
                    # This head has selected positions
                    selected_indices = selected_by_layer_head[key]
                    num_selected = len(selected_indices)

                    # Extract selected keys and values from the to-compact portion
                    K_selected = K[selected_indices, :]  # (num_selected, head_dim)
                    V_selected = V[selected_indices, :]  # (num_selected, head_dim)

                    # Compute beta for selected positions
                    if self.beta_method == 'zero':
                        beta_selected32 = torch.zeros(num_selected, dtype=torch.float32, device=device)
                    else:  # 'nnls'
                        # Use the algorithm's method (similar to highest_attention_keys.py)
                        beta_selected32 = self._compute_beta_for_selected(
                            K, queries_head, selected_indices
                        )

                    # Convert beta from fp32 to model dtype (e.g., bf16) for storage
                    beta_selected = beta_selected32.to(K.dtype)

                    # Compute C2 for selected positions
                    if self.c2_method == 'direct':
                        C2_selected = V_selected.clone()
                    else:  # 'lsq'
                        C2_selected = self.algorithm._compute_C2(
                            K_selected, beta_selected, K, V, queries_head,
                            ridge_lambda=self.c2_ridge_lambda,
                            solver=self.c2_solver,
                            ridge_scale=self.c2_ridge_scale
                        )
                else:
                    # No positions selected for this head
                    num_selected = 0
                    K_selected = K.new_zeros(0, head_dim)
                    beta_selected = torch.zeros(0, dtype=K.dtype, device=device)
                    C2_selected = V.new_zeros(0, head_dim)
                    selected_indices = []

                # Reconstruct full cache by concatenating: [before, compacted, after]
                # Create zero betas for kept portions
                beta_keep_before = K.new_zeros(K_keep_before.shape[0])
                beta_keep_after = K.new_zeros(K_keep_after.shape[0])

                # Concatenate kept and selected portions
                C1 = torch.cat([K_keep_before, K_selected, K_keep_after], dim=0)
                beta = torch.cat([beta_keep_before, beta_selected, beta_keep_after], dim=0)
                C2 = torch.cat([V_keep_before, C2_selected, V_keep_after], dim=0)

                # Track effective tokens (actual selected, not padded)
                total_effective_article_tokens += num_selected

                # Store results
                C1_heads.append(C1.unsqueeze(0).unsqueeze(0))  # (1, 1, seq_len, head_dim)
                beta_heads.append(beta.unsqueeze(0).unsqueeze(0))  # (1, 1, seq_len)
                C2_heads.append(C2.unsqueeze(0).unsqueeze(0))  # (1, 1, seq_len, head_dim)

                # Store stats
                head_stats = {
                    'layer': layer_idx,
                    'head': head_idx,
                    'num_selected': num_selected,
                    'selection_ratio': num_selected / K.shape[0] if K.shape[0] > 0 else 0.0,
                    **({'selected_indices': [int(idx) for idx in selected_indices]} if verbose_logging else {}),
                    'selected_indices_stats': {
                        'count': len(selected_indices),
                        'min': int(min(selected_indices)) if len(selected_indices) > 0 else None,
                        'max': int(max(selected_indices)) if len(selected_indices) > 0 else None,
                    },
                    **({'beta_stats': {
                        'min': float(beta_selected.min().item()) if len(beta_selected) > 0 else None,
                        'max': float(beta_selected.max().item()) if len(beta_selected) > 0 else None,
                        'mean': float(beta_selected.mean().item()) if len(beta_selected) > 0 else None,
                        'std': float(beta_selected.std().item()) if len(beta_selected) > 1 else None,
                        'num_less_than_minus_7': int((beta_selected < -7).sum().item()) if len(beta_selected) > 0 else 0,
                    }} if verbose_logging else {})
                }

                # Compute train stats if requested
                if compute_stats and num_selected > 0:
                    start_time = time.time()
                    from ..algorithms.base import evaluate_compaction

                    # Subsample queries to eval_queries_per_kv_head per KV head
                    n_train_queries = queries_head.shape[0]
                    if n_train_queries > query_config.eval_queries_per_kv_head:
                        subsample_indices = torch.randperm(n_train_queries)[:query_config.eval_queries_per_kv_head]
                        queries_subsample = queries_head[subsample_indices]
                    else:
                        queries_subsample = queries_head

                    # Evaluate on the portion that was compacted
                    train_metrics = evaluate_compaction(
                        K, V, K_selected, beta_selected, C2_selected, queries_subsample
                    )
                    head_stats['train_stats'] = {k: float(v) for k, v in train_metrics.items()}
                    eval_time = time.time() - start_time
                    all_stats['train_stats_time'] += eval_time

                all_stats['per_layer_head_metrics'][f'L{layer_idx}H{head_idx}'] = head_stats

            # Pad all heads within this layer to the same size before concatenating
            # Each layer can have different sequence lengths (nonuniform caches)
            layer_max_seq_len = max(h.shape[2] for h in C1_heads)
            for i in range(len(C1_heads)):
                curr_len = C1_heads[i].shape[2]
                if curr_len < layer_max_seq_len:
                    pad_len = layer_max_seq_len - curr_len
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
            C1_layer = torch.cat(C1_heads, dim=1)  # (1, num_heads, layer_seq_len, head_dim)
            beta_layer = torch.cat(beta_heads, dim=1)  # (1, num_heads, layer_seq_len)
            C2_layer = torch.cat(C2_heads, dim=1)  # (1, num_heads, layer_seq_len, head_dim)

            compacted_layers.append((C1_layer, beta_layer, C2_layer))

        # Compute average tensor length across all global (non-sliding) layers
        # For nonuniform caches, different layers can have different sequence lengths
        total_tensor_len = 0
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
        effective_article_tokens = total_effective_article_tokens / total_global_heads if total_global_heads > 0 else 0
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
            self._aggregate_train_stats(all_stats, query_config.eval_queries_per_kv_head)
            if all_stats['train_stats_time'] > 0:
                print(f"Total train stats computation time: {all_stats['train_stats_time']:.2f}s")

        return tuple(compacted_layers), all_stats

    def _compute_global_attention_scores(
        self,
        K_all_layers: list,
        queries: torch.Tensor,
        global_layer_indices: list,
    ) -> Tuple[torch.Tensor, list, Dict[Tuple[int, int], torch.Tensor]]:
        """
        Compute attention scores for all keys across all layers and heads.

        Parameters
        ----------
        K_all_layers : list of lists
            K_all_layers[global_idx][head_idx] is a tensor of shape (T, d)
            where T is the number of positions to score (the portion to compact)
            Only contains global (non-sliding) layers.
        queries : Tensor, shape (num_global_layers, num_heads, n_queries, head_dim)
            Query tensors for each global layer and head
        global_layer_indices : list of int
            Original layer indices for each global layer

        Returns
        -------
        global_scores : Tensor, shape (total_positions,)
            Attention scores for all positions across all layers and heads
        layer_head_info : list of tuples
            Mapping from global index to (layer_idx, head_idx, pos_idx)
            layer_idx is the ORIGINAL layer index, not the global index
        key_scores_dict : dict
            Dictionary mapping (layer_idx, head_idx) to key_scores tensor
            layer_idx is the ORIGINAL layer index
        """
        num_global_layers = len(K_all_layers)
        num_heads = len(K_all_layers[0])

        all_scores = []
        layer_head_info = []
        key_scores_dict = {}  # Store key_scores for each layer-head combination

        for global_idx in range(num_global_layers):
            layer_idx = global_layer_indices[global_idx]  # Get original layer index
            for head_idx in range(num_heads):
                K = K_all_layers[global_idx][head_idx]  # (T, d)
                queries_head = queries[global_idx, head_idx, :, :]  # (n_queries, head_dim)

                # Compute attention scores for this head (same as highest_attention_keys.py)
                n, d = queries_head.shape
                T = K.shape[0]
                inv_sqrt_d = (1.0 / d) ** 0.5

                # QK matmul in original dtype; upcast for softmax
                scores_raw = queries_head @ K.T  # (n, T)
                scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
                max_scores = scores32.max(dim=1, keepdim=True)[0]
                exp_scores = torch.exp(scores32 - max_scores)

                # Compute softmax attention weights
                sum_exp = exp_scores.sum(dim=1, keepdim=True)
                attention_weights = exp_scores / sum_exp  # (n, T)

                # Compute score for each key
                if self.score_method == 'rms':
                    key_scores = torch.sqrt((attention_weights ** 2).mean(dim=0))  # (T,)
                elif self.score_method == 'max':
                    key_scores = attention_weights.max(dim=0)[0]  # (T,)
                else:  # 'mean'
                    key_scores = attention_weights.mean(dim=0)  # (T,)

                all_scores.append(key_scores)
                key_scores_dict[(layer_idx, head_idx)] = key_scores.cpu()

                # Track which layer/head/position each score corresponds to
                for pos_idx in range(T):
                    layer_head_info.append((layer_idx, head_idx, pos_idx))

        # Concatenate all scores
        global_scores = torch.cat(all_scores, dim=0)  # (total_positions,)

        return global_scores, layer_head_info, key_scores_dict

    def _save_head_proportions_to_json(
        self,
        selected_by_layer_head: Dict[Tuple[int, int], list],
        num_layers: int,
        num_heads: int,
        compaction_ratio: float
    ) -> None:
        """
        Save per-head key allocation proportions to JSON.

        Proportion for each head = (#selected keys in this head) / (total #selected keys).
        All proportions sum to 1 (up to floating point error).
        """
        total_selected = sum(len(idxs) for idxs in selected_by_layer_head.values())
        if total_selected == 0:
            print("Warning: total_selected is 0, cannot compute allocation proportions")
            return

        proportions = {}
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                key = (layer_idx, head_idx)
                n = len(selected_by_layer_head.get(key, []))
                proportions[f"L{layer_idx}H{head_idx}"] = n / total_selected

        output_dir = Path("logs/head_proportions")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_suffix = (
            self.config_name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "_")
            if self.config_name else "nonuniform"
        )
        filename = f"head_proportions_t{compaction_ratio:.2f}_{config_suffix}_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(proportions, f, indent=2)

        print(f"Head proportions saved to: {filepath}")

    def _compute_beta_for_selected(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        selected_indices: list,
    ) -> torch.Tensor:
        """
        Compute beta values for specific selected indices using NNLS.

        This follows the same approach as highest_attention_keys.py:
        - Compute exp_scores over all keys in K
        - Extract exp_scores for selected positions
        - Solve NNLS to match the sum of all exp_scores

        Parameters
        ----------
        K : Tensor, shape (T, head_dim)
            Key matrix (the portion being compacted)
        queries : Tensor, shape (n_queries, head_dim)
            Query vectors
        selected_indices : list of int
            Indices of selected keys (relative to K)

        Returns
        -------
        beta : Tensor, shape (num_selected,)
            Beta values for selected keys
        """
        device = K.device
        d = K.shape[1]
        inv_sqrt_d = (1.0 / d) ** 0.5

        # Compute exp_scores for all positions (same as highest_attention_keys.py)
        scores_raw = queries @ K.T  # (n, T)
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
        max_scores = scores32.max(dim=1, keepdim=True)[0]
        exp_scores = torch.exp(scores32 - max_scores)  # (n, T)

        # Extract exp_scores for selected positions
        exp_scores_selected = exp_scores[:, selected_indices]  # (n, num_selected)

        # Compute target (sum of all exp_scores)
        target = exp_scores.sum(dim=1)  # (n,)

        # Solve NNLS: min ||M B - target||^2, B >= 0
        B = self.algorithm._nnls_pg(
            exp_scores_selected, target,
            self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound
        )
        beta = torch.log(B)

        return beta

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
            eval_queries_per_kv_head or 0
        )

    def _select_global_top_k(
        self,
        global_scores: torch.Tensor,
        layer_head_info: list,
        num_to_select: int,
    ) -> Dict[Tuple[int, int], list]:
        """
        Select global top-k positions across ALL (layer, head, position) entries.

        Parameters
        ----------
        global_scores : Tensor, shape (total_positions,)
            Concatenated scores for all positions across all layers and heads
        layer_head_info : list of tuples
            Mapping from global index to (layer_idx, head_idx, pos_idx)
        num_to_select : int
            Total number of positions to select globally

        Returns
        -------
        selected_by_layer_head : dict
            Dictionary mapping (layer_idx, head_idx) to list of selected position indices
        """
        # Select top-k globally
        k = min(num_to_select, len(global_scores))
        _, top_global_indices = torch.topk(global_scores, k, largest=True)

        # Group selected positions by (layer, head)
        selected_by_layer_head = {}
        for global_idx in top_global_indices.cpu().tolist():
            layer_idx, head_idx, pos_idx = layer_head_info[global_idx]
            key = (layer_idx, head_idx)
            if key not in selected_by_layer_head:
                selected_by_layer_head[key] = []
            selected_by_layer_head[key].append(pos_idx)

        # Sort indices within each head for consistent ordering
        for key in selected_by_layer_head:
            selected_by_layer_head[key] = sorted(selected_by_layer_head[key])

        return selected_by_layer_head

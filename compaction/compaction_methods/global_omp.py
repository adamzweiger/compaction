# compaction/compaction_methods/global_omp.py
"""
Note: Not tested recently or used in final paper.

Global OMP compaction method.

Uses orthogonal matching pursuit with global key selection across all layers and heads.
At each iteration, computes correlation between keys and residual for all layer/head
combinations, then selects the top chunk globally.
"""
import torch
import time
import json
import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional, Any, List

from .base import FullCacheCompactionAlgorithm
from ..algorithms.omp import OMPCompaction
from ..query_generation import QueryConfig


class GlobalOMPCompaction(FullCacheCompactionAlgorithm):
    """
    Global OMP compaction with global key selection.

    At each OMP iteration, computes the correlation between exp_scores and residual
    for all keys across all layers and heads, then selects the top chunk globally.
    Updates the compacted cache for the selected layers/heads.
    """

    def __init__(
        self,
        beta_method: str = 'nnls',
        c2_method: str = 'lsq',
        nnls_iters: int = 0,
        nnls_lower_bound: Optional[float] = None,
        nnls_upper_bound: Optional[float] = None,
        c2_ridge_lambda: float = 0,
        c2_solver: str = 'lstsq',
        c2_ridge_scale: str = 'spectral',
        k_choice: int = 1,
        nnls_interval: Optional[int] = None,
        use_abs_corr: bool = False,
        normalize_exp_scores: bool = False,
        config_name: Optional[str] = None,
        debug: bool = False,
        precompute_ratios: bool = False,
        save_head_proportions: bool = False,
    ):
        """
        Initialize global OMP compaction.

        Parameters
        ----------
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
        k_choice : int, optional
            Number of keys to select at once (chunk size) per global iteration (default: 1).
        nnls_interval : int, optional
            If provided, solve NNLS only every nnls_interval iterations.
        use_abs_corr : bool, optional
            If True, use absolute correlation for key selection (default: False).
        normalize_exp_scores : bool, optional
            If True, normalize exp_scores columns by L2 norm before computing correlation (default: False).
        config_name : str, optional
            Name of the configuration (used for logging).
        debug : bool, optional
            If True, enable debug printing (default: False).
        precompute_ratios : bool, optional
            If True, do one global correlation step to determine per-layer/head budgets,
            then run local OMP for each layer/head with that budget (default: False).
        save_head_proportions : bool, optional
            If True, save per-head key allocation proportions to JSON (default: False).
        """
        self.beta_method = beta_method
        self.c2_method = c2_method
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale
        self.k_choice = k_choice
        self.nnls_interval = nnls_interval
        self.use_abs_corr = use_abs_corr
        self.normalize_exp_scores = normalize_exp_scores
        self.config_name = config_name
        self.debug = debug
        self.precompute_ratios = precompute_ratios
        self.save_head_proportions = save_head_proportions

        # Create algorithm instance for C2 computation
        self.algorithm = OMPCompaction(
            nnls_iters=nnls_iters,
            nnls_lower_bound=nnls_lower_bound,
            nnls_upper_bound=nnls_upper_bound,
            c2_method=c2_method,
            k_choice=k_choice,
            c2_ridge_lambda=c2_ridge_lambda,
            c2_solver=c2_solver,
            c2_ridge_scale=c2_ridge_scale,
            nnls_interval=nnls_interval,
            use_abs_corr=use_abs_corr,
            normalize_exp_scores=normalize_exp_scores,
            debug=debug,
        )

    def name(self) -> str:
        """Return the config name if provided, otherwise a descriptive name."""
        if self.config_name:
            return self.config_name
        return f"nonuniform_omp_beta={self.beta_method}_c2={self.c2_method}_k={self.k_choice}"

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
        Compact KV cache using global OMP-based key selection.

        Parameters
        ----------
        past_key_values : tuple of tuples
            KV cache structure: ((keys_layer0, values_layer0), ...)
            keys/values shape: (batch_size, num_heads, seq_len, head_dim)
        target_size : int
            Target compacted sequence length for the full cache.
        indices : range, optional
            Indices of sequence positions to compact. If None, compact entire sequence.
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

        Returns
        -------
        compacted_cache : tuple of tuples
            ((C1_layer0, beta_layer0, C2_layer0), ...)
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

        num_global_layers = num_layers - len(sliding_layer_indices)
        if sliding_layer_indices:
            print(f"Using {len(sliding_layer_indices)} sliding window layers (indices: {sorted(sliding_layer_indices)})")
            print(f"Only compacting {num_global_layers} global attention layers")

        if batch_size != 1:
            raise NotImplementedError(
                "GlobalOMPCompaction currently only supports batch_size=1"
            )

        from ..query_generation import QueryGenerator

        device = past_key_values[ref_layer_idx][0].device
        dtype = past_key_values[ref_layer_idx][0].dtype

        # Handle partial compaction
        is_partial_compaction = indices is not None
        if is_partial_compaction:
            indices_list = list(indices)
            num_to_compact = len(indices_list)
            num_to_keep = seq_len - num_to_compact
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

            all_indices = torch.arange(seq_len)
            compact_mask = torch.zeros(seq_len, dtype=torch.bool)
            compact_mask[indices_list] = True
            keep_mask = ~compact_mask
            keep_indices = all_indices[keep_mask].tolist()
        else:
            indices_list = None
            keep_indices = None
            actual_target_size = target_size
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

        base_queries, query_stats, _ = generator.generate_queries(
            formatted_context=formatted_context,
            past_key_values=kv_for_queries,
        )

        print(f"Generated {query_stats['final_n_queries_per_kv_head']} queries per KV head")
        for method, method_stats in query_stats.get('methods_used', {}).items():
            print(f"  {method}: {method_stats['n_queries_actual_per_kv_head']} queries ({method_stats['fraction']:.1%})")
        print("Train queries shape:", base_queries.shape)

        # Extract keys and values to compact (only from global layers)
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
                K_full = past_key_values[layer_idx][0][0, head_idx, :, :]
                V_full = past_key_values[layer_idx][1][0, head_idx, :, :]

                if is_partial_compaction:
                    K = K_full[indices_list, :]
                    V = V_full[indices_list, :]

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

        # Run global OMP selection or precompute ratios + local OMP
        # Extract queries only for global layers
        queries_global_layers = base_queries[global_layer_indices, :, :, :]
        if self.precompute_ratios:
            print("Running precompute ratios + local OMP selection...")
            selected_by_layer_head, omp_stats = self._precompute_and_local_omp_selection(
                K_all_layers, queries_global_layers, actual_target_size, global_layer_indices
            )
        else:
            print("Running global OMP selection...")
            selected_by_layer_head, omp_stats = self._global_omp_selection(
                K_all_layers, queries_global_layers, actual_target_size, global_layer_indices
            )

        print(f"Selected positions across {len(selected_by_layer_head)} layer/head combinations")

        # Create a mapping from layer_idx to global_idx for easy lookup
        layer_to_global_idx = {layer_idx: global_idx for global_idx, layer_idx in enumerate(global_layer_indices)}

        # For nonuniform caches, we'll pad heads within each layer to that layer's maximum
        # Rather than computing a global maximum upfront

        # Build compacted cache
        compacted_layers = []
        all_stats = {
            'per_layer_head_metrics': {},
            'is_partial_compaction': is_partial_compaction,
            'train_stats_time': 0.0,
            'num_sliding_layers': len(sliding_layer_indices),
            'num_global_layers': num_global_layers,
            'global_selection_info': omp_stats,
        }

        if is_partial_compaction:
            all_stats['compaction_indices'] = {
                'start': indices_list[0],
                'end': indices_list[-1] + 1,
                'num_positions': len(indices_list),
            }

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

                # Use global_idx to access the correct position in K_all_layers
                K = K_all_layers[global_idx][head_idx]
                V = V_all_layers[global_idx][head_idx]
                K_keep_before = K_keep_before_all[global_idx][head_idx]
                V_keep_before = V_keep_before_all[global_idx][head_idx]
                K_keep_after = K_keep_after_all[global_idx][head_idx]
                V_keep_after = V_keep_after_all[global_idx][head_idx]

                queries_head = base_queries[layer_idx, head_idx, :, :]

                if key in selected_by_layer_head:
                    selected_indices = selected_by_layer_head[key]
                    num_selected = len(selected_indices)

                    K_selected = K[selected_indices, :]
                    V_selected = V[selected_indices, :]

                    # Compute beta
                    if self.beta_method == 'zero':
                        beta_selected32 = torch.zeros(num_selected, dtype=torch.float32, device=device)
                    else:
                        beta_selected32 = self._compute_beta_for_selected(
                            K, queries_head, selected_indices
                        )

                    beta_selected = beta_selected32.to(K.dtype)

                    # Compute C2
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
                    num_selected = 0
                    K_selected = K.new_zeros(0, head_dim)
                    beta_selected = torch.zeros(0, dtype=K.dtype, device=device)
                    C2_selected = V.new_zeros(0, head_dim)
                    selected_indices = []

                # Reconstruct full cache
                beta_keep_before = K.new_zeros(K_keep_before.shape[0])
                beta_keep_after = K.new_zeros(K_keep_after.shape[0])

                C1 = torch.cat([K_keep_before, K_selected, K_keep_after], dim=0)
                beta = torch.cat([beta_keep_before, beta_selected, beta_keep_after], dim=0)
                C2 = torch.cat([V_keep_before, C2_selected, V_keep_after], dim=0)

                # Track effective tokens (actual selected, not padded)
                total_effective_article_tokens += num_selected

                C1_heads.append(C1.unsqueeze(0).unsqueeze(0))
                beta_heads.append(beta.unsqueeze(0).unsqueeze(0))
                C2_heads.append(C2.unsqueeze(0).unsqueeze(0))

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

                if compute_stats and num_selected > 0:
                    start_time = time.time()
                    from ..algorithms.base import evaluate_compaction

                    n_train_queries = queries_head.shape[0]
                    if n_train_queries > query_config.eval_queries_per_kv_head:
                        subsample_indices = torch.randperm(n_train_queries)[:query_config.eval_queries_per_kv_head]
                        queries_subsample = queries_head[subsample_indices]
                    else:
                        queries_subsample = queries_head

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

            C1_layer = torch.cat(C1_heads, dim=1)
            beta_layer = torch.cat(beta_heads, dim=1)
            C2_layer = torch.cat(C2_heads, dim=1)

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

        if compute_stats:
            self._aggregate_train_stats(all_stats, query_config.eval_queries_per_kv_head)
            if all_stats['train_stats_time'] > 0:
                print(f"Total train stats computation time: {all_stats['train_stats_time']:.2f}s")

        # Save head proportions to JSON if enabled
        if self.save_head_proportions:
            self._save_head_proportions_to_json(selected_by_layer_head, num_layers, num_heads)

        return tuple(compacted_layers), all_stats

    def _global_omp_selection(
        self,
        K_all_layers: List[List[torch.Tensor]],
        queries: torch.Tensor,
        target_size: int,
        global_layer_indices: List[int],
    ) -> Tuple[Dict[Tuple[int, int], List[int]], Dict]:
        """
        Run global OMP selection across all layers and heads.

        At each iteration:
        1. Compute correlation between exp_scores and residual for all unselected keys
        2. Select top k_choice keys globally
        3. Update the approximation for affected layers/heads
        4. Repeat until we have target_size * num_layers * num_heads keys

        Parameters
        ----------
        K_all_layers : list of lists
            K_all_layers[global_idx][head_idx] is a tensor of shape (T, d)
            Only contains global (non-sliding) layers.
        queries : Tensor, shape (num_global_layers, num_heads, n_queries, head_dim)
            Query tensors for each global layer and head
        target_size : int
            Target number of keys per layer/head on average
        global_layer_indices : list of int
            Original layer indices for each global layer

        Returns
        -------
        selected_by_layer_head : dict
            Mapping from (layer_idx, head_idx) to list of selected position indices
            layer_idx is the ORIGINAL layer index
        stats : dict
            Statistics about the selection process
        """
        num_global_layers = len(K_all_layers)
        num_heads = len(K_all_layers[0])
        T = K_all_layers[0][0].shape[0]
        d = K_all_layers[0][0].shape[1]
        n_queries = queries.shape[2]
        device = K_all_layers[0][0].device

        inv_sqrt_d = (1.0 / d) ** 0.5
        total_to_select = target_size * num_global_layers * num_heads

        print(f"Global OMP: selecting {total_to_select} positions out of {T * num_global_layers * num_heads} total")

        # First pass: compute targets and max_scores for each head (one at a time to save memory)
        head_targets = {}
        head_max_scores = {}

        for global_idx in range(num_global_layers):
            layer_idx = global_layer_indices[global_idx]
            for head_idx in range(num_heads):
                K = K_all_layers[global_idx][head_idx]
                queries_head = queries[global_idx, head_idx, :, :]

                scores_raw = queries_head @ K.T
                scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
                max_scores = scores32.max(dim=1, keepdim=True)[0]

                exp_scores = torch.exp(scores32 - max_scores)
                target = exp_scores.sum(dim=1)

                head_targets[(layer_idx, head_idx)] = target
                head_max_scores[(layer_idx, head_idx)] = max_scores

                # Free large tensors immediately
                del exp_scores, scores32, scores_raw

        # Initialize per-head state (minimal storage - no exp_scores)
        head_states = {}
        for global_idx, layer_idx in enumerate(global_layer_indices):
            for head_idx in range(num_heads):
                head_states[(layer_idx, head_idx)] = {
                    'current': torch.zeros(n_queries, dtype=torch.float32, device=device),
                    'mask_selected': torch.zeros(T, dtype=torch.bool, device=device),
                    'selected_indices': [],
                    'prev_B': None,
                    'global_idx': global_idx,
                }

        # Global OMP iterations
        num_selected = 0
        iteration = 0

        while num_selected < total_to_select:
            # Compute correlation for all unselected keys across all layers/heads
            # Process one head at a time to save memory
            all_correlations = []
            correlation_info = []

            if self.debug:
                start_time = time.time()
                print("iteration", iteration, "Computing correlations")
                avg_time_exp_score_computation = 0
                avg_time_correlation_computation = 0

            for global_idx in range(num_global_layers):
                layer_idx = global_layer_indices[global_idx]
                for head_idx in range(num_heads):
                    state = head_states[(layer_idx, head_idx)]
                    target = head_targets[(layer_idx, head_idx)]
                    max_scores = head_max_scores[(layer_idx, head_idx)]
                    residual = target - state['current']
                    mask = state['mask_selected']

                    K = K_all_layers[global_idx][head_idx]
                    queries_head = queries[global_idx, head_idx, :, :]

                    # Recompute exp_scores for this head on-the-fly
                    if self.debug:
                        start_time_exp_score_computation = time.time()
                    scores_raw = queries_head @ K.T
                    scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
                    exp_scores = torch.exp(scores32 - max_scores)
                    if self.debug:
                        end_time_exp_score_computation = time.time()
                        avg_time_exp_score_computation += end_time_exp_score_computation - start_time_exp_score_computation

                    # Compute correlation
                    if self.debug:
                        start_time_correlation_computation = time.time()
                    if self.normalize_exp_scores:
                        exp_scores_norm = torch.norm(exp_scores, dim=0, keepdim=True)
                        exp_scores_normalized = exp_scores / (exp_scores_norm + 1e-12)
                        corr = (exp_scores_normalized * residual.unsqueeze(1)).sum(dim=0)
                    else:
                        corr = (exp_scores * residual.unsqueeze(1)).sum(dim=0)
                    if self.debug:
                        end_time_correlation_computation = time.time()
                        avg_time_correlation_computation += end_time_correlation_computation - start_time_correlation_computation

                    # Free large tensors immediately
                    del exp_scores, scores32, scores_raw

                    # Use absolute correlation if configured
                    if self.use_abs_corr:
                        corr_for_selection = torch.abs(corr)
                    else:
                        corr_for_selection = corr

                    # Mask already selected
                    corr_for_selection[mask] = -float('inf')

                    all_correlations.append(corr_for_selection)
                    for pos_idx in range(T):
                        correlation_info.append((layer_idx, head_idx, pos_idx))

            if self.debug:
                end_time = time.time()
                print("iteration", iteration, "Time taken to compute correlations across all layer-heads", end_time - start_time)
                print("iteration", iteration, "Total time for exp score computation", avg_time_exp_score_computation)
                print("iteration", iteration, "Total time for correlation computation", avg_time_correlation_computation)

            # Concatenate all correlations
            global_corr = torch.cat(all_correlations, dim=0)

            # Select top k_choice globally
            num_remaining = total_to_select - num_selected
            k_select = min(self.k_choice, num_remaining)

            if k_select == 0:
                break

            # Check if there are valid candidates
            valid_mask = global_corr > -float('inf')
            if not valid_mask.any():
                print(f"Warning: No valid candidates remaining at iteration {iteration}")
                break
            
            if self.debug:
                start_time = time.time()
                print("iteration", iteration, "Selecting top k_select candidates")
            top_k_values, top_k_global_indices = torch.topk(global_corr, k_select, largest=True)
            if self.debug:
                print("iteration", iteration, "Top k_select candidates selected")
                end_time = time.time()
                print("iteration", iteration, "Time taken to select top k_select candidates", end_time - start_time)

            # Track which heads got updated this iteration
            heads_updated = set()

            # Map back to (layer, head, position) and update states
            for global_idx in top_k_global_indices.cpu().tolist():
                if num_selected >= total_to_select:
                    break

                layer_idx, head_idx, pos_idx = correlation_info[global_idx]
                state = head_states[(layer_idx, head_idx)]

                # Skip if already selected
                if state['mask_selected'][pos_idx]:
                    continue

                # Add to selected
                state['mask_selected'][pos_idx] = True
                state['selected_indices'].append(pos_idx)
                num_selected += 1
                heads_updated.add((layer_idx, head_idx))

            # Update approximation only for heads that got new selections this iteration
            for (layer_idx, head_idx) in heads_updated:
                state = head_states[(layer_idx, head_idx)]
                if len(state['selected_indices']) > 0:
                    global_idx = state['global_idx']
                    K = K_all_layers[global_idx][head_idx]
                    queries_head = queries[global_idx, head_idx, :, :]
                    max_scores = head_max_scores[(layer_idx, head_idx)]
                    target = head_targets[(layer_idx, head_idx)]

                    # Compute exp_scores only for selected columns
                    selected_indices_tensor = torch.tensor(state['selected_indices'], dtype=torch.long, device=device)
                    K_selected = K[selected_indices_tensor, :]

                    scores_raw = queries_head @ K_selected.T
                    scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
                    M = torch.exp(scores32 - max_scores)

                    # Solve NNLS
                    if self.beta_method == 'zero':
                        B = torch.ones(len(state['selected_indices']), dtype=torch.float32, device=device)
                    else:
                        # Use interval-based or norm-threshold-based lazy NNLS
                        should_solve = False
                        if state['prev_B'] is None:
                            should_solve = True
                        elif self.nnls_interval is not None:
                            should_solve = (iteration % self.nnls_interval == 0)
                        else:
                            should_solve = True  # Default: always solve

                        if should_solve:
                            B = self.algorithm._nnls_pg(
                                M, target,
                                self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound
                            )
                        else:
                            # Extend previous solution
                            i = M.shape[1]
                            prev_i = state['prev_B'].shape[0]
                            B = torch.zeros(i, dtype=torch.float32, device=device)
                            B[:prev_i] = state['prev_B']
                            min_val = 1e-12 if self.nnls_lower_bound is None else self.nnls_lower_bound
                            B[prev_i:] = min_val

                    state['prev_B'] = B
                    state['current'] = M @ B

                    # Free tensors
                    del M, scores32, scores_raw

            iteration += 1

            if self.debug and iteration % 100 == 0:
                print(f"[Global OMP] Iteration {iteration}: selected {num_selected}/{total_to_select}")

        # Extract selected indices per layer/head
        selected_by_layer_head = {}
        for (layer_idx, head_idx), state in head_states.items():
            if len(state['selected_indices']) > 0:
                selected_by_layer_head[(layer_idx, head_idx)] = sorted(state['selected_indices'])

        stats = {
            'total_positions': T * num_global_layers * num_heads,
            'num_selected': num_selected,
            'selection_ratio': num_selected / (T * num_global_layers * num_heads),
            'num_iterations': iteration,
            'k_choice': self.k_choice,
        }

        return selected_by_layer_head, stats

    def _precompute_and_local_omp_selection(
        self,
        K_all_layers: List[List[torch.Tensor]],
        queries: torch.Tensor,
        target_size: int,
        global_layer_indices: List[int],
    ) -> Tuple[Dict[Tuple[int, int], List[int]], Dict]:
        """
        Precompute per-layer/head budgets via one global correlation step,
        then run local OMP for each layer/head with that budget.

        Parameters
        ----------
        K_all_layers : list of lists
            K_all_layers[global_idx][head_idx] is a tensor of shape (T, d)
            Only contains global (non-sliding) layers.
        queries : Tensor, shape (num_global_layers, num_heads, n_queries, head_dim)
            Query tensors for each global layer and head
        target_size : int
            Target number of keys per layer/head on average
        global_layer_indices : list of int
            Original layer indices for each global layer

        Returns
        -------
        selected_by_layer_head : dict
            Mapping from (layer_idx, head_idx) to list of selected position indices
            layer_idx is the ORIGINAL layer index
        stats : dict
            Statistics about the selection process
        """
        num_global_layers = len(K_all_layers)
        num_heads = len(K_all_layers[0])
        T = K_all_layers[0][0].shape[0]

        # Step 1: Compute budgets via one global correlation step
        print("Computing per-layer/head budgets via global correlation...")
        budgets = self._precompute_budgets(K_all_layers, queries, target_size, global_layer_indices)

        # Print budget distribution
        total_budget = sum(budgets.values())
        print(f"Budget distribution: total={total_budget}, "
              f"min={min(budgets.values()) if budgets else 0}, "
              f"max={max(budgets.values()) if budgets else 0}, "
              f"mean={total_budget / (num_global_layers * num_heads):.1f}")

        # Step 2: Run local OMP for each layer/head with its budget
        print("Running local OMP for each layer/head...")
        selected_by_layer_head = {}
        total_selected = 0

        for global_idx in range(num_global_layers):
            layer_idx = global_layer_indices[global_idx]
            for head_idx in range(num_heads):
                print(f"Running local OMP for layer {layer_idx}, head {head_idx} with budget {budgets.get((layer_idx, head_idx), 0)}")
                key = (layer_idx, head_idx)
                budget = budgets.get(key, 0)

                if budget > 0:
                    K = K_all_layers[global_idx][head_idx]
                    queries_head = queries[global_idx, head_idx, :, :]

                    selected_indices = self._local_omp_selection(
                        K, queries_head, budget
                    )
                    selected_by_layer_head[key] = selected_indices
                    total_selected += len(selected_indices)

        print(f"Local OMP completed: selected {total_selected} positions total")

        stats = {
            'total_positions': T * num_global_layers * num_heads,
            'num_selected': total_selected,
            'selection_ratio': total_selected / (T * num_global_layers * num_heads),
            'precompute_ratios': True,
            'budgets': {f'L{l}H{h}': b for (l, h), b in budgets.items()},
            'k_choice': self.k_choice,
        }

        return selected_by_layer_head, stats

    def _precompute_budgets(
        self,
        K_all_layers: List[List[torch.Tensor]],
        queries: torch.Tensor,
        target_size: int,
        global_layer_indices: List[int],
    ) -> Dict[Tuple[int, int], int]:
        """
        Compute per-layer/head budgets by doing one global correlation step.

        This computes the correlation for all keys across all layers/heads,
        then selects the top target_size * num_global_layers * num_heads keys globally.
        The count of keys per layer/head determines the budget.

        Parameters
        ----------
        K_all_layers : list of lists
            K_all_layers[global_idx][head_idx] is a tensor of shape (T, d)
            Only contains global (non-sliding) layers.
        queries : Tensor, shape (num_global_layers, num_heads, n_queries, head_dim)
            Query tensors for each global layer and head
        target_size : int
            Target number of keys per layer/head on average
        global_layer_indices : list of int
            Original layer indices for each global layer

        Returns
        -------
        budgets : dict
            Mapping from (layer_idx, head_idx) to budget (number of keys to select)
            layer_idx is the ORIGINAL layer index
        """
        num_global_layers = len(K_all_layers)
        num_heads = len(K_all_layers[0])
        T = K_all_layers[0][0].shape[0]
        d = K_all_layers[0][0].shape[1]

        inv_sqrt_d = (1.0 / d) ** 0.5
        total_to_select = target_size * num_global_layers * num_heads

        # Compute correlation for all keys (initial residual = target since current approx = 0)
        all_correlations = []
        correlation_info = []

        for global_idx in range(num_global_layers):
            layer_idx = global_layer_indices[global_idx]
            for head_idx in range(num_heads):
                K = K_all_layers[global_idx][head_idx]
                queries_head = queries[global_idx, head_idx, :, :]

                scores_raw = queries_head @ K.T
                scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
                max_scores = scores32.max(dim=1, keepdim=True)[0]

                exp_scores = torch.exp(scores32 - max_scores)
                target = exp_scores.sum(dim=1)

                # Initial residual is just target (current approximation is 0)
                residual = target

                # Compute correlation
                if self.normalize_exp_scores:
                    exp_scores_norm = torch.norm(exp_scores, dim=0, keepdim=True)
                    exp_scores_normalized = exp_scores / (exp_scores_norm + 1e-12)
                    corr = (exp_scores_normalized * residual.unsqueeze(1)).sum(dim=0)
                else:
                    corr = (exp_scores * residual.unsqueeze(1)).sum(dim=0)

                if self.use_abs_corr:
                    corr = torch.abs(corr)

                all_correlations.append(corr)
                for pos_idx in range(T):
                    correlation_info.append((layer_idx, head_idx, pos_idx))

                del exp_scores, scores32, scores_raw

        # Concatenate and select top k
        global_corr = torch.cat(all_correlations, dim=0)
        _, top_k_indices = torch.topk(global_corr, total_to_select, largest=True)

        # Count budget per layer/head
        budgets = {}
        for global_idx in top_k_indices.cpu().tolist():
            layer_idx, head_idx, pos_idx = correlation_info[global_idx]
            key = (layer_idx, head_idx)
            budgets[key] = budgets.get(key, 0) + 1

        # Ensure all global layer/heads have an entry (even if 0)
        for layer_idx in global_layer_indices:
            for head_idx in range(num_heads):
                key = (layer_idx, head_idx)
                if key not in budgets:
                    budgets[key] = 0

        return budgets

    def _local_omp_selection(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        target_size: int,
    ) -> List[int]:
        """
        Run OMP selection for a single layer/head to select target_size keys.

        Parameters
        ----------
        K : Tensor, shape (T, head_dim)
            Key matrix for this head
        queries : Tensor, shape (n_queries, head_dim)
            Query vectors for this head
        target_size : int
            Number of keys to select

        Returns
        -------
        selected_indices : list of int
            Indices of selected keys
        """
        T = K.shape[0]
        d = K.shape[1]
        device = K.device
        n_queries = queries.shape[0]

        if target_size <= 0:
            return []

        if target_size >= T:
            return list(range(T))

        inv_sqrt_d = (1.0 / d) ** 0.5

        # Compute exp_scores and target
        scores_raw = queries @ K.T
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
        max_scores = scores32.max(dim=1, keepdim=True)[0]

        exp_scores = torch.exp(scores32 - max_scores)
        target = exp_scores.sum(dim=1)

        # Initialize state
        current = torch.zeros(n_queries, dtype=torch.float32, device=device)
        mask_selected = torch.zeros(T, dtype=torch.bool, device=device)
        selected_indices = []
        prev_B = None

        # OMP iterations
        iteration = 0
        while len(selected_indices) < target_size:
            residual = target - current

            # Compute correlation
            if self.normalize_exp_scores:
                exp_scores_norm = torch.norm(exp_scores, dim=0, keepdim=True)
                exp_scores_normalized = exp_scores / (exp_scores_norm + 1e-12)
                corr = (exp_scores_normalized * residual.unsqueeze(1)).sum(dim=0)
            else:
                corr = (exp_scores * residual.unsqueeze(1)).sum(dim=0)

            if self.use_abs_corr:
                corr_for_selection = torch.abs(corr)
            else:
                corr_for_selection = corr

            # Mask already selected
            corr_for_selection[mask_selected] = -float('inf')

            # Select top k_choice
            num_remaining = target_size - len(selected_indices)
            k_select = min(self.k_choice, num_remaining)

            if k_select == 0:
                break

            valid_mask = corr_for_selection > -float('inf')
            if not valid_mask.any():
                break

            top_k_values, top_k_indices = torch.topk(corr_for_selection, k_select, largest=True)

            for idx in top_k_indices.cpu().tolist():
                if len(selected_indices) >= target_size:
                    break
                if mask_selected[idx]:
                    continue
                mask_selected[idx] = True
                selected_indices.append(idx)

            # Update approximation
            if len(selected_indices) > 0:
                selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)
                M = exp_scores[:, selected_indices_tensor]

                if self.beta_method == 'zero':
                    B = torch.ones(len(selected_indices), dtype=torch.float32, device=device)
                else:
                    should_solve = False
                    if prev_B is None:
                        should_solve = True
                    elif self.nnls_interval is not None:
                        should_solve = (iteration % self.nnls_interval == 0)
                    else:
                        should_solve = True

                    if should_solve:
                        B = self.algorithm._nnls_pg(
                            M, target,
                            self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound
                        )
                    else:
                        i = M.shape[1]
                        prev_i = prev_B.shape[0]
                        B = torch.zeros(i, dtype=torch.float32, device=device)
                        B[:prev_i] = prev_B
                        min_val = 1e-12 if self.nnls_lower_bound is None else self.nnls_lower_bound
                        B[prev_i:] = min_val

                prev_B = B
                current = M @ B

            iteration += 1

        return sorted(selected_indices)

    def _compute_beta_for_selected(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        selected_indices: List[int],
    ) -> torch.Tensor:
        """
        Compute beta values for specific selected indices using NNLS.

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

        scores_raw = queries @ K.T
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
        max_scores = scores32.max(dim=1, keepdim=True)[0]

        exp_scores = torch.exp(scores32 - max_scores)
        exp_scores_selected = exp_scores[:, selected_indices]
        target = exp_scores.sum(dim=1)

        B = self.algorithm._nnls_pg(
            exp_scores_selected, target,
            self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound
        )
        beta = torch.log(B)

        return beta

    def _aggregate_train_stats(self, all_stats: Dict, eval_queries_per_kv_head: int):
        """Aggregate train stats across all layers and heads into all_head_train_stats."""
        from evaluation.utils import compute_all_head_stats

        train_stats_per_head = {}
        for head_key, head_metrics in all_stats['per_layer_head_metrics'].items():
            if 'train_stats' in head_metrics:
                train_stats_per_head[head_key] = head_metrics['train_stats']

        all_stats['all_head_train_stats'] = compute_all_head_stats(
            train_stats_per_head,
            eval_queries_per_kv_head or 0
        )

    def _save_head_proportions_to_json(
        self,
        selected_by_layer_head: Dict[Tuple[int, int], List[int]],
        num_layers: int,
        num_heads: int
    ) -> None:
        """
        Save per-head key allocation proportions to JSON.

        Each head's proportion is the number of keys selected for that head
        divided by the total number of keys selected. All proportions sum to 1.

        Parameters
        ----------
        selected_by_layer_head : dict
            Dictionary mapping (layer_idx, head_idx) to list of selected indices
        num_layers : int
            Number of layers
        num_heads : int
            Number of heads per layer
        """
        if not selected_by_layer_head:
            print("Warning: No selections to save")
            return

        # Compute total selections across all heads
        total_selected = sum(len(indices) for indices in selected_by_layer_head.values())

        if total_selected == 0:
            print("Warning: Total selected is 0, cannot compute proportions")
            return

        # Compute proportion for each head
        proportions = {}
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                key = (layer_idx, head_idx)
                if key in selected_by_layer_head:
                    num_selected = len(selected_by_layer_head[key])
                    proportions[f"L{layer_idx}H{head_idx}"] = num_selected / total_selected
                else:
                    proportions[f"L{layer_idx}H{head_idx}"] = 0.0

        # Create output directory if it doesn't exist
        output_dir = Path("logs/head_proportions")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename based on config name and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_suffix = self.config_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_") if self.config_name else "global_omp"
        filename = f"head_proportions_{config_suffix}_{timestamp}.json"
        filepath = output_dir / filename

        # Write to JSON
        with open(filepath, 'w') as f:
            json.dump(proportions, f, indent=2)

        print(f"Head proportions saved to: {filepath}")

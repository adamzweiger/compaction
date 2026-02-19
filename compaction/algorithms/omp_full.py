# compaction/algorithms/omp_full.py
"""
OMP-based KV cache compaction algorithm with full attention output evaluation.

Uses orthogonal matching pursuit to greedily select keys, but evaluates candidates
based on the full attention output MSE rather than just partition function MSE.
"""
import torch
from typing import Tuple
from .base import CompactionAlgorithm


class OMPFullCompaction(CompactionAlgorithm):
    """Orthogonal Matching Pursuit based compaction with full attention evaluation."""

    def __init__(
        self,
        nnls_iters: int = 0,
        nnls_lower_bound: float = None,
        nnls_upper_bound: float = None,
        c2_method: str = 'lsq',
        num_candidates: int = 1,
        chunk_size: int = 1,
    ):
        """
        Parameters
        ----------
        nnls_iters : int
            Number of projected gradient descent iterations for NNLS.
            If 0, uses lstsq with clamping (default: 0).
        nnls_lower_bound : float, optional
            Lower bound for NNLS solution (default: None, uses 1e-12).
        nnls_upper_bound : float, optional
            Upper bound for NNLS solution (default: None, no upper bound).
        c2_method : str
            Method to compute C2: 'lsq' for least squares (default) or 'direct' for nearest neighbor selection.
        num_candidates : int
            Number of top candidates to evaluate at each greedy step (default: 1).
            If 1, uses simple correlation-based selection (fastest).
            If > 1, evaluates top num_candidates by solving NNLS for beta, then solving for C2,
            and picks best by MSE on the full attention output.
        chunk_size : int
            Number of keys to select per iteration (chunking). Matches OMP chunking behavior.
        """
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_method = c2_method
        self.num_candidates = num_candidates
        self.chunk_size = chunk_size

    def name(self) -> str:
        return "OMPFull"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Compute compacted cache using OMP-based key selection with full attention evaluation.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix
        V : Tensor, shape (T, d)
            Original value matrix
        queries : Tensor, shape (n, d)
            Query samples for training
        t : int
            Compacted size (number of keys to select)
        Returns
        -------
        C1 : Tensor, shape (t, d)
            Compacted keys
        beta : Tensor, shape (t,)
            Bias terms
        C2 : Tensor, shape (t, d)
            Compacted values
        indices : list of int
            Indices of selected keys
        """
        # Select keys using OMP with full attention evaluation
        C1, beta, indices = self._select_keys_omp_full(
            K, V, queries, t, self.num_candidates, self.chunk_size
        )

        # Compute compacted values
        C2 = self._compute_C2_with_method(C1, beta, K, V, queries, method=self.c2_method)

        return C1, beta, C2, indices

    def _select_keys_omp_full(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        num_candidates: int = 1,
        chunk_size: int = 1,
    ):
        """
        Greedy selection of t keys from K using OMP with full attention output evaluation.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix.
        V : Tensor, shape (T, d)
            Original value matrix.
        queries : Tensor, shape (n, d)
            Sampled query vectors.
        t : int
            Number of keys to select for the compacted cache.
        num_candidates : int
            Number of top candidates to evaluate at each greedy step.
            If 1, uses simple correlation-based selection (fastest).
            If > 1, evaluates top num_candidates by solving NNLS for beta, then C2,
            and picks best by MSE on full attention output on holdout set (~20%).
        chunk_size : int
            Number of keys to select per iteration (chunking). Matches OMP chunking behavior.

        Returns
        -------
        C1 : Tensor, shape (t, d)
            Selected keys (atoms) from K.
        beta : Tensor, shape (t,)
            Bias terms for each selected key.
        indices : list of int
            Indices of the selected keys in the original K.
        """
        # Shapes
        n, d = queries.shape
        T = K.shape[0]
        device = K.device
        dtype_param = K.dtype
        
        # Split queries into train/holdout sets (80/20 split)
        n_holdout = max(1, int(0.2 * n))  # At least 1 sample for holdout
        n_train = n - n_holdout
        
        # Shuffle indices for random split
        perm = torch.randperm(n, device=device)
        train_idx = perm[:n_train]
        holdout_idx = perm[n_train:]
        
        queries_train = queries[train_idx]
        queries_holdout = queries[holdout_idx]
        
        # QK matmul in original dtype; upcast only for scale + exp
        inv_sqrt_d = (1.0 / d) ** 0.5
        
        # Compute exp_scores and target for training set
        scores_raw_train = queries_train @ K.T                               # (n_train, T) original dtype
        scores32_train = scores_raw_train.to(torch.float32) * inv_sqrt_d     # (n_train, T) fp32
        max_scores_train = scores32_train.max(dim=1, keepdim=True)[0]        # (n_train, 1) fp32
        exp_scores_train = torch.exp(scores32_train - max_scores_train)      # (n_train, T) fp32
        target_train = exp_scores_train.sum(dim=1)                           # (n_train,)  fp32
        
        # Compute target attention output for holdout set (only if num_candidates > 1)
        if num_candidates > 1:
            scores_raw_holdout = queries_holdout @ K.T                           # (n_holdout, T)
            scores32_holdout = scores_raw_holdout.to(torch.float32) * inv_sqrt_d # (n_holdout, T) fp32
            max_scores_holdout = scores32_holdout.max(dim=1, keepdim=True)[0]    # (n_holdout, 1) fp32
            exp_scores_holdout = torch.exp(scores32_holdout - max_scores_holdout)# (n_holdout, T) fp32
            sum_exp_holdout = exp_scores_holdout.sum(dim=1, keepdim=True)        # (n_holdout, 1) fp32
            attn_weights_holdout = exp_scores_holdout / sum_exp_holdout          # (n_holdout, T) fp32
            target_output_holdout = attn_weights_holdout @ V.to(torch.float32)   # (n_holdout, d) fp32

        # Pre-allocate tensor for indices (avoid repeated list->tensor conversion)
        selected_indices_tensor = torch.zeros(t, dtype=torch.long, device=device)
        beta32 = torch.zeros(t, dtype=torch.float32, device=device)
        current_train = torch.zeros_like(target_train)                      # (n_train,) fp32
        mask_selected = torch.zeros(T, dtype=torch.bool, device=device)

        i = 0
        while i < t:
            residual_train = target_train - current_train                   # (n_train,)
            corr = (exp_scores_train * residual_train.unsqueeze(1)).sum(dim=0)  # (T,)
            corr[mask_selected] = -float('inf')

            num_remaining_total = T - mask_selected.sum().item()
            k_select = min(chunk_size, num_remaining_total, t - i)
            if k_select <= 0:
                break

            selections_made = 0
            while selections_made < k_select and i < t:
                if num_candidates == 1:
                    best_idx = torch.argmax(corr)
                else:
                    num_remaining = T - mask_selected.sum().item()
                    k = min(num_candidates, num_remaining)
                    if k == 0:
                        selections_made = k_select
                        break
                    top_k_values, top_k_indices = torch.topk(corr, k, largest=True)

                    best_mse = float('inf')
                    best_idx = top_k_indices[0]

                    for candidate_idx in top_k_indices:
                        temp_indices = selected_indices_tensor[:i].clone()
                        temp_indices = torch.cat([temp_indices, candidate_idx.unsqueeze(0)])

                        M_temp_train = exp_scores_train[:, temp_indices]
                        B_temp = self._nnls_pg(
                            M_temp_train,
                            target_train,
                            self.nnls_iters,
                            self.nnls_lower_bound,
                            self.nnls_upper_bound,
                        )
                        beta_temp = torch.log(B_temp)

                        C1_temp = K[temp_indices]
                        C2_temp = self._compute_C2_with_method(
                            C1_temp,
                            beta_temp,
                            K,
                            V,
                            queries_train,
                            method=self.c2_method,
                        )

                        scores_C1_raw_holdout = queries_holdout @ C1_temp.T
                        scores_C1_holdout = (
                            scores_C1_raw_holdout.to(torch.float32) * inv_sqrt_d + beta_temp.unsqueeze(0)
                        )
                        max_scores_C1_holdout = scores_C1_holdout.max(dim=1, keepdim=True)[0]
                        exp_scores_C1_holdout = torch.exp(scores_C1_holdout - max_scores_C1_holdout)
                        sum_exp_C1_holdout = exp_scores_C1_holdout.sum(dim=1, keepdim=True)
                        attn_weights_C1_holdout = exp_scores_C1_holdout / sum_exp_C1_holdout
                        pred_output_holdout = attn_weights_C1_holdout @ C2_temp.to(torch.float32)

                        mse = ((target_output_holdout - pred_output_holdout) ** 2).mean().item()

                        if mse < best_mse:
                            best_mse = mse
                            best_idx = candidate_idx

                selected_indices_tensor[i] = best_idx
                mask_selected[best_idx] = True
                corr[best_idx] = -float('inf')
                i += 1
                selections_made += 1

            if i == 0:
                break

            # Build design matrix M and solve NNLS on TRAIN set:  min ||M B - target||^2, B >= 0
            M_train = exp_scores_train[:, selected_indices_tensor[:i]]    # (n_train, i) fp32
            B = self._nnls_pg(
                M_train,
                target_train,
                self.nnls_iters,
                self.nnls_lower_bound,
                self.nnls_upper_bound,
            )       # (i,) fp32 (>=0)
            beta32[:i] = torch.log(B)
            current_train = M_train @ B                                     # (n_train,)

        # Convert to list only at the end (single GPU->CPU transfer)
        selected_indices = selected_indices_tensor.cpu().tolist()
        C1 = K[selected_indices_tensor]   # original dtype
        # Return beta in fp32; caller can downcast if desired
        return C1, beta32, selected_indices

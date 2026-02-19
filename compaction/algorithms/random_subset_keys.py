# compaction/algorithms/random_subset_keys.py
"""
Random subset key selection KV cache compaction algorithm.

Randomly selects keys.
"""
import torch
from typing import Tuple
from .base import CompactionAlgorithm
from .batched import BatchedCompactionAlgorithm


class RandomSubsetKeysCompaction(CompactionAlgorithm):
    """Random subset key selection"""
    def __init__(self, nnls_iters: int = 0, nnls_lower_bound: float = None, nnls_upper_bound: float = None,
                 c2_method: str = 'lsq', beta_method: str = 'nnls',
                 c2_ridge_lambda: float = 0, c2_solver: str = 'lstsq', c2_ridge_scale: str = 'spectral'):
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
        beta_method : str, optional
            Method to compute beta: 'nnls' to solve via NNLS (default) or 'zero' to set all beta=0.
        c2_ridge_lambda : float
            Regularization parameter for C2 ridge regression (default: 0).
        c2_solver : str
            Solver to use for C2: 'pinv', 'cholesky', or 'lstsq' (default: 'lstsq').
        c2_ridge_scale : str
            How to scale ridge_lambda: 'spectral', 'frobenius', or 'fixed' (default: 'spectral').
        """
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_method = c2_method
        if beta_method not in ['nnls', 'zero']:
            raise ValueError(f"beta_method must be 'nnls' or 'zero', got '{beta_method}'")
        self.beta_method = beta_method
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale

    def name(self) -> str:
        return "RandomCandidates"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Compute compacted cache using random candidate selection.

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
        # Select keys using random candidate selection
        C1, beta, indices = self._select_keys_random_candidate(K, queries, t)

        # Compute compacted values
        C2 = self._compute_C2_with_method(
            C1, beta, K, V, queries,
            method=self.c2_method,
            indices=indices,
            ridge_lambda=self.c2_ridge_lambda,
            solver=self.c2_solver,
            ridge_scale=self.c2_ridge_scale
        )

        return C1, beta, C2, indices

    def _select_keys_random_candidate(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int,
    ):
        """
        Random selection of t keys from K from the set of candidates (keys not yet selected).
        Randomly picks keys without replacement, then fits beta once at the end.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix.
        queries : Tensor, shape (n, d)
            Sampled query vectors.
        t : int
            Number of keys to select for the compacted cache.

        Returns
        -------
        C1 : Tensor, shape (t, d)
            Selected keys (atoms) from K.
        beta : Tensor, shape (t,)
            Bias terms for each selected key.
        indices : list of int
            Indices of the selected keys in the original K.
        """
        n, d = queries.shape
        T = K.shape[0]
        device = K.device

        # Randomly select t keys from K without replacement
        sel_idx = torch.randperm(T, device=device)[:t]
        selected_indices = sel_idx.tolist()
        C1 = K[sel_idx]  # keep original dtype

        # Compute beta based on beta_method
        if self.beta_method == 'zero':
            # Set all beta values to 0 (compute in fp32)
            beta = torch.zeros(t, dtype=torch.float32, device=device)
        else:  # 'nnls'
            # Precompute in policy style: QK in original dtype, softmax path in fp32
            inv_sqrt_d = (1.0 / d) ** 0.5
            scores_raw = queries @ K.T                                 # (n, T) original dtype
            scores32 = scores_raw.to(torch.float32) * inv_sqrt_d       # (n, T) fp32
            max_scores = scores32.max(dim=1, keepdim=True)[0]          # (n, 1) fp32
            exp_scores = torch.exp(scores32 - max_scores)              # (n, T) fp32
            target = exp_scores.sum(dim=1)                             # (n,) fp32

            # Design matrix for the selected subset and NNLS
            M = exp_scores[:, sel_idx]                                 # (n, t) fp32
            B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)              # (t,) fp32, >= 0
            beta = torch.log(B)                       # (t,) fp32

        return C1, beta, selected_indices


class BatchedRandomSubsetKeysCompaction(BatchedCompactionAlgorithm):
    """Batched random selection.

    Processes multiple (layer, head) combinations simultaneously for GPU efficiency.
    """

    def __init__(self, nnls_iters: int = 0, nnls_lower_bound: float = None, nnls_upper_bound: float = None, beta_method: str = 'nnls'):
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
        beta_method : str, optional
            Method to compute beta: 'nnls' to solve via NNLS (default) or 'zero' to set all beta=0.
        """
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        if beta_method not in ['nnls', 'zero']:
            raise ValueError(f"beta_method must be 'nnls' or 'zero', got '{beta_method}'")
        self.beta_method = beta_method

    def name(self) -> str:
        return "BatchedRandomCandidates"

    def compute_compacted_cache_batched(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute compacted cache using random candidate selection for multiple instances.

        Parameters
        ----------
        K : Tensor, shape (B, T, d)
            Original key matrices for B instances (e.g., layer×head combinations)
        V : Tensor, shape (B, T, d)
            Original value matrices
        queries : Tensor, shape (B, n, d)
            Query samples for training (B instances, n queries each)
        t : int
            Compacted size (number of keys to select)

        Returns
        -------
        C1 : Tensor, shape (B, t, d)
            Compacted keys
        beta : Tensor, shape (B, t)
            Bias terms
        C2 : Tensor, shape (B, t, d)
            Compacted values
        indices : Tensor, shape (B, t)
            Indices of selected keys for each instance
        """
        # Select keys using batched random candidate selection
        C1, beta, indices = self._select_keys_random_candidate_batched(K, queries, t)

        # Compute compacted values using shared primitive
        C2 = self.compute_C2_batched(C1, beta, K, V, queries)

        return C1, beta, C2, indices

    def _select_keys_random_candidate_batched(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched random selection of t keys from K without replacement.

        Parameters
        ----------
        K : Tensor, shape (B, T, d)
            Original key matrices.
        queries : Tensor, shape (B, n, d)
            Sampled query vectors.
        t : int
            Number of keys to select for the compacted cache.

        Returns
        -------
        C1 : Tensor, shape (B, t, d)
            Selected keys (atoms) from K.
        beta : Tensor, shape (B, t)
            Bias terms for each selected key.
        indices : Tensor, shape (B, t)
            Indices of the selected keys in the original K.
        """
        B, T, d = K.shape
        device = K.device

        # Randomly select t keys from each batch element without replacement
        # Create random permutations for each batch element
        selected_indices_tensor = torch.stack([
            torch.randperm(T, device=device)[:t]
            for _ in range(B)
        ])  # (B, t)

        # Gather selected keys: C1 = K[selected_indices]
        C1 = torch.stack([K[b, selected_indices_tensor[b]] for b in range(B)])  # (B, t, d)
        # print(f"C1.norm(dim=0, keepdim=True, p=1): {C1.norm(dim=1, keepdim=True, p=1)}")
        # Compute beta based on beta_method
        if self.beta_method == 'zero':
            # Set all beta values to 0 (compute in fp32)
            beta = torch.zeros(B, t, dtype=torch.float32, device=device)
        else:  # 'nnls'
            # Compute exp_scores and target using shared primitive
            exp_scores, target = self.compute_exp_scores_and_target_batched(K, queries)

            # Extract exp_scores for selected keys: Design matrix M
            # M: (B, n, t) - gather selected columns for each batch element
            n = queries.shape[1]
            batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, n, t)
            query_indices = torch.arange(n, device=device).view(1, n, 1).expand(B, n, t)
            selected_expanded = selected_indices_tensor.unsqueeze(1).expand(B, n, t)

            M = exp_scores[batch_indices, query_indices, selected_expanded]  # (B, n, t) fp32

            # Batched NNLS solve using shared primitive
            B_solution = self.box_ls_pg_batched(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)  # (B, t) fp32 (>=0)
            beta = torch.log(B_solution)  # (B, t) fp32

        return C1, beta, selected_indices_tensor

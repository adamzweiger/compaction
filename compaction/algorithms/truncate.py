# compaction/algorithms/truncate.py
"""
Truncation KV cache compaction algorithm.

Selects the first t keys and optionally learns beta/C2 similar to the random
subset implementation, but using deterministic truncation instead of sampling.
"""
import torch
from typing import Tuple
from .base import CompactionAlgorithm


class TruncationCompaction(CompactionAlgorithm):
    """Deterministic truncation-based key selection."""

    def __init__(
        self,
        nnls_iters: int = 0,
        nnls_lower_bound: float = None,
        nnls_upper_bound: float = None,
        c2_method: str = 'lsq',
        beta_method: str = 'nnls',
        c2_ridge_lambda: float = 0,
        c2_solver: str = 'lstsq',
        c2_ridge_scale: str = 'spectral',
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
            Method to compute C2: 'lsq' for least squares (default) or 'direct'.
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
        return "Truncation"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Compute compacted cache using the first t keys.

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
        C1, beta, indices = self._select_keys_truncated(K, queries, t)

        C2 = self._compute_C2_with_method(
            C1,
            beta,
            K,
            V,
            queries,
            method=self.c2_method,
            indices=indices,
            ridge_lambda=self.c2_ridge_lambda,
            solver=self.c2_solver,
            ridge_scale=self.c2_ridge_scale,
        )

        return C1, beta, C2, indices

    def _select_keys_truncated(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int,
    ):
        """
        Select the first t keys/values and fit beta if requested.
        """
        T = K.shape[0]
        if t > T:
            raise ValueError(f"Cannot truncate to t={t} when only {T} keys are available.")

        device = K.device
        d = K.shape[1]
        sel_idx = torch.arange(t, device=device)
        C1 = K[sel_idx]

        if self.beta_method == 'zero':
            beta = torch.zeros(t, dtype=torch.float32, device=device)
        else:
            inv_sqrt_d = (1.0 / d) ** 0.5
            scores_raw = queries @ K.T
            scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
            max_scores = scores32.max(dim=1, keepdim=True)[0]
            exp_scores = torch.exp(scores32 - max_scores)
            target = exp_scores.sum(dim=1)

            M = exp_scores[:, sel_idx]
            B = self._nnls_pg(
                M,
                target,
                self.nnls_iters,
                self.nnls_lower_bound,
                self.nnls_upper_bound,
            )
            beta = torch.log(B)

        return C1, beta, sel_idx.tolist()

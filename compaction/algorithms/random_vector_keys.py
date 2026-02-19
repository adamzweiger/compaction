# compaction/algorithms/random_vector_keys.py
"""
Fully random KV cache compaction algorithm.

Randomly selects keys from the entire set without any optimization.
"""
import torch
from typing import Tuple
from .base import CompactionAlgorithm


class RandomVectorKeysCompaction(CompactionAlgorithm):
    """Fully random key selection."""

    def __init__(self, nnls_iters: int = 0, nnls_lower_bound: float = None, nnls_upper_bound: float = None):
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
        """
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound

    def name(self) -> str:
        return "RandomVectorKeys"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Compute compacted cache using fully random key selection.

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
        # Select keys using fully random selection
        C1, beta, indices = self._select_keys_random(K, queries, t)

        # Compute compacted values
        C2 = self._compute_C2(C1, beta, K, V, queries)

        return C1, beta, C2, indices

    def _select_keys_random(self, K: torch.Tensor, queries: torch.Tensor, t: int):
        """
        Generate t random vectors from normal distribution.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix (used for computing target and device/dtype info).
        queries : Tensor, shape (n, d)
            Sampled query vectors.
        t : int
            Number of random keys to generate for the compacted cache.

        Returns
        -------
        C1 : Tensor, shape (t, d)
            Randomly generated keys from normal distribution.
        beta : Tensor, shape (t,)
            Bias terms for each random key.
        indices : list of int
            Empty list (no indices since keys are not selected from K).
        """
        n, d = queries.shape
        device = K.device
        dtype = K.dtype

        # Generate random keys from normal distribution
        C1 = torch.randn(t, d, device=device, dtype=dtype)
        selected_indices = []  # No indices since we're not selecting from K

        # Precompute in policy style: QK in original dtype, softmax path in fp32
        inv_sqrt_d = (1.0 / d) ** 0.5
        scores_raw = queries @ K.T                                 # (n, T) original dtype
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d       # (n, T) fp32
        max_scores = scores32.max(dim=1, keepdim=True)[0]          # (n, 1)
        exp_scores = torch.exp(scores32 - max_scores)              # (n, T) fp32
        target = exp_scores.sum(dim=1)                             # (n,) fp32

        # Design matrix for the random keys and NNLS
        scores_random_raw = queries @ C1.T                         # (n, t) original dtype
        scores_random32 = scores_random_raw.to(torch.float32) * inv_sqrt_d  # (n, t) fp32
        M = torch.exp(scores_random32 - max_scores)                # (n, t) fp32
        B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)                    # (t,) fp32, >= 0
        beta = torch.log(B)                       # (t,) fp32

        return C1, beta, selected_indices

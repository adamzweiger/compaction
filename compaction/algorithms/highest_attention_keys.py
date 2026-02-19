# compaction/algorithms/highest_attention_keys.py
"""
Highest Attention Keys KV cache compaction algorithm.

Selects keys with the highest average attention scores across all training queries,
then solves for beta using NNLS and C2 using ridge regression.
"""
import torch
import torch.nn.functional as F
from typing import Tuple
from .base import CompactionAlgorithm


class HighestAttentionKeysCompaction(CompactionAlgorithm):
    """Select keys with highest average attention scores (or RMS if configured)."""

    def __init__(self, nnls_iters: int = 0, nnls_lower_bound: float = None, nnls_upper_bound: float = None,
                 score_method: str = 'max', c2_method: str = 'lsq', beta_method: str = 'nnls',
                 c2_ridge_lambda: float = 0, c2_solver: str = 'lstsq', c2_ridge_scale: str = 'spectral',
                 pooling: str = None, kernel_size: int = 7):
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
        score_method : str, optional
            Method to score keys: 'max' for maximum attention (default),
            'mean' for mean attention, or 'rms' for rms attention.
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
        pooling : str, optional
            Pooling method to apply to attention scores: 'avgpool', 'maxpool', or None (default: None).
        kernel_size : int
            Kernel size for pooling operation (default: 7).
        """
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        if score_method not in ['mean', 'rms', 'max']:
            raise ValueError(f"score_method must be 'mean', 'rms', or 'max', got '{score_method}'")
        self.score_method = score_method
        self.c2_method = c2_method
        if beta_method not in ['nnls', 'zero']:
            raise ValueError(f"beta_method must be 'nnls' or 'zero', got '{beta_method}'")
        self.beta_method = beta_method
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale
        if pooling is not None and pooling not in ['avgpool', 'maxpool']:
            raise ValueError(f"pooling must be 'avgpool', 'maxpool', or None, got '{pooling}'")
        self.pooling = pooling
        self.kernel_size = kernel_size

    def name(self) -> str:
        return "HighestAttentionKeys"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Compute compacted cache using highest attention key selection.

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
        attention_bias : Tensor, optional
            Additive attention bias for the original cache (broadcastable to (n, T)).
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
        # Select keys based on highest average attention
        C1, beta, indices = self._select_keys_highest_attention(K, queries, t, attention_bias)

        # Compute compacted values
        C2 = self._compute_C2_with_method(
            C1, beta, K, V, queries,
            method=self.c2_method,
            indices=indices,
            attention_bias=attention_bias,
            ridge_lambda=self.c2_ridge_lambda,
            solver=self.c2_solver,
            ridge_scale=self.c2_ridge_scale
        )

        return C1, beta, C2, indices

    def _select_keys_highest_attention(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ):
        """
        Select t keys from K with highest attention scores across training queries.
        Uses score_method to determine how to score keys: 'mean', 'rms', or 'max'.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix.
        queries : Tensor, shape (n, d)
            Sampled query vectors.
        t : int
            Number of keys to select for the compacted cache.
        attention_bias : Tensor, optional
            Additive attention bias for the original cache (broadcastable to (n, T)).

        Returns
        -------
        C1 : Tensor, shape (t, d)
            Selected keys from K.
        beta : Tensor, shape (t,)
            Bias terms for each selected key.
        indices : list of int
            Indices of the selected keys in the original K.
        """
        n, d = queries.shape
        T = K.shape[0]
        device = K.device
        dtype_param = K.dtype

        # Compute attention scores in fp32 for numerical stability
        inv_sqrt_d = (1.0 / d) ** 0.5

        # QK matmul in original dtype; upcast for softmax
        scores_raw = queries @ K.T                                 # (n, T) original dtype
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d       # (n, T) fp32
        if attention_bias is not None:
            try:
                bias32 = torch.broadcast_to(
                    attention_bias.to(torch.float32),
                    scores32.shape
                )
                scores32 = scores32 + bias32                       # (n, T)
            except Exception as e:
                raise ValueError(
                    f"attention_bias must be broadcastable to {scores32.shape}, "
                    f"got {tuple(attention_bias.shape)}"
                ) from e
        max_scores = scores32.max(dim=1, keepdim=True)[0]          # (n, 1) fp32
        exp_scores = torch.exp(scores32 - max_scores)              # (n, T) fp32

        # Compute softmax attention weights
        sum_exp = exp_scores.sum(dim=1, keepdim=True)              # (n, 1)
        attention_weights = exp_scores / sum_exp                   # (n, T) normalized attention

        # Compute score for each key across all queries
        if self.score_method == 'rms':
            # RMS attention score: sqrt(mean(attention^2))
            key_scores = torch.sqrt((attention_weights ** 2).mean(dim=0))  # (T,) fp32
        elif self.score_method == 'max':
            # Maximum attention score across all queries
            key_scores = attention_weights.max(dim=0)[0]  # (T,) fp32
        else:  # 'mean'
            # Average attention score
            key_scores = attention_weights.mean(dim=0)              # (T,) fp32

        # Apply pooling if specified
        if self.pooling is not None:
            # key_scores is (T,), need to add batch and channel dims for pooling: (1, 1, T)
            key_scores_pooled = key_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
            if self.pooling == 'avgpool':
                key_scores_pooled = F.avg_pool1d(key_scores_pooled, kernel_size=self.kernel_size,
                                                  padding=self.kernel_size // 2, stride=1)
            elif self.pooling == 'maxpool':
                key_scores_pooled = F.max_pool1d(key_scores_pooled, kernel_size=self.kernel_size,
                                                  padding=self.kernel_size // 2, stride=1)
            key_scores = key_scores_pooled.squeeze(0).squeeze(0)  # (T,) fp32

        # Select top-t keys by score
        _, top_indices = torch.topk(key_scores, t, largest=True)
        selected_indices_tensor = top_indices

        # Extract selected keys
        C1 = K[selected_indices_tensor]  # (t, d) original dtype

        # Compute beta based on beta_method
        if self.beta_method == 'zero':
            # Set all beta values to 0 (compute in fp32, then convert to model dtype)
            beta32 = torch.zeros(t, dtype=torch.float32, device=device)
        else:  # 'nnls'
            # Compute target for NNLS
            target = exp_scores.sum(dim=1)  # (n,) fp32

            # Build design matrix M for NNLS
            M = exp_scores[:, selected_indices_tensor]  # (n, t) fp32

            # Solve NNLS: min ||M B - target||^2, B >= 0
            B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)  # (t,) fp32 (>=0)
            beta32 = torch.log(B)  # (t,) fp32

        # Convert beta from fp32 to model dtype (e.g., bf16) for storage
        beta = beta32.to(dtype_param)

        # Convert indices to list
        selected_indices = selected_indices_tensor.cpu().tolist()

        return C1, beta, selected_indices

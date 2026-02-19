# compaction/algorithms/batched.py
"""
Base class and shared utilities for batched KV cache compaction algorithms.

Provides batched versions of common operations used across multiple algorithms.
All operations expect a batch dimension as the first dimension.
"""
import torch
from typing import Tuple
from abc import ABC, abstractmethod


class BatchedCompactionAlgorithm(ABC):
    """Base class for batched KV cache compaction algorithms."""

    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name for logging."""
        pass

    @abstractmethod
    def compute_compacted_cache_batched(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute compacted cache for multiple instances simultaneously.

        Parameters
        ----------
        K : Tensor, shape (B, T, d)
            Original key matrices for B instances
        V : Tensor, shape (B, T, d)
            Original value matrices
        queries : Tensor, shape (B, n, d)
            Query samples for training
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
        pass

    def _compute_C2_batched(
        self,
        C1: torch.Tensor,
        beta: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        ridge_lambda: float = 0,
        solver: str = 'lstsq',
        ridge_scale: str = 'spectral'
    ) -> torch.Tensor:
        """
        Batched solve for C2 across multiple instances.

        Solve for C2 in softmax(q·Kᵀ)V ≈ softmax(q·C1ᵀ + beta)·C2 for all queries.

        Matches: exp(qK^T)V / sum_j exp(qK_j^T) = [exp(qC1^T + beta) / sum_j exp(q(C1)_j^T + beta_j)] C2

        Follows precision policy:
        - QK and (softmax·V) matmuls in original dtype (kernel does fp32 accumulation)
        - Upcast to fp32 only for scaling + softmax/LSE/exp
        - Ridge regression for C2

        Parameters
        ----------
        C1 : Tensor, shape (B, t, d)
            Reduced set of keys.
        beta : Tensor, shape (B, t)
            Bias terms for the reduced keys.
        K, V : Tensors, shapes (B, T, d)
            Original keys and values.
        queries : Tensor, shape (B, n, d)
            Sampled query vectors.
        ridge_lambda : float
            Regularization parameter for ridge regression (default: 0).
        solver : str
            Solver to use: 'pinv', 'cholesky', or 'lstsq' (default: 'lstsq').
        ridge_scale : str
            How to scale ridge_lambda: 'spectral' (||X||_2^2), 'frobenius' (||X||_F^2),
            or 'fixed' (no scaling, just ridge_lambda) (default: 'spectral').

        Returns
        -------
        C2 : Tensor, shape (B, t, d)
            Compacted value matrix.
        """
        # matmuls in original dtype; softmax bits in fp32
        B, T, d = K.shape
        n = queries.shape[1]
        t = C1.shape[1]
        dtype_param = K.dtype
        inv_sqrt_d = (1.0 / d) ** 0.5

        # Y = softmax((QK)/sqrt(d)) @ V
        sK_raw = torch.bmm(queries, K.transpose(1, 2))  # (B, n, T) original dtype
        sK32 = sK_raw.to(torch.float32) * inv_sqrt_d  # (B, n, T)
        m_K = sK32.max(dim=2, keepdim=True)[0]  # (B, n, 1) for stability
        exp_sK = torch.exp(sK32 - m_K)  # (B, n, T)
        sum_exp_K = exp_sK.sum(dim=2, keepdim=True)  # (B, n, 1)
        attn_K = exp_sK / sum_exp_K  # (B, n, T) normalized
        Y = torch.bmm(attn_K, V.to(torch.float32))  # (B, n, d) fp32

        # X = softmax((Q C1^T)/sqrt(d) + beta)
        sC_raw = torch.bmm(queries, C1.transpose(1, 2))  # (B, n, t) original dtype
        sC32 = sC_raw.to(torch.float32) * inv_sqrt_d + beta.to(torch.float32).unsqueeze(1)  # (B, n, t)
        m_C = sC32.max(dim=2, keepdim=True)[0]  # (B, n, 1) for stability
        exp_sC = torch.exp(sC32 - m_C)  # (B, n, t)
        sum_exp_C = exp_sC.sum(dim=2, keepdim=True)  # (B, n, 1)
        X = exp_sC / sum_exp_C  # (B, n, t) normalized

        # Note: See https://blogs.rstudio.com/ai/posts/2022-10-13-torch-linalg/
        # Scale ridge lambda by spectral norm or frobenius norm / t or use fixed value
        # Frobenius^2 = tr(XtX) = sum (singular value)^2 of X = sum eigenvalue of XtX. Frobenius^2/t = average eigenvalue of XtX.
        if ridge_lambda == 0:
            lam = torch.zeros(B, device=X.device, dtype=torch.float32)
        else:
            if ridge_scale == 'spectral':
                try:
                    lam = ridge_lambda * (torch.linalg.matrix_norm(X, ord=2, dim=(1, 2))**2)  # (B,) ridge to PD (largest eigenvalue)
                except Exception as e:
                    # fallback to frobenius norm
                    print(f"Warning: Spectral norm computation failed ({e}), falling back to Frobenius norm")
                    lam = ridge_lambda * ((torch.linalg.matrix_norm(X, ord='fro', dim=(1, 2))**2) / t)  # average eigenvalue
            elif ridge_scale == 'frobenius':
                lam = ridge_lambda * ((torch.linalg.matrix_norm(X, ord='fro', dim=(1, 2))**2) / t)  # average eigenvalue of XtX
            elif ridge_scale == 'fixed':
                lam = torch.full((B,), ridge_lambda, device=X.device, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown ridge_scale: {ridge_scale}. Must be 'spectral', 'frobenius', or 'fixed'.")

        # Solve based on selected method
        if solver == 'lstsq':
            # Use lstsq solver
            # Solves: X @ C2 = Y
            try:
                C2_32 = torch.linalg.lstsq(X, Y, driver='gels').solution    # (B, t, d)
            except Exception as e:
                print(f"lstsq failed ({e}), increasing lambda to 1e-5*(||X||_2^2) and retrying")
                lam = 1e-5 * (torch.linalg.matrix_norm(X, ord=2, dim=(1, 2))**2)
                # Fall back to cholesky method with increased lambda
                if n < t:
                    XXt = torch.bmm(X, X.transpose(1, 2))
                    XXt = 0.5 * (XXt + XXt.transpose(1, 2))
                    XXt.diagonal(dim1=1, dim2=2).add_(lam.unsqueeze(1))
                    L = torch.linalg.cholesky(XXt)
                    Z = torch.cholesky_solve(Y, L)
                    C2_32 = torch.bmm(X.transpose(1, 2), Z)
                else:
                    XtX = torch.bmm(X.transpose(1, 2), X)
                    XtX = 0.5 * (XtX + XtX.transpose(1, 2))
                    XtX.diagonal(dim1=1, dim2=2).add_(lam.unsqueeze(1))
                    L = torch.linalg.cholesky(XtX)
                    XtY = torch.bmm(X.transpose(1, 2), Y)
                    C2_32 = torch.cholesky_solve(XtY, L)
        elif solver == 'pinv':
            # Use pseudoinverse
            if (lam == 0).all():
                X_pinv = torch.linalg.pinv(X)
                C2_32 = torch.bmm(X_pinv, Y)
            elif n >= t:
                # C2 = (XᵀX + λI)^{-1} Xᵀ Y
                XtX = torch.bmm(X.transpose(1, 2), X)
                XtX = 0.5 * (XtX + XtX.transpose(1, 2))
                XtX.diagonal(dim1=1, dim2=2).add_(lam.unsqueeze(1))
                XtY = torch.bmm(X.transpose(1, 2), Y)
                C2_32 = torch.bmm(torch.linalg.pinv(XtX), XtY)         # (B, t, d)
            else:
                # n < t: use underdetermined formulation
                XXt = torch.bmm(X, X.transpose(1, 2))
                XXt = 0.5 * (XXt + XXt.transpose(1, 2))
                XXt.diagonal(dim1=1, dim2=2).add_(lam.unsqueeze(1))
                XXt_pinv = torch.linalg.pinv(XXt)
                Z = torch.bmm(XXt_pinv, Y)
                C2_32 = torch.bmm(X.transpose(1, 2), Z)
        elif solver == 'cholesky':
            # Use Cholesky decomposition
            if n < t:
                # C2 = Xᵀ (XXᵀ + λI)^{-1} Y
                XXt = torch.bmm(X, X.transpose(1, 2))
                XXt = 0.5 * (XXt + XXt.transpose(1, 2))
                XXt.diagonal(dim1=1, dim2=2).add_(lam.unsqueeze(1))
                L = torch.linalg.cholesky(XXt)               # (B, n, n)
                Z = torch.cholesky_solve(Y, L)               # solves (XXᵀ+λI)Z = Y
                C2_32 = torch.bmm(X.transpose(1, 2), Z)      # (B, t, d)
            else:
                # C2 = (XᵀX + λI)^{-1} Xᵀ Y
                XtX = torch.bmm(X.transpose(1, 2), X)
                XtX = 0.5 * (XtX + XtX.transpose(1, 2))
                XtX.diagonal(dim1=1, dim2=2).add_(lam.unsqueeze(1))
                L = torch.linalg.cholesky(XtX)               # (B, t, t)
                XtY = torch.bmm(X.transpose(1, 2), Y)
                C2_32 = torch.cholesky_solve(XtY, L)         # (B, t, d)
        else:
            raise ValueError(f"Unknown solver: {solver}. Must be 'pinv', 'cholesky', or 'lstsq'.")

        return C2_32.to(dtype_param)

    def _direct_C2_batched(
        self,
        C1: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        indices: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Batched version: Select C2 directly from V by finding nearest neighbors in K for each key in C1.

        For each key in C1, finds the index in K that is the nearest neighbor (by L2 distance),
        and selects the corresponding value from V.

        If indices are provided (i.e., C1 is a subset of K), uses those indices directly
        since the nearest neighbor of K[i] in K is K[i] itself.

        Parameters
        ----------
        C1 : Tensor, shape (B, t, d)
            Compacted keys.
        K : Tensor, shape (B, T, d)
            Original keys.
        V : Tensor, shape (B, T, d)
            Original values.
        indices : Tensor, shape (B, t), optional
            Indices of selected keys from K. If provided, C2 is directly selected
            from V using these indices (default: None).

        Returns
        -------
        C2 : Tensor, shape (B, t, d)
            Compacted values, selected from V based on nearest neighbor matching.
        """
        B, t, d = C1.shape
        T = K.shape[1]

        # If indices are provided, C1 is a subset of K, so just use those indices
        if indices is not None:
            # Use advanced indexing to gather values
            # indices: (B, t), V: (B, T, d)
            batch_indices = torch.arange(B, device=V.device).view(B, 1).expand(B, t)  # (B, t)
            C2 = V[batch_indices, indices]  # (B, t, d)
            return C2

        # Otherwise, find nearest neighbors for each batch element
        # Compute pairwise squared distances: ||C1[b,i] - K[b,j]||^2
        # = ||C1[b,i]||^2 + ||K[b,j]||^2 - 2*C1[b,i]·K[b,j]
        C1_norms_sq = (C1 ** 2).sum(dim=2)  # (B, t)
        K_norms_sq = (K ** 2).sum(dim=2)  # (B, T)
        pairwise_dots = torch.bmm(C1, K.transpose(1, 2))  # (B, t, T)

        # Squared distances: (B, t, T)
        squared_distances = C1_norms_sq.unsqueeze(2) + K_norms_sq.unsqueeze(1) - 2 * pairwise_dots

        # Find nearest neighbor index for each key in C1
        nearest_indices = squared_distances.argmin(dim=2)  # (B, t)

        # Select corresponding values from V using advanced indexing
        batch_indices = torch.arange(B, device=V.device).view(B, 1).expand(B, t)  # (B, t)
        C2 = V[batch_indices, nearest_indices]  # (B, t, d)

        return C2

    def _compute_C2_with_method_batched(
        self,
        C1: torch.Tensor,
        beta: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        method: str = 'lsq',
        indices: torch.Tensor = None,
        ridge_lambda: float = 0,
        solver: str = 'lstsq',
        ridge_scale: str = 'spectral',
        **kwargs
    ) -> torch.Tensor:
        """
        Batched version: Compute C2 using the specified method.

        Parameters
        ----------
        C1 : Tensor, shape (B, t, d)
            Compacted keys.
        beta : Tensor, shape (B, t)
            Bias terms (not used for 'direct' method).
        K : Tensor, shape (B, T, d)
            Original keys.
        V : Tensor, shape (B, T, d)
            Original values.
        queries : Tensor, shape (B, n, d)
            Query samples (not used for 'direct' method).
        method : str
            Method to use: 'lsq' for least squares (default) or 'direct' for nearest
            neighbor selection.
        indices : Tensor, shape (B, t), optional
            Indices of selected keys (used for 'direct' method when C1 is a subset of K).
        ridge_lambda : float
            Regularization parameter for ridge regression (default: 0).
        solver : str
            Solver to use: 'auto', 'pinv', 'cholesky', or 'lstsq' (default: 'auto').
        ridge_scale : str
            How to scale ridge_lambda: 'spectral', 'frobenius', or 'fixed' (default: 'spectral').

        Returns
        -------
        C2 : Tensor, shape (B, t, d)
            Compacted values.
        """
        if method == 'direct':
            return self._direct_C2_batched(C1, K, V, indices)
        elif method == 'lsq':
            return self._compute_C2_batched(C1, beta, K, V, queries, ridge_lambda=ridge_lambda, solver=solver, ridge_scale=ridge_scale)
        else:
            raise ValueError(f"Unknown C2 computation method: {method}. Must be 'lsq' or 'direct'.")

    @staticmethod
    def _nnls_pg_batched(M: torch.Tensor, y: torch.Tensor, iters: int = 0,
                         lower_bound: float = 1e-12, upper_bound: float = None, debug: bool = False) -> torch.Tensor:
        """
        Batched box-constrained non-negative least squares solver with projected gradient.

        If iters == 0: Use ridge normal-equations solve + clamp with clamping to bounds.
        If iters > 0: Use projected-gradient descent with specified iterations.

        Solves: min_B ||M B - y||_2^2  s.t. lower_bound <= B <= upper_bound
        Step size 1/L with L ≈ ||M||_2^2 via power iteration.
        Expects fp32 inputs; returns fp32 B with box constraints.

        Parameters
        ----------
        M : Tensor, shape (B, n, t)
            Design matrices
        y : Tensor, shape (B, n)
            Target vectors
        iters : int
            Number of projected gradient iterations (0 = use clamped least squares)
        lower_bound : float
            Lower bound for B values (default: 1e-12)
        upper_bound : float, optional
            Upper bound for B values (default: None, no upper bound)
        debug : bool
            Whether to print debug information (default: False)

        Returns
        -------
        B : Tensor, shape (B, t)
            Solutions with box constraints
        """
        B_batch, n, t = M.shape
        device = M.device
        min_val = 1e-12 if lower_bound is None else lower_bound

        # lam = 1e-4 * (M.norm()**2 / t)
        # lam = 1e-4 * (torch.linalg.matrix_norm(M, ord=2)**2)
        #  We also estimate spectral norm below and could put that here and use that instead. lam=0 seems to be best though.
        lam = 0

        if lam == 0:
            # Use lstsq when no regularization
            # Solves: M @ B = y
            B = torch.linalg.lstsq(M, y.unsqueeze(2), driver='gels').solution.squeeze(2)  # (B, t)
        else:
            # Ridge regression path
            # Check if underdetermined (n < t) or overdetermined (n >= t)
            # For batched case, assume same condition for all batches
            if n < t:  # underdetermined
                MMt = torch.bmm(M, M.transpose(1, 2))  # (B, n, n)
                MMt.diagonal(dim1=1, dim2=2).add_(lam)
                R = torch.linalg.cholesky(MMt)  # SPD factor
                alpha = torch.cholesky_solve(y.unsqueeze(2), R).squeeze(2)  # (B, n)
                B = torch.bmm(M.transpose(1, 2), alpha.unsqueeze(2)).squeeze(2)  # (B, t)
            else:  # overdetermined
                MtM = torch.bmm(M.transpose(1, 2), M)  # (B, t, t)
                MtM.diagonal(dim1=1, dim2=2).add_(lam)
                R = torch.linalg.cholesky(MtM)
                Mty = torch.bmm(M.transpose(1, 2), y.unsqueeze(2)).squeeze(2)  # (B, t)
                B = torch.cholesky_solve(Mty.unsqueeze(2), R).squeeze(2)  # (B, t)

        # Apply bounds
        B = B.clamp_min_(min_val)
        if upper_bound is not None:
            B = B.clamp_max_(upper_bound)

        if iters == 0:
            return B

        # Power iteration for spectral norm (batched)
        u = torch.randn(B_batch, t, device=device, dtype=M.dtype)
        u = u / (u.norm(dim=1, keepdim=True) + 1e-12)

        for _ in range(3):  # converges very fast usually
            v = torch.bmm(M, u.unsqueeze(2)).squeeze(2)  # (B, n)
            v_norm = v.norm(dim=1, keepdim=True)
            if (v_norm == 0).any():
                break
            v = v / v_norm

            u = torch.bmm(M.transpose(1, 2), v.unsqueeze(2)).squeeze(2)  # (B, t)
            u_norm = u.norm(dim=1, keepdim=True)
            if (u_norm == 0).any():
                break
            u = u / u_norm

        # Compute spectral norms: ~||M||_2
        Mu = torch.bmm(M, u.unsqueeze(2)).squeeze(2)  # (B, n)
        MtMu = torch.bmm(M.transpose(1, 2), Mu.unsqueeze(2)).squeeze(2)  # (B, t)
        sigma = (u * MtMu).sum(dim=1).sqrt().clamp_min(1e-6)  # (B,)
        L = (sigma ** 2).clamp_min(1e-6)
        eta = 1.0 / L  # (B,)

        # Projected gradient descent for B
        for _ in range(iters):
            MB = torch.bmm(M, B.unsqueeze(2)).squeeze(2)  # (B, n)
            grad = torch.bmm(M.transpose(1, 2), (MB - y).unsqueeze(2)).squeeze(2)  # (B, t)
            B = B - eta.unsqueeze(1) * grad
            # Project to feasible set
            B = B.clamp_min_(min_val)
            if upper_bound is not None:
                B = B.clamp_max_(upper_bound)

        return B

    @staticmethod
    def _compute_exp_scores_and_target_batched(
        K: torch.Tensor,
        queries: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched computation of exp scores and target for key selection.

        This is the common preprocessing step used by both OMP and RandomCandidates.

        Parameters
        ----------
        K : Tensor, shape (B, T, d)
            Key matrices
        queries : Tensor, shape (B, n, d)
            Query vectors

        Returns
        -------
        exp_scores : Tensor, shape (B, n, T)
            Exponentiated scores (fp32)
        target : Tensor, shape (B, n)
            Target values (sum of exp_scores) (fp32)
        """
        B, T, d = K.shape
        inv_sqrt_d = (1.0 / d) ** 0.5

        # Compute scores: (B, n, T)
        scores_raw = torch.bmm(queries, K.transpose(1, 2))  # (B, n, T) in original dtype
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d  # (B, n, T) fp32
        max_scores = scores32.max(dim=2, keepdim=True)[0]  # (B, n, 1) fp32
        exp_scores = torch.exp(scores32 - max_scores)  # (B, n, T) fp32
        target = exp_scores.sum(dim=2)  # (B, n) fp32

        return exp_scores, target

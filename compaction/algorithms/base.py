# compaction/algorithms/base.py
"""
Base class and shared utilities for KV cache compaction algorithms.

All algorithms must implement the compute_compacted_cache function
with the standardized signature.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Dict
from abc import ABC, abstractmethod


class CompactionAlgorithm(ABC):
    """Base class for KV cache compaction algorithms."""

    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name for logging."""
        pass

    @abstractmethod
    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Compute compacted cache representation.

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
            Additive attention bias for the *original* cache scores.
            Shape (T,) or (n, T); broadcastable to (n, T). Defaults to None.

        Returns
        -------
        C1 : Tensor, shape (t, d)
            Compacted keys
        beta : Tensor, shape (t,)
            Bias terms for each compacted key
        C2 : Tensor, shape (t, d)
            Compacted values
        indices : list of int
            Indices of selected keys (if applicable)
        """
        pass

    def _compute_C2(
        self,
        C1: torch.Tensor,
        beta: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        attention_bias: torch.Tensor = None,
        ridge_lambda: float = 0,
        solver: str = 'lstsq',
        ridge_scale: str = 'spectral'
    ) -> torch.Tensor:
        """
        Solve for C2 in softmax(q·Kᵀ)V ≈ softmax(q·C1ᵀ + beta)·C2 for all queries.

        Matches: exp(qK^T)V / sum_j exp(qK_j^T) = [exp(qC1^T + beta) / sum_j exp(q(C1)_j^T + beta_j)] C2

        Follows precision policy:
        - QK and (softmax·V) matmuls in original dtype (kernel does fp32 accumulation)
        - Upcast to fp32 only for scaling + softmax/LSE/exp
        - Ridge regression for C2

        Parameters
        ----------
        C1 : Tensor, shape (t, d)
            Reduced set of keys.
        beta : Tensor, shape (t,)
            Bias terms for the reduced keys.
        K, V : Tensors, shapes (T, d)
            Original keys and values.
        queries : Tensor, shape (n, d)
            Sampled query vectors.
        attention_bias : Tensor, optional
            Additive attention bias for the original cache (broadcastable to (n, T)).
        ridge_lambda : float
            Regularization parameter for ridge regression (default: 0).
        solver : str
            Solver to use: 'pinv', 'cholesky', or 'lstsq' (default: 'lstsq').
        ridge_scale : str
            How to scale ridge_lambda: 'spectral' (||X||_2^2), 'frobenius' (||X||_F^2),
            or 'fixed' (no scaling, just ridge_lambda) (default: 'spectral').

        Returns
        -------
        C2 : Tensor, shape (t, d)
            Compacted value matrix.
        """
        # matmuls in original dtype; softmax bits in fp32
        dtype_param = K.dtype
        d = K.shape[1]
        inv_sqrt_d = (1.0 / d) ** 0.5

        # Y = softmax((QK)/sqrt(d)) @ V
        sK_raw = queries @ K.T                                   # (n, T) original dtype
        sK32 = sK_raw.to(torch.float32) * inv_sqrt_d             # (n, T)
        if attention_bias is not None:
            try:
                bias32 = torch.broadcast_to(
                    attention_bias.to(torch.float32),
                    sK32.shape
                )
                sK32 = sK32 + bias32                             # (n, T)
            except Exception as e:
                raise ValueError(
                    f"attention_bias must be broadcastable to {sK32.shape}, "
                    f"got {tuple(attention_bias.shape)}"
                ) from e
        m_K = sK32.max(dim=1, keepdim=True)[0]                   # (n, 1) for stability
        exp_sK = torch.exp(sK32 - m_K)                           # (n, T)
        sum_exp_K = exp_sK.sum(dim=1, keepdim=True)              # (n, 1)
        attn_K = exp_sK / sum_exp_K                              # (n, T) normalized
        Y = attn_K @ V.to(torch.float32)                         # (n, d) fp32

        # X = softmax((Q C1^T)/sqrt(d) + beta)
        sC_raw = queries @ C1.T                                  # (n, t) original dtype
        sC32 = sC_raw.to(torch.float32) * inv_sqrt_d + beta.to(torch.float32)  # (n, t)
        m_C = sC32.max(dim=1, keepdim=True)[0]                   # (n, 1) for stability
        exp_sC = torch.exp(sC32 - m_C)                           # (n, t)
        sum_exp_C = exp_sC.sum(dim=1, keepdim=True)              # (n, 1)
        X = exp_sC / sum_exp_C                                   # (n, t) normalized

        # Note: See https://blogs.rstudio.com/ai/posts/2022-10-13-torch-linalg/
        n, t = X.shape
        # Scale ridge lambda by spectral norm or frobenius norm / t or use fixed value
        # Frobenius^2 = tr(XtX) = sum (singular value)^2 of X = sum eigenvalue of XtX. Frobenius^2/t = average eigenvalue of XtX.
        if ridge_lambda == 0:
            lam = 0
        else:
            if ridge_scale == 'spectral':
                try:
                    lam = ridge_lambda * (torch.linalg.matrix_norm(X, ord=2)**2) # ridge to PD (largest eigenvalue)
                except Exception as e:
                    # fallback to frobenius norm
                    print(f"Warning: Spectral norm computation failed ({e}), falling back to Frobenius norm")
                    lam = ridge_lambda * ((torch.linalg.matrix_norm(X, ord='fro')**2) / t)  # average eigenvalue
            elif ridge_scale == 'frobenius':
                lam = ridge_lambda * ((torch.linalg.matrix_norm(X, ord='fro')**2) / t)  # average eigenvalue of XtX
            elif ridge_scale == 'fixed':
                lam = ridge_lambda
            else:
                raise ValueError(f"Unknown ridge_scale: {ridge_scale}. Must be 'spectral', 'frobenius', or 'fixed'.")

        # Solve based on selected method
        if solver == 'lstsq':
            # Use lstsq solver
            # Solves: X @ C2 = Y
            try:
                C2_32 = torch.linalg.lstsq(X, Y, driver='gels').solution    # (t,d)
                if torch.isnan(C2_32).any():
                    raise RuntimeError("NaNs in lstsq solution")
            except Exception as e:
                print(f"lstsq failed ({e}), increasing lambda to 1e-6 and retrying")
                print(f"  C1 has NaN: {torch.isnan(C1).any().item()}, Inf: {torch.isinf(C1).any().item()}")
                print(f"  beta has NaN: {torch.isnan(beta).any().item()}, Inf: {torch.isinf(beta).any().item()}")
                print(f"  K has NaN: {torch.isnan(K).any().item()}, Inf: {torch.isinf(K).any().item()}")
                print(f"  V has NaN: {torch.isnan(V).any().item()}, Inf: {torch.isinf(V).any().item()}")
                print(f"  queries has NaN: {torch.isnan(queries).any().item()}, Inf: {torch.isinf(queries).any().item()}")
                print(f"  X has NaN: {torch.isnan(X).any().item()}, Inf: {torch.isinf(X).any().item()}")
                print(f"  Y has NaN: {torch.isnan(Y).any().item()}, Inf: {torch.isinf(Y).any().item()}")

                lam = 1e-6
                # Fall back to cholesky method with increased lambda
                if n < t:
                    XXt = X @ X.T
                    XXt = 0.5 * (XXt + XXt.T)
                    XXt.diagonal().add_(lam)
                    L = torch.linalg.cholesky(XXt)
                    Z = torch.cholesky_solve(Y, L)
                    C2_32 = X.T @ Z
                else:
                    XtX = X.T @ X
                    XtX = 0.5 * (XtX + XtX.T)
                    XtX.diagonal().add_(lam)
                    L = torch.linalg.cholesky(XtX)
                    XtY = X.T @ Y
                    C2_32 = torch.cholesky_solve(XtY, L)
                if torch.isnan(C2_32).any():
                    raise RuntimeError("NaNs in cholesky solution")
        elif solver == 'pinv':
            # Use pseudoinverse
            if lam == 0:
                X_pinv = torch.linalg.pinv(X)
                C2_32 = X_pinv @ Y
            elif n >= t:
                # C2 = (XᵀX + λI)^{-1} Xᵀ Y
                XtX = X.T @ X
                XtX = 0.5 * (XtX + XtX.T)
                XtX.diagonal().add_(lam)
                XtY = X.T @ Y
                C2_32 = torch.linalg.pinv(XtX) @ XtY         # (t,d)
            else:
                # n < t: use underdetermined formulation
                XXt = X @ X.T
                XXt = 0.5 * (XXt + XXt.T)
                XXt.diagonal().add_(lam)
                XXt_pinv = torch.linalg.pinv(XXt)
                Z = XXt_pinv @ Y
                C2_32 = X.T @ Z
        elif solver == 'cholesky':
            # Use Cholesky decomposition
            if n < t:
                # C2 = Xᵀ (XXᵀ + λI)^{-1} Y
                XXt = X @ X.T
                XXt = 0.5 * (XXt + XXt.T)
                XXt.diagonal().add_(lam)
                L = torch.linalg.cholesky(XXt)               # (n,n)
                Z = torch.cholesky_solve(Y, L)               # solves (XXᵀ+λI)Z = Y
                C2_32 = X.T @ Z                              # (t,d)
            else:
                # C2 = (XᵀX + λI)^{-1} Xᵀ Y
                XtX = X.T @ X
                XtX = 0.5 * (XtX + XtX.T)
                XtX.diagonal().add_(lam)
                L = torch.linalg.cholesky(XtX)               # (t,t)
                XtY = X.T @ Y
                C2_32 = torch.cholesky_solve(XtY, L)         # (t,d)
        else:
            raise ValueError(f"Unknown solver: {solver}. Must be 'pinv', 'cholesky', or 'lstsq'.")

        return C2_32.to(dtype_param)

    def _compute_C2_on_policy(
        self,
        C1: torch.Tensor,
        beta: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        queries_original: torch.Tensor,
        queries_generated: torch.Tensor,
        attention_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        UNUSED
        Instead, on-policy compaction uses _compute_C2 with just the on-policy queries.
        Solve for C2 in softmax(q_original·Kᵀ)V ≈ softmax(q_generated·C1ᵀ + beta)·C2 for all queries.

        Matches: exp(q_originalK^T)V / sum_j exp(q_originalK_j^T) = [exp(q_generatedC1^T + beta) / sum_j exp(q_generated(C1)_j^T + beta_j)] C2

        Follows precision policy:
        - QK and (softmax·V) matmuls in original dtype (kernel does fp32 accumulation)
        - Upcast to fp32 only for scaling + softmax/LSE/exp
        - Ridge regression for C2

        Parameters
        ----------
        C1 : Tensor, shape (t, d)
            Reduced set of keys.
        beta : Tensor, shape (t,)
            Bias terms for the reduced keys.
        K, V : Tensors, shapes (T, d)
            Original keys and values.
        queries_original : Tensor, shape (n, d)
            Original query vectors.
        queries_generated : Tensor, shape (n, d)
            Sampled query vectors.

        Returns
        -------
        C2 : Tensor, shape (t, d)
            Compacted value matrix.
        """
        # matmuls in original dtype; softmax bits in fp32
        dtype_param = K.dtype
        d = K.shape[1]
        inv_sqrt_d = (1.0 / d) ** 0.5

        # Y = softmax((Q_original K)/sqrt(d)) @ V
        sK_raw = queries_original @ K.T                           # (n, T) original dtype
        sK32 = sK_raw.to(torch.float32) * inv_sqrt_d             # (n, T)
        if attention_bias is not None:
            try:
                bias32 = torch.broadcast_to(
                    attention_bias.to(torch.float32),
                    sK32.shape
                )
                sK32 = sK32 + bias32
            except Exception as e:
                raise ValueError(
                    f"attention_bias must be broadcastable to {sK32.shape}, "
                    f"got {tuple(attention_bias.shape)}"
                ) from e
        m_K = sK32.max(dim=1, keepdim=True)[0]                   # (n, 1) for stability
        exp_sK = torch.exp(sK32 - m_K)                           # (n, T)
        sum_exp_K = exp_sK.sum(dim=1, keepdim=True)              # (n, 1)
        attn_K = exp_sK / sum_exp_K                              # (n, T) normalized
        Y = attn_K @ V.to(torch.float32)                         # (n, d) fp32

        # X = softmax((Q_generated C1^T)/sqrt(d) + beta)
        sC_raw = queries_generated @ C1.T                         # (n, t) original dtype
        sC32 = sC_raw.to(torch.float32) * inv_sqrt_d + beta.to(torch.float32)  # (n, t)
        m_C = sC32.max(dim=1, keepdim=True)[0]                   # (n, 1) for stability
        exp_sC = torch.exp(sC32 - m_C)                           # (n, t)
        sum_exp_C = exp_sC.sum(dim=1, keepdim=True)              # (n, 1)
        X = exp_sC / sum_exp_C                                   # (n, t) normalized

        # Note: See https://blogs.rstudio.com/ai/posts/2022-10-13-torch-linalg/
        n, t = X.shape
        lam = 0

        # Use lstsq with lam=0 and try-except fallback
        try:
            C2_32 = torch.linalg.lstsq(X, Y, driver='gels').solution    # (t,d)
            if torch.isnan(C2_32).any():
                raise RuntimeError("NaNs in lstsq solution")
        except Exception as e:
            print(f"lstsq failed ({e}), increasing lambda to 1e-6 and retrying")
            lam = 1e-6
            # Fall back to cholesky method with increased lambda
            if n < t:
                XXt = X @ X.T
                XXt = 0.5 * (XXt + XXt.T)
                XXt.diagonal().add_(lam)
                L = torch.linalg.cholesky(XXt)
                Z = torch.cholesky_solve(Y, L)
                C2_32 = X.T @ Z
            else:
                XtX = X.T @ X
                XtX = 0.5 * (XtX + XtX.T)
                XtX.diagonal().add_(lam)
                L = torch.linalg.cholesky(XtX)
                XtY = X.T @ Y
                C2_32 = torch.cholesky_solve(XtY, L)
            if torch.isnan(C2_32).any():
                raise RuntimeError("NaNs in cholesky solution")

        return C2_32.to(dtype_param)

    def _direct_C2(
        self,
        C1: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        indices: list = None
    ) -> torch.Tensor:
        """
        Select C2 directly from V by finding nearest neighbors in K for each key in C1.

        For each key in C1, finds the index in K that is the nearest neighbor (by L2 distance),
        and selects the corresponding value from V.

        If indices are provided (i.e., C1 is a subset of K), uses those indices directly
        since the nearest neighbor of K[i] in K is K[i] itself.

        Parameters
        ----------
        C1 : Tensor, shape (t, d)
            Compacted keys.
        K : Tensor, shape (T, d)
            Original keys.
        V : Tensor, shape (T, d)
            Original values.
        indices : list of int, optional
            Indices of selected keys from K. If provided, C2 is directly selected
            from V using these indices (default: None).

        Returns
        -------
        C2 : Tensor, shape (t, d)
            Compacted values, selected from V based on nearest neighbor matching.
        """
        # If indices are provided, C1 is a subset of K, so just use those indices
        if indices is not None:
            if isinstance(indices, list):
                indices = torch.tensor(indices, device=V.device, dtype=torch.long)
            C2 = V[indices]  # (t, d)
            return C2

        # Otherwise, find nearest neighbors
        t, d = C1.shape
        T = K.shape[0]

        # Compute pairwise squared distances: ||C1[i] - K[j]||^2
        # = ||C1[i]||^2 + ||K[j]||^2 - 2*C1[i]·K[j]
        C1_norms_sq = (C1 ** 2).sum(dim=1)  # (t,)
        K_norms_sq = (K ** 2).sum(dim=1)    # (T,)
        pairwise_dots = C1 @ K.T            # (t, T)

        # Squared distances: (t, T)
        squared_distances = C1_norms_sq.unsqueeze(1) + K_norms_sq.unsqueeze(0) - 2 * pairwise_dots

        # Find nearest neighbor index for each key in C1
        nearest_indices = squared_distances.argmin(dim=1)  # (t,)

        # Select corresponding values from V
        C2 = V[nearest_indices]  # (t, d)

        return C2

    def _compute_C2_with_method(
        self,
        C1: torch.Tensor,
        beta: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        method: str = 'lsq',
        indices: list = None,
        attention_bias: torch.Tensor = None,
        ridge_lambda: float = 0,
        solver: str = 'lstsq',
        ridge_scale: str = 'spectral',
        **kwargs
    ) -> torch.Tensor:
        """
        Compute C2 using the specified method.

        Parameters
        ----------
        C1 : Tensor, shape (t, d)
            Compacted keys.
        beta : Tensor, shape (t,)
            Bias terms (not used for 'direct' method).
        K : Tensor, shape (T, d)
            Original keys.
        V : Tensor, shape (T, d)
            Original values.
        queries : Tensor, shape (n, d)
            Query samples (not used for 'direct' method).
        method : str
            Method to use: 'lsq' for least squares (default) or 'direct' for nearest
            neighbor selection.
        indices : list of int, optional
            Indices of selected keys (used for 'direct' method when C1 is a subset of K).
        attention_bias : Tensor, optional
            Additive attention bias for the original cache (broadcastable to (n, T)).
        ridge_lambda : float
            Regularization parameter for ridge regression (default: 0).
        solver : str
            Solver to use: 'auto', 'pinv', 'cholesky', or 'lstsq' (default: 'auto').
        ridge_scale : str
            How to scale ridge_lambda: 'spectral', 'frobenius', or 'fixed' (default: 'spectral').

        Returns
        -------
        C2 : Tensor, shape (t, d)
            Compacted values.
        """
        if method == 'direct':
            return self._direct_C2(C1, K, V, indices)
        elif method == 'lsq':
            return self._compute_C2(
                C1, beta, K, V, queries,
                attention_bias=attention_bias,
                ridge_lambda=ridge_lambda,
                solver=solver,
                ridge_scale=ridge_scale
            )
        else:
            raise ValueError(f"Unknown C2 computation method: {method}. Must be 'lsq' or 'direct'.")

    @staticmethod
    def _nnls_pg(M: torch.Tensor, y: torch.Tensor, iters: int = 0,
                   lower_bound: float = 1e-12, upper_bound: float = None, debug: bool = False) -> torch.Tensor:
        """
        Box-constrained non-negative least squares solver with projected gradient.

        If iters == 0: Use ridge normal-equations solve + clamp with clamping to bounds.
        If iters > 0: Use projected-gradient descent with specified iterations.

        Solves: min_B 0.5 * ||M B - y||_2^2  s.t. lower_bound <= B <= upper_bound
        Step size 1/L with L ≈ ||M||_2^2 via power iteration.
        Expects fp32 inputs; returns fp32 B with box constraints.

        Parameters
        ----------
        M : Tensor, shape (n, t)
            Design matrix
        y : Tensor, shape (n,)
            Target vector
        iters : int
            Number of projected gradient iterations (0 = use clamped least squares)
        lower_bound : float
            Lower bound for B values (default: 1e-12)
        upper_bound : float, optional
            Upper bound for B values (default: None, no upper bound)
        """
        n, t = M.shape
        min_val = 1e-12 if lower_bound is None else lower_bound

        lam = 0

        if lam == 0:
            # Use lstsq when no regularization
            # Solves: M @ B = y
            try:
                B = torch.linalg.lstsq(M, y.unsqueeze(1), driver='gels').solution.squeeze(1)
                if torch.isnan(B).any():
                    raise RuntimeError("NaNs in NNLS lstsq solution")
            except Exception as e:
                print(f"lstsq failed ({e}), increasing lambda to 1e-5*(||M||_2^2) and retrying")
                print(f"  M has NaN: {torch.isnan(M).any().item()}, Inf: {torch.isinf(M).any().item()}")
                print(f"  y has NaN: {torch.isnan(y).any().item()}, Inf: {torch.isinf(y).any().item()}")
                lam = 1e-6
                # Fall back to cholesky method with increased lambda
                if n < t:
                    MMt = M @ M.T
                    MMt = 0.5 * (MMt + MMt.T)
                    MMt.diagonal().add_(lam)
                    R = torch.linalg.cholesky(MMt)
                    alpha = torch.cholesky_solve(y.unsqueeze(1), R).squeeze(1)
                    B = M.T @ alpha
                else:
                    MtM = M.T @ M
                    MtM = 0.5 * (MtM + MtM.T)
                    MtM.diagonal().add_(lam)
                    R = torch.linalg.cholesky(MtM)
                    Mty = M.T @ y
                    B = torch.cholesky_solve(Mty.unsqueeze(1), R).squeeze(1)
                if torch.isnan(B).any():
                    raise RuntimeError("NaNs in NNLS cholesky solution")
        elif n < t:  # underdetermined
            MMt = M @ M.T
            MMt.diagonal().add_(lam)
            R = torch.linalg.cholesky(MMt)  # SPD factor
            alpha = torch.cholesky_solve(y.unsqueeze(1), R).squeeze(1)
            B = M.T @ alpha
        else:  # overdetermined
            MtM = M.T @ M
            MtM.diagonal().add_(lam)
            R = torch.linalg.cholesky(MtM)
            Mty = M.T @ y
            B = torch.cholesky_solve(Mty.unsqueeze(1), R).squeeze(1)

        # Debug: print statistics before clamping
        if debug:
            n_total = B.numel()
            n_below_min = (B < min_val).sum().item()
            n_above_upper = (B > upper_bound).sum().item() if upper_bound is not None else 0
            print(f"[NNLS Debug] Before clamping: total_values={n_total}, below_min={n_below_min}, above_upper={n_above_upper}")
            print(f"[NNLS Debug] B range before clamping: min={B.min().item():.6e}, max={B.max().item():.6e}")

        # Apply bounds
        B = B.clamp_min_(min_val)
        if upper_bound is not None:
            B = B.clamp_max_(upper_bound)

        # Debug: print B range after clamping
        if debug:
            print(f"[NNLS Debug] B range after clamping: min={B.min().item():.6e}, max={B.max().item():.6e}")

        if iters == 0:
            if debug:
                residual = M @ B - y
                loss = (residual ** 2).sum().item()
                print(f"[NNLS Debug] Initial solution (iters=0): loss={loss:.6e}")
            return B

        # Power iteration for spectral norm
        u = torch.randn(t, device=M.device, dtype=M.dtype)
        u = u / (u.norm() + 1e-12)
        for _ in range(3):  # converges very fast usually
            v = M @ u
            if v.norm() == 0:
                break
            v = v / v.norm()
            u = M.T @ v
            if u.norm() == 0:
                break
            u = u / u.norm()
        sigma = (u @ (M.T @ (M @ u))).sqrt().clamp_min(1e-6)  # ~||M||_2
        L = (sigma ** 2).clamp_min(1e-6)
        eta = 1.0 / L

        # Debug: print initial loss before PGD
        if debug:
            residual = M @ B - y
            loss = (residual ** 2).sum().item()
            print(f"[NNLS Debug] PGD iteration 0/{iters}: loss={loss:.6e}")

        # Projected gradient descent for B
        for iter_idx in range(iters):
            grad = M.T @ (M @ B - y)
            B = B - eta * grad
            # Project to feasible set
            B = B.clamp_min_(min_val)
            if upper_bound is not None:
                B = B.clamp_max_(upper_bound)

            # Debug: print loss at each iteration (1-based indexing)
            if debug and ((iter_idx + 1) % max(1, iters // 10) == 0 or iter_idx == iters - 1):
                residual = M @ B - y
                loss = (residual ** 2).sum().item()
                print(f"[NNLS Debug] PGD iteration {iter_idx + 1}/{iters}: loss={loss:.6e}")

        return B


def compute_attention(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    UNUSED
    Compute standard attention output for query q.

    Parameters
    ----------
    q : Tensor, shape (1, d) or (d,)
        Single query vector.
    K, V : Tensors, shapes (T, d)
        Keys and values.

    Returns
    -------
    output : Tensor, shape (d,)
        Attention output.
    attn_weights : Tensor, shape (T,)
        Attention weights (softmax over keys).
    lse : float
        Log-sum-exp of the scores.
    """
    if q.ndim == 1:
        q = q.unsqueeze(0)
    d = K.shape[1]
    inv_sqrt_d = (1.0 / d) ** 0.5
    # QK matmul in original dtype (kernel will accumulate in fp32)
    scores_raw = q @ K.T                              # (1, T) original dtype
    # Upcast only for scale + softmax
    scores32 = (scores_raw.to(torch.float32) * inv_sqrt_d).squeeze(0)  # (T,) fp32
    lse32 = torch.logsumexp(scores32, dim=0)           # fp32
    attn_w32 = torch.exp(scores32 - lse32)             # (T,) fp32
    # Downcast weights to original dtype for matmul with V
    attn_w = attn_w32.to(V.dtype)
    out = attn_w @ V                                   # (d,) original dtype (fp32 acc in kernel)
    return out, attn_w, float(lse32.item())


def evaluate_compaction(
    K: torch.Tensor,
    V: torch.Tensor,
    C1: torch.Tensor,
    beta: torch.Tensor,
    C2: torch.Tensor,
    test_queries: torch.Tensor,
    attention_bias: torch.Tensor = None
) -> Dict:
    """
    Evaluate the quality of KV cache compaction.

    Parameters
    ----------
    K, V : Tensors, shapes (T, d)
        Original keys and values.
    C1 : Tensor, shape (t, d)
        Compacted keys.
    beta : Tensor, shape (t,)
        Bias terms.
    C2 : Tensor, shape (t, d)
        Compacted values.
    test_queries : Tensor, shape (n_test, d)
        Test query vectors.
    attention_bias : Tensor, optional
        Additive attention bias for the original cache scores.
        Shape (T,) or (n_test, T); broadcastable to (n_test, T). Defaults to None.

    Returns
    -------
    metrics : dict
        Dictionary containing evaluation metrics:
        - mean_output_mse: Mean MSE across queries (averaged over output dimension)
        - mean_output_mse_std: Standard deviation of mean MSE
        - max_output_mse: Maximum MSE across queries
        - rms_output_mse: RMS MSE across queries (max over output dim, RMS over queries)
        - mean_output_relative_l2_error: Mean relative L2 error
        - mean_output_relative_l2_error_std: Standard deviation of relative L2 error
        - max_output_relative_l2_error: Maximum relative L2 error across queries
        - rms_output_relative_l2_error: RMS relative L2 error
        - mean_output_cosine_sim: Mean cosine similarity
        - mean_output_cosine_sim_std: Standard deviation of cosine similarity
        - min_output_cosine_sim: Minimum cosine similarity across queries
        - rms_output_cosine_sim: RMS cosine similarity
        - mean_sumexp_relative_error: Mean sumexp relative error
        - mean_sumexp_relative_error_std: Standard deviation of sumexp relative error
        - max_sumexp_relative_error: Maximum sumexp relative error across queries
        - rms_sumexp_relative_error: RMS sumexp relative error
        - compaction_ratio: T/t
    """
    n_test = test_queries.shape[0]
    T, d = K.shape
    t = C1.shape[0]

    # Original attention: Q @ K.T -> softmax -> @ V
    # Compacted attention: Q @ C1.T + beta -> softmax -> @ C2

    # Prepare for batched SDPA: (batch=1, heads=1, seq_len, head_dim)
    # SDPA expects: (batch, heads, seq_len_q, head_dim)
    # Ensure contiguous memory layout for SDPA kernels (required for stride 1 on last dim)
    queries_batch = test_queries.unsqueeze(0).unsqueeze(0).contiguous()  # (1, 1, n_test, d)
    K_batch = K.unsqueeze(0).unsqueeze(0).contiguous()  # (1, 1, T, d)
    V_batch = V.unsqueeze(0).unsqueeze(0).contiguous()  # (1, 1, T, d)
    C1_batch = C1.unsqueeze(0).unsqueeze(0).contiguous()  # (1, 1, t, d)
    C2_batch = C2.unsqueeze(0).unsqueeze(0).contiguous()  # (1, 1, t, d)

    # Build optional attention mask for the original cache
    orig_attn_mask = None
    if attention_bias is not None:
        try:
            bias = torch.broadcast_to(
                attention_bias.to(queries_batch.dtype),
                (1, 1, n_test, T)
            ).contiguous()
            orig_attn_mask = bias
        except Exception as e:
            raise ValueError(
                f"attention_bias must be broadcastable to (1, 1, {n_test}, {T}), "
                f"got {tuple(attention_bias.shape)}"
            ) from e

    # Compute original attention using SDPA
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out_orig = torch.nn.functional.scaled_dot_product_attention(
            queries_batch, 
            K_batch, 
            V_batch,
            attn_mask=orig_attn_mask,
            is_causal=False
        )  # (1, 1, n_test, d)
    out_orig = out_orig.squeeze(0).squeeze(0)  # (n_test, d)

    # For compacted attention, add beta bias via attn_mask
    # Create bias mask: (1, 1, n_test, t) with beta values broadcast across queries
    beta_mask = beta.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, t)
    beta_mask = beta_mask.expand(1, 1, n_test, t).contiguous()  # (1, 1, n_test, t)
    # Convert to match query dtype (e.g., bf16)
    beta_mask = beta_mask.to(queries_batch.dtype)

    # Compute compacted attention using SDPA with beta bias
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out_comp = torch.nn.functional.scaled_dot_product_attention(
            queries_batch[:, :, :, :C1.shape[1]],  # (1, 1, n_test, d)
            C1_batch,  # (1, 1, t, d)
            C2_batch,  # (1, 1, t, d)
            attn_mask=beta_mask,  # (1, 1, n_test, t)
            is_causal=False
        )  # (1, 1, n_test, d)
    out_comp = out_comp.squeeze(0).squeeze(0)  # (n_test, d)

    # Check for NaNs in outputs
    out_orig_has_nan = torch.isnan(out_orig).any().item()
    out_comp_has_nan = torch.isnan(out_comp).any().item()
    if out_orig_has_nan or out_comp_has_nan:
        print(f"NaN detected in outputs: out_orig={out_orig_has_nan}, out_comp={out_comp_has_nan}")
        print(f"  Input NaN status: test_queries={torch.isnan(test_queries).any().item()}, "
              f"K={torch.isnan(K).any().item()}, V={torch.isnan(V).any().item()}, "
              f"C1={torch.isnan(C1).any().item()}, C2={torch.isnan(C2).any().item()}")

    # Compute scores for LSE metrics (need manual computation for this)
    inv_sqrt_d = (1.0 / d) ** 0.5
    scores_orig = (test_queries @ K.T) * inv_sqrt_d  # (n_test, T)
    if attention_bias is not None:
        try:
            bias = torch.broadcast_to(
                attention_bias.to(scores_orig.dtype),
                scores_orig.shape
            )
            scores_orig = scores_orig + bias
        except Exception as e:
            raise ValueError(
                f"attention_bias must be broadcastable to {scores_orig.shape}, "
                f"got {tuple(attention_bias.shape)}"
            ) from e
    lse_orig = torch.logsumexp(scores_orig, dim=1)  # (n_test,)

    scores_comp = (test_queries @ C1.T) * inv_sqrt_d + beta  # (n_test, t)
    lse_comp = torch.logsumexp(scores_comp, dim=1)  # (n_test,)

    # Compute all metrics in batched form
    # Output MSE (mean over output dimension for each query)
    output_errors = torch.mean((out_orig - out_comp) ** 2, dim=1)  # (n_test,)

    # Relative L2 error
    orig_norms = torch.norm(out_orig, dim=1)  # (n_test,)
    diff_norms = torch.norm(out_orig - out_comp, dim=1)  # (n_test,)
    output_relative_errors = diff_norms / (orig_norms + 1e-10)  # (n_test,)

    # Cosine similarity (normalize with eps to avoid blowing up on tiny norms)
    cosine_sims = F.cosine_similarity(out_orig, out_comp, dim=1, eps=1e-8)  # (n_test,)

    # Sum-exp relative error
    lse_diff = lse_comp - lse_orig  # (n_test,)
    # For small differences, use |lse_diff| as approximation
    # For larger differences, compute |exp(lse_diff) - 1|
    small_diff_mask = torch.abs(lse_diff) < 1e-10
    sumexp_relative_errors = torch.where(
        small_diff_mask,
        torch.abs(lse_diff),
        torch.abs(torch.exp(lse_diff) - 1.0)
    )  # (n_test,)

    metrics = {
        # Mean metrics
        'mean_output_mse': output_errors.mean().item(),
        'mean_output_mse_std': output_errors.std().item(),
        'mean_output_relative_l2_error': output_relative_errors.mean().item(),
        'mean_output_relative_l2_error_std': output_relative_errors.std().item(),
        'mean_output_cosine_sim': cosine_sims.mean().item(),
        'mean_output_cosine_sim_std': cosine_sims.std().item(),
        'mean_sumexp_relative_error': sumexp_relative_errors.mean().item(),
        'mean_sumexp_relative_error_std': sumexp_relative_errors.std().item(),

        # Max metrics
        'max_output_mse': output_errors.max().item(),
        'max_output_relative_l2_error': output_relative_errors.max().item(),
        'min_output_cosine_sim': cosine_sims.min().item(),
        'max_sumexp_relative_error': sumexp_relative_errors.max().item(),

        # RMS metrics
        'rms_output_mse': torch.sqrt(torch.mean(output_errors ** 2)).item(),
        'rms_output_relative_l2_error': torch.sqrt(torch.mean(output_relative_errors ** 2)).item(),
        'rms_output_cosine_sim': torch.sqrt(torch.mean(cosine_sims ** 2)).item(),
        'rms_sumexp_relative_error': torch.sqrt(torch.mean(sumexp_relative_errors ** 2)).item(),

        'compaction_ratio': T / t,
    }

    return metrics

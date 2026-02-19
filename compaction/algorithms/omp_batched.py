# compaction/algorithms/omp_batched.py
# unused

import torch
from typing import Tuple
from .batched import BatchedCompactionAlgorithm

class BatchedOMPCompaction(BatchedCompactionAlgorithm):
    """Batched Orthogonal Matching Pursuit based compaction.

    Processes multiple (layer, head) combinations simultaneously for GPU efficiency.
    """

    def __init__(
        self,
        nnls_iters: int = 0,
        nnls_lower_bound: float = None,
        nnls_upper_bound: float = None,
        c2_method: str = 'lsq',
        k_choice: int = 1,
        nnls_interval: int = None,
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
        k_choice : int
            Number of keys to select at once (chunking). Selects top k_choice candidates
            by correlation magnitude with LSE residual (default: 1).
        nnls_interval : int, optional
            If provided, skips NNLS solves except once every nnls_interval iterations.
        """
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_method = c2_method
        self.k_choice = k_choice
        self.nnls_interval = nnls_interval

    def name(self) -> str:
        return "BatchedOMP"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Non-batched interface wrapper for compatibility with run_experiments.py.

        Adds a batch dimension, calls the batched method, then removes batch dimension.

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
        # Add batch dimension
        K_batched = K.unsqueeze(0)  # (1, T, d)
        V_batched = V.unsqueeze(0)  # (1, T, d)
        queries_batched = queries.unsqueeze(0)  # (1, n, d)

        # Call batched method
        C1_batched, beta_batched, C2_batched, indices_batched = self.compute_compacted_cache_batched(
            K_batched, V_batched, queries_batched, t
        )

        # Remove batch dimension
        C1 = C1_batched.squeeze(0)  # (t, d)
        beta = beta_batched.squeeze(0)  # (t,)
        C2 = C2_batched.squeeze(0)  # (t, d)
        indices = indices_batched[0].cpu().tolist()  # list of int

        return C1, beta, C2, indices

    def compute_compacted_cache_batched(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute compacted cache using OMP-based key selection for multiple instances.

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
        # Select keys using batched OMP
        C1, beta, indices = self._select_keys_omp_batched(K, queries, t, self.k_choice)

        # Compute compacted values using shared primitive
        C2 = BatchedCompactionAlgorithm._compute_C2_with_method_batched(
            C1, beta, K, V, queries, method=self.c2_method, indices=indices
        )

        return C1, beta, C2, indices

    def _solve_nnls_batched(
        self,
        M: torch.Tensor,
        target: torch.Tensor,
        prev_B: torch.Tensor,
        iteration: int,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Unified batched NNLS solver that supports both eager and lazy solving modes.

        Modes:
        - Eager (default): Always solve NNLS at every iteration (nnls_interval=1)
        - Lazy: Conditionally solve based on interval-based triggering

        Parameters
        ----------
        M : Tensor, shape (B, n, i)
            Design matrices with i selected columns
        target : Tensor, shape (B, n)
            Target vectors
        prev_B : Tensor, shape (B, i-1), optional
            Previous solutions (if available)
        iteration : int
            Current iteration number

        Returns
        -------
        B : Tensor, shape (B, i)
            Solution vectors
        solved : bool
            Whether NNLS was actually solved (True) or solution was reused/updated (False)
        """
        # Determine if we should solve based on the configured mode
        should_solve = False

        if prev_B is None:
            # Always solve if no previous solution exists
            should_solve = True
        elif self.nnls_interval is not None:
            # Interval-based triggering
            should_solve = (iteration % self.nnls_interval == 0)

        # Solve or reuse based on the decision
        if should_solve:
            B = BatchedCompactionAlgorithm._nnls_pg_batched(
                M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound
            )
            return B, True
        else:
            # Reuse previous solution and extend with lower-bound values for new entries
            B_batch, _, i = M.shape
            prev_i = prev_B.shape[1]
            B = torch.zeros(B_batch, i, dtype=torch.float32, device=M.device)
            B[:, :prev_i] = prev_B
            min_val = 1e-12 if self.nnls_lower_bound is None else self.nnls_lower_bound
            B[:, prev_i:] = min_val
            return B, False

    def _select_keys_omp_batched(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        k_choice: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched greedy selection of t keys from K using OMP-style approximation.

        Parameters
        ----------
        K : Tensor, shape (B, T, d)
            Original key matrices.
        queries : Tensor, shape (B, n, d)
            Sampled query vectors.
        t : int
            Number of keys to select for the compacted cache.
        k_choice : int
            Number of keys to select at once (chunking). Selects top k_choice candidates
            by correlation magnitude with LSE residual.

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
        n = queries.shape[1]
        device = K.device

        # Compute exp_scores and target using shared primitive
        exp_scores, target = BatchedCompactionAlgorithm._compute_exp_scores_and_target_batched(K, queries)

        # Pre-allocate tensors
        selected_indices_tensor = torch.zeros(B, t, dtype=torch.long, device=device)
        beta32 = torch.zeros(B, t, dtype=torch.float32, device=device)
        current = torch.zeros_like(target)  # (B, n) fp32
        mask_selected = torch.zeros(B, T, dtype=torch.bool, device=device)

        i = 0
        prev_B = None  # Track previous solution for lazy NNLS
        iteration = 0
        
        while i < t:
            residual = target - current  # (B, n)
            # Vectorized correlation with residual: (B, n, T) * (B, n, 1) -> sum over n -> (B, T)
            corr = (exp_scores * residual.unsqueeze(2)).sum(dim=1)  # (B, T)
            corr[mask_selected] = -float('inf')
            
            # Select top k_choice candidates by correlation magnitude for each batch element
            num_remaining = T - mask_selected.sum(dim=1)  # (B,)
            k_select = min(k_choice, num_remaining.min().item(), t - i)  # Don't exceed remaining slots
            
            if k_select == 0:
                break
            
            # Get top k_select by absolute correlation magnitude for each batch element
            corr_abs = torch.abs(corr)  # (B, T)
            top_k_values, top_k_indices = torch.topk(corr_abs, k_select, dim=1, largest=True)  # (B, k_select)
            
            # Add selected indices for each batch element
            for j in range(k_select):
                if i >= t:
                    break
                best_idx = top_k_indices[:, j]  # (B,)
                selected_indices_tensor[:, i] = best_idx
                # Update mask: mark selected indices as True
                mask_selected[torch.arange(B, device=device), best_idx] = True
                i += 1

            # Build design matrix M and solve NNLS lazily for each batch element
            # M: (B, n, i) - gather selected columns for each batch element
            # Use advanced indexing with broadcasting
            batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, n, i)
            query_indices = torch.arange(n, device=device).view(1, n, 1).expand(B, n, i)
            selected_so_far = selected_indices_tensor[:, :i].unsqueeze(1).expand(B, n, i)

            M = exp_scores[batch_indices, query_indices, selected_so_far]  # (B, n, i) fp32

            # Unified batched NNLS solve
            B_solution, _ = self._solve_nnls_batched(
                M, target, prev_B, iteration
            )  # (B, i) fp32 (>=0)
            prev_B = B_solution  # Update for next iteration
            beta32[:, :i] = torch.log(B_solution)

            # Update current: (B, n, i) @ (B, i, 1) -> (B, n)
            current = torch.bmm(M, B_solution.unsqueeze(2)).squeeze(2)  # (B, n)
            iteration += 1

        # Gather selected keys using indices
        # C1: (B, t, d) - use simple loop approach for clarity
        C1 = torch.stack([K[b, selected_indices_tensor[b]] for b in range(B)])

        # Convert beta from fp32 to model dtype (e.g., bf16) for storage
        beta = beta32.to(K.dtype)

        return C1, beta, selected_indices_tensor

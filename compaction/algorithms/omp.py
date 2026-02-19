# compaction/algorithms/omp.py
"""
OMP-based KV cache compaction algorithm.

Uses orthogonal matching pursuit to greedily select keys that best
approximate the partition function over attention scores.
"""
import torch
from typing import Tuple, List, Optional
from .base import CompactionAlgorithm


class SimpleOMPCompaction:
    """
    OMP algorithm for reference/documentation purposes.

    This class implements the core OMP algorithm without any hyperparameters
    or optimizations.

    The algorithm greedily selects keys that best approximate the partition
    function (sum of exp scores) over attention scores using orthogonal
    matching pursuit.
    """

    def select_keys(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Greedy selection of t keys from K using OMP.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix.
        queries : Tensor, shape (n, d)
            Sampled query vectors.
        t : int
            Number of keys to select.
        attention_bias : Tensor, optional
            Additive attention bias for the original cache (broadcastable to (n, T)).

        Returns
        -------
        C1 : Tensor, shape (t, d)
            Selected keys.
        beta : Tensor, shape (t,)
            Log-weights for each selected key.
        indices : list of int
            Indices of the selected keys in K.
        """
        n, d = queries.shape
        T = K.shape[0]
        device = K.device
        dtype = K.dtype

        # Compute exp(q·k / sqrt(d)) for all query-key pairs
        inv_sqrt_d = (1.0 / d) ** 0.5
        scores = (queries @ K.T).to(torch.float32) * inv_sqrt_d  # (n, T)
        if attention_bias is not None:
            try:
                bias32 = torch.broadcast_to(
                    attention_bias.to(torch.float32),
                    scores.shape
                )
                scores = scores + bias32
            except Exception as e:
                raise ValueError(
                    f"attention_bias must be broadcastable to {scores.shape}, "
                    f"got {tuple(attention_bias.shape)}"
                ) from e
        max_scores = scores.max(dim=1, keepdim=True)[0]  # (n, 1)
        exp_scores = torch.exp(scores - max_scores)  # (n, T)

        # Target: sum of all exp scores per query (partition function)
        target = exp_scores.sum(dim=1)  # (n,)

        # Greedy selection
        selected_indices = []
        mask = torch.zeros(T, dtype=torch.bool, device=device)
        current = torch.zeros_like(target)  # Current approximation

        for _ in range(t):
            # Compute residual
            residual = target - current  # (n,)

            # Correlation of each key's exp_scores with residual
            corr = (exp_scores * residual.unsqueeze(1)).sum(dim=0)  # (T,)
            corr[mask] = -float('inf')  # Exclude already selected

            # Select key with highest correlation
            idx = corr.argmax().item()
            selected_indices.append(idx)
            mask[idx] = True

            # Solve NNLS: find weights B such that M @ B ≈ target
            M = exp_scores[:, selected_indices]  # (n, i)
            B = torch.linalg.lstsq(M, target.unsqueeze(1)).solution.squeeze(1)
            B = B.clamp(min=1e-12)  # Non-negative constraint

            # Update current approximation
            current = M @ B

        # Extract results
        indices_tensor = torch.tensor(selected_indices, device=device)
        C1 = K[indices_tensor]
        beta = torch.log(B).to(dtype)

        return C1, beta, selected_indices


# Default progressive schedule for OMP:
# - First 300 keys: standard OMP (k_choice=1, interval=1)
# - Keys 301-1500: k_choice=2, interval=2
# - Keys 1501+: k_choice=4, interval=2
# Each tuple is (max_keys, k_choice, nnls_interval)
DEFAULT_PROGRESSIVE_SCHEDULE = [
    (300, 1, 1),
    (1500, 2, 2),
    (None, 4, 2),
]

class OMPCompaction(CompactionAlgorithm):
    """
    Orthogonal Matching Pursuit based compaction.

    This class has many hyperparameters, but most are pretty minor variations 
    or optimizations. See SimpleOMPCompaction above for the core algorithm.
    The key extensions here are:
    - k_choice / progressive_schedule: select multiple keys per iteration for speed
    - nnls_interval: skip NNLS solves for speed
    - drop_key_beta_cutoff: refinement phase to drop low-weight keys (unimportant in practice because low-weight keys are rare and usually mean we are doing well already on the train set)
    """

    def __init__(self, nnls_iters: int = 0, nnls_lower_bound: float = None, nnls_upper_bound: float = None,
                 c2_method: str = 'lsq', k_choice: int = 1,
                 c2_ridge_lambda: float = 0, c2_solver: str = 'lstsq', c2_ridge_scale: str = 'spectral',
                 nnls_interval: int = 1, use_abs_corr: bool = False, normalize_exp_scores: bool = False,
                 debug: bool = False, drop_key_beta_cutoff: float = None,
                 progressive_schedule: Optional[List[Tuple[float, int, int]]] = None,
                 zerobeta: bool = False):
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
        c2_ridge_lambda : float
            Regularization parameter for C2 ridge regression (default: 0).
        c2_solver : str
            Solver to use for C2: 'pinv', 'cholesky', or 'lstsq' (default: 'lstsq').
        c2_ridge_scale : str
            How to scale ridge_lambda: 'spectral', 'frobenius', or 'fixed' (default: 'spectral').
        nnls_interval : int, optional
            If provided, skips NNLS solves except once every nnls_interval iterations.
        use_abs_corr : bool
            If True, use absolute correlation for key selection. If False, use raw correlation (default: False).
        normalize_exp_scores : bool
            If True, normalize exp_scores columns by L2 norm before computing correlation (default: False).
        debug : bool
            If True, enable debug printing (default: False).
        drop_key_beta_cutoff : float, optional
            If provided, enables a refinement phase after selecting t keys. Keys with beta (log weight)
            below this threshold are dropped and excluded from future selection. The algorithm continues
            selecting replacement keys until t stable keys are obtained, or until all available keys
            are exhausted (in which case fewer than t keys may be returned) (default: None).
        progressive_schedule : list of (max_keys, k_choice, nnls_interval), optional
            Progressive schedule for OMP. Each tuple specifies:
            - max_keys: use this config until we've selected this many keys
            - k_choice: number of keys to select per iteration
            - nnls_interval: how often to solve NNLS
            If provided, overrides k_choice and nnls_interval based on current key count.
            Use DEFAULT_PROGRESSIVE_SCHEDULE for the default schedule (default: None).
        zerobeta : bool
            If True, set all betas to 0 before computing C2 (default: False).
        """
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_method = c2_method
        self.k_choice = k_choice
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale
        self.use_abs_corr = use_abs_corr
        self.normalize_exp_scores = normalize_exp_scores
        self.debug = debug
        self.drop_key_beta_cutoff = drop_key_beta_cutoff
        self.progressive_schedule = progressive_schedule

        # Default: always solve (nnls_interval = 1)
        # For lazy solving, explicitly set nnls_interval > 1
        self.nnls_interval = nnls_interval
        self.zerobeta = zerobeta

    def name(self) -> str:
        return "OMP"

    def _get_schedule_params(self, num_selected: int) -> Tuple[int, int]:
        """
        Get k_choice and nnls_interval for current number of selected keys.

        Parameters
        ----------
        num_selected : int
            Current number of selected keys

        Returns
        -------
        k_choice : int
            Number of keys to select at once
        nnls_interval : int
            How often to solve NNLS
        """
        if self.progressive_schedule is None:
            return self.k_choice, self.nnls_interval

        for max_keys, k_choice, nnls_interval in self.progressive_schedule:
            if max_keys is None or num_selected < max_keys:
                return k_choice, nnls_interval

        # Fallback to last entry if we exceed all thresholds
        return self.progressive_schedule[-1][1], self.progressive_schedule[-1][2]

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        cached_selection_order: Optional[List[int]] = None,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Compute compacted cache using OMP-based key selection.

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
        cached_selection_order : list of int, optional
            Pre-computed selection order from a previous run with larger t.
            If provided, uses the first t indices from this list instead of
            running OMP, and only computes beta via NNLS.
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
        if cached_selection_order is not None:
            # Use cached selection order - just take first t indices
            C1, beta, indices = self._select_keys_from_cached_order(
                K, queries, t, cached_selection_order, attention_bias
            )
        else:
            # Select keys using OMP
            C1, beta, indices = self._select_keys_omp(K, queries, t, self.k_choice, attention_bias)

        # Zero out betas if requested (before C2 computation)
        if self.zerobeta:
            beta = torch.zeros_like(beta)

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

    def get_full_selection_order(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        max_keys: Optional[int] = None,
        attention_bias: torch.Tensor = None,
    ) -> List[int]:
        """
        Get the full greedy selection order for all keys (or up to max_keys).

        This runs OMP to select keys and returns the order in which they were
        selected. This order can be cached and reused when varying the number
        of keys to select.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix
        queries : Tensor, shape (n, d)
            Query samples for training
        max_keys : int, optional
            Maximum number of keys to select. If None, selects all T keys.
        attention_bias : Tensor, optional
            Additive attention bias for the original cache (broadcastable to (n, T)).

        Returns
        -------
        selection_order : list of int
            Indices in the order they were selected by greedy OMP
        """
        T = K.shape[0]
        if max_keys is None:
            max_keys = T

        # Run OMP to get the selection order
        _, _, indices = self._select_keys_omp(K, queries, max_keys, self.k_choice, attention_bias)
        return indices

    def _select_keys_from_cached_order(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        cached_order: List[int],
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Select first t keys from a cached selection order and compute beta.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix
        queries : Tensor, shape (n, d)
            Query samples for training
        t : int
            Number of keys to select
        cached_order : list of int
            Pre-computed selection order
        attention_bias : Tensor, optional
            Additive attention bias for the original cache (broadcastable to (n, T)).

        Returns
        -------
        C1 : Tensor, shape (t, d)
            Selected keys
        beta : Tensor, shape (t,)
            Bias terms
        indices : list of int
            Indices of selected keys (first t from cached_order)
        """
        n, d = queries.shape
        device = K.device
        dtype_param = K.dtype

        # Take first t indices from cached order
        indices = cached_order[:t]
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)

        # Compute exp_scores for NNLS
        inv_sqrt_d = (1.0 / d) ** 0.5
        scores_raw = queries @ K.T
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
        if attention_bias is not None:
            try:
                bias32 = torch.broadcast_to(
                    attention_bias.to(torch.float32),
                    scores32.shape
                )
                scores32 = scores32 + bias32
            except Exception as e:
                raise ValueError(
                    f"attention_bias must be broadcastable to {scores32.shape}, "
                    f"got {tuple(attention_bias.shape)}"
                ) from e
        max_scores = scores32.max(dim=1, keepdim=True)[0]
        exp_scores = torch.exp(scores32 - max_scores)
        target = exp_scores.sum(dim=1)

        # Build design matrix and solve NNLS
        M = exp_scores[:, indices_tensor]
        B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)
        beta32 = torch.log(B)

        # Extract selected keys
        C1 = K[indices_tensor]
        beta = beta32.to(dtype_param)

        return C1, beta, indices

    def _solve_nnls(
        self,
        M: torch.Tensor,
        target: torch.Tensor,
        prev_B: torch.Tensor,
        iteration: int,
        nnls_interval: Optional[int] = None,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Unified NNLS solver that supports both eager and lazy solving modes.

        Modes:
        - Eager (default): Always solve NNLS at every iteration (nnls_interval=1)
        - Lazy: Conditionally solve based on interval-based triggering

        Parameters
        ----------
        M : Tensor, shape (n, i)
            Design matrix with i selected columns
        target : Tensor, shape (n,)
            Target vector
        prev_B : Tensor, shape (i-1,), optional
            Previous solution (if available)
        iteration : int
            Current iteration number
        nnls_interval : int, optional
            Override for nnls_interval (used by progressive schedule).
            If None, uses self.nnls_interval.
        debug : bool
            Enable debug printing

        Returns
        -------
        B : Tensor, shape (i,)
            Solution vector
        solved : bool
            Whether NNLS was actually solved (True) or solution was reused/updated (False)
        """
        # Use provided interval or fall back to instance default
        interval = nnls_interval if nnls_interval is not None else self.nnls_interval

        # Determine if we should solve based on the configured mode
        should_solve = False

        if prev_B is None:
            # Always solve if no previous solution exists
            should_solve = True
        elif interval is not None:
            # Interval-based triggering
            should_solve = (iteration % interval == 0)

        # Solve or reuse based on the decision
        if should_solve:
            B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound, debug=debug)
            return B, True
        else:
            # Reuse previous solution and extend with lower-bound values for new entries
            i = M.shape[1]
            prev_i = prev_B.shape[0]
            B = torch.zeros(i, dtype=torch.float32, device=M.device)
            B[:prev_i] = prev_B
            min_val = 1e-12 if self.nnls_lower_bound is None else self.nnls_lower_bound
            B[prev_i:] = min_val
            return B, False

    def _select_keys_omp(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        k_choice: int = 1,
        attention_bias: torch.Tensor = None,
    ):
        """
        Greedy selection of t keys from K using an orthogonal-matching-pursuit-style
        approximation of the log-sum-exp over queries.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix.
        queries : Tensor, shape (n, d)
            Sampled query vectors.
        t : int
            Number of keys to select for the compacted cache.
        k_choice : int
            Number of keys to select at once (chunking). Selects top k_choice candidates
            by correlation magnitude with LSE residual.
        attention_bias : Tensor, optional
            Additive attention bias for the original cache (broadcastable to (n, T)).

        Returns
        -------
        C1 : Tensor, shape (num_selected, d)
            Selected keys (atoms) from K. Usually num_selected=t, but may be less if
            drop_key_beta_cutoff causes all remaining keys to be exhausted.
        beta : Tensor, shape (num_selected,)
            Bias terms for each selected key.
        indices : list of int
            Indices of the selected keys in the original K. Length equals num_selected.
        """
        # Shapes
        n, d = queries.shape
        T = K.shape[0]
        device = K.device
        dtype_param = K.dtype
        # QK matmul in original dtype; upcast only for scale + exp
        inv_sqrt_d = (1.0 / d) ** 0.5
        scores_raw = queries @ K.T                               # (n, T) original dtype
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d     # (n, T) fp32
        if attention_bias is not None:
            try:
                bias32 = torch.broadcast_to(
                    attention_bias.to(torch.float32),
                    scores32.shape
                )
                scores32 = scores32 + bias32                     # (n, T)
            except Exception as e:
                raise ValueError(
                    f"attention_bias must be broadcastable to {scores32.shape}, "
                    f"got {tuple(attention_bias.shape)}"
                ) from e
        max_scores = scores32.max(dim=1, keepdim=True)[0]        # (n, 1) fp32
        exp_scores = torch.exp(scores32 - max_scores)            # (n, T) fp32
        target = exp_scores.sum(dim=1)                           # (n,)  fp32

        # Pre-allocate tensor for indices (avoid repeated list->tensor conversion)
        selected_indices_tensor = torch.zeros(t, dtype=torch.long, device=device)
        beta32 = torch.zeros(t, dtype=torch.float32, device=device)
        current = torch.zeros_like(target)                      # (n,) fp32
        mask_selected = torch.zeros(T, dtype=torch.bool, device=device)
        mask_excluded = torch.zeros(T, dtype=torch.bool, device=device)  # Track permanently excluded keys

        i = 0
        prev_B = None  # Track previous solution for lazy NNLS
        iteration = 0
        refinement_count = 0  # Track number of refinement phase entries

        while True:
            # Phase 1: Initial selection (i < t)
            # Phase 2: Refinement phase (i == t, but may drop and re-add keys)

            if i < t:
                # Normal selection phase
                residual = target - current                         # (n,)

                # Vectorized correlation with residual, O(T)
                # Normalize exp_scores columns if requested (only for selection, not for M matrix)
                if self.normalize_exp_scores:
                    exp_scores_norm = torch.norm(exp_scores, dim=0, keepdim=True)  # (1, T)
                    exp_scores_normalized = exp_scores / (exp_scores_norm + 1e-12)  # (n, T)
                    corr = (exp_scores_normalized * residual.unsqueeze(1)).sum(dim=0)  # (T,)
                else:
                    corr = (exp_scores * residual.unsqueeze(1)).sum(dim=0)  # (T,)

                # Get k_choice and nnls_interval based on progressive schedule (if any)
                current_k_choice, current_nnls_interval = self._get_schedule_params(i)

                # Select top k_choice candidates by correlation magnitude
                # Count keys that are neither selected nor excluded
                num_remaining = T - (mask_selected | mask_excluded).sum().item()
                k_select = min(current_k_choice, num_remaining, t - i)  # Don't exceed remaining slots

                if k_select == 0:
                    # No more keys available - return what we have
                    if self.debug:
                        print(f"[OMP Debug] Iteration {iteration}: Ran out of keys. Returning {i}/{t} keys.")
                    break

                # Get top k_select by absolute correlation magnitude or correlation magnitude
                if self.use_abs_corr:
                    corr_abs = torch.abs(corr)
                else:
                    corr_abs = corr

                # Always mask already-selected and excluded indices after any absolute-value transform
                # so they don't re-enter the pool (abs(-inf) = inf would otherwise break this).
                corr_abs[mask_selected] = -float('inf')
                corr_abs[mask_excluded] = -float('inf')

                top_k_values, top_k_indices = torch.topk(corr_abs, k_select, largest=True)  # (k_select,)
                # Track newly selected indices for lazy NNLS
                new_indices_list = []
                prev_i = i
                # Add selected indices, avoiding duplicates
                for idx in top_k_indices:
                    if i >= t:
                        break
                    # Skip if already selected (shouldn't happen due to mask, but check for safety)
                    if mask_selected[idx]:
                        continue
                    selected_indices_tensor[i] = idx
                    mask_selected[idx] = True
                    new_indices_list.append(idx.item())
                    i += 1

                # Build design matrix M and solve NNLS lazily:  min ||M B - target||^2, B >= 0
                M = exp_scores[:, selected_indices_tensor[:i]]    # (n, i) fp32

                # Unified NNLS solving with different triggering conditions
                B, solved = self._solve_nnls(
                    M, target, prev_B, iteration,
                    nnls_interval=current_nnls_interval, debug=self.debug
                )  # (i,) fp32 (>=0)
                prev_B = B  # Update for next iteration

                # Debug: count number of non-zero elements and print statistics
                if self.debug:
                    tolerance = 1e-6
                    non_zero_B = torch.where(B > tolerance)[0].sum().item()
                    print(f"[OMP Debug] Iteration {iteration}: non_zero_B={non_zero_B}, max B={B.max().item():.6f}, min B={B.min().item():.6e}")

                beta32[:i] = torch.log(B)
                current = M @ B
                iteration += 1

            elif i == t and self.drop_key_beta_cutoff is not None:
                # Refinement phase: we have exactly t keys, now check if any should be dropped
                refinement_count += 1
                if self.debug:
                    print(f"[OMP Debug] Refinement phase entry {refinement_count} at iteration {iteration}")

                # Exit if we've entered refinement too many times without converging
                if refinement_count > 3:
                    if self.debug:
                        print(f"[OMP Debug] Refinement limit reached (3 entries). Exiting with {i} keys.")
                    break

                # Always solve NNLS in refinement phase to get accurate beta values for drop decision
                # (ignore nnls_interval here - we need accurate coefficients to decide which keys to drop)
                M = exp_scores[:, selected_indices_tensor[:i]]
                B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound, debug=self.debug)
                prev_B = B

                # Debug statistics
                if self.debug:
                    tolerance = 1e-6
                    non_zero_B = torch.where(B > tolerance)[0].sum().item()
                    print(f"[OMP Debug] Refinement iteration {iteration}: non_zero_B={non_zero_B}, max B={B.max().item():.6f}, min B={B.min().item():.6e}")

                beta32[:i] = torch.log(B)

                # Check for keys to drop based on beta cutoff
                keys_to_drop_mask = beta32[:i] < self.drop_key_beta_cutoff
                num_keys_to_drop = keys_to_drop_mask.sum().item()

                if num_keys_to_drop > 0:
                    if self.debug:
                        print(f"[OMP Debug] Refinement iteration {iteration}: Dropping {num_keys_to_drop} keys with beta < {self.drop_key_beta_cutoff}")

                    # Get indices of keys to drop
                    drop_positions = torch.where(keys_to_drop_mask)[0]
                    keys_to_exclude = selected_indices_tensor[drop_positions]

                    # Mark these keys as permanently excluded
                    mask_excluded[keys_to_exclude] = True

                    # Remove dropped keys from selected set
                    keep_mask = ~keys_to_drop_mask
                    keep_positions = torch.where(keep_mask)[0]
                    num_kept = keep_positions.shape[0]

                    # Compact the selected indices and beta arrays
                    selected_indices_tensor[:num_kept] = selected_indices_tensor[keep_positions]
                    beta32[:num_kept] = beta32[keep_positions]

                    # Update mask_selected to reflect dropped keys
                    mask_selected[keys_to_exclude] = False

                    # Update i to reflect the new number of selected keys
                    i = num_kept

                    # Update current approximation
                    M = exp_scores[:, selected_indices_tensor[:i]]
                    B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound, debug=self.debug)
                    current = M @ B

                    # Reset prev_B to None since we're going back to phase 1 with a different key set
                    # The B we just computed is for the reduced set, but after selecting more keys,
                    # the indices will be different, so we can't reuse it
                    prev_B = None

                    iteration += 1
                    # Continue loop - we'll go back to i < t and select more keys
                    continue
                else:
                    # No keys to drop - we have a stable set of t keys
                    if self.debug:
                        print(f"[OMP Debug] Refinement converged at iteration {iteration} with {i} stable keys")
                    break
            else:
                # i == t and no drop_key_beta_cutoff configured
                # Do a final NNLS solve if we may have skipped it due to nnls_interval
                if self.nnls_interval > 1:
                    M = exp_scores[:, selected_indices_tensor[:i]]
                    B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound, debug=self.debug)
                    beta32[:i] = torch.log(B)
                break

        # Convert to list only at the end (single GPU->CPU transfer)
        # Use only the first i valid indices (in case refinement phase dropped keys)
        selected_indices = selected_indices_tensor[:i].cpu().tolist()
        C1 = K[selected_indices_tensor[:i]]   # original dtype
        # Convert beta from fp32 to model dtype (e.g., bf16) for storage
        beta = beta32[:i].to(dtype_param)
        return C1, beta, selected_indices

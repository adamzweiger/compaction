# compaction/algorithms/optim_batched.py
"""
Batched gradient-based optimization algorithms for KV cache compaction.

Implements batched OptimJointCompaction that processes multiple (layer, head) pairs simultaneously.
"""
import torch
from typing import Tuple
from .batched import BatchedCompactionAlgorithm


class BatchedOptimJointCompaction(BatchedCompactionAlgorithm):
    """
    Batched version of OptimJointCompaction that jointly optimizes C1, beta, and C2 for multiple instances.

    Loss: L = ||softmax(qK^T)V - softmax(qC1^T + beta)C2||^2
              + lambda * (logsumexp(qK^T) - logsumexp(qC1^T + beta))^2
              + lambda_l2 * (||C1||^2 + ||beta||^2 + ||C2||^2)

    Processes all (layer, head) pairs simultaneously for better GPU utilization.
    """

    def __init__(
        self,
        lr: float,
        num_steps: int,
        lam: float,
        patience: int,
        optimizer: str = 'adam',
        lam_l2: float = 0.0,
        use_lr_decay: bool = True,
        eta_min: float = 0.0,
        adam_steps: int = 5000,
        lbfgs_steps: int = 5000
    ):
        """
        Parameters
        ----------
        lr : float
            Learning rate for gradient descent
        num_steps : int
            Maximum number of optimization steps
        lam : float
            Regularization weight for partition function matching
        patience : int
            Early stopping patience
        optimizer : str
            Optimizer to use ('adam', 'lbfgs', or 'adam_lbfgs'). Default is 'adam'.
            If 'adam_lbfgs', uses Adam for adam_steps then switches to LBFGS for lbfgs_steps.
        lam_l2 : float
            Regularization weight for L2 regularization on C1, beta, and C2. Default is 0.0.
        use_lr_decay : bool
            Whether to use cosine annealing learning rate decay. Default is True.
        eta_min : float
            Minimum learning rate for cosine annealing. Default is 0.0.
        adam_steps : int
            Number of Adam optimization steps when using 'adam_lbfgs' optimizer. Default is 5000.
        lbfgs_steps : int
            Number of LBFGS optimization steps when using 'adam_lbfgs' optimizer. Default is 5000.
        """
        self.lr = lr
        self.num_steps = num_steps
        self.lam = lam
        self.patience = patience
        self.optimizer_type = optimizer.lower()
        self.lam_l2 = lam_l2
        self.use_lr_decay = use_lr_decay
        self.eta_min = eta_min
        self.adam_steps = adam_steps
        self.lbfgs_steps = lbfgs_steps

    def name(self) -> str:
        return f"BatchedOptimJointCompaction_lr{self.lr}_steps{self.num_steps}_lam{self.lam}_laml2{self.lam_l2}"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Non-batched interface wrapper for compatibility.

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
            Empty list (no discrete selection for optim methods)
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
        C1 = C1_batched[0]  # (t, d)
        beta = beta_batched[0]  # (t,)
        C2 = C2_batched[0]  # (t, d)
        indices = indices_batched[0].cpu().tolist()  # list of int (all zeros for optim methods)

        return C1, beta, C2, indices

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
            Optimized compacted keys
        beta : Tensor, shape (B, t)
            Optimized bias terms
        C2 : Tensor, shape (B, t, d)
            Optimized compacted values
        indices : Tensor, shape (B, t)
            Empty tensor (no discrete selection)
        """
        C1, beta, C2 = self._optimize_joint_batched(K, V, queries, t)

        # Return empty indices tensor
        B = K.shape[0]
        indices = torch.zeros(B, t, dtype=torch.long, device=K.device)

        return C1, beta, C2, indices

    def _optimize_joint_batched(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Jointly optimize C1, beta, and C2 for all instances simultaneously.

        Parameters
        ----------
        K : Tensor, shape (B, T, d)
            Original key matrices
        V : Tensor, shape (B, T, d)
            Original value matrices
        queries : Tensor, shape (B, n, d)
            Query samples
        t : int
            Number of compacted keys

        Returns
        -------
        C1 : Tensor, shape (B, t, d)
            Optimized keys
        beta : Tensor, shape (B, t)
            Optimized bias terms
        C2 : Tensor, shape (B, t, d)
            Optimized values
        """
        B, T, d = K.shape
        n = queries.shape[1]
        device = K.device
        dtype_param = K.dtype
        inv_sqrt_d = (1.0 / d) ** 0.5

        # Initialize C1 by randomly selecting t keys from K for each instance
        init_indices = torch.stack([torch.randperm(T, device=device)[:t] for _ in range(B)])  # (B, t)
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, t)  # (B, t)
        C1 = K[batch_indices, init_indices].clone().to(torch.float32).contiguous()  # (B, t, d)
        beta = torch.zeros(B, t, dtype=torch.float32, device=device).contiguous()  # (B, t)

        # Initialize C2 using ridge regression (warm start)
        with torch.no_grad():
            C2_init = self._compute_C2_batched(
                C1.to(dtype_param),
                beta,
                K,
                V,
                queries
            )
        C2 = C2_init.clone().to(torch.float32).contiguous()  # (B, t, d)

        # Make them require gradients
        C1.requires_grad_(True)
        beta.requires_grad_(True)
        C2.requires_grad_(True)

        # Precompute target attention outputs
        # scores_K: (B, n, T)
        scores_K_raw = torch.bmm(queries, K.transpose(1, 2))  # (B, n, T) original dtype
        scores_K = scores_K_raw.to(torch.float32) * inv_sqrt_d  # (B, n, T) fp32

        # For numerical stability, use max-normalized softmax
        max_scores_K = scores_K.max(dim=2, keepdim=True)[0]  # (B, n, 1)
        exp_scores_K = torch.exp(scores_K - max_scores_K)  # (B, n, T)
        sum_exp_K = exp_scores_K.sum(dim=2, keepdim=True)  # (B, n, 1)
        attn_weights_K = exp_scores_K / sum_exp_K  # (B, n, T)
        target_output = torch.bmm(attn_weights_K, V.to(torch.float32))  # (B, n, d) fp32

        # Also precompute target logsumexp
        target_lse = torch.logsumexp(scores_K, dim=2)  # (B, n) fp32

        # Helper function to compute loss
        def compute_loss():
            # Compute predicted attention output
            # scores_C1: (B, n, t)
            scores_C1_raw = torch.bmm(queries.to(torch.float32), C1.transpose(1, 2))  # (B, n, t) fp32
            scores_C1 = scores_C1_raw * inv_sqrt_d + beta.unsqueeze(1)  # (B, n, t) fp32

            # Numerical stability: use max normalization
            max_scores_C1 = scores_C1.max(dim=2, keepdim=True)[0]  # (B, n, 1)
            exp_scores_C1 = torch.exp(scores_C1 - max_scores_C1)  # (B, n, t)
            sum_exp_C1 = exp_scores_C1.sum(dim=2, keepdim=True)  # (B, n, 1)
            attn_weights_C1 = exp_scores_C1 / sum_exp_C1  # (B, n, t)
            pred_output = torch.bmm(attn_weights_C1, C2)  # (B, n, d) fp32

            # Output reconstruction loss (mean over batch, queries, and dimensions)
            output_loss = torch.mean((target_output - pred_output) ** 2)

            # Partition function matching loss (using logsumexp)
            pred_lse = torch.logsumexp(scores_C1, dim=2)  # (B, n) fp32
            partition_loss = torch.mean((target_lse - pred_lse) ** 2)

            # L2 regularization (mean over batch and elements)
            reg_loss = self.lam_l2 * (torch.mean(C1 ** 2) + torch.mean(beta ** 2) + torch.mean(C2 ** 2))

            # Total loss
            loss = output_loss + self.lam * partition_loss + reg_loss
            return loss, output_loss, partition_loss

        best_loss = float('inf')
        no_improve_count = 0
        iteration = 0

        if self.optimizer_type == 'adam_lbfgs':
            # Two-stage optimization: Adam first, then LBFGS
            # Stage 1: Adam optimization
            if self.adam_steps > 0:
                print(f"Starting Adam optimization for {self.adam_steps} steps...")
                optimizer_adam = torch.optim.Adam([C1, beta, C2], lr=self.lr)
                # Create cosine annealing scheduler if enabled
                if self.use_lr_decay:
                    scheduler_adam = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer_adam, T_max=self.adam_steps, eta_min=self.eta_min
                    )
                else:
                    scheduler_adam = None

                # Run Adam optimization
                for step in range(self.adam_steps):
                    optimizer_adam.zero_grad()
                    loss, output_loss, partition_loss = compute_loss()

                    if step % 1000 == 0:
                        print(f"Adam Step {step}: Total loss: {loss.item():.6e}, Output loss: {output_loss.item():.6e}, Partition loss: {partition_loss.item():.6e}")

                    loss.backward()
                    optimizer_adam.step()

                    # Step scheduler if available
                    if scheduler_adam is not None:
                        scheduler_adam.step()

                    # Early stopping check
                    current_loss = loss.item()
                    if current_loss < best_loss - 1e-6:
                        best_loss = current_loss
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if no_improve_count >= self.patience:
                            print(f"Early stopping at Adam step {step}")
                            break

            # Stage 2: LBFGS optimization
            if self.lbfgs_steps > 0:
                print(f"Switching to LBFGS optimization for {self.lbfgs_steps} steps...")
                # Reset early stopping counters for LBFGS phase
                best_loss = float('inf')
                no_improve_count = 0

                optimizer_lbfgs = torch.optim.LBFGS([C1, beta, C2], lr=self.lr, max_iter=5, history_size=10)

                def closure():
                    optimizer_lbfgs.zero_grad()
                    loss, _, _ = compute_loss()
                    loss.backward()
                    return loss

                # Run LBFGS optimization
                while iteration < self.lbfgs_steps and no_improve_count < self.patience:
                    optimizer_lbfgs.step(closure)

                    # Check loss after this iteration
                    with torch.no_grad():
                        loss, output_loss, partition_loss = compute_loss()
                        current_loss = loss.item()

                    if iteration % 1000 == 0:
                        print(f"LBFGS Step {iteration}: Total loss: {current_loss:.6e}, Output loss: {output_loss.item():.6e}, Partition loss: {partition_loss.item():.6e}")

                    # Early stopping check
                    if current_loss < best_loss - 1e-6:
                        best_loss = current_loss
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    iteration += 1

        elif self.optimizer_type == 'lbfgs':
            # Create optimizer
            optimizer = torch.optim.LBFGS([C1, beta, C2], lr=self.lr, max_iter=5, history_size=10)
            scheduler = None  # LBFGS doesn't work well with schedulers

            # LBFGS requires a closure function
            def closure():
                optimizer.zero_grad()
                loss, _, _ = compute_loss()
                loss.backward()
                return loss

            # Run LBFGS optimization
            while iteration < self.num_steps and no_improve_count < self.patience:
                optimizer.step(closure)

                # Check loss after this iteration
                with torch.no_grad():
                    loss, output_loss, partition_loss = compute_loss()
                    current_loss = loss.item()

                if iteration % 1000 == 0:
                    print(f"Step {iteration}: Total loss: {current_loss:.6e}, Output loss: {output_loss.item():.6e}, Partition loss: {partition_loss.item():.6e}")

                # Early stopping check
                if current_loss < best_loss - 1e-9:
                    best_loss = current_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                iteration += 1

                # Step scheduler if available (for LBFGS, step after each outer iteration)
                if scheduler is not None:
                    scheduler.step()
        else:
            # Adam optimizer loop
            optimizer = torch.optim.Adam([C1, beta, C2], lr=self.lr)
            # Create cosine annealing scheduler if enabled
            if self.use_lr_decay:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.num_steps, eta_min=self.eta_min
                )
            else:
                scheduler = None

            for step in range(self.num_steps):
                optimizer.zero_grad()
                loss, output_loss, partition_loss = compute_loss()

                if step % 1000 == 0:
                    print(f"Step {step}: Total loss: {loss.item():.6e}, Output loss: {output_loss.item():.6e}, Partition loss: {partition_loss.item():.6e}")

                loss.backward()
                optimizer.step()

                # Step scheduler if available
                if scheduler is not None:
                    scheduler.step()

                # Early stopping check
                current_loss = loss.item()
                if current_loss < best_loss - 1e-6:
                    best_loss = current_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= self.patience:
                        break

        # Convert back to original dtype
        C1_final = C1.detach().to(dtype_param)
        beta_final = beta.detach().to(dtype_param)
        C2_final = C2.detach().to(dtype_param)

        return C1_final, beta_final, C2_final

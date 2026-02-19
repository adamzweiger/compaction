# compaction/algorithms/optim.py
"""
Gradient-based optimization algorithms for KV cache compaction.

Implements two variants:
1. OptimC1BetaCompaction: Optimize C1 and beta to match partition function, then solve for C2
2. OptimJointCompaction: Jointly optimize C1, beta, and C2 with partition function regularization
"""
import torch
from typing import Tuple
from .base import CompactionAlgorithm


class OptimC1BetaCompaction(CompactionAlgorithm):
    """
    Optimize C1 and beta via gradient descent to match partition function.

    Loss: L = (logsumexp(qK^T) - logsumexp(qC1^T + beta))^2
              + lambda * (||C1||^2 + ||beta||^2)
    Then compute C2 using ridge regression as in base class.
    """

    def __init__(self, lr: float, num_steps: int, patience: int, optimizer: str = 'lbfgs', lam: float = 0.0, adam_steps: int = 500, lbfgs_steps: int = 500):
        """
        Parameters
        ----------
        lr : float
            Learning rate for gradient descent
        num_steps : int
            Maximum number of optimization steps
        patience : int
            Early stopping patience (stop if no improvement for this many steps)
        optimizer : str
            Optimizer to use ('adam' or 'lbfgs'). Default is 'lbfgs'.
        lam : float
            Regularization weight for L2 regularization on C1 and beta. Default is 0.0.
        adam_steps : int
            Number of Adam optimization steps when using 'adam_lbfgs' optimizer. Default is 500.
        lbfgs_steps : int
            Number of LBFGS optimization steps when using 'adam_lbfgs' optimizer. Default is 500.
        """
        self.lr = lr
        self.num_steps = num_steps
        self.patience = patience
        self.optimizer_type = optimizer.lower()
        self.lam = lam
        self.adam_steps = adam_steps
        self.lbfgs_steps = lbfgs_steps

    def name(self) -> str:
        return f"OptimC1BetaCompaction_lr{self.lr}_steps{self.num_steps}_lam{self.lam}"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Compute compacted cache using gradient-based optimization of C1 and beta.

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
            Optimized compacted keys
        beta : Tensor, shape (t,)
            Optimized bias terms
        C2 : Tensor, shape (t, d)
            Compacted values (solved via ridge regression)
        indices : list of int
            Empty list (no discrete selection)
        """
        # Optimize C1 and beta
        C1, beta = self._optimize_C1_beta(K, queries, t)

        # Compute C2 using ridge regression
        C2 = self._compute_C2(C1, beta, K, V, queries)

        return C1, beta, C2, []

    def _optimize_C1_beta(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimize C1 and beta to match partition function via gradient descent.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix
        queries : Tensor, shape (n, d)
            Query samples
        t : int
            Number of compacted keys

        Returns
        -------
        C1 : Tensor, shape (t, d)
            Optimized keys
        beta : Tensor, shape (t,)
            Optimized bias terms
        """
        n, d = queries.shape
        T = K.shape[0]
        device = K.device
        dtype_param = K.dtype
        inv_sqrt_d = (1.0 / d) ** 0.5

        # Initialize C1 by randomly selecting t keys from K
        init_indices = torch.randperm(T, device=device)[:t]
        C1 = K[init_indices].clone().to(torch.float32).contiguous()
        beta = torch.zeros(t, dtype=torch.float32, device=device).contiguous()

        # Make them require gradients
        C1.requires_grad_(True)
        beta.requires_grad_(True)

        # Precompute target logsumexp for original keys
        # QK matmul in original dtype, then upcast for softmax
        scores_K_raw = queries @ K.T  # (n, T) original dtype
        scores_K = scores_K_raw.to(torch.float32) * inv_sqrt_d  # (n, T) fp32
        target_lse = torch.logsumexp(scores_K, dim=1)  # (n,) fp32

        # Create optimizer
        if self.optimizer_type == 'lbfgs':
            optimizer = torch.optim.LBFGS([C1, beta], lr=self.lr, max_iter=20, history_size=100)
        else:
            optimizer = torch.optim.Adam([C1, beta], lr=self.lr)

        best_loss = float('inf')
        no_improve_count = 0
        iteration = 0

        if self.optimizer_type == 'lbfgs':
            # LBFGS requires a closure function
            def closure():
                nonlocal best_loss
                optimizer.zero_grad()
                
                # Compute logsumexp for compacted representation
                scores_C1_raw = queries.to(torch.float32) @ C1.T  # (n, t) fp32
                scores_C1 = scores_C1_raw * inv_sqrt_d + beta.unsqueeze(0)  # (n, t) fp32
                pred_lse = torch.logsumexp(scores_C1, dim=1)  # (n,) fp32
                
                # Loss: mean squared error of logsumexp
                # partition_loss = torch.mean((target_lse - pred_lse) ** 2)
                partition_loss = torch.max(torch.abs(target_lse - pred_lse))
                
                # L2 regularization
                reg_loss = self.lam * (torch.sum(C1 ** 2) + torch.sum(beta ** 2))
                
                loss = partition_loss + reg_loss
                
                loss.backward()
                return loss

            # Run LBFGS optimization
            while iteration < self.num_steps and no_improve_count < self.patience:
                optimizer.step(closure)
                
                # Check loss after this iteration
                with torch.no_grad():
                    scores_C1_raw = queries.to(torch.float32) @ C1.T  # (n, t) fp32
                    scores_C1 = scores_C1_raw * inv_sqrt_d + beta.unsqueeze(0)  # (n, t) fp32
                    pred_lse = torch.logsumexp(scores_C1, dim=1)  # (n,) fp32
                    partition_loss = torch.mean((target_lse - pred_lse) ** 2)
                    reg_loss = self.lam * (torch.sum(C1 ** 2) + torch.sum(beta ** 2))
                    current_loss = (partition_loss + reg_loss).item()
                
                # Early stopping check
                if current_loss < best_loss - 1e-6:
                    best_loss = current_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                iteration += 1
        else:
            # Adam optimizer loop
            for step in range(self.num_steps):
                optimizer.zero_grad()

                # Compute logsumexp for compacted representation
                # Q @ C1^T / sqrt(d) + beta
                scores_C1_raw = queries.to(torch.float32) @ C1.T  # (n, t) fp32
                scores_C1 = scores_C1_raw * inv_sqrt_d + beta.unsqueeze(0)  # (n, t) fp32
                pred_lse = torch.logsumexp(scores_C1, dim=1)  # (n,) fp32

                # Loss: mean squared error of logsumexp
                partition_loss = torch.mean((target_lse - pred_lse) ** 2)
                # partition_loss = torch.max(torch.abs(target_lse - pred_lse))
                
                # L2 regularization
                reg_loss = self.lam * (torch.sum(C1 ** 2) + torch.sum(beta ** 2))
                
                loss = partition_loss + reg_loss

                loss.backward()
                optimizer.step()

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
        # Convert beta from fp32 to model dtype (e.g., bf16) for storage
        beta_final = beta.detach().to(dtype_param)

        return C1_final, beta_final


class OptimJointCompaction(CompactionAlgorithm):
    """
    Jointly optimize C1, beta, and C2 via gradient descent.

    Loss: L = ||softmax(qK^T)V - softmax(qC1^T + beta)C2||^2
              + lambda * (logsumexp(qK^T) - logsumexp(qC1^T + beta))^2
              + lambda_l2 * (||C1||^2 + ||beta||^2 + ||C2||^2)
    """

    def __init__(
        self,
        lr: float,
        num_steps: int,
        lam: float,
        patience: int,
        optimizer: str = 'lbfgs',
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
            Optimizer to use ('adam', 'lbfgs', or 'adam_lbfgs'). Default is 'lbfgs'.
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
        return f"OptimJointCompaction_lr{self.lr}_steps{self.num_steps}_lam{self.lam}_laml2{self.lam_l2}"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Compute compacted cache using joint gradient-based optimization.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix
        V : Tensor, shape (T, d)
            Original value matrix
        queries : Tensor, shape (n, d)
            Query samples for training
        t : int
            Compacted size

        Returns
        -------
        C1 : Tensor, shape (t, d)
            Optimized compacted keys
        beta : Tensor, shape (t,)
            Optimized bias terms
        C2 : Tensor, shape (t, d)
            Optimized compacted values
        indices : list of int
            Empty list (no discrete selection)
        """
        C1, beta, C2 = self._optimize_joint(K, V, queries, t)
        return C1, beta, C2, []

    def _optimize_joint(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Jointly optimize C1, beta, and C2.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix
        V : Tensor, shape (T, d)
            Original value matrix
        queries : Tensor, shape (n, d)
            Query samples
        t : int
            Number of compacted keys

        Returns
        -------
        C1 : Tensor, shape (t, d)
            Optimized keys
        beta : Tensor, shape (t,)
            Optimized bias terms
        C2 : Tensor, shape (t, d)
            Optimized values
        """
        n, d = queries.shape
        T = K.shape[0]
        device = K.device
        dtype_param = K.dtype
        inv_sqrt_d = (1.0 / d) ** 0.5

        # Initialize C1 by randomly selecting t keys from K
        init_indices = torch.randperm(T, device=device)[:t]
        C1 = K[init_indices].clone().to(torch.float32).contiguous()
        beta = torch.zeros(t, dtype=torch.float32, device=device).contiguous()

        # Initialize C2 using ridge regression (warm start)
        with torch.no_grad():
            C2_init = self._compute_C2(
                C1.to(dtype_param),
                beta,
                K,
                V,
                queries
            )
        C2 = C2_init.clone().to(torch.float32).contiguous()

        # Make them require gradients
        C1.requires_grad_(True)
        beta.requires_grad_(True)
        C2.requires_grad_(True)

        # Precompute target attention outputs
        # exp((QK)/sqrt(d) - m) @ V / sum(exp(...))
        scores_K_raw = queries @ K.T  # (n, T) original dtype
        scores_K = scores_K_raw.to(torch.float32) * inv_sqrt_d  # (n, T) fp32

        # For numerical stability, use max-normalized softmax
        max_scores_K = scores_K.max(dim=1, keepdim=True)[0]  # (n, 1)
        exp_scores_K = torch.exp(scores_K - max_scores_K)  # (n, T)
        sum_exp_K = exp_scores_K.sum(dim=1, keepdim=True)  # (n, 1)
        attn_weights_K = exp_scores_K / sum_exp_K  # (n, T)
        target_output = attn_weights_K @ V.to(torch.float32)  # (n, d) fp32

        # Also precompute target logsumexp
        target_lse = torch.logsumexp(scores_K, dim=1)  # (n,) fp32

        # Helper function to compute loss
        def compute_loss():
            # Compute predicted attention output
            # exp((QC1^T)/sqrt(d) + beta - m) @ C2 / sum(exp(...))
            scores_C1_raw = queries.to(torch.float32) @ C1.T  # (n, t) fp32
            scores_C1 = scores_C1_raw * inv_sqrt_d + beta.unsqueeze(0)  # (n, t) fp32
            
            # Numerical stability: use max normalization
            max_scores_C1 = scores_C1.max(dim=1, keepdim=True)[0]  # (n, 1)
            exp_scores_C1 = torch.exp(scores_C1 - max_scores_C1)  # (n, t)
            sum_exp_C1 = exp_scores_C1.sum(dim=1, keepdim=True)  # (n, 1)
            attn_weights_C1 = exp_scores_C1 / sum_exp_C1  # (n, t)
            pred_output = attn_weights_C1 @ C2  # (n, d) fp32
            
            # Output reconstruction loss
            output_loss = torch.mean((target_output - pred_output) ** 2)
            
            # Partition function matching loss (using logsumexp)
            pred_lse = torch.logsumexp(scores_C1, dim=1)  # (n,) fp32
            partition_loss = torch.mean((target_lse - pred_lse) ** 2)
            
            # L2 regularization
            reg_loss = self.lam_l2 * (torch.sum(C1 ** 2) + torch.sum(beta ** 2) + torch.sum(C2 ** 2))
            
            # Total loss
            loss = output_loss + self.lam * partition_loss + reg_loss
            return loss, output_loss, partition_loss, scores_C1

        best_loss = float('inf')
        no_improve_count = 0
        iteration = 0

        if self.optimizer_type == 'adam_lbfgs':
            # Two-stage optimization: Adam first, then LBFGS
            # Stage 1: Adam optimization
            if self.adam_steps > 0:
                # print(f"Starting Adam optimization for {self.adam_steps} steps...")
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
                    loss, output_loss, partition_loss, _ = compute_loss()

                    if step%1000==0:
                        print(f"Adam Step {step}: Total loss: {loss.item()}, Output loss: {output_loss.item()}, Partition loss: {partition_loss.item()}")

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
                            # print(f"Early stopping at Adam step {step}")
                            break

            # Stage 2: LBFGS optimization
            if self.lbfgs_steps > 0:
                # print(f"Switching to LBFGS optimization for {self.lbfgs_steps} steps...")
                # Reset early stopping counters for LBFGS phase
                best_loss = float('inf')
                no_improve_count = 0

                optimizer_lbfgs = torch.optim.LBFGS([C1, beta, C2], lr=self.lr, max_iter=5, history_size=10)

                def closure():
                    optimizer_lbfgs.zero_grad()
                    loss, _, _, _ = compute_loss()
                    loss.backward()
                    return loss

                # Run LBFGS optimization
                while iteration < self.lbfgs_steps and no_improve_count < self.patience:
                    optimizer_lbfgs.step(closure)

                    # Check loss after this iteration
                    with torch.no_grad():
                        loss, output_loss, partition_loss, _ = compute_loss()
                        current_loss = loss.item()

                    if (iteration % 1000 == 0):
                        print(f"LBFGS Step {iteration}: Total loss: {current_loss}, Output loss: {output_loss.item()}, Partition loss: {partition_loss.item()}")

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
                loss, _, _, _ = compute_loss()
                loss.backward()
                return loss

            # Run LBFGS optimization
            while iteration < self.num_steps and no_improve_count < self.patience:
                optimizer.step(closure)

                # Check loss after this iteration
                with torch.no_grad():
                    loss, output_loss, partition_loss, _ = compute_loss()
                    current_loss = loss.item()

                if (iteration % 1000 == 0):
                    print(f"Step {iteration}: Total loss: {current_loss}, Output loss: {output_loss.item()}, Partition loss: {partition_loss.item()}")

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
                loss, output_loss, partition_loss, _ = compute_loss()

                if (step % 1000 == 0):
                    print(f"Step {step}: Total loss: {loss.item()}, Output loss: {output_loss.item()}, Partition loss: {partition_loss.item()}")

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
        # Convert beta from fp32 to model dtype (e.g., bf16) for storage
        beta_final = beta.detach().to(dtype_param)
        C2_final = C2.detach().to(dtype_param)

        return C1_final, beta_final, C2_final

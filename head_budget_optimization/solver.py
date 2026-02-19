# head_budget_optimization/solver.py
"""
Greedy solver for optimal head budget allocation.

Given per-head influence curves, this module finds the optimal allocation
of budget across heads to minimize perplexity for a given total compaction ratio.
"""
import json
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


class HeadBudgetSolver:
    """
    Greedy solver that allocates budget to heads based on influence curves.

    The solver works by starting from zero allocation and greedily adding
    budget to the head that provides the greatest log perplexity reduction
    at each step.

    Note: Only global (non-sliding window) layers are optimized. The solver
    infers which layers are global based on the head_curves keys provided.
    """

    def __init__(
        self,
        head_curves: Dict[str, List[Tuple[float, float]]],
        num_layers: int,
        num_heads: int,
        smoothing_window: int = 0,
    ):
        """
        Initialize the solver.

        Parameters
        ----------
        head_curves : dict
            Mapping from head key (e.g., "L0H0") to influence curve.
            Each curve is a list of (ratio, delta_log_perplexity) tuples,
            where ratio is the proportion of original keys (0 to 1),
            and delta is the change in log(perplexity) relative to baseline.
            Note: Only heads from global (non-sliding) layers should be included.
        num_layers : int
            Total number of layers in the model (including sliding)
        num_heads : int
            Number of KV heads per layer
        smoothing_window : int
            Window size for sliding window smoothing of curves (default: 0 = no smoothing).
            Smoothed curves are used for decision-making, but final loss is computed
            on original curves.
        """
        self.head_curves = head_curves
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.smoothing_window = smoothing_window

        # Infer global layer indices from the head_curves keys
        self.global_layer_indices = set()
        for head_key in head_curves.keys():
            layer_idx = int(head_key[1:].split('H')[0])
            self.global_layer_indices.add(layer_idx)
        self.global_layer_indices = sorted(self.global_layer_indices)
        self.num_global_layers = len(self.global_layer_indices)

        # Total heads = only global layer heads
        self.total_heads = self.num_global_layers * num_heads

        # Pre-process curves for efficient interpolation
        self._prepare_interpolators()

    def _prepare_interpolators(self):
        """Prepare interpolation data for each head curve."""
        self.interpolators = {}
        self.smoothed_interpolators = {}
        self.head_max_ratios = {}  # Track max evaluated ratio per head

        for head_key, curve in self.head_curves.items():
            # Sort by ratio (should already be sorted, but ensure it)
            sorted_curve = sorted(curve, key=lambda x: x[0])
            ratios = np.array([p[0] for p in sorted_curve])
            deltas = np.array([p[1] for p in sorted_curve])

            self.interpolators[head_key] = {
                'ratios': ratios,
                'deltas': deltas,
            }
            # Store the max ratio this head was evaluated at
            self.head_max_ratios[head_key] = float(ratios[-1])

            # Create smoothed interpolators if smoothing is enabled
            if self.smoothing_window > 0:
                smoothed_deltas = self._sliding_window_smooth(deltas, self.smoothing_window)
                self.smoothed_interpolators[head_key] = {
                    'ratios': ratios,
                    'deltas': smoothed_deltas,
                }

    def _sliding_window_smooth(self, values: np.ndarray, window: int) -> np.ndarray:
        """Apply sliding window averaging to smooth values."""
        if window <= 1 or len(values) <= 1:
            return values.copy()

        smoothed = np.zeros_like(values)
        half_window = window // 2

        for i in range(len(values)):
            start = max(0, i - half_window)
            end = min(len(values), i + half_window + 1)
            smoothed[i] = np.mean(values[start:end])

        return smoothed

    def interpolate_delta(self, head_key: str, ratio: float) -> float:
        """
        Interpolate the delta log perplexity for a head at a given ratio.

        Parameters
        ----------
        head_key : str
            Head identifier (e.g., "L0H0")
        ratio : float
            Ratio to interpolate at (0 to 1)

        Returns
        -------
        delta : float
            Interpolated delta log perplexity
        """
        interp = self.interpolators[head_key]
        ratios = interp['ratios']
        deltas = interp['deltas']

        # Clamp ratio to valid range
        ratio = max(ratios[0], min(ratios[-1], ratio))

        # Linear interpolation
        return float(np.interp(ratio, ratios, deltas))

    def interpolate_delta_smoothed(self, head_key: str, ratio: float) -> float:
        """
        Interpolate using smoothed curves (for decision-making).
        Falls back to original if no smoothing is enabled.
        """
        if self.smoothing_window > 0 and head_key in self.smoothed_interpolators:
            interp = self.smoothed_interpolators[head_key]
        else:
            interp = self.interpolators[head_key]

        ratios = interp['ratios']
        deltas = interp['deltas']
        ratio = max(ratios[0], min(ratios[-1], ratio))
        return float(np.interp(ratio, ratios, deltas))

    def interpolate_marginal_benefit(
        self,
        head_key: str,
        current_ratio: float,
        delta_ratio: float,
    ) -> float:
        """
        Compute the marginal benefit (reduction in delta) of increasing a head's ratio.

        Uses smoothed curves if smoothing is enabled (for better decision-making).

        Parameters
        ----------
        head_key : str
            Head identifier
        current_ratio : float
            Current ratio for this head
        delta_ratio : float
            Amount to increase the ratio by

        Returns
        -------
        marginal_benefit : float
            The reduction in delta log perplexity (positive = improvement).
            Computed as delta(current) - delta(current + delta_ratio).
        """
        current_delta = self.interpolate_delta_smoothed(head_key, current_ratio)
        new_delta = self.interpolate_delta_smoothed(head_key, current_ratio + delta_ratio)

        # Benefit is the reduction in delta (which reduces perplexity)
        # A negative delta means better than baseline, so we want to minimize delta
        return current_delta - new_delta

    def interpolate_marginal_cost(
        self,
        head_key: str,
        current_ratio: float,
        delta_ratio: float,
    ) -> float:
        """
        Compute the marginal cost (increase in delta) of decreasing a head's ratio.

        Uses smoothed curves if smoothing is enabled (for better decision-making).

        Parameters
        ----------
        head_key : str
            Head identifier
        current_ratio : float
            Current ratio for this head
        delta_ratio : float
            Amount to decrease the ratio by (positive value)

        Returns
        -------
        marginal_cost : float
            The increase in delta log perplexity (positive = worse).
            Computed as delta(current - delta_ratio) - delta(current).
        """
        current_delta = self.interpolate_delta_smoothed(head_key, current_ratio)
        new_delta = self.interpolate_delta_smoothed(head_key, current_ratio - delta_ratio)

        # Cost is the increase in delta from removing budget
        return new_delta - current_delta

    def solve_greedy(
        self,
        target_total_ratio: float,
        step_size: float = 0.001,
        min_ratio_per_head: float = 0.0,
        max_ratio_per_head: float = 1.0,
    ) -> Dict[str, float]:
        """
        Greedily allocate budget to minimize perplexity.

        Starting from min_ratio_per_head for each head, iteratively allocate
        budget to the head with the greatest marginal benefit until the
        target total ratio is reached.

        Parameters
        ----------
        target_total_ratio : float
            Target total compaction ratio (e.g., 0.05 means 5% of original keys).
            This is the average ratio across all heads.
        step_size : float
            Step size for greedy allocation (default: 0.001)
        min_ratio_per_head : float
            Minimum ratio per head (default: 0.0)
        max_ratio_per_head : float
            Maximum ratio per head (default: 1.0)

        Returns
        -------
        head_ratios : dict
            Mapping from head key to allocated ratio (0 to 1 range,
            representing proportion of original keys for that head)
        """
        # Initialize all global layer heads at minimum ratio
        head_ratios = {
            f"L{l}H{h}": min_ratio_per_head
            for l in self.global_layer_indices
            for h in range(self.num_heads)
        }

        # Current total ratio (sum of all head ratios / num_heads = average ratio)
        current_total = sum(head_ratios.values()) / self.total_heads

        # Target total is the average ratio we want
        target = target_total_ratio

        print(f"Greedy allocation: target={target:.4f}, step_size={step_size:.4f}")
        print(f"Starting from {current_total:.4f} average ratio")

        iteration = 0
        while current_total < target:
            # Find head with greatest marginal benefit
            best_head = None
            best_benefit = float('-inf')

            for head_key in head_ratios:
                current_ratio = head_ratios[head_key]

                # Skip if already at max (use per-head max from curve data)
                head_max = min(max_ratio_per_head, self.head_max_ratios.get(head_key, 1.0))
                if current_ratio >= head_max:
                    continue

                # Compute marginal benefit of adding step_size to this head
                benefit = self.interpolate_marginal_benefit(
                    head_key, current_ratio, step_size
                )

                if benefit > best_benefit:
                    best_benefit = benefit
                    best_head = head_key

            if best_head is None:
                print(f"Warning: All heads at max ratio, stopping at {current_total:.4f}")
                break

            # Allocate budget to best head (respect per-head max from curve data)
            best_head_max = min(max_ratio_per_head, self.head_max_ratios.get(best_head, 1.0))
            head_ratios[best_head] = min(
                head_ratios[best_head] + step_size,
                best_head_max
            )

            current_total = sum(head_ratios.values()) / self.total_heads
            iteration += 1

            if iteration % 100 == 0:
                print(f"  Iteration {iteration}: total={current_total:.4f}, "
                      f"last allocated to {best_head} (benefit={best_benefit:.6f})")

        print(f"Completed in {iteration} iterations, final total={current_total:.4f}")

        return head_ratios

    def solve_swap(
        self,
        target_total_ratio: float,
        step_size: float = 0.001,
        min_ratio_per_head: float = 0.0,
        max_ratio_per_head: float = 1.0,
        max_iterations: int = 100000,
    ) -> Dict[str, float]:
        """
        Solve by starting at uniform allocation and swapping budget between heads.

        This approach addresses the U-shaped curve problem where greedy-from-zero
        gets stuck at local minima. By starting at uniform allocation, all heads
        begin past any initial "bump" in their curves.

        At each iteration:
        1. Find the head with the best forward marginal benefit (recipient)
        2. Find the head with the lowest backward marginal cost (donor)
        3. If net gain (benefit - cost) > 0, transfer budget from donor to recipient
        4. Stop when no beneficial swap exists

        Parameters
        ----------
        target_total_ratio : float
            Target total compaction ratio (average ratio across all heads).
        step_size : float
            Step size for budget transfers (default: 0.001)
        min_ratio_per_head : float
            Minimum ratio per head (default: 0.0)
        max_ratio_per_head : float
            Maximum ratio per head (default: 1.0)
        max_iterations : int
            Maximum iterations to prevent infinite loops (default: 100000)

        Returns
        -------
        head_ratios : dict
            Mapping from head key to allocated ratio
        """
        # Initialize all global layer heads at uniform ratio = target_total_ratio
        head_ratios = {
            f"L{l}H{h}": target_total_ratio
            for l in self.global_layer_indices
            for h in range(self.num_heads)
        }

        # Clamp initial values to valid range for each head
        for head_key in head_ratios:
            head_max = min(max_ratio_per_head, self.head_max_ratios.get(head_key, 1.0))
            head_ratios[head_key] = max(min_ratio_per_head, min(head_max, head_ratios[head_key]))

        # Compute initial total log perplexity delta (loss)
        initial_loss = sum(
            self.interpolate_delta(head_key, ratio)
            for head_key, ratio in head_ratios.items()
        )

        print(f"Swap-based allocation: target={target_total_ratio:.4f}, step_size={step_size:.4f}")
        print(f"Starting from uniform allocation at {target_total_ratio:.4f} per head")
        print(f"Initial total delta (loss): {initial_loss:.6f}")

        iteration = 0
        no_improvement_count = 0

        while iteration < max_iterations:
            # Find best recipient (head that benefits most from +step_size)
            best_recipient = None
            best_benefit = float('-inf')

            for head_key in head_ratios:
                current_ratio = head_ratios[head_key]
                head_max = min(max_ratio_per_head, self.head_max_ratios.get(head_key, 1.0))

                # Skip if already at max
                if current_ratio >= head_max:
                    continue

                benefit = self.interpolate_marginal_benefit(head_key, current_ratio, step_size)
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_recipient = head_key

            # Find best donor (head that costs least to remove -step_size)
            best_donor = None
            best_cost = float('inf')

            for head_key in head_ratios:
                current_ratio = head_ratios[head_key]

                # Skip if already at min
                if current_ratio <= min_ratio_per_head:
                    continue

                # Can't donate to yourself
                if head_key == best_recipient:
                    continue

                cost = self.interpolate_marginal_cost(head_key, current_ratio, step_size)
                if cost < best_cost:
                    best_cost = cost
                    best_donor = head_key

            # Check if swap is beneficial
            if best_recipient is None or best_donor is None:
                print(f"No valid swap found (recipient={best_recipient}, donor={best_donor})")
                break

            net_gain = best_benefit - best_cost

            if net_gain <= 0:
                no_improvement_count += 1
                if no_improvement_count >= 10:  # Allow some tolerance
                    print(f"No beneficial swap found (net_gain={net_gain:.6f}), stopping")
                    break
                # Try continuing in case of numerical issues
                iteration += 1
                continue

            no_improvement_count = 0

            # Perform the swap
            head_ratios[best_recipient] = min(
                head_ratios[best_recipient] + step_size,
                min(max_ratio_per_head, self.head_max_ratios.get(best_recipient, 1.0))
            )
            head_ratios[best_donor] = max(
                head_ratios[best_donor] - step_size,
                min_ratio_per_head
            )

            iteration += 1

            if iteration % 100 == 0:
                current_loss = sum(
                    self.interpolate_delta(head_key, ratio)
                    for head_key, ratio in head_ratios.items()
                )
                print(f"  Iteration {iteration}: loss={current_loss:.6f}, "
                      f"last swap: {best_donor} -> {best_recipient} "
                      f"(benefit={best_benefit:.6f}, cost={best_cost:.6f}, net={net_gain:.6f})")

        # Final statistics
        final_loss = sum(
            self.interpolate_delta(head_key, ratio)
            for head_key, ratio in head_ratios.items()
        )
        current_total = sum(head_ratios.values()) / self.total_heads

        print(f"Completed in {iteration} iterations")
        print(f"Final total delta (loss): {final_loss:.6f} (improvement: {initial_loss - final_loss:.6f})")
        print(f"Final average ratio: {current_total:.4f}")

        return head_ratios

    def solve_annealing(
        self,
        target_total_ratio: float,
        step_size: float = 0.001,
        min_ratio_per_head: float = 0.0,
        max_ratio_per_head: float = 1.0,
        max_iterations: int = 100000,
        initial_temp: float = 0.01,
        final_temp: float = 1e-6,
        cooling_rate: float = 0.99995,
    ) -> Dict[str, float]:
        """
        Solve using simulated annealing - like swap but occasionally accepts bad moves.

        This approach can escape local minima by accepting disadvantageous swaps
        with a probability that decreases as the temperature cools.

        Parameters
        ----------
        target_total_ratio : float
            Target total compaction ratio (average ratio across all heads).
        step_size : float
            Step size for budget transfers (default: 0.001)
        min_ratio_per_head : float
            Minimum ratio per head (default: 0.0)
        max_ratio_per_head : float
            Maximum ratio per head (default: 1.0)
        max_iterations : int
            Maximum iterations (default: 100000)
        initial_temp : float
            Starting temperature (default: 0.01). Should be on the order of typical
            net_gain values (usually 0.001-0.01 per swap).
        final_temp : float
            Temperature at which to stop (default: 1e-6)
        cooling_rate : float
            Multiply temperature by this each iteration (default: 0.99995).
            With these defaults, takes ~92,000 iterations to cool from 0.01 to 1e-6.

        Returns
        -------
        head_ratios : dict
            Mapping from head key to allocated ratio
        """
        import random

        # Initialize all global layer heads at uniform ratio = target_total_ratio
        head_ratios = {
            f"L{l}H{h}": target_total_ratio
            for l in self.global_layer_indices
            for h in range(self.num_heads)
        }

        # Clamp initial values to valid range for each head
        for head_key in head_ratios:
            head_max = min(max_ratio_per_head, self.head_max_ratios.get(head_key, 1.0))
            head_ratios[head_key] = max(min_ratio_per_head, min(head_max, head_ratios[head_key]))

        # Compute initial total loss
        initial_loss = self.compute_total_loss(head_ratios)
        best_loss = initial_loss
        best_ratios = head_ratios.copy()

        print(f"Simulated annealing: target={target_total_ratio:.4f}, step_size={step_size:.4f}")
        print(f"Temperature: {initial_temp} -> {final_temp}, cooling_rate={cooling_rate}")
        print(f"Starting from uniform allocation at {target_total_ratio:.4f} per head")
        print(f"Initial total delta (loss): {initial_loss:.6f}")

        temperature = initial_temp
        iteration = 0
        accepted_bad_moves = 0

        # Build list of head keys for random selection
        head_keys = list(head_ratios.keys())

        while iteration < max_iterations and temperature > final_temp:
            # Pick a RANDOM recipient (head that can receive budget)
            valid_recipients = [
                h for h in head_keys
                if head_ratios[h] < min(max_ratio_per_head, self.head_max_ratios.get(h, 1.0))
            ]

            if not valid_recipients:
                print("No valid recipients remaining")
                break

            recipient = random.choice(valid_recipients)

            # Pick a RANDOM donor (head that can give budget, different from recipient)
            valid_donors = [
                h for h in head_keys
                if head_ratios[h] > min_ratio_per_head and h != recipient
            ]

            if not valid_donors:
                print("No valid donors remaining")
                break

            donor = random.choice(valid_donors)

            # Compute the net gain of this random swap
            benefit = self.interpolate_marginal_benefit(recipient, head_ratios[recipient], step_size)
            cost = self.interpolate_marginal_cost(donor, head_ratios[donor], step_size)
            net_gain = benefit - cost

            # Decide whether to accept the swap
            accept = False
            if net_gain > 0:
                # Always accept improving moves
                accept = True
            else:
                # Accept bad moves with probability exp(net_gain / temperature)
                # net_gain is negative here, so this gives a probability < 1
                acceptance_prob = np.exp(net_gain / temperature)
                if random.random() < acceptance_prob:
                    accept = True
                    accepted_bad_moves += 1

            if accept:
                # Perform the swap
                head_ratios[recipient] = min(
                    head_ratios[recipient] + step_size,
                    min(max_ratio_per_head, self.head_max_ratios.get(recipient, 1.0))
                )
                head_ratios[donor] = max(
                    head_ratios[donor] - step_size,
                    min_ratio_per_head
                )

                # Track best solution found
                current_loss = self.compute_total_loss(head_ratios)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_ratios = head_ratios.copy()

            # Cool down
            temperature *= cooling_rate
            iteration += 1

            if iteration % 1000 == 0:
                current_loss = self.compute_total_loss(head_ratios)
                print(f"  Iteration {iteration}: temp={temperature:.6f}, loss={current_loss:.6f}, "
                      f"best_loss={best_loss:.6f}, bad_moves_accepted={accepted_bad_moves}")

        # Use the best solution found during annealing
        head_ratios = best_ratios
        final_loss = best_loss
        current_total = sum(head_ratios.values()) / self.total_heads

        print(f"Completed in {iteration} iterations")
        print(f"Final temperature: {temperature:.6f}")
        print(f"Bad moves accepted: {accepted_bad_moves}")
        print(f"Final total delta (loss): {final_loss:.6f} (improvement: {initial_loss - final_loss:.6f})")
        print(f"Final average ratio: {current_total:.4f}")

        return head_ratios

    def convert_to_proportions(
        self,
        head_ratios: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Convert head ratios to proportions format used by head_budgets files.

        The head_budgets format stores:
            proportion[head] = (keys for this head) / (total keys across all heads)

        So proportions sum to 1.0.

        Parameters
        ----------
        head_ratios : dict
            Mapping from head key to ratio (proportion of original keys for that head)

        Returns
        -------
        proportions : dict
            Mapping from head key to proportion (fraction of total allocated keys)
        """
        # Total allocated = sum of ratios (each ratio is proportion of article_len)
        total_allocated = sum(head_ratios.values())

        if total_allocated == 0:
            # Uniform if nothing allocated
            uniform = 1.0 / self.total_heads
            return {head_key: uniform for head_key in head_ratios}

        # Each head's proportion is its share of the total
        proportions = {
            head_key: ratio / total_allocated
            for head_key, ratio in head_ratios.items()
        }

        return proportions

    def compute_total_loss(self, head_ratios: Dict[str, float]) -> float:
        """Compute total delta (loss) for a given allocation."""
        return sum(
            self.interpolate_delta(head_key, ratio)
            for head_key, ratio in head_ratios.items()
        )

    def solve_for_ratios(
        self,
        target_ratios: List[float],
        step_size: float = 0.001,
        min_ratio_per_head: float = 0.0,
        max_ratio_per_head: float = 1.0,
        method: str = "swap",
    ) -> Tuple[Dict[float, Dict[str, float]], Dict[float, Dict]]:
        """
        Solve for optimal allocations at multiple target ratios.

        Parameters
        ----------
        target_ratios : list of float
            List of target ratios to solve for
        step_size : float
            Step size for greedy allocation
        min_ratio_per_head : float
            Minimum ratio per head
        max_ratio_per_head : float
            Maximum ratio per head
        method : str
            Solver method: "greedy" (start from zero, build up),
            "swap" (start from uniform, swap between heads), or
            "annealing" (simulated annealing). Default: "swap"

        Returns
        -------
        all_proportions : dict
            Mapping from target ratio to proportions dict
        solve_stats : dict
            Mapping from target ratio to solve statistics (loss, uniform_loss, etc.)
        """
        all_proportions = {}
        solve_stats = {}

        for target in sorted(target_ratios):
            print(f"\n{'='*60}")
            print(f"Solving for target ratio: {target} (method={method})")
            print(f"{'='*60}")

            if method == "greedy":
                head_ratios = self.solve_greedy(
                    target_total_ratio=target,
                    step_size=step_size,
                    min_ratio_per_head=min_ratio_per_head,
                    max_ratio_per_head=max_ratio_per_head,
                )
            elif method == "swap":
                head_ratios = self.solve_swap(
                    target_total_ratio=target,
                    step_size=step_size,
                    min_ratio_per_head=min_ratio_per_head,
                    max_ratio_per_head=max_ratio_per_head,
                )
            elif method == "annealing":
                head_ratios = self.solve_annealing(
                    target_total_ratio=target,
                    step_size=step_size,
                    min_ratio_per_head=min_ratio_per_head,
                    max_ratio_per_head=max_ratio_per_head,
                )
            else:
                raise ValueError(f"Unknown solver method: {method}. Use 'greedy', 'swap', or 'annealing'.")

            proportions = self.convert_to_proportions(head_ratios)
            all_proportions[target] = proportions

            # Print summary statistics
            ratios_list = list(head_ratios.values())
            props_list = list(proportions.values())

            print(f"\nHead ratio statistics:")
            print(f"  Min: {min(ratios_list):.4f}")
            print(f"  Max: {max(ratios_list):.4f}")
            print(f"  Mean: {np.mean(ratios_list):.4f}")
            print(f"  Std: {np.std(ratios_list):.4f}")

            print(f"\nProportion statistics:")
            print(f"  Sum: {sum(props_list):.6f}")
            print(f"  Min: {min(props_list):.6f}")
            print(f"  Max: {max(props_list):.6f}")

            # Find most and least allocated heads
            sorted_by_ratio = sorted(head_ratios.items(), key=lambda x: x[1], reverse=True)
            print(f"\nTop 5 allocated heads:")
            for head_key, ratio in sorted_by_ratio[:5]:
                print(f"  {head_key}: ratio={ratio:.4f}, prop={proportions[head_key]:.6f}")
            print(f"\nBottom 5 allocated heads:")
            for head_key, ratio in sorted_by_ratio[-5:]:
                print(f"  {head_key}: ratio={ratio:.4f}, prop={proportions[head_key]:.6f}")

            # Compute and store statistics
            final_loss = self.compute_total_loss(head_ratios)

            # Also compute uniform baseline loss for comparison
            uniform_ratios = {key: target for key in head_ratios}
            uniform_loss = self.compute_total_loss(uniform_ratios)

            solve_stats[target] = {
                'method': method,
                'final_loss': float(final_loss),
                'uniform_loss': float(uniform_loss),
                'improvement_over_uniform': float(uniform_loss - final_loss),
                'step_size': step_size,
                'smoothing_window': self.smoothing_window,
            }

            print(f"\nLoss comparison:")
            print(f"  Uniform loss: {uniform_loss:.6f}")
            print(f"  Optimized loss: {final_loss:.6f}")
            print(f"  Improvement: {uniform_loss - final_loss:.6f}")

        return all_proportions, solve_stats

    def save_proportions(
        self,
        proportions: Dict[str, float],
        output_path: str,
    ):
        """
        Save proportions to a JSON file in the head_budgets format.

        Parameters
        ----------
        proportions : dict
            Head proportions to save
        output_path : str
            Path to save to
        """
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(proportions, f, indent=2)

        print(f"Saved proportions to {output_path}")

    def save_all_proportions(
        self,
        all_proportions: Dict[float, Dict[str, float]],
        output_dir: str,
        prefix: str = "optimized",
    ):
        """
        Save proportions for all target ratios.

        Parameters
        ----------
        all_proportions : dict
            Mapping from target ratio to proportions
        output_dir : str
            Directory to save files
        prefix : str
            Prefix for filenames
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for target_ratio, proportions in all_proportions.items():
            # Format ratio for filename (e.g., 0.05 -> "t0.05")
            ratio_str = f"{target_ratio:.4f}".rstrip('0').rstrip('.')
            filename = f"{prefix}_t{ratio_str}.json"
            filepath = output_path / filename

            self.save_proportions(proportions, str(filepath))

    def save_solve_stats(
        self,
        solve_stats: Dict[float, Dict],
        output_path: str,
    ):
        """
        Save solve statistics to a JSON file.

        Parameters
        ----------
        solve_stats : dict
            Mapping from target ratio to statistics dict
        output_path : str
            Path to save to
        """
        # Convert float keys to strings for JSON
        stats_for_json = {
            str(ratio): stats for ratio, stats in solve_stats.items()
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(stats_for_json, f, indent=2)

        print(f"Saved solve stats to {output_path}")


    def proportions_to_ratios_at_target(
        self,
        proportions: Dict[str, float],
        target_ratio: float,
    ) -> Dict[str, float]:
        """
        Convert proportions to per-head ratios at a given target compaction ratio.

        Parameters
        ----------
        proportions : dict
            Per-head proportions (sum to 1.0)
        target_ratio : float
            Overall target ratio (e.g., 0.05)

        Returns
        -------
        ratios : dict
            Per-head ratios (each in range [0, 1])
        """
        # Total budget = target_ratio * total_heads (in units of "article_len per head")
        # Each head's ratio = proportion * total_budget = proportion * target_ratio * total_heads
        ratios = {}
        for head_key, proportion in proportions.items():
            ratios[head_key] = proportion * target_ratio * self.total_heads
        return ratios

    def compute_loss_at_ratio(
        self,
        proportions: Dict[str, float],
        target_ratio: float,
    ) -> float:
        """
        Compute total delta (loss) for proportions at a specific target ratio.

        Parameters
        ----------
        proportions : dict
            Per-head proportions (sum to 1.0)
        target_ratio : float
            Target compaction ratio

        Returns
        -------
        loss : float
            Total delta log perplexity
        """
        ratios = self.proportions_to_ratios_at_target(proportions, target_ratio)
        return sum(
            self.interpolate_delta(head_key, ratio)
            for head_key, ratio in ratios.items()
        )

    def compute_average_loss(
        self,
        proportions: Dict[str, float],
        target_ratios: List[float],
        weights: List[float] = None,
    ) -> float:
        """
        Compute average loss across multiple target ratios.

        Parameters
        ----------
        proportions : dict
            Per-head proportions (sum to 1.0)
        target_ratios : list of float
            Target ratios to evaluate at
        weights : list of float, optional
            Weights for each target ratio (default: uniform)

        Returns
        -------
        avg_loss : float
            Weighted average loss across all target ratios
        """
        if weights is None:
            weights = [1.0] * len(target_ratios)

        total_weight = sum(weights)
        weighted_loss = sum(
            w * self.compute_loss_at_ratio(proportions, t)
            for t, w in zip(target_ratios, weights)
        )
        return weighted_loss / total_weight

    def compute_marginal_benefit_across_ratios(
        self,
        head_key: str,
        proportions: Dict[str, float],
        step_size: float,
        target_ratios: List[float],
        weights: List[float],
    ) -> float:
        """
        Compute the marginal benefit of increasing a head's proportion,
        averaged across all target ratios.

        Uses smoothed curves if smoothing is enabled (for better decision-making).

        Returns the reduction in average loss (positive = good).
        """
        total_weight = sum(weights)
        benefit = 0.0

        for t, w in zip(target_ratios, weights):
            # Convert proportion to ratio at this target
            current_ratio = proportions[head_key] * t * self.total_heads
            new_ratio = (proportions[head_key] + step_size) * t * self.total_heads

            # Compute delta improvement for this head at this target ratio
            # Use smoothed curves for decision-making
            current_delta = self.interpolate_delta_smoothed(head_key, current_ratio)
            new_delta = self.interpolate_delta_smoothed(head_key, new_ratio)

            # Benefit is reduction in delta (lower delta = better)
            benefit += w * (current_delta - new_delta)

        return benefit / total_weight

    def compute_marginal_cost_across_ratios(
        self,
        head_key: str,
        proportions: Dict[str, float],
        step_size: float,
        target_ratios: List[float],
        weights: List[float],
    ) -> float:
        """
        Compute the marginal cost of decreasing a head's proportion,
        averaged across all target ratios.

        Uses smoothed curves if smoothing is enabled (for better decision-making).

        Returns the increase in average loss (positive = bad).
        """
        total_weight = sum(weights)
        cost = 0.0

        for t, w in zip(target_ratios, weights):
            # Convert proportion to ratio at this target
            current_ratio = proportions[head_key] * t * self.total_heads
            new_ratio = (proportions[head_key] - step_size) * t * self.total_heads

            # Compute delta change for this head at this target ratio
            # Use smoothed curves for decision-making
            current_delta = self.interpolate_delta_smoothed(head_key, current_ratio)
            new_delta = self.interpolate_delta_smoothed(head_key, new_ratio)

            # Cost is increase in delta (higher delta = worse)
            cost += w * (new_delta - current_delta)

        return cost / total_weight

    def solve_ratio_agnostic_swap(
        self,
        target_ratios: List[float],
        step_size: float = 0.001,
        max_iterations: int = 100000,
        weights: List[float] = None,
    ) -> Dict[str, float]:
        """
        Solve for a single set of proportions that works well across all target ratios.

        This method optimizes proportions (which sum to 1) rather than ratios.
        At each step, it finds the best swap (increase one head, decrease another)
        to minimize average loss across all target ratios.

        Parameters
        ----------
        target_ratios : list of float
            Target ratios to optimize across (e.g., [0.01, 0.02, 0.05, 0.1])
        step_size : float
            Step size for proportion adjustments (default: 0.001)
        max_iterations : int
            Maximum iterations (default: 100000)
        weights : list of float, optional
            Weights for each target ratio (default: uniform).
            Use higher weights for ratios you care more about.

        Returns
        -------
        proportions : dict
            Mapping from head key to proportion (sums to 1.0)
        """
        if weights is None:
            weights = [1.0] * len(target_ratios)

        # Initialize with uniform proportions (global layers only)
        uniform_prop = 1.0 / self.total_heads
        proportions = {
            f"L{l}H{h}": uniform_prop
            for l in self.global_layer_indices
            for h in range(self.num_heads)
        }

        # Compute initial loss
        initial_loss = self.compute_average_loss(proportions, target_ratios, weights)

        print(f"Ratio-agnostic swap solver")
        print(f"Target ratios: {target_ratios}")
        print(f"Weights: {weights}")
        print(f"Step size: {step_size}")
        print(f"Starting from uniform proportions: {uniform_prop:.6f} per head")
        print(f"Initial average loss: {initial_loss:.6f}")

        # Print initial per-ratio losses
        for t in target_ratios:
            loss_at_t = self.compute_loss_at_ratio(proportions, t)
            print(f"  Loss at t={t}: {loss_at_t:.6f}")

        iteration = 0
        no_improvement_count = 0

        head_keys = list(proportions.keys())

        while iteration < max_iterations:
            # Find the best recipient (head that benefits most from +step_size)
            best_recipient = None
            best_benefit = float('-inf')

            for head_key in head_keys:
                benefit = self.compute_marginal_benefit_across_ratios(
                    head_key, proportions, step_size, target_ratios, weights
                )
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_recipient = head_key

            # Find the best donor (head that costs least to give -step_size)
            best_donor = None
            best_cost = float('inf')

            for head_key in head_keys:
                # Skip if at minimum or same as recipient
                if proportions[head_key] <= step_size:
                    continue
                if head_key == best_recipient:
                    continue

                cost = self.compute_marginal_cost_across_ratios(
                    head_key, proportions, step_size, target_ratios, weights
                )
                if cost < best_cost:
                    best_cost = cost
                    best_donor = head_key

            # Check if swap is beneficial
            if best_recipient is None or best_donor is None:
                print(f"No valid swap found (recipient={best_recipient}, donor={best_donor})")
                break

            net_gain = best_benefit - best_cost

            if net_gain <= 0:
                no_improvement_count += 1
                if no_improvement_count >= 10:
                    print(f"No beneficial swap found after {no_improvement_count} tries, stopping")
                    break
                iteration += 1
                continue

            no_improvement_count = 0

            # Perform the swap
            proportions[best_recipient] += step_size
            proportions[best_donor] -= step_size

            iteration += 1

            if iteration % 100 == 0:
                current_loss = self.compute_average_loss(proportions, target_ratios, weights)
                print(f"  Iteration {iteration}: avg_loss={current_loss:.6f}, "
                      f"last swap: {best_donor} -> {best_recipient} "
                      f"(benefit={best_benefit:.6f}, cost={best_cost:.6f}, net={net_gain:.6f})")

        # Final statistics
        final_loss = self.compute_average_loss(proportions, target_ratios, weights)

        print(f"\nCompleted in {iteration} iterations")
        print(f"Final average loss: {final_loss:.6f} (improvement: {initial_loss - final_loss:.6f})")

        # Print per-ratio losses
        print(f"\nPer-ratio losses:")
        for t in target_ratios:
            loss_at_t = self.compute_loss_at_ratio(proportions, t)
            uniform_loss_at_t = self.compute_loss_at_ratio(
                {k: uniform_prop for k in proportions}, t
            )
            print(f"  t={t}: optimized={loss_at_t:.6f}, uniform={uniform_loss_at_t:.6f}, "
                  f"improvement={uniform_loss_at_t - loss_at_t:.6f}")

        # Normalize proportions to ensure they sum to 1
        total = sum(proportions.values())
        proportions = {k: v / total for k, v in proportions.items()}

        return proportions

    def solve_ratio_agnostic(
        self,
        target_ratios: List[float],
        step_size: float = 0.001,
        max_iterations: int = 100000,
        weights: List[float] = None,
        method: str = "swap",
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Solve for ratio-agnostic proportions and return stats.

        Parameters
        ----------
        target_ratios : list of float
            Target ratios to optimize across
        step_size : float
            Step size for optimization
        max_iterations : int
            Maximum iterations
        weights : list of float, optional
            Weights for each target ratio
        method : str
            Solver method (currently only "swap" is supported)

        Returns
        -------
        proportions : dict
            Optimized proportions (sum to 1)
        stats : dict
            Statistics including per-ratio losses
        """
        if method == "swap":
            proportions = self.solve_ratio_agnostic_swap(
                target_ratios=target_ratios,
                step_size=step_size,
                max_iterations=max_iterations,
                weights=weights,
            )
        else:
            raise ValueError(f"Unknown method: {method}. Currently only 'swap' is supported.")

        # Compute stats
        uniform_prop = 1.0 / self.total_heads
        uniform_proportions = {k: uniform_prop for k in proportions}

        stats = {
            'target_ratios': target_ratios,
            'weights': weights,
            'method': method,
            'step_size': step_size,
            'per_ratio_stats': {},
        }

        for t in target_ratios:
            optimized_loss = self.compute_loss_at_ratio(proportions, t)
            uniform_loss = self.compute_loss_at_ratio(uniform_proportions, t)
            stats['per_ratio_stats'][t] = {
                'optimized_loss': float(optimized_loss),
                'uniform_loss': float(uniform_loss),
                'improvement': float(uniform_loss - optimized_loss),
            }

        stats['average_optimized_loss'] = float(self.compute_average_loss(proportions, target_ratios, weights))
        stats['average_uniform_loss'] = float(self.compute_average_loss(uniform_proportions, target_ratios, weights))
        stats['average_improvement'] = stats['average_uniform_loss'] - stats['average_optimized_loss']

        return proportions, stats


def analyze_head_curves(
    head_curves: Dict[str, List[Tuple[float, float]]],
) -> Dict:
    """
    Analyze head curves to understand which heads are most important.

    Parameters
    ----------
    head_curves : dict
        Head curves to analyze

    Returns
    -------
    analysis : dict
        Analysis results including:
        - auc: Area under curve (integral of delta over ratio range)
        - delta_at_zero: Delta when head has 0 keys (how bad is removing this head?)
        - delta_at_max: Delta when head has max keys (how good is keeping all keys?)
        - slope_at_low_ratio: How quickly does adding keys help?
    """
    # For each head, compute various importance metrics

    importance_metrics = {}

    for head_key, curve in head_curves.items():
        ratios = [p[0] for p in curve]
        deltas = [p[1] for p in curve]

        # Area under curve (integral of delta)
        # More negative area = head is more important (helps more)
        auc = np.trapz(deltas, ratios)

        # Delta at ratio=0 (how bad is it to remove this head entirely?)
        delta_at_zero = deltas[0] if ratios[0] == 0 else np.interp(0, ratios, deltas)

        # Delta at max ratio (how good is it to keep all keys for this head?)
        max_ratio = max(ratios)
        delta_at_max = deltas[-1] if ratios[-1] == max_ratio else np.interp(max_ratio, ratios, deltas)

        # Slope at low ratios (how quickly does adding keys help?)
        if len(ratios) >= 2:
            slope_low = (deltas[1] - deltas[0]) / (ratios[1] - ratios[0]) if ratios[1] != ratios[0] else 0
        else:
            slope_low = 0

        importance_metrics[head_key] = {
            'auc': float(auc),
            'delta_at_zero': float(delta_at_zero),
            'delta_at_max': float(delta_at_max),
            'slope_at_low_ratio': float(slope_low),
        }

    # Rank heads by importance
    # Higher delta_at_zero = removing head hurts more = more important
    heads_by_importance = sorted(
        importance_metrics.items(),
        key=lambda x: x[1]['delta_at_zero'],
        reverse=True
    )

    return {
        'per_head_metrics': importance_metrics,
        'heads_ranked_by_importance': [h[0] for h in heads_by_importance],
        'top_10_most_important': heads_by_importance[:10],
        'top_10_least_important': heads_by_importance[-10:],
    }

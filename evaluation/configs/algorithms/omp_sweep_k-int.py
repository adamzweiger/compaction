# Sweep over k_choice and nnls_interval for fast OMP variants.
# Fixed: nnls_iters=2, lower_bound=exp(-5), upper_bound=exp(5), normalize_exp_scores=False

import math

exp = math.exp

config = {}

# Sweep over:
# - k_choice: 1, 2, 4, 8
# - nnls_interval: 1, 2, 4, 8
# Fixed: (2, -5, 5), normalize_exp_scores=False

k_choice_values = [1, 2, 4, 8]
nnls_interval_values = [1, 2, 4, 8]

for k_choice in k_choice_values:
    for nnls_interval in nnls_interval_values:
        # Build config name
        name_parts = ['omp', 'nnls2_-5_5']

        if k_choice != 1:
            name_parts.append(f'k{k_choice}')

        if nnls_interval != 1:
            name_parts.append(f'int{nnls_interval}')

        name = '_'.join(name_parts)

        # Build config dict
        config[name] = {
            'algorithm': 'omp',
            'nnls_iters': 2,
            'nnls_lower_bound': exp(-5),
            'nnls_upper_bound': exp(5),
            'c2_method': 'lsq',
            'normalize_exp_scores': False,
            'k_choice': k_choice,
            'nnls_interval': nnls_interval,
        }

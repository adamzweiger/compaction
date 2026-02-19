# Sweep over NNLS configurations and normalize_exp_scores.
# Fixed: k_choice=1, nnls_interval=1

import math

exp = math.exp

config = {}

# Sweep over:
# - nnls_iters, lower_bound, upper_bound: (0, -inf, inf), (0, -5, 5), (2, -inf, inf), (2, -5, 5)
# - normalize_exp_scores: False, True
# Fixed: k_choice=1, nnls_interval=1

nnls_configs = [
    (0, None, None, 'nnls0_-inf_inf'),
    (0, exp(-5), exp(5), 'nnls0_-5_5'),
    (2, None, None, 'nnls2_-inf_inf'),
    (2, exp(-5), exp(5), 'nnls2_-5_5'),
]

normalize_exp_scores_values = [False, True]

for nnls_iters, lower_bound, upper_bound, nnls_label in nnls_configs:
    for normalize_exp_scores in normalize_exp_scores_values:
        # Build config name
        name_parts = ['omp', nnls_label]

        if normalize_exp_scores:
            name_parts.append('norm')

        name = '_'.join(name_parts)

        # Build config dict
        config_dict = {
            'algorithm': 'omp',
            'nnls_iters': nnls_iters,
            'c2_method': 'lsq',
            'normalize_exp_scores': normalize_exp_scores,
            'k_choice': 1,
            'nnls_interval': 1,
        }

        if lower_bound is not None:
            config_dict['nnls_lower_bound'] = lower_bound
        if upper_bound is not None:
            config_dict['nnls_upper_bound'] = upper_bound

        config[name] = config_dict

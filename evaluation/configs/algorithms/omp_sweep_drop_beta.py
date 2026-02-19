# Sweep over drop_key_beta_cutoff configurations.
#
# After NNLS, if any beta value (log weight) is below the cutoff threshold,
# that key is dropped and won't be considered again. We then select more keys
# until we hit the target size.

config = {}

# Sweep over different cutoff values
# beta = log(B), so if B < exp(-10), then beta < -10
# Testing different cutoff thresholds
cutoff_values = [None, -10]

# Also sweep over whether to normalize exp_scores
normalize_exp_scores_values = [False]

# Also sweep over NNLS configurations
nnls_configs = [
    (0, None, None, 'nnls0'),
    (2, None, None, 'nnls2'),
    (5, None, None, 'nnls5'),
]

for nnls_iters, lower_bound, upper_bound, nnls_label in nnls_configs:
    for cutoff in cutoff_values:
        for normalize_exp_scores in normalize_exp_scores_values:
            # Build config name
            name_parts = ['omp', nnls_label, f'drop{cutoff}']

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
                'drop_key_beta_cutoff': cutoff,
            }

            if lower_bound is not None:
                config_dict['nnls_lower_bound'] = lower_bound
            if upper_bound is not None:
                config_dict['nnls_upper_bound'] = upper_bound

            config[name] = config_dict

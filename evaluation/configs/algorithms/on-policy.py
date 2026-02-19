import math

exp = math.exp

config = {
    'highest_attn_keys_rms_nnls2_-3_3_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
    },
    'highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
        'on_policy': True,
    },
    'global_highest_attn_keys_rms_nnls2_-3_3_lsq': {
        'algorithm': 'global_highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
    },
    'global_highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy': {
        'algorithm': 'global_highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
        'on_policy': True,
    },
    'omp_nnls0_-inf_7_drop-7_lsq': {
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'lsq',
    },
    'omp_nnls0_-inf_7_drop-7_lsq_on-policy': {
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'lsq',
        'on_policy': True,
    },
}

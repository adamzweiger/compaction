import math

from compaction.algorithms.omp import DEFAULT_PROGRESSIVE_SCHEDULE

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
    'highest_attn_keys_max_nnls2_-3_3_lsq_on-policy': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
        'on_policy': True,
    },
    'omp_nnls0_-inf_7_lsq': {
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'c2_method': 'lsq',
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
        'on_policy': True
    },
    'omp_nnls0_-inf_7_drop-7_lsq_progressive': {
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'lsq',
        'progressive_schedule': DEFAULT_PROGRESSIVE_SCHEDULE
    },
    'omp_nnls0_-inf_7_drop-7_lsq_progressive_on-policy': {
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'lsq',
        'progressive_schedule': DEFAULT_PROGRESSIVE_SCHEDULE,
        'on_policy': True
    },
    'omp_nnls0_-inf_7_drop-7_lsq_k4int2': {
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'lsq',
        'k_choice': 4,
        'nnls_interval': 2,
    },
    'omp_nnls0_-inf_7_drop-7_lsq_k4int2_on-policy': {
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'lsq',
        'k_choice': 4,
        'nnls_interval': 2,
        'on_policy': True,
    },
    'omp_nnls0_-inf_7_drop-7_lsq_k4int4_on-policy': {
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'lsq',
        'k_choice': 4,
        'nnls_interval': 4,
        'on_policy': True,
    },
}

import math

from compaction.algorithms.omp import DEFAULT_PROGRESSIVE_SCHEDULE

exp = math.exp

config = {
    'random_vector_keys_nnls2_-3_3_lsq_on-policy': {
        'algorithm': 'random_vector_keys',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'on_policy': True,
    },
    'kmeans_keys_nnls2_-3_3_lsq_on-policy': {
        'algorithm': 'kmeans',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'on_policy': True,
    },
    'truncate_nnls2_-3_3_lsq_on-policy': {
        'algorithm': 'truncate',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'on_policy': True,
    },
    'random_subset_keys_nnls2_-3_3_lsq_on-policy': {
        'algorithm': 'random_subset_keys',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
        'on_policy': True,
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
    'omp_nnls0_-inf_7_drop-7_lsq_progressive_on-policy': {
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'lsq',
        'progressive_schedule': DEFAULT_PROGRESSIVE_SCHEDULE,
        'on_policy': True
    },
}

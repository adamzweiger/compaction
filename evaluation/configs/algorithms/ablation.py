import math

from compaction.algorithms.omp import DEFAULT_PROGRESSIVE_SCHEDULE

exp = math.exp

config = {
    'omp_nnls0_-inf_7_drop-7_lsq_progressive_on-policy': { # best
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'lsq',
        'progressive_schedule': DEFAULT_PROGRESSIVE_SCHEDULE,
        'on_policy': True
    },
    'omp_nnls0_-inf_7_drop-7_direct_progressive_on-policy': { # no-lsq
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'direct',
        'progressive_schedule': DEFAULT_PROGRESSIVE_SCHEDULE,
        'on_policy': True
    },
    'omp_nnls0_-inf_7_drop-7_zerobeta_lsq_progressive_on-policy': { # no-beta
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'zerobeta': True,
        'c2_method': 'lsq',
        'progressive_schedule': DEFAULT_PROGRESSIVE_SCHEDULE,
        'on_policy': True
    },
    'omp_nnls0_-inf_7_drop-7_zerobeta_direct_progressive_on-policy': { # no-beta, no-lsq
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'zerobeta': True,
        'c2_method': 'direct',
        'progressive_schedule': DEFAULT_PROGRESSIVE_SCHEDULE,
        'on_policy': True
    },
    'omp_nnls0_-inf_7_drop-7_lsq_progressive': { # no-on-policy
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'lsq',
        'progressive_schedule': DEFAULT_PROGRESSIVE_SCHEDULE,
    },
}

import math

exp = math.exp

config = {
    'baseline': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'mean',
        'beta_method': 'zero',
        'c2_method': 'direct',
    },
    'AM-HighestAttnKeys-basic': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
    },
    'AM-HighestAttnKeys': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
        'on_policy': True,
    },
    'AM-OMP-basic': {
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'lsq'
    },
    'AM-OMP': {
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'lsq',
        'on_policy': True
    },
    'AM-OMP-fast': {
        'algorithm': 'omp',
        'nnls_iters': 0,
        'nnls_upper_bound': exp(7),
        'drop_key_beta_cutoff': -7,
        'c2_method': 'lsq',
        'k_choice': 4,
        'nnls_interval': 2,
        'on_policy': True,
    },
}

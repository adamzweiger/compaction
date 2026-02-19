import math

exp = math.exp

config = {
    'highest_attention_keys_rms_2_-3_3_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
    },
}

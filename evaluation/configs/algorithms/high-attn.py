import math

exp = math.exp

config = {
    'highest_attn_keys_max_nobeta_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'max',
        'beta_method': 'zero',
        'c2_method': 'lsq',
    },
    'highest_attn_keys_max_nnls0_-3_3_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'max',
        'nnls_iters': 0,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
    },
    'highest_attn_keys_max_nnls0_-inf_inf_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'max',
        'nnls_iters': 0,
        'c2_method': 'lsq',
    },
    'highest_attn_keys_max_nnls2_-3_3_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'max',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
    },
    'highest_attn_keys_max_nnls2_-inf_inf_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'max',
        'nnls_iters': 2,
        'c2_method': 'lsq',
    },
    'highest_attn_keys_mean_nobeta_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'mean',
        'beta_method': 'zero',
        'c2_method': 'lsq',
    },
    'highest_attn_keys_mean_nnls0_-3_3_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'mean',
        'nnls_iters': 0,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
    },
    'highest_attn_keys_mean_nnls0_-inf_inf_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'mean',
        'nnls_iters': 0,
        'c2_method': 'lsq',
    },
    'highest_attn_keys_mean_nnls2_-3_3_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'mean',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
    },
    'highest_attn_keys_mean_nnls2_-inf_inf_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'mean',
        'nnls_iters': 2,
        'c2_method': 'lsq',
    },
    'highest_attn_keys_rms_nobeta_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'beta_method': 'zero',
        'c2_method': 'lsq',
    },
    'highest_attn_keys_rms_nnls0_-3_3_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 0,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
    },
    'highest_attn_keys_rms_nnls0_-inf_inf_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 0,
        'c2_method': 'lsq',
    },
    'highest_attn_keys_rms_nnls2_-3_3_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
    },
    'highest_attn_keys_rms_nnls2_-inf_inf_lsq': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 2,
        'c2_method': 'lsq',
    },
}

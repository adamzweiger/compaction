config = {
    'highest_attn_keys_max_nobeta_direct': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'max',
        'beta_method': 'zero',
        'c2_method': 'direct',
    },
    'highest_attn_keys_rms_nobeta_direct': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'beta_method': 'zero',
        'c2_method': 'direct',
    },
    'highest_attn_keys_mean_nobeta_direct': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'mean',
        'beta_method': 'zero',
        'c2_method': 'direct',
    },
}

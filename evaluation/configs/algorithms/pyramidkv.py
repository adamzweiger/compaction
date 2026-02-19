config = {
    'pyramidkv_max': {
        'algorithm': 'highest_attention_keys',
        'pooling': 'maxpool',
        'score_method': 'max',
        'beta_method': 'zero',
        'c2_method': 'direct',
    },
    'pyramidkv_rms': {
        'algorithm': 'highest_attention_keys',
        'pooling': 'maxpool',
        'score_method': 'rms',
        'beta_method': 'zero',
        'c2_method': 'direct',
    },
    'pyramidkv_mean': {
        'algorithm': 'highest_attention_keys',
        'pooling': 'maxpool',
        'score_method': 'mean',
        'beta_method': 'zero',
        'c2_method': 'direct',
    },
}

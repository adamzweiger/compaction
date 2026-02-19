config = {
    'optim_joint': {
        'algorithm': 'optim_joint',
        'lr': 0.02,
        'num_steps': 2000,
        'lam': 0.05,
        'patience': 2000,
        'optimizer': 'adam',
        'use_lr_decay': True,
        'on_policy': True,
    },
    'batched_optim_joint': {
        'algorithm': 'optim_joint',
        'use_batched': True,
        'use_per_layer_batching': True,
        'lr': 0.02,
        'num_steps': 2000,
        'lam': 0.05,
        'patience': 2000,
        'optimizer': 'adam',
        'use_lr_decay': True,
        'on_policy': True,
    },
}

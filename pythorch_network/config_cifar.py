config = {
    'name': 'CIFAR10_Generalization_Test',

    # Architecture Params
    'num_classes': 10,
    'final_layer_filters': 512,
    'dropout_rate': 0.4,
    'use_batch_norm': True,

    # Training Params
    'lr': 0.001,
    'opt_type': 'Adam',
    'weight_decay': 1e-4,
    'batch_size': 128,
    'num_epochs': 40,

    # Optimization & Reliability
    'early_stop_patience': 6,
    'init_type': 'kaiming'
}
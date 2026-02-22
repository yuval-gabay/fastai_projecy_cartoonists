# Configuration for the Wide Network
config = {
    'name': 'Wide_Experiment',
    'lr': 0.001,
    'dropout_rate': 0.7,#0.6
    'use_batch_norm': True,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'opt_type': 'Adam',
    'num_epochs': 20,
    'batch_size': 32,
    'init_type': 'xavier',
    'aug_rotation': 20,      # סיבוב במעלות
    'aug_zoom_scale': 0.8,

}
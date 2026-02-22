# config_deep.py
from sympy import false

config = {
    'name': 'Deep_Experiment',
    'lr': 0.0003,#0.0005-> 0.0003->0.0001->0.0003
    'dropout_rate': 0.3,#0.3->0.6->0.3
    'use_batch_norm': True,
    'weight_decay': 1e-3,
    'momentum': 0.9,#0.9
    'opt_type': 'Adam',#Adam->SGD->Adam
    'num_epochs': 50,#15->30->50
    'batch_size': 32,#48
    'init_type': 'kaiming',
    'aug_rotation': 20,      # סיבוב במעלות
    'aug_zoom_scale': 0.8,
    'early_stop_patience': 7, # כמה איפוקים לחכות ללא שיפור ב-Accuracy
    'final_layer_filters': 512

}
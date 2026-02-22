config = {
    'name': 'ResNet50_Transfer_Learning',
    'data_path': r"C:\Users\Surface\PycharmProjects\PythonProject2\RESNET_CARTOONIST_DATA",
    'lr': 0.0005,                # קצב למידה לראש (Phase 1)
    'lr_fine_tune': 1e-3,        # קצב למידה לשכבות עמוקות (Phase 2)
    'opt_type': 'Adam',          # הוספנו למניעת KeyError
    'momentum': 0.9,             # הוספנו לאחידות
    'dropout_rate': 0.5,         # מניעת Overfitting
    'weight_decay': 1e-2,        # רגולריזציה כמו בחלק 1
    'num_epochs_warmup': 5,
    'num_epochs_fine_tune': 20,
    'batch_size': 16,
    'img_size': 224,             # ResNet דורשת 224
    'aug_rotation': 30,
    'aug_zoom_scale': 0.7,
    'early_stop_patience': 5     # זהה למנגנון בחלק 1
}
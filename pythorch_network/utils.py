import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import ctypes

def prevent_sleep():
    """
    מונע מהמחשב להיכנס למצב שינה או לכבות את המסך בזמן שהקוד רץ.
    עובד על מערכות Windows.
    """
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    # שומר על המערכת והתצוגה דולקים
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
    print(">>> System sleep prevented. Keeping PC awake for training...")

def allow_sleep():
    """
    מחזיר את הגדרות השינה למצב הרגיל של המערכת.
    """
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print(">>> System sleep settings restored.")

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1) # Gamma
            nn.init.constant_(m.bias, 0)   # Beta
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


# ספירת פרמטרים (עבור שקף 11)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# יצירת מטריצת בלבול (עבור שקף 14)
def get_confusion_matrix(model, loader, device, classes=['Pendleton', 'Tartakovsky', 'Timm']):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm
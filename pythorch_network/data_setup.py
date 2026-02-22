import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_dataloaders(data_dir, cfg):
    data_dir = os.path.abspath(data_dir)
    batch_size = cfg['batch_size']

    # טרנספורמציות לאימון (Training) - כולל זום וסיבוב
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(cfg['aug_rotation']),  # שימוש ב-config
        transforms.RandomResizedCrop(224, scale=(cfg['aug_zoom_scale'], 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # טרנספורמציות לולידציה (Validation) - נקיות וקבועות
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # טעינת הדאטה מהתיקייה שבה נמצאים האמנים
    full_dataset = datasets.ImageFolder(root=data_dir)

    # פיצול 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # החלת הטרנספורמציות הנפרדות
    # אנחנו משתמשים ב-Wrapper פשוט כדי להבטיח שה-Train וה-Val יקבלו טרנספורמציות שונות
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader
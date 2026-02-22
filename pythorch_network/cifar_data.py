import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar_loaders(cfg):
    """
    Downloads CIFAR-10 to './data' and returns DataLoaders.
    """
    # Transformations for 32x32 images
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Loading the Dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    print(f"✅ CIFAR-10 data is ready in './data' folder")
    print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")

    return train_loader, test_loader
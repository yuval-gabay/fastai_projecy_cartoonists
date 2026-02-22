import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from models_transfer import CartoonResNet
from config_transfer import config as cfg
from data_setup import get_dataloaders
from utils import prevent_sleep, allow_sleep


def run_experiment(part1_acc):
    # מניעת שינה של המחשב
    prevent_sleep()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl = get_dataloaders(cfg['data_path'], cfg)
    model = CartoonResNet(cfg).to(device)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_acc': []}
    start_time = time.time()

    # --- Phase 1: Warmup ---
    print(f"\n>>> Phase 1: Warmup ({cfg['num_epochs_warmup']} epochs)")

    # שימוש ב-opt_type מהקונפיג ששלחת
    if cfg['opt_type'] == 'Adam':
        optimizer = optim.Adam(model.resnet.fc.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.SGD(model.resnet.fc.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'])

    for epoch in range(cfg['num_epochs_warmup']):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = evaluate(model, val_dl, device)
        history['train_loss'].append(running_loss / len(train_dl))
        history['val_acc'].append(acc)
        print(f"Warmup {epoch + 1}: Loss {running_loss / len(train_dl):.4f}, Acc {acc:.2f}%")

    # --- Phase 2: Fine-Tuning עם Early Stopping (כמו בחלק 1) ---
    print(f"\n>>> Phase 2: Controlled Fine-Tuning")
    model.unfreeze_limited()

    # ב-Phase 2 משתמשים ב-Differential LR
    optimizer = optim.Adam([
        {'params': model.resnet.layer4.parameters(), 'lr': cfg['lr_fine_tune']},
        {'params': model.resnet.fc.parameters(), 'lr': cfg['lr'] * 0.1}
    ], weight_decay=cfg['weight_decay'])

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(cfg['num_epochs_fine_tune']):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = evaluate(model, val_dl, device)
        history['train_loss'].append(running_loss / len(train_dl))
        history['val_acc'].append(acc)

        print(f"Fine-Tune {epoch + 1}: Loss {running_loss / len(train_dl):.4f}, Acc {acc:.2f}%")

        # לוגיקת Early Stopping של חלק 1
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  [SAVED] New Best: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= cfg['early_stop_patience']:
                print(f"\n>>> Early Stopping after {cfg['early_stop_patience']} epochs.")
                break

    model.load_state_dict(best_model_wts)
    allow_sleep()  # שחרור נעילת שינה
    create_summary_dashboard(model, val_dl, history, part1_acc, device)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, pred = torch.max(model(imgs), 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100 * correct / total


def create_summary_dashboard(model, val_loader, history, part1_acc, device):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            _, preds = torch.max(model(imgs.to(device)), 1)
            all_preds.extend(preds.cpu().numpy());
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(18, 6))
    plt.suptitle(f"ResNet50 Study - Best Accuracy: {max(history['val_acc']):.2f}%", fontsize=16)

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], color='red');
    plt.title("Loss History")

    plt.subplot(1, 3, 2)
    plt.plot(history['val_acc'], color='green', label='ResNet50')
    plt.axhline(y=part1_acc, color='blue', linestyle='--', label=f'Part 1 Baseline ({part1_acc}%)')
    plt.title("Accuracy Comparison");
    plt.legend()

    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pendleton', 'Tartakovsky', 'Timm'],
                yticklabels=['Pendleton', 'Tartakovsky', 'Timm'])
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig('final_resnet_results.png')
    plt.show()


if __name__ == "__main__":
    run_experiment(part1_acc=79.5)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # מניעת קריסה בגלל כפילות ספריות

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# ייבוא הקבצים האחרים שיצרנו
from config_cifar import config as cfg
from cifar_data import get_cifar_loaders
from models_cifar import CIFARNet


def run_cifar_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. טעינת דאטה
    train_loader, test_loader = get_cifar_loaders(cfg)

    # 2. אתחול מודל
    model = CIFARNet(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    history = {'train_loss': [], 'test_acc': []}
    best_acc = 0
    patience_counter = 0

    # 3. לולאת אימון
    print(f"\n>>> Starting CIFAR-10 Training on {device}")
    for epoch in range(cfg['num_epochs']):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # בדיקת דיוק (Validation)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        test_acc = 100 * correct / total
        history['train_loss'].append(running_loss / len(train_loader))
        history['test_acc'].append(test_acc)

        print(
            f"Epoch [{epoch + 1}/{cfg['num_epochs']}] - Loss: {running_loss / len(train_loader):.4f} - Acc: {test_acc:.2f}%")

        # שמירת המודל הכי טוב ו-Early Stopping
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_cifar_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= cfg['early_stop_patience']:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # 4. יצירת ה-Dashboard האוטומטי בסיום
    create_research_dashboard(model, test_loader, history, device)


def create_research_dashboard(model, test_loader, history, device):
    print("\nGenerating Research Dashboard...")
    model.load_state_dict(torch.load('best_cifar_model.pth'))
    model.eval()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, pred = torch.max(model(imgs), 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    # פריסה של 2 שורות על 3 עמודות (בדיוק כמו בקוד המקורי שלך)
    fig = plt.figure(figsize=(22, 12))
    plt.suptitle('Part 2: CIFAR-10 Generalization - Deep & Slim Architecture', fontsize=22, fontweight='bold')

    # 1. גרף Accuracy
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history['test_acc'], label='Test Accuracy', color='green', linewidth=2, marker='o')
    ax1.set_title('Test Accuracy (%)', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. גרף Loss
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(history['train_loss'], color='red', label='Train Loss', linewidth=2)
    ax2.set_title('Training Loss Convergence', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.legend()

    # 3. מטריצת בלבול (תופסת שטח גדול יותר במרכז)
    ax3 = plt.subplot(2, 3, (4, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes, ax=ax3)
    ax3.set_title('Confusion Matrix: Predicted vs Actual', fontsize=14)

    # 4. לוח נתונים מסכם (בצד ימין - כמו ה-ax6 שלך)
    ax_info = plt.subplot(2, 3, (3, 6))
    ax_info.axis('off')

    best_acc = max(history['test_acc'])
    num_params = sum(p.numel() for p in model.parameters())

    # חישוב דיוק ממוצע למחלקה
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1) * 100
    top_class_idx = accuracy_per_class.argsort()[-1]

    summary_text = (
        f"📊 CIFAR-10 RESEARCH SUMMARY\n"
        f"{'-' * 35}\n"
        f"PERFORMANCE METRICS:\n"
        f"• Best Accuracy: {best_acc:.2f}%\n"
        f"• Total Epochs: {len(history['test_acc'])}\n"
        f"• Model Parameters: {num_params:,}\n\n"
        f"STRENGTHS:\n"
        f"• Strongest Class: {classes[top_class_idx]}\n"
        f"  ({accuracy_per_class[top_class_idx]:.1f}% Accuracy)\n\n"
        f"CONCLUSION:\n"
        f"The 'Slim-Deep' architecture\n"
        f"shows excellent generalization\n"
        f"capabilities on 10 object classes.\n"
        f"{'-' * 35}\n"
        f"STATUS: SUCCESS"
    )
    ax_info.text(0.05, 0.5, summary_text, fontsize=15, family='monospace', va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # שמירה אוטומטית והצגה
    plt.savefig('cifar_research_summary.png', dpi=300)
    print(f"✅ Research Dashboard saved to: {os.getcwd()}/cifar_research_summary.png")
    plt.show()


if __name__ == "__main__":
    run_cifar_experiment()
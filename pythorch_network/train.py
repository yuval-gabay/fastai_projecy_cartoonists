import os

from sympy import false

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from models import DeepCartoonNet, WideCartoonNet
from utils import initialize_weights, count_parameters, get_confusion_matrix
from data_setup import get_dataloaders
from visualize_final import plot_final_summary

# ייבוא הקונפיגורציות
from config_deep import config as deep_cfg
from config_wide import config as wide_cfg


def run_experiment(model, train_loader, val_loader, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    initialize_weights(model)

    criterion = nn.CrossEntropyLoss()

    if cfg['opt_type'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'])

    # Dynamic LR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=3
    )

    # Early Stopping Initialization
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = cfg.get('early_stop_patience', 7)

    history = {'train_loss': [], 'val_acc': [], 'weights_std': [], 'time': 0}
    start_time = time.time()

    print(f"\n>>> Starting Training: {cfg['name']}")
    print(f">>> Configuration: Filters: {cfg.get('final_layer_filters')}, Patience: {patience_limit}\n")

    for epoch in range(cfg['num_epochs']):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # --- Stability & Metrics Tracking ---
        with torch.no_grad():
            weights = list(model.parameters())[0].cpu().numpy()
            history['weights_std'].append(weights.std())

        # --- Validation Phase ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_acc)

        # Update Scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        # --- Early Stopping Logic ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save the best weights found so far
            torch.save(model.state_dict(), f"best_{cfg['name']}.pth")
            status_msg = " [NEW BEST!]"
        else:
            patience_counter += 1
            status_msg = ""

        # Progress update
        print(
            f"Epoch [{epoch + 1}/{cfg['num_epochs']}] - Loss: {avg_train_loss:.4f} | Acc: {val_acc:.2f}% | LR: {current_lr:.6f} | Patience: {patience_counter}/{patience_limit}{status_msg}")

        if patience_counter >= patience_limit:
            print(f"\n[!] Early Stopping triggered at epoch {epoch + 1}. Restoring best weights ({best_val_acc:.2f}%).")
            break

    # Load the best weights before returning history
    model.load_state_dict(torch.load(f"best_{cfg['name']}.pth"))
    history['time'] = time.time() - start_time
    return history





if __name__ == "__main__":
    run_wide = False  # שנה ל-True אם תרצה להריץ שוב את הרשת הרחבה
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = r"C:\Users\Surface\PycharmProjects\PythonProject2\imageData"

    # טעינת דאטה (לפי הקונפיגורציה של הרשת העמוקה)
    train_dl, val_dl = get_dataloaders(data_path, deep_cfg)

    # --- ניסוי 1: רשת עמוקה ---
    print(f"\n--- Starting Experiment: {deep_cfg['name']} ---")
    deep_model = DeepCartoonNet(deep_cfg)
    deep_params = count_parameters(deep_model)
    deep_hist = run_experiment(deep_model, train_dl, val_dl, deep_cfg)
    deep_cm = get_confusion_matrix(deep_model, val_dl, device)

    # --- ניסוי 2: רשת רחבה (מותנה במשתנה run_wide) ---
    if run_wide:
        print(f"\n--- Starting Experiment: {wide_cfg['name']} ---")
        wide_model = WideCartoonNet(wide_cfg)
        wide_params = count_parameters(wide_model)
        wide_hist = run_experiment(wide_model, train_dl, val_dl, wide_cfg)
        wide_cm = get_confusion_matrix(wide_model, val_dl, device)
    else:
        print("\n--- Skipping Wide Experiment. Using dummy data for plots. ---")
        wide_hist = {
            'val_acc': [0],
            'train_loss': [0],
            'weights_std': [0],
            'time': 0
        }
        wide_cm = np.zeros((3, 3), dtype=int)
        wide_params = 0

    # הפקת הדו"ח הסופי למצגת - השורה הזו מיושרת לשמאל בתוך ה-main
    plot_final_summary(deep_hist, wide_hist, deep_cm, wide_cm, deep_params, wide_params)
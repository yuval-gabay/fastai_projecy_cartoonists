import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_final_summary(deep_hist, wide_hist, deep_cm, wide_cm, deep_params, wide_params):
    # הגדרת פריסה של 2 שורות על 3 עמודות
    fig = plt.figure(figsize=(20, 12))
    plt.suptitle('Research Results: Deep vs Wide Architecture Analysis', fontsize=22, fontweight='bold')

    # 1. גרף Accuracy (דיוק לאורך זמן)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(deep_hist['val_acc'], label='Deep Net', color='blue', linewidth=2, marker='o')
    ax1.plot(wide_hist['val_acc'], label='Wide Net', color='red', linewidth=2, marker='s')
    ax1.set_title('Validation Accuracy (%)', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. גרף Loss (התכנסות הרשת)
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(deep_hist['train_loss'], color='blue', label='Deep Loss')
    ax2.plot(wide_hist['train_loss'], color='red', label='Wide Loss')
    ax2.set_title('Training Loss Convergence', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # 3. יציבות משקלים (Batch Norm Effect)
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(deep_hist['weights_std'], color='blue', linestyle='--', label='Deep Std')
    ax3.plot(wide_hist['weights_std'], color='red', linestyle='--', label='Wide Std')
    ax3.set_title('Weight Stability (Standard Deviation)', fontsize=14)
    ax3.set_xlabel('Epoch')
    ax3.legend()

    # 4. מטריצת בלבול - רשת עמוקה
    ax4 = plt.subplot(2, 3, 4)
    classes = ['Pendleton', 'Tartakovsky', 'Timm']
    sns.heatmap(deep_cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax4,
                cbar=False)
    ax4.set_title('Deep Net: Confusion Matrix', fontsize=14)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')

    # 5. מטריצת בלבול - רשת רחבה
    ax5 = plt.subplot(2, 3, 5)
    sns.heatmap(wide_cm, annot=True, fmt='d', cmap='Reds', xticklabels=classes, yticklabels=classes, ax=ax5, cbar=False)
    ax5.set_title('Wide Net: Confusion Matrix', fontsize=14)
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('Actual')

    # 6. לוח נתונים מסכם (Final Metrics)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    best_deep = max(deep_hist['val_acc'])
    best_wide = max(wide_hist['val_acc'])

    summary_text = (
            f"📊 FINAL STATISTICS\n"
            f"{'-' * 30}\n"
            f"DEEP NETWORK:\n"
            f"• Max Accuracy: {best_deep:.2f}%\n"
            f"• Parameters: {deep_params:,}\n"
            f"• Train Time: {deep_hist['time']:.1f}s\n\n"
            f"WIDE NETWORK:\n"
            f"• Max Accuracy: {best_wide:.2f}%\n"
            f"• Parameters: {wide_params:,}\n"
            f"• Train Time: {wide_hist['time']:.1f}s\n"
            f"{'-' * 30}\n"
            f"WINNER: " + ("Deep Architecture" if best_deep > best_wide else "Wide Architecture")
    )
    ax6.text(0.05, 0.5, summary_text, fontsize=15, family='monospace', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('research_final_results.png', dpi=300)  # שמירה למצגת באיכות גבוהה
    plt.show()
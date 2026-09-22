"""
utils/metrics.py
Evaluation metrics for binary classification (Real=0 / Fake=1).
"""
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1_score":  f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def print_metrics(metrics):
    print("\n" + "=" * 45)
    print("          E V A L U A T I O N   R E P O R T")
    print("=" * 45)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("=" * 45 + "\n")


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    labels = labels or ["Real", "Fake"]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses,   label="Val Loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(epochs, train_accs, label="Train Acc")
    axes[1].plot(epochs, val_accs,   label="Val Acc")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, title="", save_path=None, figsize=(8,6)):
    """
    Plot a confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix " + title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

# ===================== New Functions =====================

def plot_loss(train_loss_history, save_path,title=''):
    """
    Plot training loss curve.
    """
    plt.figure()
    plt.plot(range(1,len(train_loss_history)+1), train_loss_history, label="Train Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss Curve")
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_f1(val_f1_history, save_path,title=''):
    """
    Plot validation F1 score curve.
    """
    plt.figure()
    plt.plot(range(1,len(val_f1_history)+1), val_f1_history, label="Validation F1", marker='o', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title(f"{title} F1 Curve")
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

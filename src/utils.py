import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_training_history(history, save_path='metrics/loss_accuracy.png'):
    """
    Plots training and validation metrics.
    Handles both epoch-level and step-level history.
    """
    plt.figure(figsize=(15, 6))
    
    # 1. Plot Loss
    plt.subplot(1, 2, 1)
    if 'train_loss_steps' in history:
        steps = range(1, len(history['train_loss_steps']) + 1)
        plt.plot(steps, history['train_loss_steps'], alpha=0.3, color='blue', label='Train Loss (Batch)')
        plt.plot(steps, smooth_curve(history['train_loss_steps']), color='blue', label='Train Loss (Smoothed)')
    elif 'train_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
        
    if 'val_loss' in history:
        # If we have steps, we map epochs to steps. 
        # For simplicity, if we have both, we place val_loss markers at the end of epochs.
        if 'train_loss_steps' in history and len(history['train_loss']) > 0:
            step_interval = len(history['train_loss_steps']) // len(history['val_loss'])
            val_steps = [(i+1) * step_interval for i in range(len(history['val_loss']))]
            plt.plot(val_steps, history['val_loss'], 'rs-', linewidth=2, label='Val Loss (Epoch)')
        else:
            epochs = range(1, len(history['val_loss']) + 1)
            plt.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')

    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Progress (Steps/Epochs)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Plot Accuracy
    plt.subplot(1, 2, 2)
    if 'train_acc_steps' in history:
        steps = range(1, len(history['train_acc_steps']) + 1)
        plt.plot(steps, history['train_acc_steps'], alpha=0.3, color='green', label='Train Acc (Batch)')
        plt.plot(steps, smooth_curve(history['train_acc_steps']), color='green', label='Train Acc (Smoothed)')
    elif 'train_acc' in history:
        epochs = range(1, len(history['train_acc']) + 1)
        plt.plot(epochs, history['train_acc'], 'g-o', label='Train Acc')
        
    if 'val_acc' in history:
        if 'train_acc_steps' in history and len(history['train_acc']) > 0:
            step_interval = len(history['train_acc_steps']) // len(history['val_acc'])
            val_steps = [(i+1) * step_interval for i in range(len(history['val_acc']))]
            plt.plot(val_steps, history['val_acc'], 'rs-', linewidth=2, label='Val Acc (Epoch)')
        else:
            epochs = range(1, len(history['val_acc']) + 1)
            plt.plot(epochs, history['val_acc'], 'r-o', label='Val Acc')

    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Progress (Steps/Epochs)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    print(f"Saved detailed training history plot to {save_path}")
    plt.close()

def plot_confusion_matrix_sns(y_true, y_pred, classes, save_path='metrics/confusion_matrix.png'):
    """
    Plots a confusion matrix using seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved confusion matrix plot to {save_path}")
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

def get_risk_level(risk_score):
    """
    Maps continuous risk score to 4 discrete categories.
    Using inclusive thresholds to ensure boundary samples (like 0.85) are captured.
    """
    if risk_score >= 0.9:
        return 3 # CRITICAL
    elif risk_score >= 0.85:
        return 2 # HIGH
    elif risk_score >= 0.8:
        return 1 # MODERATE
    else:
        return 0 # LOW

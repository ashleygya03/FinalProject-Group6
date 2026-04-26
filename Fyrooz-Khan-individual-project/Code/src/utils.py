"""Utility functions for data processing and visualization."""
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def add_class_name_prefix(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Adds a class name prefix to the given filename column in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input pandas DataFrame.
        col_name (str): Name of the column containing filenames.
        
    Returns:
        pd.DataFrame: DataFrame with modified column.
    """
    df_copy = df.copy()
    df_copy[col_name] = df_copy[col_name].apply(
        lambda x: x[re.search("_", x).start() + 1 : re.search("_", x).start() + 2]
        + "/" + x
    )
    return df_copy

def plot_class_distribution(df: pd.DataFrame, categories: dict, save_path: str = None):
    """
    Plots the distribution of classes in the given dataset.
    """
    label, count = np.unique(df.category, return_counts=True)
    uni = pd.DataFrame(data=count, index=categories.values(), columns=['Count'])
    plt.figure(figsize=(14, 4), dpi=200)
    sns.barplot(data=uni, x=uni.index, y='Count', palette='icefire', width=0.4).set_title('Class distribution in Dataset', fontsize=15)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_training_history(history, save_path: str = None):
    """
    Plots training loss and accuracy from history.
    Accepts either a Keras History object or a dict loaded from JSON.
    """
    # Support both Keras history objects and plain dicts (loaded from JSON)
    if hasattr(history, 'history'):
        data = history.history
    else:
        data = history

    error = pd.DataFrame(data)
    plt.figure(figsize=(18, 5), dpi=200)
    sns.set_style('darkgrid')

    plt.subplot(121)
    plt.title('Cross Entropy Loss', fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.plot(error['loss'], label='Train')
    plt.plot(error['val_loss'], label='Validation')
    plt.legend()

    plt.subplot(122)
    plt.title('Classification Accuracy', fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.plot(error['accuracy'], label='Train')
    plt.plot(error['val_accuracy'], label='Validation')
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_confusion_matrix(confusion_mtx, categories_values, save_path: str = None):
    """
    Plots a confusion matrix heatmap.
    """
    sns.set_style('ticks')
    f, ax = plt.subplots(figsize=(20, 18), dpi=200)
    sns.heatmap(confusion_mtx, annot=True, 
                linewidths=0.1, cmap="gist_yarg_r", 
                linecolor="black", fmt='.0f', ax=ax, 
                cbar=False, xticklabels=categories_values, 
                yticklabels=categories_values)
    plt.xlabel("Predicted Label", fontdict={'color': 'red', 'size': 20})
    plt.ylabel("True Label", fontdict={'color': 'green', 'size': 20})
    plt.title("Confusion Matrix", fontdict={'color': 'brown', 'size': 25})
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_per_class_f1(y_true, y_pred, class_names, save_path: str = None):
    """
    Plots a per-class F1-score bar chart.
    """
    from sklearn.metrics import f1_score
    per_class_f1 = f1_score(y_true, y_pred, average=None)

    plt.figure(figsize=(16, 5), dpi=200)
    sns.set_style('darkgrid')
    colors = sns.color_palette('viridis', len(class_names))
    sns.barplot(x=class_names, y=per_class_f1, palette=colors)
    plt.title('Per-Class F1-Score', fontsize=15)
    plt.xlabel('ASL Sign', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

    return per_class_f1

def print_top_confused_pairs(confusion_mtx, class_names, top_n: int = 10, save_path: str = None):
    """
    Extracts and prints the top N most confused class pairs from a confusion matrix.
    Optionally saves the table to a CSV file.
    """
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and confusion_mtx[i][j] > 0:
                confused_pairs.append({
                    'True Label': class_names[i],
                    'Predicted As': class_names[j],
                    'Count': int(confusion_mtx[i][j])
                })

    df = pd.DataFrame(confused_pairs)
    df = df.sort_values('Count', ascending=False).head(top_n).reset_index(drop=True)

    print(f"\nTop {top_n} Most Confused Pairs:")
    print(df.to_string(index=False))

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    return df

def compute_top_k_accuracy(predictions, y_true, k: int = 5):
    """
    Computes Top-K accuracy from prediction probabilities.
    
    Args:
        predictions: numpy array of shape (n_samples, n_classes) with probabilities.
        y_true: numpy array of true labels (integer indices).
        k: number of top predictions to consider.
    
    Returns:
        float: Top-K accuracy as a decimal (e.g. 0.95 for 95%).
    """
    top_k_correct = 0
    for i in range(len(predictions)):
        top_k_preds = np.argsort(predictions[i])[-k:]
        if y_true[i] in top_k_preds:
            top_k_correct += 1
    return top_k_correct / len(y_true)

def save_history(history, save_path: str):
    """
    Saves a Keras training history object to a JSON file.
    """
    import json
    history_dict = history.history
    # Convert numpy floats to Python floats for JSON serialization
    serializable = {k: [float(v) for v in vals] for k, vals in history_dict.items()}
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved training history: {save_path}")

def load_history(load_path: str) -> dict:
    """
    Loads a training history dict from a JSON file.
    """
    import json
    with open(load_path, 'r') as f:
        return json.load(f)
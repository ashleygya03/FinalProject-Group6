"""Utility functions for data processing and visualization."""
import re
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
        lambda x: x[re.search("_", x).start() + 1: re.search("_", x).start() + 2]
                  + "/" + x
    )
    return df_copy

def plot_class_distribution(df: pd.DataFrame, categories: dict):
    """
    Plots the distribution of classes in the given dataset.
    """
    label, count = np.unique(df.category, return_counts=True)
    uni = pd.DataFrame(data=count, index=categories.values(), columns=['Count'])
    plt.figure(figsize=(14, 4), dpi=200)
    sns.barplot(data=uni, x=uni.index, y='Count', palette='icefire', width=0.4).set_title('Class distribution in Dataset', fontsize=15)
    plt.show()



def plot_training_history(history):
    """
    Plots training loss and accuracy from history.
    """
    error = pd.DataFrame(history.history)
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

    plt.show()

def plot_confusion_matrix(confusion_mtx, categories_values):
    """
    Plots a confusion matrix heatmap.
    """
    sns.set_style('ticks')
    f, ax = plt.subplots(figsize=(20, 8), dpi=200)
    sns.heatmap(confusion_mtx, annot=True,
                linewidths=0.1, cmap="gist_yarg_r",
                linecolor="black", fmt='.0f', ax=ax,
                cbar=False, xticklabels=categories_values,
                yticklabels=categories_values)
    plt.xlabel("Predicted Label", fontdict={'color': 'red', 'size': 20})
    plt.ylabel("True Label", fontdict={'color': 'green', 'size': 20})
    plt.title("Confusion Matrix", fontdict={'color': 'brown', 'size': 25})
    plt.show()

# this is the starting point for the project

"""Main module to run the ASL model training and evaluation."""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.config import IMAGE_SIZE, IMG_CHANNEL, N_CLASSES, CATEGORIES
from src.dataset import ASLDataset
from src.model import ASLModel
from src.utils import plot_training_history, plot_confusion_matrix


def main():
    # 1. Prepare Data
    print("Preparing dataset...")
    dataset = ASLDataset()

    # Check if data is already split, if not, split it
    import os
    if not os.path.exists(dataset.train_path):
        print("Splitting raw data into train/val/test...")
        dataset.split_data()

    train_data, val_data, test_data = dataset.get_generators()
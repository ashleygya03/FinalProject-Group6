"""Data handling class for the ASL project."""
import os
import shutil
import pandas as pd
import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import BASE_PATH, WORKING_PATH, CATEGORIES, IMAGE_SIZE, BATCH_SIZE


class ASLDataset:
    """Class to manage dataset loading, splitting, and generators."""

    def __init__(self, base_path: str = BASE_PATH, working_path: str = WORKING_PATH):
        """Initialize the dataloader with paths."""
        self.base_path = base_path
        self.working_path = working_path
        self.train_path = os.path.join(self.working_path, 'train')
        self.val_path = os.path.join(self.working_path, 'val')
        self.test_path = os.path.join(self.working_path, 'test')
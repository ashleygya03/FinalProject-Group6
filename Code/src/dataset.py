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

    def generate_dataframe(self) -> pd.DataFrame:
        """Generates a dataframe with filenames and categories from the raw dataset."""
        filenames_list = []
        categories_list = []
        for category in CATEGORIES:
            path = os.path.join(self.base_path, CATEGORIES[category])
            if os.path.exists(path):
                filenames = os.listdir(path)
                filenames_list.extend(filenames)
                categories_list.extend([category] * len(filenames))

        df = pd.DataFrame({"filename": filenames_list, "category": categories_list})
        df = df.sample(frac=1).reset_index(drop=True)
        return df

    def split_data(self, seed: int = 1333, ratio: tuple = (0.8, 0.1, 0.1)):
        """Splits raw data into train, val, test folders using splitfolders."""
        splitfolders.ratio(self.base_path, output=self.working_path, seed=seed, ratio=ratio)

        # Cleanup the nested folder if exists (to match the kaggle environment behavior)
        for split in ['train', 'val', 'test']:
            nested_path = os.path.join(self.working_path, split, 'asl_dataset')
            if os.path.exists(nested_path):
                shutil.rmtree(nested_path)

    def get_generators(self):
        """Returns train, validation, and test image data generators."""
        datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_data = datagen.flow_from_directory(
            directory=self.train_path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        val_data = datagen.flow_from_directory(
            directory=self.val_path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        test_data = datagen.flow_from_directory(
            directory=self.test_path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        return train_data, val_data, test_data
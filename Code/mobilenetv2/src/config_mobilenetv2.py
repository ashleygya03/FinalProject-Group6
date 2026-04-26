"""Configuration constants for the ASL classification project."""
import os

# Original Kaggle-only path (use this if NOT merging datasets):
# BASE_PATH = "../Data/asl_alphabet_train/asl_alphabet_train/"

# Merged dataset path (use this AFTER running merge_datasets.py):
BASE_PATH = "../Data_3/merged-asl"
WORKING_PATH = "./working/"
TRAIN_PATH = os.path.join(WORKING_PATH, 'train')
VAL_PATH = os.path.join(WORKING_PATH, 'val')
TEST_PATH = os.path.join(WORKING_PATH, 'test')

IMAGE_SIZE = 224
IMG_CHANNEL = 3
BATCH_SIZE = 32
N_CLASSES = 26
SEED = 1333

CATEGORIES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
}

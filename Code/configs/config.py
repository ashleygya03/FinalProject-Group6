"""Configuration constants for the ASL classification project."""
import os

# Original Kaggle-only path (use this if NOT merging datasets):
# BASE_PATH = "../Data/asl_alphabet_train/asl_alphabet_train/"

# Merged dataset path (use this AFTER running merge_datasets.py):
BASE_PATH = os.path.join("data", "merged_asl_dataset")
WORKING_PATH = "./working/"
TRAIN_PATH = os.path.join(WORKING_PATH, 'train')
VAL_PATH = os.path.join(WORKING_PATH, 'val')
TEST_PATH = os.path.join(WORKING_PATH, 'test')

OUTPUTS_PATH = "./outputs/"
ARTIFACTS_PATH = "./artifacts/"

# Image Settings
IMAGE_SIZE = 224
IMG_CHANNEL = 3
BATCH_SIZE = 32
SEED = 1333

CATEGORIES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
}

N_CLASSES = len(CATEGORIES)

# Training Settings
TRAINING_CONFIG = {
    "epochs_phase1": 10,
    "epochs_phase2": 10,
    "batch_size": BATCH_SIZE,
    "optimizer": "adam",
    "loss": "categorical_crossentropy"
}

# MobileNetV2 Settings
MOBILENETV2_CONFIG = {
    "name": "mobilenetv2",
    "fine_tune_at": 100,
    "learning_rates": {
        "phase1": 1e-3,
        "phase2": 1e-5
    },
    "dropout_rate": 0.3,
    "output_dir": os.path.join(OUTPUTS_PATH, "mobilenetv2"),
    "checkpoint_path": os.path.join(ARTIFACTS_PATH, "best_mobilenetv2.keras")
}

# ResNet50 Settings
RESNET50_CONFIG = {
    "name": "resnet50",
    "fine_tune_at": 140,
    "learning_rates": {
        "phase1": 1e-3,
        "phase2": 1e-5
    },
    "dropout_rate": 0.3,
    "output_dir": os.path.join(OUTPUTS_PATH, "resnet50"),
    "checkpoint_path": os.path.join(ARTIFACTS_PATH, "best_resnet50.keras")
}

# Shared Callbacks
CALLBACKS_CONFIG = {
    "early_stopping": {
        "monitor": "val_loss",
        "patience": 3
    },
    "reduce_lr": {
        "monitor": "val_loss",
        "factor": 0.2,
        "patience": 2
    },
    "checkpoint": {
        "monitor": "val_accuracy",
        "save_best_only": True
    }
}
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# Path Resolution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train"

def get_data_generators(batch_size, img_size, val_split=0.2):
    """
    Creates training and validation generators with real-time augmentation.
    """
    print(f"--- Initializing Data Generators from: {DATA_DIR} ---")

    # 1. Define Augmentation for Training
    train_datagen = ImageDataGenerator(
        rescale=1./255,             # Normalize pixels to [0, 1]
        rotation_range=15,          # Randomly rotate images
        width_shift_range=0.1,      # Randomly translate horizontally
        height_shift_range=0.1,     # Randomly translate vertically
        brightness_range=[0.9, 1.1],# Simulate different lighting
        horizontal_flip=True,       # Some signs might be reversible
        validation_split=val_split   # Reserve part of data for validation
    )

    # 2. Validation generator only rescales (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split
    )

    # 3. Create the Training Stream
    train_generator = train_datagen.flow_from_directory(
        str(DATA_DIR),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',     # Multi-class labels
        subset='training',
        shuffle=True,
        seed=123
    )

    # 4. Create the Validation Stream
    val_generator = val_datagen.flow_from_directory(
        str(DATA_DIR),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=123
    )

    return train_generator, val_generator
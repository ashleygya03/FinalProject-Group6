import os
import cv2
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from pathlib import Path

# ==========================================
# CONFIGURATION / CONSTANTS
# ==========================================
# --- Paths ---

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# Define paths relative to the project root
DATA_DIR = PROJECT_ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train"
OUTPUT_DIR = PROJECT_ROOT / "models"

# --- Hyperparameters ---
N_EPOCH = 15
BATCH_SIZE = 128   
LR = 1e-4          
IMAGE_SIZE = 224   
VAL_SPLIT = 0.2
SAVE_MODEL = True


# ==========================================

# ==========================================

def build_network(num_classes):
    """Defines the MobileNetV2 architecture."""
    print("--- Building Model Architecture ---")
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    
    # Preprocessing integrated into the model
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Base Model (Fully trainable)
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = True

    # Classification Head
    x = base_model(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model

def read_data():
    """Loads dataset using the dynamic DATA_DIR path."""
    print(f"--- Searching for data in: {DATA_DIR} ---")
    
    # Verify the directory exists before starting
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Could not find data at {DATA_DIR}. Please ensure your dataset is in the 'data' folder.")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(DATA_DIR),
        validation_split=VAL_SPLIT,
        subset="training",
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(DATA_DIR),
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
    )
    
    class_names = train_ds.class_names
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names

def model_setup(num_classes):
    """Configures optimizer and compilation."""
    model = build_network(num_classes)
    optimizer = keras.optimizers.Adam(learning_rate=LR)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    # 1. Data Setup
    train_ds, val_ds, class_names = read_data()

    # 2. Model Setup
    model = model_setup(len(class_names))
    model.summary()
    
    # 3. Output Directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "mobilenetv2_best.h5"),
            monitor="val_accuracy",
            save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.5, 
            patience=2, 
            verbose=1
        )
    ]

    # 5. Training
    print(f"\n--- Starting Unified Training for {N_EPOCH} Epochs ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=N_EPOCH,
        callbacks=callbacks
    )

    # 6. Save Artifacts
    if SAVE_MODEL:
        model.save(OUTPUT_DIR / "mobilenetv2_final.h5")
        with open(OUTPUT_DIR / "class_indices.json", "w") as f:
            json.dump({"class_names": class_names}, f)
        print(f"--- Training Complete. Files saved to {OUTPUT_DIR} ---")

if __name__ == "__main__":
    # Auto-detect GPU for AWS/Mac/Local
    device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    with tf.device(device_name):
        main()
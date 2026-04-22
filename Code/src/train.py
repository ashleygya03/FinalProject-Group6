import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from pathlib import Path
import json

# Import from preprocess.py
from preprocess import get_data_generators

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "models"

# --- Hyperparameters ---
N_EPOCH = 15
BATCH_SIZE = 128   
LR = 1e-4          
IMAGE_SIZE = 224   

def create_model(num_classes):
    """Defines the MobileNetV2 architecture."""
    print("--- Building Model Architecture ---")

    # Base Model (MobileNetV2)
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = True

    # Classification Head
    model = keras.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        base_model,
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

def main():
    # 1. Load Data from preprocess.py
    train_ds, val_ds = get_data_generators(BATCH_SIZE, IMAGE_SIZE)
    num_classes = train_ds.num_classes

    # 2. Model Setup
    model = create_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy", # CRITICAL CHANGE
        metrics=["accuracy"]
    )

    # 2. Map indices to labels for Streamlit
    class_indices = train_ds.class_indices
    class_names = list(class_indices.keys())
    
    # 3. Output Directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the class mapping immediately
    with open(OUTPUT_DIR / "class_indices.json", "w") as f:
        json.dump({"class_names": class_names}, f)

    # 4. Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "mobilenetv2_best.keras"),
            monitor="val_accuracy",
            save_best_only=True
        ),
        keras.callbacks.EarlyStopping(patience=3, monitor="val_loss")
    ]

    # 5. Training
    print(f"\n--- Starting Unified Training for {N_EPOCH} Epochs ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=N_EPOCH,
        callbacks=callbacks
    )

    # 7. Save Artifacts
    model.save(OUTPUT_DIR / "mobilenetv2_final.keras")
    print(f"--- Training Complete. Models saved to {OUTPUT_DIR} ---")

if __name__ == "__main__":
    main()
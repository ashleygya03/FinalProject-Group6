import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

"""
Model Evaluation & Diagnostic Reporting
---------------------------------------
Runs the trained .keras model against an unseen test set. Generates 
a Classification Report (Precision, Recall, F1) and a Confusion Matrix 
to identify specific gesture misclassifications and background biases.
"""

# Import from preprocess.py
from preprocess import get_data_generators

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parent / "models" / "mobilenetv2_robust.keras"
MAPPING_PATH = SCRIPT_DIR.parent / "models" / "class_indices.json"

def run_test():
    # 1. Load Model and Class Names
    print("Loading model and class mapping...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(MAPPING_PATH, 'r') as f:
        class_names = json.load(f)["class_names"]

    # 2. Get Validation/Test Data
    
    _, val_gen = get_data_generators(batch_size=1, img_size=224)
    val_gen.shuffle = False  
    val_gen.reset()

    y_true = val_gen.classes
    print(f"Evaluating model on {len(y_true)} images...")

    # 3. Generate Predictions
    preds = model.predict(val_gen, steps=len(val_gen), verbose=1)
    y_pred = np.argmax(preds, axis=1)

    # 4. Classification Report (Precision, Recall, F1-Score)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n--- Classification Report ---")
    print(report)

    # 5. Confusion Matrix Visualization
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("ASL Alphabet Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Save chart
    plt.savefig(SCRIPT_DIR.parent / "models" / "confusion_matrix.png")
    print(f"\nSuccess! Confusion Matrix saved to {SCRIPT_DIR.parent}/models/")

if __name__ == "__main__":
    run_test()
"""Main module to run the ASL model training and evaluation."""
import os
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from src.config_mobilenetv2 import IMAGE_SIZE, IMG_CHANNEL, N_CLASSES, CATEGORIES
from src.dataset_mobilenetv2 import ASLDataset
from src.model_mobilenetv2 import ASLModel
from src.utils import (
    plot_training_history, plot_confusion_matrix, plot_per_class_f1,
    print_top_confused_pairs, compute_top_k_accuracy, save_history
)

# Output directory for all saved figures and results
OUTPUT_DIR = "outputs_mobilenetv2"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Prepare Data
    print("Preparing dataset...")
    dataset = ASLDataset()

    if not os.path.exists(dataset.train_path):
        print("Splitting raw data into train/val/test...")
        dataset.split_data()

    train_data, val_data, test_data = dataset.get_generators()

    # 2. Build Model
    print("Building model...")
    asl_model_class = ASLModel(IMAGE_SIZE, IMG_CHANNEL, N_CLASSES)
    model = asl_model_class.get_model()
    model.summary()
    print(f"\nTotal Parameters: {model.count_params():,}")

    # 3. Train Model
    print("Training model...")

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        ),
        ModelCheckpoint(
            "best_asl_model.keras",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    # phase 1: train classifier head
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    # Save phase 1 training history
    save_history(history1, os.path.join(OUTPUT_DIR, "history_phase1.json"))

    # phase 2: fine-tuning
    asl_model_class.unfreeze_top_layers(fine_tune_at=100)

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    # Save phase 2 training history
    save_history(history2, os.path.join(OUTPUT_DIR, "history_phase2.json"))

    # load best saved model before evaluation
    model = load_model("best_asl_model.keras")

    # save class names for Streamlit inference
    class_names = list(CATEGORIES.values())
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    # 4. Evaluate Model
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    loss, acc = model.evaluate(train_data, verbose=0)
    print(f'Training Accuracy: {acc*100:.2f}%, Loss: {loss:.4f}')

    loss, acc = model.evaluate(val_data, verbose=0)
    print(f'Validation Accuracy: {acc*100:.2f}%, Loss: {loss:.4f}')

    test_loss, test_acc = model.evaluate(test_data, verbose=0)
    print(f'Testing Accuracy: {test_acc*100:.2f}%, Loss: {test_loss:.4f}')

    # Plot training history (both phases)
    plot_training_history(history1, save_path=os.path.join(OUTPUT_DIR, "training_curves_phase1.png"))
    plot_training_history(history2, save_path=os.path.join(OUTPUT_DIR, "training_curves_phase2.png"))

    # 5. Predict & Metrics
    print("\nRunning predictions...")
    result = model.predict(test_data, verbose=0)
    y_pred = np.argmax(result, axis=1)
    y_true = test_data.labels

    # ---- Top-5 Accuracy ----
    top5_acc = compute_top_k_accuracy(result, y_true, k=5)
    print(f'\nTop-5 Accuracy: {top5_acc*100:.2f}%')

    # ---- Macro Precision, Recall, F1 ----
    macro_prec = precision_score(y_true, y_pred, average='macro')
    macro_rec = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    print(f'\n{"="*40}')
    print("METRICS SUMMARY TABLE (for slides)")
    print(f'{"="*40}')
    print(f'{"Metric":<20} {"Value":>10}')
    print(f'{"-"*40}')
    print(f'{"Accuracy":<20} {test_acc*100:>9.2f}%')
    print(f'{"Macro Precision":<20} {macro_prec*100:>9.2f}%')
    print(f'{"Macro Recall":<20} {macro_rec*100:>9.2f}%')
    print(f'{"Macro F1-Score":<20} {macro_f1*100:>9.2f}%')
    print(f'{"Top-5 Accuracy":<20} {top5_acc*100:>9.2f}%')
    print(f'{"Parameters":<20} {model.count_params():>10,}')

    # Save metrics to JSON for easy reference
    metrics = {
        "accuracy": round(test_acc * 100, 2),
        "macro_precision": round(macro_prec * 100, 2),
        "macro_recall": round(macro_rec * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "top5_accuracy": round(top5_acc * 100, 2),
        "parameters": model.count_params(),
        "test_loss": round(test_loss, 4)
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f'\nMetrics saved to {OUTPUT_DIR}/metrics.json')

    # ---- Full Classification Report ----
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # Save classification report to text file
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # ---- Confusion Matrix ----
    print("\nPlotting Confusion Matrix...")
    conf_mtx = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(conf_mtx, class_names, save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # ---- Per-Class F1 Bar Chart ----
    print("\nPlotting Per-Class F1-Score...")
    plot_per_class_f1(y_true, y_pred, class_names, save_path=os.path.join(OUTPUT_DIR, "per_class_f1.png"))

    # ---- Top 10 Most Confused Pairs ----
    print_top_confused_pairs(conf_mtx, class_names, top_n=10, save_path=os.path.join(OUTPUT_DIR, "top_confused_pairs.csv"))

    print(f'\n{"="*60}')
    print(f'All outputs saved to: {OUTPUT_DIR}/')
    print(f'{"="*60}')

if __name__ == "__main__":
    main()
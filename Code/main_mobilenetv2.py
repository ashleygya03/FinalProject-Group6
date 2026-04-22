"""Main module to run the ASL model training and evaluation."""
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from src.config_mobilenetv2 import IMAGE_SIZE, IMG_CHANNEL, N_CLASSES, CATEGORIES
from src.dataset_mobilenetv2 import ASLDataset
from src.model_mobilenetv2 import ASLModel
from src.utils import plot_training_history, plot_confusion_matrix

def main():
    # 1. Prepare Data
    print("Preparing dataset...")
    dataset = ASLDataset()

    import os
    if not os.path.exists(dataset.train_path):
        print("Splitting raw data into train/val/test...")
        dataset.split_data()

    train_data, val_data, test_data = dataset.get_generators()

    # 2. Build Model
    print("Building model...")
    asl_model_class = ASLModel(IMAGE_SIZE, IMG_CHANNEL, N_CLASSES)
    model = asl_model_class.get_model()
    model.summary()

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

    # load best saved model before evaluation
    model = load_model("best_asl_model.keras")

    # save class names for Streamlit inference
    import json

    class_names = list(CATEGORIES.values())
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    # 4. Evaluate Model
    print("Evaluating model...")
    loss, acc = model.evaluate(train_data, verbose=0)
    print(f'Training Accuracy: {acc*100:.2f}%, Loss: {loss:.4f}')

    loss, acc = model.evaluate(val_data, verbose=0)
    print(f'Validation Accuracy: {acc*100:.2f}%, Loss: {loss:.4f}')

    loss, acc = model.evaluate(test_data, verbose=0)
    print(f'Testing Accuracy: {acc*100:.2f}%, Loss: {loss:.4f}')

    # Plot history
    plot_training_history(history1)
    plot_training_history(history2)

    # 5. Predict & Metrics
    print("Running predictions...")
    result = model.predict(test_data, verbose=0)
    y_pred = np.argmax(result, axis=1)
    y_true = test_data.labels

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(CATEGORIES.values())))

    print("\nPlotting Confusion Matrix...")
    conf_mtx = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(conf_mtx, list(CATEGORIES.values()))

if __name__ == "__main__":
    main()
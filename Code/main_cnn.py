# this is the starting point for the project

"""Main module to run the ASL model training and evaluation."""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.config_cnn import IMAGE_SIZE, IMG_CHANNEL, N_CLASSES, CATEGORIES
from src.dataset_cnn import ASLDataset
from src.model_cnn import ASLModel
from src.utils import plot_training_history, plot_confusion_matrix



def main():
    # 1. Prepare Data
    print("Preparing dataset...")
    dataset = ASLDataset()

    # Check if data is already split, if not, split it
    import os
    if not os.path.exists(dataset.train_path):
        print("Splitting raw data into train/val/test...")
        dataset.split_data()

    train_data, val_data, test_data = dataset.get_generators()

    # 2. Build Model
    print("Building model...")
    asl_model_class = ASLModel(IMAGE_SIZE, IMG_CHANNEL, N_CLASSES)
    model = asl_model_class.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 3. Train Model
    print("Training model...")
    early_stoping = EarlyStopping(
        monitor='val_loss', min_delta=0.001, patience=5,
        restore_best_weights=True, verbose=0
    )
    reduce_learning_rate = ReduceLROnPlateau(
        monitor='val_accuracy', patience=2, factor=0.5, verbose=1
    )

    # Starting the training
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,
        callbacks=[early_stoping, reduce_learning_rate],
        verbose=1
    )

    # 4. Evaluate Model
    print("Evaluating model...")
    loss, acc = model.evaluate(train_data, verbose=0)
    print(f'Training Accuracy: {acc * 100:.2f}%, Loss: {loss:.4f}')

    loss, acc = model.evaluate(val_data, verbose=0)
    print(f'Validation Accuracy: {acc * 100:.2f}%, Loss: {loss:.4f}')

    loss, acc = model.evaluate(test_data, verbose=0)
    print(f'Testing Accuracy: {acc * 100:.2f}%, Loss: {loss:.4f}')

    # Plot history using utils
    plot_training_history(history)
    print("Closing plots to continue execution...")

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

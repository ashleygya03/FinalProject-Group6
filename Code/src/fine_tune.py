import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# 1. Configuration
DATA_DIR = "/home/ubuntu/FinalProject-Group6/data/asl_alphabet_train/asl_alphabet_train"
MODEL_PATH = "/home/ubuntu/FinalProject-Group6/Code/models/mobilenetv2_final.keras"
NEW_MODEL_PATH = "/home/ubuntu/FinalProject-Group6/Code/models/mobilenetv2_robust.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 128 

# 2. Data Generators 
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation'
)

# 3. Load & Prep Model
print("--- Loading existing 96% accuracy model ---")
model = load_model(MODEL_PATH)

# Freeze the first 100 layers (basic shapes/edges)
# Suggested by Gemini - This prevents the model from "forgetting" its foundational knowledge
for layer in model.layers[:100]:
    layer.trainable = False

# 4. Re-compile with a VERY low learning rate for careful adjustments 
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Fine-Tune
print("--- Starting Fine-Tuning for 5 Epochs ---")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(NEW_MODEL_PATH, save_best_only=True)
    ]
)

print(f"--- Robust model saved to {NEW_MODEL_PATH} ---")
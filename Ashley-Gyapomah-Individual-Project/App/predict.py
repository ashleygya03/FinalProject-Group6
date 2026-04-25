import json
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMAGE_SIZE = 224

def predict_image(image, 
                  model_path="../models/mobilenetv2_robust.keras", 
                  class_file="../models/class_indices.json"):
    
    # 1. Load the full model directly 
    # This automatically includes the architecture + weights
    model = tf.keras.models.load_model(model_path)

    # 2. Load Class Names (Handle your specific JSON structure)
    with open(class_file, "r") as f:
        data = json.load(f)
        # Handles both {"class_names": [...]} and [...] structures
        class_names = data["class_names"] if isinstance(data, dict) and "class_names" in data else data

    # 3. Preprocess the PIL image
    # Ensure it is RGB (3 channels) and the correct size
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

# Standard MobileNetV2 normalization (-1 to 1)
    arr = np.array(image, dtype=np.float32)
    arr = preprocess_input(arr)
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    arr = np.expand_dims(arr, axis=0)

    # 4. Inference
    preds = model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)

    return class_names[idx], float(preds[idx]), preds
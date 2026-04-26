import json
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.model_mobilenetv2 import ASLModel
from src.config_mobilenetv2 import IMAGE_SIZE, IMG_CHANNEL, N_CLASSES

def predict_image(image, model_path="best_asl_model.keras", class_file="class_names.json", image_size=IMAGE_SIZE):
    # Rebuild the model architecture
    asl = ASLModel(image_size, IMG_CHANNEL, N_CLASSES)
    model = asl.get_model()
    
    # Load only the weights to bypass Keras 2 vs Keras 3 architecture incompatibilities
    model.load_weights(model_path)

    with open(class_file, "r") as f:
        class_names = json.load(f)

    # Resize image based on expected model input
    image = image.convert("RGB").resize((image_size, image_size))
    arr = np.array(image, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)

    return class_names[idx], float(preds[idx]), preds

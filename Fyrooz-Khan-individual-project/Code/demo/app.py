import os
import json
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from predict import predict_image

st.set_page_config(page_title="ASL Sign Detector", layout="centered")
st.title("ASL Sign Language Detection")
st.write("Take a photo of your Right hand sign. The app will detect the hand, crop it, and predict the sign.")

HAND_MODEL_PATH = "hand_landmarker.task"   # keep this file in the same folder as app.py

@st.cache_resource
def load_hand_landmarker():
    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )
    return vision.HandLandmarker.create_from_options(options)

def detect_and_crop_hand(pil_image, padding=0.10):
    image_rgb = np.array(pil_image.convert("RGB"))
    h, w, _ = image_rgb.shape

    detector = load_hand_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None, None, None

    hand_landmarks = result.hand_landmarks[0]

    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]

    min_x = max(int(min(xs) * w - padding * w), 0)
    min_y = max(int(min(ys) * h - padding * h), 0)
    max_x = min(int(max(xs) * w + padding * w), w)
    max_y = min(int(max(ys) * h + padding * h), h)

    boxed = image_rgb.copy()
    cv2.rectangle(boxed, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    cropped = image_rgb[min_y:max_y, min_x:max_x]

    boxed_pil = Image.fromarray(boxed)
    cropped_pil = Image.fromarray(cropped)

    bbox = (min_x, min_y, max_x, max_y)
    return boxed_pil, cropped_pil, bbox

camera_image = st.camera_input("Capture a hand sign")

if camera_image is not None:
    image = Image.open(camera_image).convert("RGB")
    st.image(image, caption="Captured Image")

    boxed_img, cropped_img, bbox = detect_and_crop_hand(image)

    if cropped_img is None:
        st.error("No hand detected. Try again with better lighting and keep only one hand in the frame.")
    else:
        st.subheader("Detected hand region")
        st.image(boxed_img, caption="Bounding box")

        st.subheader("Cropped hand")
        st.image(cropped_img, caption="Image used for prediction")

        label, confidence, preds = predict_image(cropped_img)

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2%}")

        with open("class_names.json", "r") as f:
            class_names = json.load(f)

        top_idx = np.argsort(preds)[::-1][:5]
        top_df = {
            "Class": [class_names[i] for i in top_idx],
            "Probability": [float(preds[i]) for i in top_idx]
        }

        st.subheader("Top 5 predictions")
        st.dataframe(top_df)

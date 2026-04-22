import streamlit as st
from PIL import Image
from predict import predict_image
import numpy as np
import pandas as pd
import json

st.set_page_config(page_title="ASL Sign Detector", layout="centered")
st.title("ASL Sign Language Detection")
st.write("Take a photo of your hand sign and the model will predict the class.")

camera_image = st.camera_input("Capture a hand sign")

if camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption="Captured Image")

    label, confidence, preds = predict_image(image)

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2%}")

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    df = pd.DataFrame({
        "Class": class_names,
        "Probability": preds
    }).sort_values("Probability", ascending=False).head(5)

    st.subheader("Top 5 predictions")
    st.dataframe(df, use_container_width=True)
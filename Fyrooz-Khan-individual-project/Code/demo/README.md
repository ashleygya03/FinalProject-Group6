# ASL Streamlit Demo

This folder contains a Streamlit demo for American Sign Language (ASL) sign prediction using a trained MobileNetV2-based model. The app captures an image, processes it, and predicts the hand sign class.

## Files
- `app.py` — Streamlit app
- `predict.py` — prediction logic
- `requirements.txt` — required packages
- `class_names.json` — class labels
- `hand_landmarker.task` — MediaPipe hand detection model
- `best_asl_model.keras` or `best_asl.weights.h5` — trained model file
- `src/` — model and config files

## Run the demo

Open terminal in this `demo` folder and run:

```bash
streamlit run app.py

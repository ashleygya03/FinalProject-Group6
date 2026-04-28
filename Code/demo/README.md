# ASL Streamlit Demo

This folder contains a Streamlit demo for American Sign Language (ASL) sign prediction using a trained MobileNetV2-based model. The app captures an image, processes it, and predicts the hand sign class.

## Files
- `ASL_Alphabet_list.jpg` — American Alphabets chart
- `app.py` — Streamlit app
- `predict.py` — prediction logic
- `requirements_demo.txt` — required packages
- `class_names.json` — class labels
- `hand_landmarker.task` — MediaPipe hand detection model
- `best_asl_model.keras` — trained model file (this file must be in this directory to run `app.py`)
- `src/` — model and config files


## Download the trained model

- [Model](https://drive.google.com/file/d/1XS6T9E1l-4LDbOpx_vQQmSwPpCUsSE2t/view?usp=drive_link)

Or download it from terminal inside the `demo` folder:

```bash
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1XS6T9E1l-4LDbOpx_vQQmSwPpCUsSE2t" -O best_asl_model.keras
```

## Install dependencies:

```bash
pip install -r requirements_demo.txt
```

## Run the demo

```bash
streamlit run app.py

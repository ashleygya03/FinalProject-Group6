# Code

This folder contains all code for the final project, including data preprocessing, model training, and the Streamlit demo.

## Folder structure

```text
Code/
│
├── data-preprocessing/
│   ├── README.md
│   ├── compare_datasets.py
│   ├── preprocess.py
│   ├── crop_hands.py
│   ├── brightness_filter.py
│   ├── merge_datasets.py
│   ├── hand_landmarker.task
│   └── requirements.txt
│
├── mobilenetv2/
│   ├── README.md
│   ├── main_mobilenetv2.py
│   ├── requirements.txt
│   ├── src/
│   └── tests/
│
├── resnet50/
│   ├── README.md
│   ├── main_resnet50.py
│   ├── requirements.txt
│   ├── src/
│   └── tests/
│
└── demo/
    ├── README.md
    ├── app.py
    ├── predict.py
    ├── requirements_demo.txt
    ├── class_names.json
    ├── hand_landmarker.task
    ├── best_asl_model.keras
    └── src/

## Run Order

### 1. `data-preprocessing/`
Run scripts in this order:
1. `compare_datasets.py` — compares the two ASL datasets
2. `preprocess.py` — crops and filters the 87K dataset
3. `merge_datasets.py` — merges datasets into the final balanced dataset

---

### 2. `resnet50/` and `mobilenetv2/`
Run after preprocessing is complete:
- `main_resnet50.py` — trains and evaluates the ResNet50 model
- `main_mobilenetv2.py` — trains and evaluates the MobileNetV2 model

---

### 3. `demo/`
Run after training the MobileNetV2 model:
- `app.py` — launches the Streamlit app for live ASL sign prediction

> Note: Place the trained model file (`best_asl_model.keras`) in the `demo/` folder before running.

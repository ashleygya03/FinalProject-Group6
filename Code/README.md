# Code

This folder contains all code for the final project, including data preprocessing, model training, and the Streamlit demo. The README.md section of each folder contains detailed description of running the codes.

## Folder structure

```text
│
├── README.md
│
├── data-preprocessing/
│   ├── README.md
│   ├── compare_datasets.py
│   ├── preprocess.py
│   ├── crop_hands.py
│   ├── brightness_filter.py
│   ├── merge_datasets.py
│   └── hand_landmarker.task
│
├── models/
│   ├── mobilenetv2/
│   │   └── main_mobilenetv2.py
│   ├── resnet50/
│       └── main_resnet50.py
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

```
## Run Order

### 1. `data-preprocessing/`
Run scripts in this order:
1. `compare_datasets.py` — compares the two ASL datasets
2. `preprocess.py` — crops and filters the 87K dataset
3. `merge_datasets.py` — merges datasets into the final balanced dataset

> Note: Place the two datasets in the `data-preprocessing/` folder before running.
> **Datasets:**
- [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/code?datasetId=23079&sortBy=voteCount)
- [SignAlphaSet](https://data.mendeley.com/datasets/8fmvr9m98w/2)

---
### 2. `models/`
Run the following on the new produced dataset:
- `main_resnet50.py` — trains and evaluates the ResNet50 model
- `main_mobilenetv2.py` — trains and evaluates the MobileNetV2 model

---

### 3. `demo/`
Run after training the MobileNetV2 model:
- `app.py` — launches the Streamlit app for live ASL sign prediction

> Note: Place the trained MobileNetV2 model file (`best_asl_model.keras`) in the `demo/` folder before running.



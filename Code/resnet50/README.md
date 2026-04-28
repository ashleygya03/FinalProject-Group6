# ResNet50 ASL Classification

This folder contains the ResNet50-based American Sign Language (ASL) classification project

## Main files
- `main_resnet50.py` — main training and evaluation script
- `src/` — source code for config, dataset, model, and utilities
- `tests/` — test files
- `requirements.txt` — required packages

## Run the project

Open terminal in this `resnet50` folder and run to generate model and outputs:

```bash
python main_resnet50.py
```

## Download the trained model

- [Model](https://drive.google.com/file/d/15V-j7nm8cDqXsXp7hmWk-o0-50tQiiSn/view?usp=drive_link)

Or download it from terminal inside the `demo` folder:

```bash
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=15V-j7nm8cDqXsXp7hmWk-o0-50tQiiSn" -O best_asl_model.keras
```

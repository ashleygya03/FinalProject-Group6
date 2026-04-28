# ASL Sign Language Detection - AG - Individual Project Archive

## Project Overview
This folder contains the individual contributions and research conducted for the ASL Alphabet detection capstone. My focus was on implementing a robust MobileNetV2 architecture capable of high-accuracy inference in real-world lighting conditions via a Streamlit web interface.

## Folder Structure
- **/App**: Contains the deployment logic.
    - `app.py`: Streamlit interface utilizing AWS EC2 for hosting.
    - `predict.py`: Inference engine with MobileNetV2 preprocessing.
- **/Code**: Core machine learning pipeline.
    - `train.py`: Initial training with frozen MobileNetV2 weights.
    - `fine_tune.py`: Deep optimization of upper layers for ASL-specific nuances.
    - `test.py`: Performance evaluation (Confusion Matrix/Classification Reports).

## Key Results
- **Model Accuracy**: 97% on the validation set.
- **Top-5 Logic**: Implemented a probability-weighted top-5 prediction table to analyze model confidence and edge cases.
- **Robustness**: Fine-tuned to handle variable backgrounds and slight motion blur.

## How to Run
1. Ensure dependencies are installed: `pip install -r requirements.txt`
2. Launch the app: `streamlit run App/app.py`

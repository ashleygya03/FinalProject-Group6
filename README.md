# 🧠 American Sign Language (ASL) Alphabet Classification using Deep Learning  
### A Comparative Study of ResNet-50 and MobileNetV2

**Members:** Fyrooz Khan | Ashley Gyapomah  
**Course:** DATS 6303 – Deep Learning  

---

## Overview

This project develops a deep learning–based computer vision system for **static American Sign Language (ASL) alphabet recognition** from images. The goal is to build a model that can accurately classify ASL hand signs and help reduce communication barriers between Deaf and Hard-of-Hearing users and hearing non-signers.

We compare two transfer learning architectures:

- **MobileNetV2** (lightweight, efficient)
- **ResNet50** (deeper, higher capacity)

To improve real-world performance, we designed a three-phase data pipeline:

> **Audit → Preprocess → Merge**

This pipeline standardizes, cleans, and combines publicly available ASL datasets into a balanced dataset for training.

The full workflow includes:
- Data preprocessing  
- Model training (transfer learning + fine-tuning)  
- Evaluation (metrics + confusion matrices)  
- Deployment via a **Streamlit-based real-time demo**

---

## Datasets

- [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/code?datasetId=23079&sortBy=voteCount)  
- [SignAlphaSet Dataset (Mendeley)](https://data.mendeley.com/datasets/8fmvr9m98w/2)

---

## Project Structure

```text
FinalProject-Group6/
├── Ashley-Gyapomah-Individual-Project/
├── Fyrooz-Khan-individual-project/
├── Code/
│   ├── data_preprocessing/
│   ├── models/
│   │   ├── mobilenetv2/
│   │   ├── resnet50/
│   ├── demo/
│   
├── Final-Group-Presentation/
├── Final-Group-Project-Report/
├── Group-Proposal/
└── README.md

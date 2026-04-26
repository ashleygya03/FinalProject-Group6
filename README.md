# FinalProject-Group6

**Course:** DATS 6303 – Deep Learning  

**Project:** Static American Sign Language (ASL) Alphabet Recognition  

## Overview
This project develops a deep learning–based computer vision system for **static American Sign Language (ASL) alphabet recognition** from images. The goal is to improve accessibility by building a model that can classify ASL hand signs and help reduce communication barriers between Deaf and Hard-of-Hearing users and hearing non-signers.

The project compares two transfer-learning image classification models, **MobileNetV2** and **ResNet-50**, for ASL letter recognition. To improve real-world performance, we designed a three-phase data pipeline — **Audit → Preprocess → Merge** — to standardize, clean, and combine publicly available ASL datasets into a balanced final dataset. The workflow includes image preprocessing, model training, evaluation, and deployment through a **Streamlit demo** based hand detection.

## GitHub Directory Structure
```text
FinalProject-Group6/
├── Ashley-Gyapomah-Individual-Project/   # Ashley's individual project files
├── Fyrooz-Khan-individual-project/       # Fyrooz's individual project files
├── Code/                                 # Source code for preprocessing, models, and Streamlit demo
├── Final-Group-Presentation/             # Final presentation slides and materials
├── Final-Group-Project-Report/           # Final group report and documentation
├── Group-Proposal/                       # Initial group proposal
├── README.md                             # Project overview (this file)
├── requirements.txt                      # Project dependencies
└── .gitignore                            # Git ignore file
```

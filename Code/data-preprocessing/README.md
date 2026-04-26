# ASL Dataset Standardization Pipeline

A reproducible preprocessing and merging pipeline for constructing a unified, balanced ASL (American Sign Language) dataset from two source collections: an **87K-image dataset** and a **26K-image dataset**.

The pipeline produces a final dataset of **26,000 images** (1,000 per letter, A–Z), suitable for training sign language recognition models.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Pipeline Overview](#pipeline-overview)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Phase 1 — Dataset Comparison & Audit](#phase-1--dataset-comparison--audit)
5. [Phase 2 — Preprocessing (Crop & Filter)](#phase-2--preprocessing-crop--filter)
6. [Phase 3 — Merge](#phase-3--merge)
7. [Letter Classification Table](#letter-classification-table)
8. [File Reference](#file-reference)
9. [Reproducibility](#reproducibility)

---

## Motivation

The two source datasets differ significantly in framing, orientation, and quality:

- **87K dataset**: Images are captured from a wider angle, including the forearm and background. The hand occupies only a portion of each 200×200 frame, and many images exhibit low contrast where the hand appears as a dark silhouette against a brighter background.
- **26K dataset**: Images are tightly framed on the hand at 296×296, with consistent centering and adequate lighting.

Additionally, a visual audit revealed that for certain letters, the two datasets depict the sign from **different angles** (e.g., palm-side vs. back-of-hand), use **different hands** (left vs. right), or show **entirely different sign poses**. Merging such images without curation would introduce label noise and degrade model performance.

This pipeline addresses these issues through a systematic, three-phase approach: **audit → preprocess → merge**.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Phase 1: AUDIT                           │
│  compare_datasets.py                                            │
│  Sample 10 random images per letter from each dataset           │
│  Generate side-by-side comparison grids for visual inspection   │
│  → Decision: which letters are safe to merge?                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Phase 2: PREPROCESS                         │
│  preprocess.py  (imports: crop_hands.py, brightness_filter.py)  │
│                                                                 │
│  For each 87K image:                                            │
│    1. Brightness check (mean V-channel ≥ 40)                    │
│    2. Hand detection & crop via MediaPipe landmarks             │
│    3. Resize to 296×296                                         │
│    4. Post-crop contrast check (foreground std dev ≥ 17)        │
│  → Output: cleaned, cropped 87K images + confidence scores      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Phase 3: MERGE                            │
│  merge_datasets.py                                              │
│                                                                 │
│  Using the letter classification table:                         │
│    • 26K-only letters: 1,000 from 26K                           │
│    • 87K-only letters: 1,000 from 87K (cropped)                 │
│    • Merged letters: 500 from each dataset                      │
│  Selection: confidence-filtered random sampling (seed=42)       │
│  → Output: 26,000 images (1,000 per letter)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites & Installation

- **Python 3.10+**
- **Dependencies**: `mediapipe`, `opencv-python`

```bash
# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the MediaPipe hand-landmarker model (~7.5 MB, one-time)
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## Phase 1 — Dataset Comparison & Audit

Before any processing, we perform a systematic visual audit to determine which letters can be safely merged across datasets.

### How it works

`compare_datasets.py` randomly samples 10 images per letter from each dataset (using a fixed seed for reproducibility) and generates a side-by-side comparison grid for all 26 letters.

```bash
python compare_datasets.py
```

This produces one image per letter in the `comparison_output/` directory (e.g., `compare_A.jpg`), showing:
- **Row 1**: 10 samples from the 87K dataset
- **Row 2**: 10 samples from the 26K dataset

### What we found

By inspecting these grids, each letter was classified into one of four categories:

| Category | Description | Example |
|----------|-------------|---------|
| **Same sign, compatible angle** | Safe to merge | B, C, H, K, L |
| **Same sign, mirrored/flipped** | Included (mirror is acceptable augmentation) | C (back vs. palm — both recognizable) |
| **Different viewing angle hiding key features** | Excluded from merge — features critical for inter-class discrimination are obscured | M, N, T (thumb-between-fingers hidden in 87K's back view) |
| **Different sign pose entirely** | Excluded from merge — 87K labels appear contaminated or use a non-standard pose | D (mixed D/O/F poses), X (index straight up vs. hooked) |

An additional handedness issue was identified: **J** in the 26K dataset uses the left hand, while the 87K dataset correctly uses the right hand (the ASL standard for dominant-hand signs). For J, only 87K images are used.

See the [Letter Classification Table](#letter-classification-table) for the full decision matrix.

---

## Phase 2 — Preprocessing (Crop & Filter)

The 87K dataset images are captured from a wider angle with the forearm and background visible. To standardize them to the same framing as the 26K dataset (hand-only, 296×296), we apply a three-stage quality pipeline in a single pass.

### Pipeline stages

| Stage | Method | Purpose |
|-------|--------|---------|
| **1. Brightness filter** | Mean of V-channel in HSV color space (Gonzalez & Woods, *Digital Image Processing*) | Remove globally dark/underexposed images where hand features are not discernible |
| **2. Hand crop** | MediaPipe `HandLandmarker` — detects 21 hand landmarks, computes a bounding box with padding, crops and resizes to 296×296 | Isolate the hand region, removing arm and background |
| **3. Post-crop contrast filter** | Otsu's automatic thresholding (Otsu, 1979) separates hand from background, then measures the standard deviation of V-channel brightness within the hand region | Remove images where the hand is a featureless dark silhouette — these pass the brightness check (bright background inflates the mean) but lack visible finger edges, knuckle creases, and skin texture |

### Running the preprocessor

```bash
python preprocess.py \
    -i ~/Downloads/87K-ASL/asl_alphabet_train/asl_alphabet_train \
    -o ~/Downloads/87K-ASL/output \
    --brightness-threshold 40 \
    --contrast-threshold 17 \
    -w 8
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | — | Input directory (flat or nested A–Z) |
| `-o` / `--output` | — | Output directory for processed images |
| `--brightness-threshold` | `40` | Minimum mean brightness (0–255). Set to 0 to disable. |
| `--contrast-threshold` | `17` | Minimum foreground contrast (std dev of V-channel in cropped hand). Set to 0 to disable. |
| `-p` / `--padding` | `20` | Extra pixels around the hand bounding box |
| `-r` / `--resize` | *none* | Resize crops to SIZE×SIZE pixels |
| `-w` / `--workers` | *CPU count* | Number of parallel worker processes |

### Single-image mode

```bash
python preprocess.py -f hand.jpg -o cropped_hand.jpg
```

### Outputs

| File | Description |
|------|-------------|
| `{A..Z}/*.jpg` | Cropped, filtered images preserving the A–Z directory structure |
| `confidence_scores.csv` | Per-image MediaPipe hand detection confidence (used in merge phase) |
| `preprocessing_report.csv` | Full report: every image with its status (OK / BRIGHT_FAIL / CROP_FAIL / CONTRAST_FAIL) |

### Results from our run

| Metric | Count | % |
|--------|------:|---:|
| Total input images | 87,000 | 100% |
| **Successfully kept** | **49,614** | **57.0%** |
| No hand detected | 23,409 | 26.9% |
| Low contrast (post-crop) | 13,977 | 16.1% |
| Brightness filtered | 0 | 0.0% |

---

## Phase 3 — Merge

`merge_datasets.py` combines the preprocessed 87K output with the 26K dataset into a single, balanced dataset using the letter classification decisions from Phase 1.

### Running the merge

```bash
python merge_datasets.py \
    --dataset1 ~/Downloads/87K-ASL/output \
    --dataset2 ~/Downloads/26K-ASL \
    --output ~/Downloads/merged-asl \
    --dataset2-only D,G,M,N,P,T,X,Z \
    --dataset1-only J \
    --confidence-csv ~/Downloads/87K-ASL/output/confidence_scores.csv \
    --confidence-threshold 0.7 \
    --seed 42
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset1` | — | First dataset directory (e.g., preprocessed 87K) |
| `--dataset2` | — | Second dataset directory (e.g., 26K) |
| `--output` | — | Output directory for merged dataset |
| `--count` | `500` | Number of images per dataset per letter |
| `--seed` | `42` | Random seed for reproducible sampling |
| `--dataset1-only` | — | Comma-separated letters sourced exclusively from dataset1 |
| `--dataset2-only` | — | Comma-separated letters sourced exclusively from dataset2 |
| `--confidence-csv` | — | Path to `confidence_scores.csv` from preprocessing |
| `--confidence-threshold` | `0.0` | Minimum confidence for dataset1 images |

### Sampling methodology

Images are selected via **confidence-filtered uniform random sampling**:

1. **Quality gate**: Only 87K images with a MediaPipe hand detection confidence ≥ 0.7 are eligible for selection.
2. **Uniform random draw**: From the eligible pool, 500 images are drawn uniformly at random using a fixed seed (`--seed 42`).
3. **Reproducibility**: The same seed guarantees the exact same subset on every run, preventing any possibility of cherry-picking.

This approach ensures an unbiased, high-quality sample while preserving the natural distribution of visual variations in the source data.

### Outputs

| File | Description |
|------|-------------|
| `{A..Z}/` | 1,000 images per letter. Filenames are prefixed `ds1_` or `ds2_` to indicate source. |
| `merge_report.csv` | Every image traced to its source (letter, dataset, original path, output filename) |
| `merge_summary.txt` | Per-letter breakdown of how many images came from each dataset |

### Results from our run

**26,000 images total** — exactly 1,000 per letter, perfectly balanced.

---

## Letter Classification Table

This table summarizes the visual audit findings and the merge strategy for each letter.

| Letter | Strategy | Source(s) | Reason |
|--------|----------|-----------|--------|
| **A** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **B** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **C** | Merged | 500 × 87K + 500 × 26K | Same sign (back vs. palm view — both recognizable) |
| **D** | 26K only | 1,000 × 26K | 87K has contaminated labels — D, O, F poses mixed together |
| **E** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **F** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **G** | 26K only | 1,000 × 26K | Mirrored + opposite side — points opposite directions |
| **H** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **I** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **J** | 87K only | 1,000 × 87K | 26K uses left hand; 87K uses right hand (ASL standard) |
| **K** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **L** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **M** | 26K only | 1,000 × 26K | 87K back view hides 3-fingers-over-thumb defining feature |
| **N** | 26K only | 1,000 × 26K | 87K back view hides 2-fingers-over-thumb defining feature |
| **O** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **P** | 26K only | 1,000 × 26K | Mirrored + opposite side — downward hand from opposite perspectives |
| **Q** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **R** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **S** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **T** | 26K only | 1,000 × 26K | Thumb-between-fingers feature hidden in 87K back view |
| **U** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **V** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **W** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **X** | 26K only | 1,000 × 26K | 87K shows index pointing up (looks like "1"); 26K shows correct hooked index |
| **Y** | Merged | 500 × 87K + 500 × 26K | Same sign, compatible angle |
| **Z** | 26K only | 1,000 × 26K | Dynamic letter + mirrored + inconsistent captures |

**Summary**: 17 merged · 8 from 26K only · 1 from 87K only

---

## File Reference

| File | Purpose |
|------|---------|
| `compare_datasets.py` | Phase 1 — Visual audit tool. Generates side-by-side comparison grids for every letter. |
| `brightness_filter.py` | Core module — `compute_brightness()` (mean V-channel) and `compute_foreground_contrast()` (Otsu + std dev). No CLI. |
| `crop_hands.py` | Core module — `create_landmarker()` and `crop_hand()` using MediaPipe HandLandmarker. No CLI. |
| `preprocess.py` | Phase 2 — Orchestrator. CLI, directory traversal, multiprocessing, brightness + crop + contrast in one pass. Imports from `brightness_filter.py` and `crop_hands.py`. |
| `merge_datasets.py` | Phase 3 — Balanced merge with per-letter source overrides, confidence filtering, and random sampling. |
| `requirements.txt` | Python dependencies (`mediapipe`, `opencv-python`). |
| `hand_landmarker.task` | MediaPipe model file (not tracked in git — download via curl). |

---

## Reproducibility

Every step in this pipeline is fully reproducible:

| Aspect | How |
|--------|-----|
| **Visual audit** | `compare_datasets.py` uses `random.seed(42)` — same 10 samples every run |
| **Preprocessing** | Deterministic MediaPipe detection; thresholds are explicit CLI arguments |
| **Merge sampling** | `merge_datasets.py` uses `random.seed(42)` — same 500 images selected every run |
| **Traceability** | `merge_report.csv` maps every output image back to its exact source file |
| **Auditability** | `preprocessing_report.csv` logs every image's status and scores |

To reproduce the entire pipeline from scratch:

```bash
# Phase 1: Visual audit
python compare_datasets.py

# Phase 2: Preprocess the 87K dataset
python preprocess.py \
    -i ~/Downloads/87K-ASL/asl_alphabet_train/asl_alphabet_train \
    -o ~/Downloads/87K-ASL/output \
    --brightness-threshold 40 \
    --contrast-threshold 17 \
    -w 8

# Phase 3: Merge into final dataset
python merge_datasets.py \
    --dataset1 ~/Downloads/87K-ASL/output \
    --dataset2 ~/Downloads/26K-ASL \
    --output ~/Downloads/merged-asl \
    --dataset2-only D,G,M,N,P,T,X,Z \
    --dataset1-only J \
    --confidence-csv ~/Downloads/87K-ASL/output/confidence_scores.csv \
    --confidence-threshold 0.7 \
    --seed 42
```

---

## Supported Image Formats

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`, `.tif`

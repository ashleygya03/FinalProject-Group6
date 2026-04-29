## Data Preprocessing Pipeline

This project uses a three-stage preprocessing pipeline:

> **Compare → Preprocess → Merge**

1. `compare_datasets.py`  
   - Compares the 87K and 26K ASL datasets and creates visual comparison images.

2. `preprocess.py`  
   - Crops and filters the 87K dataset.
   - Uses `crop_hands.py` and `brightness_filter.py`.

3. `merge_datasets.py`  
   - Merges the cleaned 87K dataset with the 26K dataset.
   - Creates the final balanced ASL dataset.

All commands should be run from the project root:

```bash
cd FinalProject-Group6


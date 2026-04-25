#!/usr/bin/env python3
"""
Compare two ASL datasets by sampling random images from each letter A–Z
and generating side-by-side comparison grids.

Produces one image per letter: 2 rows × 10 columns.
  Row 1 = 10 random samples from dataset 1 (87k-asl)
  Row 2 = 10 random samples from dataset 2 (26k-asl)
"""

import os
import random
import string
import cv2
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_1 = os.path.expanduser(
    "~/Downloads/87k-asl/asl_alphabet_train/asl_alphabet_train"
)
DATASET_2 = os.path.expanduser("~/Downloads/26k-asl")
LABEL_1 = "87k-asl"
LABEL_2 = "26k-asl"

SAMPLES = 10           # images per dataset per letter
THUMB = 150            # thumbnail size (px)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_output")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(42)        # reproducible sampling


def list_images(directory: str) -> list[str]:
    """Return sorted list of image file paths in *directory*."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )


def load_thumb(path: str, size: int = THUMB) -> np.ndarray:
    """Load an image and resize to a square thumbnail."""
    img = cv2.imread(path)
    if img is None:
        # return a grey placeholder
        return np.full((size, size, 3), 180, dtype=np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def put_label(img: np.ndarray, text: str, position: str = "top") -> None:
    """Draw a label on the image (in-place)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    color = (255, 255, 255)
    bg = (0, 0, 0)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    if position == "top":
        org = (4, th + 6)
        cv2.rectangle(img, (0, 0), (tw + 8, th + 12), bg, -1)
    else:
        h = img.shape[0]
        org = (4, h - 6)
        cv2.rectangle(img, (0, h - th - 12), (tw + 8, h), bg, -1)
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def build_comparison(letter: str) -> str:
    """Build a comparison grid for one letter and save it. Returns path."""
    dir1 = os.path.join(DATASET_1, letter)
    dir2 = os.path.join(DATASET_2, letter)

    imgs1 = list_images(dir1)
    imgs2 = list_images(dir2)

    sample1 = random.sample(imgs1, min(SAMPLES, len(imgs1))) if imgs1 else []
    sample2 = random.sample(imgs2, min(SAMPLES, len(imgs2))) if imgs2 else []

    # Pad to SAMPLES with None
    sample1 += [None] * (SAMPLES - len(sample1))
    sample2 += [None] * (SAMPLES - len(sample2))

    def make_row(samples: list, dataset_label: str) -> np.ndarray:
        thumbs = []
        for i, path in enumerate(samples):
            if path:
                t = load_thumb(path)
                put_label(t, f"#{i+1}", "top")
            else:
                t = np.full((THUMB, THUMB, 3), 60, dtype=np.uint8)
                put_label(t, "N/A", "top")
            thumbs.append(t)
        row = np.hstack(thumbs)
        return row

    row1 = make_row(sample1, LABEL_1)
    row2 = make_row(sample2, LABEL_2)

    # Add dataset labels on the left side
    label_w = 90
    h = THUMB

    def make_label_col(text: str) -> np.ndarray:
        col = np.full((h, label_w, 3), 30, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        x = (label_w - tw) // 2
        y = (h + th) // 2
        cv2.putText(col, text, (x, y), font, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return col

    row1 = np.hstack([make_label_col(LABEL_1), row1])
    row2 = np.hstack([make_label_col(LABEL_2), row2])

    # Title bar
    total_w = row1.shape[1]
    title_h = 40
    title_bar = np.full((title_h, total_w, 3), 20, dtype=np.uint8)
    title = f"Letter '{letter}'  —  {LABEL_1} vs {LABEL_2}  ({SAMPLES} random samples each)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(title, font, 0.6, 1)
    cv2.putText(title_bar, title, ((total_w - tw) // 2, (title_h + th) // 2),
                font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Separator
    sep = np.full((3, total_w, 3), 80, dtype=np.uint8)

    grid = np.vstack([title_bar, row1, sep, row2])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"compare_{letter}.jpg")
    cv2.imwrite(out_path, grid)
    return out_path


def main():
    print(f"📂  Dataset 1: {DATASET_1}")
    print(f"📂  Dataset 2: {DATASET_2}")
    print(f"📂  Output:    {OUTPUT_DIR}")
    print(f"🎲  Samples:   {SAMPLES} per letter per dataset")
    print()

    for letter in string.ascii_uppercase:
        path = build_comparison(letter)
        print(f"  ✓  {letter} → {os.path.basename(path)}")

    print(f"\n✅  Done! {26} comparison images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

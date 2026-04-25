#!/usr/bin/env python3
"""
Hand Cropper — Core cropping logic for a single image.

Uses MediaPipe Hands to detect the 21 hand landmarks and derives a tight
bounding box (with configurable padding) around them.

This module exposes two importable functions:
    - create_landmarker()  — creates a reusable MediaPipe HandLandmarker
    - crop_hand()          — crops a single image to the hand region

This file has NO CLI, NO directory traversal, NO multiprocessing.
For batch processing, see preprocess.py.
"""

import os
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── MediaPipe setup ──────────────────────────────────────────────────────────
# Resolve model path relative to this script so it works from any cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_SCRIPT_DIR, "hand_landmarker.task")

# Supported image extensions (shared constant for other modules)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def create_landmarker() -> mp_vision.HandLandmarker:
    """Create a MediaPipe HandLandmarker configured for single-image mode.

    Returns
    -------
    mp_vision.HandLandmarker
        A ready-to-use landmarker instance.  Caller is responsible for
        calling ``.close()`` when finished.

    Raises
    ------
    SystemExit
        If the model file ``hand_landmarker.task`` is not found next to
        this script.
    """
    if not os.path.isfile(_MODEL_PATH):
        print(f"❌  Model file not found: {_MODEL_PATH}")
        print("   Download it with:")
        print("   curl -L -o hand_landmarker.task "
              "https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/"
              "hand_landmarker.task")
        sys.exit(1)
    base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def crop_hand(image_path: str, output_path: str, padding: int = 20,
              resize: int | None = None,
              landmarker: mp_vision.HandLandmarker | None = None) -> tuple[bool, float]:
    """Detect a hand in *image_path*, crop to the landmark bounding box, and
    save the result to *output_path*.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    output_path : str
        Destination path for the cropped image.
    padding : int
        Extra pixels to add around the detected bounding box.
    resize : int or None
        If provided, resize the cropped image to (resize × resize) pixels.
    landmarker : HandLandmarker or None
        A reusable landmarker instance.  If ``None`` a temporary one is
        created (slower when processing many images).

    Returns
    -------
    tuple[bool, float]
        ``(success, confidence)``.  ``success`` is True if a hand was
        detected and the file was saved.  ``confidence`` is the hand
        detection confidence score (0.0–1.0), or -1.0 on failure.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False, -1.0

    h, w = img.shape[:2]

    # MediaPipe Tasks requires an mp.Image object
    mp_image = mp.Image.create_from_file(image_path)

    own_landmarker = landmarker is None
    if own_landmarker:
        landmarker = create_landmarker()

    try:
        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return False, -1.0  # No hand detected

        landmarks = result.hand_landmarks[0]  # first hand
        confidence = result.handedness[0][0].score  # detection confidence

        # Bounding box from all 21 landmarks (coordinates are normalised 0-1)
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x_min = max(0, int(min(xs)) - padding)
        y_min = max(0, int(min(ys)) - padding)
        x_max = min(w, int(max(xs)) + padding)
        y_max = min(h, int(max(ys)) + padding)

        cropped = img[y_min:y_max, x_min:x_max]

        if resize is not None:
            cropped = cv2.resize(cropped, (resize, resize),
                                interpolation=cv2.INTER_AREA)

        cv2.imwrite(output_path, cropped)
        return True, confidence

    finally:
        if own_landmarker:
            landmarker.close()

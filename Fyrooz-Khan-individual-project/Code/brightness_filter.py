#!/usr/bin/env python3
"""
Brightness Filter — Core brightness and contrast analysis for a single image.

Provides two quality checks for hand images:

1. **Mean brightness** (V-channel of HSV):
   Basic luminance check. Catches uniformly dark images.

2. **Foreground contrast** (Otsu + V-channel std dev):
   Uses Otsu's automatic thresholding to separate the hand (foreground)
   from the background, then measures the standard deviation of brightness
   within the hand region. A low std dev means the hand is a featureless
   silhouette — finger edges, knuckle creases, and skin texture are not
   visible, making the image useless for training.

   This is the key metric: a hand photographed too dark against a bright
   background will pass the mean brightness test (bright background
   inflates the average), but fail the foreground contrast test.

References:
    - HSV brightness: Gonzalez & Woods, "Digital Image Processing"
    - Otsu's method: Otsu (1979), "A Threshold Selection Method from
      Gray-Level Histograms", IEEE Trans. SMC

This module exposes two importable functions:
    - compute_brightness()          — mean V-channel brightness
    - compute_foreground_contrast() — contrast within the hand region

This file has NO CLI, NO directory traversal, NO multiprocessing.
For batch processing, see preprocess.py.
"""

import cv2
import numpy as np


def compute_brightness(image_path: str) -> tuple[float, str | None]:
    """Compute the mean brightness of an image using the V channel of HSV.

    The V (Value) channel in HSV color space represents the perceived
    brightness/luminance of each pixel, independent of hue and saturation.
    The mean of this channel gives a single scalar summary of overall
    image brightness on a 0–255 scale.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    tuple[float, str | None]
        ``(brightness_score, error_message)``.
        On success, ``error_message`` is None and ``brightness_score``
        is a float in the range [0, 255].
        On failure, ``brightness_score`` is -1.0 and ``error_message``
        describes the issue.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return -1.0, "Failed to read image"

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]  # V = brightness/value channel
        mean_brightness = float(v_channel.mean())

        return round(mean_brightness, 2), None
    except Exception as e:
        return -1.0, str(e)


def compute_foreground_contrast(image_path: str) -> tuple[float, float, str | None]:
    """Measure the contrast within the hand (foreground) region of an image.

    Uses Otsu's automatic thresholding to separate the darker foreground
    (hand) from the brighter background. Then computes the standard
    deviation of brightness (V-channel) within the foreground region.

    A low foreground std dev indicates a featureless silhouette — the hand
    is too dark or too uniformly lit to reveal finger edges, knuckle
    creases, or skin texture that a model needs for classification.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    tuple[float, float, str | None]
        ``(foreground_mean, foreground_std, error_message)``.
        - ``foreground_mean``: mean brightness of the hand region (0–255).
        - ``foreground_std``: standard deviation of brightness in the hand
          region. Higher = more visible features. Typical range:
          - < 15: silhouette, no features visible
          - 15–25: marginal, some features
          - > 25: good feature visibility
        - ``error_message``: None on success, or a description of the error.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return -1.0, -1.0, "Failed to read image"

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        # Otsu's method finds the optimal threshold to separate two classes
        # (foreground/background) by maximizing inter-class variance.
        _, mask = cv2.threshold(
            v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Foreground = dark pixels (mask == 0), which is the hand
        foreground_pixels = v_channel[mask == 0]

        if len(foreground_pixels) == 0:
            # No foreground detected (entire image is uniform)
            return float(v_channel.mean()), 0.0, None

        fg_mean = float(foreground_pixels.mean())
        fg_std = float(foreground_pixels.std())

        return round(fg_mean, 2), round(fg_std, 2), None

    except Exception as e:
        return -1.0, -1.0, str(e)

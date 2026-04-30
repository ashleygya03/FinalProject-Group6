#!/usr/bin/env python3
"""
Preprocess — Orchestrator for the ASL image preprocessing pipeline.

Combines brightness filtering, hand cropping, and post-crop contrast analysis
into a single pipeline:
    1. Scan input directory for images (flat or nested A-Z structure)
    2. For each image, check mean brightness (V-channel in HSV)
    3. Crop the hand using MediaPipe landmarks
    4. Check foreground contrast on the CROPPED image (Otsu + V-channel std dev)
    5. If all checks pass, keep the crop; otherwise discard
    6. Log confidence scores and generate reports

The post-crop contrast check catches dark silhouette hands that look fine in the
original image (bright background inflates the mean) but are featureless once
cropped to just the hand region.

Supports both single-image and batch directory modes.

Usage:
    # Single image
    python preprocess.py -f hand.jpg -o cropped.jpg

    # Batch directory (nested A-Z)
    python preprocess.py -i ~/Downloads/87K-ASL/asl_alphabet_train/asl_alphabet_train \\
                         -o ~/Downloads/87K-ASL/output \\
                         --brightness-threshold 40 \\
                         --contrast-threshold 17 \\
                         --resize 296 \\
                         -w 8
    
    Example:
    python Code/data-preprocessing/preprocess.py \
    -i data/asl_alphabet \
    -o data/processed_asl \
    --brightness-threshold 40 \
    --contrast-threshold 17 \
    --resize 296 \
    -w 8
                         
    # Skip brightness filter (crop + contrast only)
    python preprocess.py -i ./dataset -o ./cropped --brightness-threshold 0

    # Skip contrast filter (crop + brightness only)
    python preprocess.py -i ./dataset -o ./cropped --contrast-threshold 0
"""

import argparse
import csv
import multiprocessing
import os
import sys
import time

from brightness_filter import compute_brightness, compute_foreground_contrast
from crop_hands import IMAGE_EXTENSIONS, create_landmarker, crop_hand
from mediapipe.tasks.python import vision as mp_vision


# ── Image collection ─────────────────────────────────────────────────────────

def collect_images(input_dir: str) -> tuple[list[tuple[str, str]], bool]:
    """Walk *input_dir* and return ``(image_list, is_flat)``.

    ``image_list`` contains ``(relative_path, label)`` pairs.  ``label`` is
    the subdirectory name for nested layouts or an empty string for flat
    layouts.  The layout is auto-detected: if the directory contains at
    least one image file directly, it is treated as flat; otherwise the
    subdirectory-based (nested) layout is assumed.
    """
    # ── Check for flat layout (images directly in input_dir) ─────────────
    flat_images: list[tuple[str, str]] = []
    for fname in sorted(os.listdir(input_dir)):
        fpath = os.path.join(input_dir, fname)
        if os.path.isfile(fpath):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                flat_images.append((fname, ""))

    if flat_images:
        return flat_images, True

    # ── Nested layout (A–Z subdirectories) ───────────────────────────────
    nested_images: list[tuple[str, str]] = []
    for label in sorted(os.listdir(input_dir)):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                rel = os.path.join(label, fname)
                nested_images.append((rel, label))

    return nested_images, False


# ── Progress display ─────────────────────────────────────────────────────────

def _print_progress(done: int, total: int, ok: int, bright_fail: int,
                    crop_fail: int, contrast_fail: int) -> None:
    """Render a single-line progress bar to stdout."""
    pct = done / total * 100 if total > 0 else 0
    bar_len = 40
    filled = int(bar_len * done // total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(
        f"\r  [{bar}] {pct:5.1f}%  ({done}/{total})  "
        f"✓ {ok}  ☀ -{bright_fail}  ✗ -{crop_fail}  ◐ -{contrast_fail}"
    )
    sys.stdout.flush()


# ── Multiprocessing helpers ──────────────────────────────────────────────────
# Each worker process owns its own HandLandmarker (the model is not picklable).

_worker_landmarker: mp_vision.HandLandmarker | None = None


def _worker_init() -> None:
    """Pool initializer — create a per-process HandLandmarker."""
    global _worker_landmarker
    _worker_landmarker = create_landmarker()


def _worker_task(job: tuple) -> tuple[str, str, float, float, bool, str]:
    """Process a single image through the full pipeline.

    Pipeline: brightness check → crop hand → contrast check on crop.

    Parameters
    ----------
    job : tuple
        (src_path, dst_path, brightness_threshold, contrast_threshold,
         padding, resize, rel_path)

    Returns
    -------
    tuple
        (src_path, status, brightness_score, confidence, success, rel_path)
        status is one of: "OK", "BRIGHT_FAIL", "CROP_FAIL",
        "CONTRAST_FAIL", "READ_ERROR"
    """
    src, dst, brightness_threshold, contrast_threshold, padding, resize, rel_path = job

    # Step 1: Mean brightness check on the RAW image
    brightness, error = compute_brightness(src)

    if error is not None:
        return (src, "READ_ERROR", -1.0, -1.0, False, rel_path)

    if brightness_threshold > 0 and brightness < brightness_threshold:
        return (src, "BRIGHT_FAIL", brightness, -1.0, False, rel_path)

    # Step 2: Crop hand (no re-detection validation — too strict for small images)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    success, confidence = crop_hand(
        src, dst,
        padding=padding,
        resize=resize,
        landmarker=_worker_landmarker,
    )

    if not success:
        return (src, "CROP_FAIL", brightness, confidence, False, rel_path)

    # Step 3: Contrast check on the CROPPED image (post-crop quality gate)
    if contrast_threshold > 0:
        fg_mean, fg_std, error = compute_foreground_contrast(dst)
        if error is None and fg_std < contrast_threshold:
            # Remove the low-contrast crop
            os.remove(dst)
            return (src, "CONTRAST_FAIL", brightness, confidence, False, rel_path)

    return (src, "OK", brightness, confidence, True, rel_path)


# ── Report generation ────────────────────────────────────────────────────────

def save_reports(results: list[tuple], output_dir: str) -> tuple[str, str]:
    """Write CSV reports for the preprocessing run.

    Generates:
        - confidence_scores.csv  — successfully processed images with scores
        - preprocessing_report.csv — full report with all images and statuses
    """
    confidence_path = os.path.join(output_dir, "confidence_scores.csv")
    report_path = os.path.join(output_dir, "preprocessing_report.csv")

    # Full report
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "status", "brightness", "confidence"])
        for src, status, brightness, confidence, _, rel_path in sorted(results, key=lambda x: x[5]):
            writer.writerow([rel_path, status, brightness, confidence])

    # Confidence scores (only successful crops)
    with open(confidence_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "confidence_score"])
        for src, status, brightness, confidence, success, rel_path in sorted(results, key=lambda x: x[5]):
            if success:
                writer.writerow([rel_path, confidence])

    return confidence_path, report_path


# ── Core pipeline ────────────────────────────────────────────────────────────

def process_single(image_path: str, output_path: str,
                   brightness_threshold: float = 40.0,
                   contrast_threshold: float = 17.0,
                   padding: int = 20, resize: int | None = None) -> None:
    """Process a single image: brightness → crop → contrast check → save."""

    print(f"🖼   Input:    {os.path.abspath(image_path)}")
    print(f"📂  Output:   {os.path.abspath(output_path)}")
    print(f"☀   Brightness threshold: {brightness_threshold}")
    print(f"◐   Contrast threshold:   {contrast_threshold}")
    print(f"📏  Padding:  {padding}px")
    if resize:
        print(f"🔲  Resize:   {resize}×{resize}px")
    print()

    # Step 1: Mean brightness
    brightness, error = compute_brightness(image_path)
    if error:
        print(f"❌  Failed to read image: {error}")
        sys.exit(1)

    print(f"☀   Brightness: {brightness:.1f} / 255", end="")
    if brightness_threshold > 0 and brightness < brightness_threshold:
        print(f"  → FILTERED (below threshold {brightness_threshold})")
        sys.exit(1)
    else:
        print("  → PASSED")

    # Step 2: Crop
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    success, confidence = crop_hand(
        image_path, output_path,
        padding=padding, resize=resize
    )

    if not success:
        print("⚠   No hand detected in the image.")
        sys.exit(1)

    print(f"✂   Cropped (confidence: {confidence:.3f})")

    # Step 3: Contrast check on cropped image
    fg_mean, fg_std, error = compute_foreground_contrast(output_path)
    if error:
        print(f"⚠   Contrast check error: {error}")
    else:
        print(f"◐   Foreground contrast: mean={fg_mean:.1f}, std={fg_std:.1f}", end="")
        if contrast_threshold > 0 and fg_std < contrast_threshold:
            os.remove(output_path)
            print(f"  → FILTERED (std {fg_std:.1f} below threshold {contrast_threshold})")
            sys.exit(1)
        else:
            print("  → PASSED")

    print(f"✅  Done!")


def process_directory(input_dir: str, output_dir: str,
                      brightness_threshold: float = 40.0,
                      contrast_threshold: float = 17.0,
                      padding: int = 20, resize: int | None = None,
                      workers: int = 1) -> None:
    """Process all images in a directory through the full pipeline."""

    images, is_flat = collect_images(input_dir)
    if not images:
        print(f"⚠  No images found in '{input_dir}'.")
        print("   Provide either:")
        print("     • a directory with images directly inside, or")
        print("     • a directory with A–Z subdirectories containing images.")
        sys.exit(1)

    total = len(images)
    start = time.time()

    mode = "flat" if is_flat else "nested (A–Z subdirs)"
    print(f"📂  Input:    {os.path.abspath(input_dir)}")
    print(f"📂  Output:   {os.path.abspath(output_dir)}")
    print(f"📋  Layout:   {mode}")
    print(f"🖼   Images:   {total:,}")
    print(f"☀   Brightness threshold: {brightness_threshold}")
    print(f"◐   Contrast threshold:   {contrast_threshold} (post-crop)")
    print(f"📏  Padding:  {padding}px")
    if resize:
        print(f"🔲  Resize:   {resize}×{resize}px")
    print(f"⚡  Workers:  {workers}")
    print()
    print("  Legend:  ✓ ok   ☀ too dark   ✗ no hand   ◐ low contrast")
    print()

    # Build job list
    jobs = []
    for rel_path, label in images:
        src = os.path.join(input_dir, rel_path)
        dst_dir = os.path.join(output_dir, label) if label else output_dir
        dst = os.path.join(dst_dir, os.path.basename(rel_path))
        jobs.append((src, dst, brightness_threshold, contrast_threshold, padding, resize, rel_path))

    # Process
    results = []
    ok_count = 0
    bright_fail = 0
    crop_fail = 0
    contrast_fail = 0

    if workers <= 1:
        # ── Sequential ───────────────────────────────────────────────────
        landmarker = create_landmarker()

        # Override worker landmarker for sequential mode
        global _worker_landmarker
        _worker_landmarker = landmarker

        for idx, job in enumerate(jobs, 1):
            result = _worker_task(job)
            results.append(result)

            if result[1] == "OK":
                ok_count += 1
            elif result[1] == "BRIGHT_FAIL":
                bright_fail += 1
            elif result[1] == "CONTRAST_FAIL":
                contrast_fail += 1
            else:
                crop_fail += 1

            _print_progress(idx, total, ok_count, bright_fail, crop_fail, contrast_fail)

        landmarker.close()
        _worker_landmarker = None
    else:
        # ── Parallel ─────────────────────────────────────────────────────
        with multiprocessing.Pool(processes=workers,
                                  initializer=_worker_init) as pool:
            for idx, result in enumerate(
                pool.imap_unordered(_worker_task, jobs), 1
            ):
                results.append(result)

                if result[1] == "OK":
                    ok_count += 1
                elif result[1] == "BRIGHT_FAIL":
                    bright_fail += 1
                elif result[1] == "CONTRAST_FAIL":
                    contrast_fail += 1
                else:
                    crop_fail += 1

                _print_progress(idx, total, ok_count, bright_fail, crop_fail, contrast_fail)

    # ── Reports ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    confidence_path, report_path = save_reports(results, output_dir)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start
    read_errors = sum(1 for r in results if r[1] == "READ_ERROR")

    print(f"\n\n{'='*60}")
    print(f"  PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total images        : {total:,}")
    print(f"  Successfully kept   : {ok_count:,}  ({ok_count/total*100:.1f}%)")
    print(f"  Brightness filtered : {bright_fail:,}  ({bright_fail/total*100:.1f}%)")
    print(f"  No hand detected    : {crop_fail:,}  ({crop_fail/total*100:.1f}%)")
    print(f"  Low contrast (post) : {contrast_fail:,}  ({contrast_fail/total*100:.1f}%)")
    print(f"  Read errors         : {read_errors:,}")
    print(f"  Time                : {elapsed:.1f}s")
    print(f"{'='*60}")

    # Per-letter breakdown
    letter_stats: dict[str, dict[str, int]] = {}
    for _, status, _, _, _, rel_path in results:
        parts = rel_path.split(os.sep)
        letter = parts[0] if len(parts) > 1 else "flat"
        if letter not in letter_stats:
            letter_stats[letter] = {"total": 0, "ok": 0, "bright": 0,
                                     "crop": 0, "contrast": 0, "error": 0}
        letter_stats[letter]["total"] += 1
        if status == "OK":
            letter_stats[letter]["ok"] += 1
        elif status == "BRIGHT_FAIL":
            letter_stats[letter]["bright"] += 1
        elif status == "CROP_FAIL":
            letter_stats[letter]["crop"] += 1
        elif status == "CONTRAST_FAIL":
            letter_stats[letter]["contrast"] += 1
        else:
            letter_stats[letter]["error"] += 1

    print(f"\n  Per-letter breakdown:")
    print(f"  {'Letter':<8} {'Total':>6} {'OK':>6} {'Dark':>6} {'NoCrop':>7} {'LowCtst':>8} {'Err':>4}")
    print(f"  {'-'*48}")
    for letter in sorted(letter_stats.keys()):
        s = letter_stats[letter]
        print(f"  {letter:<8} {s['total']:>6} {s['ok']:>6} {s['bright']:>6} "
              f"{s['crop']:>7} {s['contrast']:>8} {s['error']:>4}")

    print(f"\n  Reports:")
    print(f"    {confidence_path}")
    print(f"    {report_path}")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess ASL hand images: brightness filter → "
                    "hand crop → post-crop contrast filter. Outputs clean, "
                    "cropped images ready for model training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  Single image:  python preprocess.py -f hand.jpg -o cropped.jpg\n"
            "  Directory:     python preprocess.py -i ./dataset -o ./cropped\n\n"
            "Supported directory layouts (auto-detected):\n"
            "  Nested:  input_dir/A/*.jpg, input_dir/B/*.png, …\n"
            "  Flat:    input_dir/*.jpg, input_dir/*.png, …\n\n"
            "Examples:\n"
            "  python preprocess.py -f photo.jpg -o cropped_photo.jpg\n"
            "  python preprocess.py -i ./dataset -o ./cropped --resize 296 -w 8\n"
            "  python preprocess.py -i ./dataset -o ./cropped --contrast-threshold 17\n"
            "  python preprocess.py -i ./dataset -o ./cropped --brightness-threshold 0\n"
        ),
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to a single image file to process. When used, -o is the "
             "output file path (not a directory).",
    )
    parser.add_argument(
        "-i", "--input",
        help="Directory containing hand images. Can be a flat folder of "
             "images or a root directory with A–Z subdirectories.",
    )
    _default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess_output")
    parser.add_argument(
        "-o", "--output", default=_default_output,
        help="Output path. A file path when using --file, or a directory "
             f"when using --input. (default: {_default_output})",
    )
    parser.add_argument(
        "-p", "--padding", type=int, default=20,
        help="Pixel padding around the detected hand bounding box "
             "(default: 20).",
    )
    parser.add_argument(
        "-r", "--resize", type=int, default=None,
        help="Optional: resize cropped images to SIZE×SIZE pixels.",
    )
    parser.add_argument(
        "--brightness-threshold", type=float, default=40,
        help="Minimum mean brightness (0-255). Images below this are "
             "filtered out. Set to 0 to disable. (default: 40)",
    )
    parser.add_argument(
        "--contrast-threshold", type=float, default=17,
        help="Minimum foreground contrast (std dev of V-channel in the "
             "cropped hand region, 0-128). Applied AFTER cropping to catch "
             "dark silhouette hands. Set to 0 to disable. (default: 17)",
    )
    parser.add_argument(
        "-w", "--workers", type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel worker processes "
             f"(default: {multiprocessing.cpu_count()}, your CPU count).",
    )

    args = parser.parse_args()

    # Validate: exactly one of --file or --input must be provided
    if args.file and args.input:
        parser.error("Use either --file (-f) or --input (-i), not both.")
    if not args.file and not args.input:
        parser.error("One of --file (-f) or --input (-i) is required.")

    return args


def main() -> None:
    args = parse_args()

    # ── Single-image mode ────────────────────────────────────────────────
    if args.file:
        if not os.path.isfile(args.file):
            print(f"❌  File does not exist: {args.file}")
            sys.exit(1)

        process_single(
            image_path=args.file,
            output_path=args.output,
            brightness_threshold=args.brightness_threshold,
            contrast_threshold=args.contrast_threshold,
            padding=args.padding,
            resize=args.resize,
        )
        return

    # ── Directory mode ───────────────────────────────────────────────────
    if not os.path.isdir(args.input):
        print(f"❌  Input directory does not exist: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    process_directory(
        input_dir=args.input,
        output_dir=args.output,
        brightness_threshold=args.brightness_threshold,
        contrast_threshold=args.contrast_threshold,
        padding=args.padding,
        resize=args.resize,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

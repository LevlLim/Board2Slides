"""Marker residue / blemish removal (hurdle slide 6).

The three functions below are BlemishRemovalTest.py (Computer Vision & Deep Learning/BlemishRemovalTest.py,
md5 ee7738c3e1f25154c81fe389104e90b7) unchanged except that the size limits of the blob filter are parameters
of remove_residue as well (the original fixed them at their defaults inside remove_residue). The CLI and the
matplotlib figure were left out.

Method (from the original): illumination division -> non-local-means denoising -> Otsu threshold ->
auto-escalation: if more than target_ratio of the image is "ink", denoise harder and remove blobs that are too
big / too small to be letters. Output: BINARY, white background, black ink.

What changes for line crops (clean_text_region):
  * the original blob filter drops anything taller than 15 % of the IMAGE height. On a single text line, every
    letter is taller than that, so the filter would erase the writing. clean_text_region sets the limit from
    the LINE height instead (a letter is at most ~1.5 line heights tall, a smear on the board is much taller).
  * it works on the board text region (all lines at once, like the original on a photo), not per line crop:
    NL-means needs the surrounding board to estimate what "background" is, and it is ~1 call per photo
    instead of ~10.
  * it returns BGR (3 channels), because the reader expects colour images.
It binarises: whether this helps the reader is MEASURED in the notebook (variant V4), not assumed.
"""
from __future__ import annotations

import time

import cv2
import numpy as np


def _normalize_illumination(img_gray, sigma_frac=1 / 15.0):
    h, w = img_gray.shape[:2]
    sigma = max(h, w) * sigma_frac
    background = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.divide(img_gray, background, scale=255)
    return normalized


def _threshold_and_filter(normalized, denoise_strength, use_blob_filter,
                          min_area=3, max_area_frac=0.004, max_height_frac=0.15):
    h_img, w_img = normalized.shape[:2]

    denoised = cv2.fastNlMeansDenoising(
        normalized, h=denoise_strength, templateWindowSize=7, searchWindowSize=21
    )
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if use_blob_filter:
        max_area = max_area_frac * (h_img * w_img)
        max_height = max_height_frac * h_img
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # same rule as the original loop, vectorised (the per-label loop takes minutes on a full photo)
        area, height = stats[:, cv2.CC_STAT_AREA], stats[:, cv2.CC_STAT_HEIGHT]
        keep = (area >= min_area) & (area <= max_area) & (height <= max_height)
        keep[0] = False
        binary = np.where(keep[labels], 255, 0).astype(np.uint8)

    # tiny morphological opening - removes 1-2px leftover speckle, safe at any strength
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary, denoised


def remove_residue(img_gray, target_ratio=0.06, max_attempts=4, verbose=False,
                   min_area=3, max_area_frac=0.004, max_height_frac=0.15):
    """Original auto-escalating cleaner. Returns (final, denoised, binary); final = white background, black ink.
    NOTE (kept from the original): max_attempts=4 cuts the 5-step schedule to its first 4 steps."""
    normalized = _normalize_illumination(img_gray)
    schedule = [(10, False), (30, False), (50, True), (70, True), (90, True)]
    schedule = schedule[:max_attempts] if max_attempts < len(schedule) else schedule
    best = None
    best_ratio = None
    for attempt, (denoise_strength, use_blob_filter) in enumerate(schedule, 1):
        binary, denoised = _threshold_and_filter(normalized, denoise_strength, use_blob_filter,
                                                 min_area, max_area_frac, max_height_frac)
        ratio = np.sum(binary > 0) / binary.size
        if verbose:
            print(f"  attempt {attempt}: denoise_strength={denoise_strength}, "
                  f"blob_filter={use_blob_filter} -> foreground ratio={ratio*100:.2f}%")
        if best_ratio is None or ratio < best_ratio:
            best, best_ratio = (binary, denoised), ratio
        if ratio <= target_ratio:
            break
    binary, denoised = best
    final = cv2.bitwise_not(binary)
    if np.mean(final) < 127:
        final = cv2.bitwise_not(final)
    return final, denoised, cv2.bitwise_not(binary)


def clean_text_region(img_bgr, line_h, target_ratio=0.06, max_attempts=4, max_letter_lines=1.5):
    """remove_residue for a board text region whose lines are about line_h px tall. Returns (bgr, info).

    The blob filter keeps components at most max_letter_lines x line_h tall; its area limit is a whole word
    (line_h x 6 line_h) rather than 0.4 % of the region, which on a small region is smaller than one letter.
    """
    t0 = time.time()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    h, w = gray.shape[:2]
    max_height_frac = min(1.0, max_letter_lines * line_h / h)
    max_area_frac = min(1.0, max(0.004, 6.0 * line_h * line_h / (h * w)))
    final, _, binary = remove_residue(gray, target_ratio, max_attempts, False,
                                      min_area=3, max_area_frac=max_area_frac, max_height_frac=max_height_frac)
    ink = float((final < 128).mean())
    info = dict(seconds=round(time.time() - t0, 2), ink_ratio=round(ink, 4),
                max_height_frac=round(max_height_frac, 3), max_area_frac=round(max_area_frac, 4))
    return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR), info

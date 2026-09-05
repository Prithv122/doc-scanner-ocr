"""Contrast/threshold enhancement applied before OCR.

Two methods are offered because they trade off differently for a *learned*
text recognizer (PaddleOCR) rather than a classical engine like Tesseract:

- CLAHE preserves grayscale gradients and anti-aliasing, which is closer to
  the natural-image style PaddleOCR's recognizer was trained on.
- Adaptive thresholding hard-binarizes the image, which can help isolate text
  from uneven lighting/shadows but risks clipping thin strokes.

Which one actually wins is measured empirically in the README, not assumed.
"""

from __future__ import annotations

import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def enhance_clahe(image: np.ndarray) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization, output as 3-channel BGR."""
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def enhance_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Binarize via adaptive Gaussian thresholding, output as 3-channel BGR."""
    gray = _to_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def enhance_for_ocr(image: np.ndarray, method: str = "clahe") -> np.ndarray:
    """Dispatch to the named enhancement method ("clahe", "adaptive", or "none")."""
    if method == "clahe":
        return enhance_clahe(image)
    if method == "adaptive":
        return enhance_adaptive_threshold(image)
    if method == "none":
        return image
    raise ValueError(f"Unknown enhancement method: {method!r}")

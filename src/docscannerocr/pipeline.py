"""End-to-end orchestration: detect -> warp -> enhance -> OCR -> structured result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .detect import find_document_contour
from .enhance import enhance_for_ocr
from .ocr import run_ocr
from .warp import warp_document


def scan_document(image_path: str | Path, enhance_method: str = "clahe") -> dict[str, Any]:
    """Run the full scan pipeline on the image at `image_path`.

    If no document quadrilateral is found, the pipeline falls back to running
    OCR on the original, un-warped image rather than failing outright — a
    photo that's already a flat, front-on shot of a page still has text worth
    extracting even though there was no perspective to correct.
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    corners = find_document_contour(image)
    if corners is not None:
        working = warp_document(image, corners)
        detection: dict[str, Any] = {"found": True, "corners": corners.astype(int).tolist()}
    else:
        working = image
        detection = {"found": False, "corners": None}

    enhanced = enhance_for_ocr(working, method=enhance_method)
    lines = run_ocr(enhanced)

    return {
        "source": str(image_path),
        "detection": detection,
        "enhance_method": enhance_method,
        "lines": lines,
    }

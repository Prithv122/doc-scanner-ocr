"""Perspective-correct a photographed document to a flat top-down view."""

from __future__ import annotations

import cv2
import numpy as np

from .detect import order_points


def compute_output_size(ordered_pts: np.ndarray) -> tuple[int, int]:
    """Derive the output width/height from the corners' own geometry.

    Using the longer of each pair of opposite edges (rather than a fixed
    aspect ratio) keeps the correction faithful to a document photographed
    at an angle, where the near and far edges project to different lengths.
    """
    tl, tr, br, bl = ordered_pts
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    return max_width, max_height


def warp_document(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Return a flat, top-down crop of `image` bounded by the quadrilateral `corners`."""
    ordered = order_points(corners)
    max_width, max_height = compute_output_size(ordered)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))

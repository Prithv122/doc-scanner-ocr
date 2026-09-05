"""Locate a document's quadrilateral outline in a photographed image."""

from __future__ import annotations

import cv2
import numpy as np

_SCALE_HEIGHT = 500.0


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 arbitrary points as top-left, top-right, bottom-right, bottom-left.

    The top-left point has the smallest x+y sum and the bottom-right the largest;
    the top-right point has the smallest x-y difference and the bottom-left the largest.
    """
    pts = np.asarray(pts, dtype="float32")
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).flatten()
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    return ordered


def find_document_contour(
    image: np.ndarray, min_area_ratio: float = 0.05, max_area_ratio: float = 0.95
) -> np.ndarray | None:
    """Find the largest convex quadrilateral in `image`, assumed to be a document.

    Edge detection runs on a downscaled copy for speed; the returned corners are
    rescaled back to `image`'s original pixel coordinates. Returns an unordered
    (4, 2) float32 array of corners, or None if no quadrilateral in the
    [`min_area_ratio`, `max_area_ratio`] band is found.

    `min_area_ratio` defaults low (0.05) because real phone photos of a
    document held at arm's length or set on a table put the page at roughly
    9-15% of the frame (measured against the SmartDoc 2015 sample corpus —
    see README section 5) — a naive "the document fills a big chunk of the
    photo" assumption, closer to 0.2, misses every one of those real frames.

    The upper bound matters more than it looks: heavy dilation over a noisy or
    highly textured background can merge edges into one blob whose outline is
    just the image border, which `approxPolyDP` happily simplifies to a "clean"
    4-point quad covering ~100% of the frame. A real photographed document
    almost always leaves at least a sliver of background margin, so a
    frame-filling quad is treated as a detection artifact, not a document.
    """
    orig_h, orig_w = image.shape[:2]
    ratio = orig_h / _SCALE_HEIGHT if orig_h > _SCALE_HEIGHT else 1.0
    resized = (
        cv2.resize(image, (int(orig_w / ratio), int(_SCALE_HEIGHT)))
        if ratio != 1.0
        else image.copy()
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    resized_area = resized.shape[0] * resized.shape[1]
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area_ratio = cv2.contourArea(approx) / resized_area
            if min_area_ratio <= area_ratio <= max_area_ratio:
                return approx.reshape(4, 2).astype("float32") * ratio

    return None

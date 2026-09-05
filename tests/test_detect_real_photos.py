"""Regression test locking in the min_area_ratio finding against real photos.

These are real smartphone-camera video frames (SmartDoc 2015 Challenge 1
sample, CC BY 4.0 — see data/smartdoc_sample/ATTRIBUTION.md), not synthetic
fixtures. The committed ground_truth.json corners come from the dataset's own
XML annotations, independent of this project's code.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from docscannerocr.detect import find_document_contour, order_points

DATA_DIR = Path(__file__).parent.parent / "data" / "smartdoc_sample"


def _load_ground_truth() -> dict:
    with open(DATA_DIR / "ground_truth.json") as f:
        return json.load(f)


@pytest.mark.parametrize("filename", sorted(_load_ground_truth().keys()))
def test_detects_real_photographed_documents_within_tolerance(filename):
    gt = _load_ground_truth()[filename]
    image = cv2.imread(str(DATA_DIR / "frames" / filename))
    corners = gt["corners"]
    true_pts = order_points(
        np.array([corners["tl"], corners["tr"], corners["br"], corners["bl"]], dtype="float32")
    )

    found = find_document_contour(image)

    assert found is not None, f"{filename}: no document detected"
    mean_corner_dist = np.linalg.norm(order_points(found) - true_pts, axis=1).mean()
    # Measured ~5px on this corpus (1920x1080 frames); 30px leaves headroom
    # for OpenCV/platform variance without masking a real regression.
    assert mean_corner_dist < 30, f"{filename}: mean corner error {mean_corner_dist:.1f}px"

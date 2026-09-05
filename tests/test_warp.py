import cv2
import numpy as np

from docscannerocr.warp import compute_output_size, warp_document


def test_compute_output_size_matches_edge_lengths():
    ordered = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype="float32")

    width, height = compute_output_size(ordered)

    assert width == 100
    assert height == 50


def test_compute_output_size_uses_the_longer_of_each_opposite_edge_pair():
    # Top edge (120) is longer than the bottom edge (80); left/right both 60.
    ordered = np.array([[0, 0], [120, 0], [100, 60], [10, 60]], dtype="float32")

    width, _ = compute_output_size(ordered)

    assert width == 120


def test_warp_document_produces_a_flat_rectangle_of_the_expected_shape():
    canvas = np.zeros((300, 400, 3), dtype=np.uint8)
    corners = np.array([[50, 40], [350, 20], [370, 260], [30, 280]], dtype="float32")
    cv2.fillConvexPoly(canvas, corners.astype(np.int32), (255, 255, 255))

    warped = warp_document(canvas, corners)
    width, height = compute_output_size(corners)

    assert warped.shape[:2] == (height, width)
    # The document interior (all white) should fill the corrected frame's center.
    assert tuple(warped[height // 2, width // 2]) == (255, 255, 255)

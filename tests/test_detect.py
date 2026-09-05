import cv2
import numpy as np

from docscannerocr.detect import find_document_contour, order_points


def _make_skewed_document(corners, canvas_size=(600, 800)):
    """Draw a white quadrilateral (the "document") over a noisy dark background."""
    canvas = np.full((*canvas_size, 3), 40, dtype=np.uint8)
    rng = np.random.default_rng(0)
    canvas = cv2.add(canvas, rng.integers(0, 30, canvas.shape, dtype=np.uint8))
    cv2.fillConvexPoly(canvas, np.array(corners, dtype=np.int32), (255, 255, 255))
    return canvas


def test_order_points_orders_regardless_of_input_order():
    true_order = np.array([[10, 10], [200, 20], [190, 300], [5, 290]], dtype="float32")
    shuffled = true_order[[2, 0, 3, 1]]

    ordered = order_points(shuffled)

    np.testing.assert_allclose(ordered, true_order, atol=1e-3)


def test_find_document_contour_recovers_known_quadrilateral():
    true_corners = [(80, 60), (700, 40), (740, 520), (60, 540)]
    image = _make_skewed_document(true_corners)

    found = find_document_contour(image)

    assert found is not None
    ordered_found = order_points(found)
    ordered_true = order_points(np.array(true_corners, dtype="float32"))
    # Tolerance covers Canny/approxPolyDP rounding and the downscale/rescale roundtrip.
    np.testing.assert_allclose(ordered_found, ordered_true, atol=15)


def test_find_document_contour_returns_none_without_a_document():
    rng = np.random.default_rng(1)
    noisy = rng.integers(0, 255, (400, 600, 3), dtype=np.uint8).astype(np.uint8)

    assert find_document_contour(noisy) is None


def test_find_document_contour_respects_min_area_ratio():
    # A small quadrilateral that covers well under the default 20% area threshold.
    small_corners = [(370, 280), (430, 280), (430, 320), (370, 320)]
    image = _make_skewed_document(small_corners)

    assert find_document_contour(image, min_area_ratio=0.2) is None
    assert find_document_contour(image, min_area_ratio=0.001) is not None

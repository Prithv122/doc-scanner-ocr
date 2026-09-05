import cv2
import numpy as np
import pytest

from docscannerocr.pipeline import scan_document


def _synthetic_photo():
    """A "page" with text, perspective-warped into a larger, darker "scene".

    This stands in for a phone photo of a document taken at an angle: the
    pipeline has to find the page, undo the perspective, and then read text
    that was never axis-aligned in the source image.
    """
    page = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.putText(page, "INVOICE 42", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
    cv2.putText(page, "TOTAL 199.99", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    page_h, page_w = page.shape[:2]

    src_pts = np.array(
        [[0, 0], [page_w - 1, 0], [page_w - 1, page_h - 1], [0, page_h - 1]], dtype="float32"
    )
    dst_pts = np.array([[120, 90], [780, 60], [820, 620], [90, 650]], dtype="float32")

    scene = np.full((700, 900, 3), 30, dtype=np.uint8)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_page = cv2.warpPerspective(page, matrix, (scene.shape[1], scene.shape[0]))
    page_mask = cv2.warpPerspective(
        np.full((page_h, page_w), 255, dtype=np.uint8), matrix, (scene.shape[1], scene.shape[0])
    )

    scene[page_mask > 0] = warped_page[page_mask > 0]
    return scene, dst_pts


@pytest.mark.network
def test_scan_document_end_to_end(tmp_path):
    scene, true_corners = _synthetic_photo()
    image_path = tmp_path / "photo.png"
    cv2.imwrite(str(image_path), scene)

    result = scan_document(image_path)

    assert result["detection"]["found"] is True
    from docscannerocr.detect import order_points

    np.testing.assert_allclose(
        order_points(np.array(result["detection"]["corners"], dtype="float32")),
        order_points(true_corners),
        atol=20,
    )

    joined_text = " ".join(line["text"] for line in result["lines"]).upper()
    assert "INVOICE" in joined_text
    assert "TOTAL" in joined_text


def test_scan_document_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does-not-exist.png"

    with pytest.raises(ValueError, match="Could not read image"):
        scan_document(missing)

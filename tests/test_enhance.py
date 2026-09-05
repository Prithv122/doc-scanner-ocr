import numpy as np
import pytest

from docscannerocr.enhance import enhance_adaptive_threshold, enhance_clahe, enhance_for_ocr


def _gradient_image():
    row = np.linspace(0, 255, 300, dtype=np.uint8)
    gray = np.tile(row, (200, 1))
    return np.stack([gray, gray, gray], axis=-1)


def test_enhance_clahe_preserves_multiple_gray_levels():
    result = enhance_clahe(_gradient_image())

    assert result.shape[-1] == 3
    assert len(np.unique(result)) > 2  # not binarized


def test_enhance_adaptive_threshold_produces_a_binary_image():
    result = enhance_adaptive_threshold(_gradient_image())

    assert result.shape[-1] == 3
    assert set(np.unique(result)).issubset({0, 255})


def test_enhance_for_ocr_none_is_a_passthrough():
    image = _gradient_image()

    assert enhance_for_ocr(image, method="none") is image


def test_enhance_for_ocr_dispatches_to_adaptive():
    result = enhance_for_ocr(_gradient_image(), method="adaptive")

    assert set(np.unique(result)).issubset({0, 255})


def test_enhance_for_ocr_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown enhancement method"):
        enhance_for_ocr(_gradient_image(), method="bogus")

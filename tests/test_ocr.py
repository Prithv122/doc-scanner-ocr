import cv2
import numpy as np
import pytest

from docscannerocr.ocr import run_ocr


@pytest.mark.network
def test_run_ocr_reads_clean_synthetic_text():
    image = np.full((200, 700, 3), 255, dtype=np.uint8)
    cv2.putText(image, "HELLO WORLD 12345", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

    lines = run_ocr(image)

    assert len(lines) == 1
    line = lines[0]
    assert line["text"].strip().upper() == "HELLO WORLD 12345"
    assert line["confidence"] > 0.9
    assert len(line["box"]) == 4


@pytest.mark.network
def test_run_ocr_returns_empty_list_for_blank_image():
    blank = np.full((200, 700, 3), 255, dtype=np.uint8)

    assert run_ocr(blank) == []

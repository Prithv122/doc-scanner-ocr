"""PaddleOCR wrapper producing a documented, structured output schema.

Each detected text line becomes:
    {"text": str, "confidence": float, "box": [[x, y], [x, y], [x, y], [x, y]]}
where "box" is the 4-point polygon PaddleOCR's detector returns (not
necessarily axis-aligned — text lines can be slightly rotated).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from paddleocr import PaddleOCR

_ocr_engine: PaddleOCR | None = None


def get_engine() -> PaddleOCR:
    """Lazily construct the shared PaddleOCR engine — model load is expensive.

    Document orientation classification and unwarping are disabled: this
    project's own detect/warp stages already produce an upright, flattened
    image, so PaddleOCR's built-in equivalents would be redundant work.
    """
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
        )
    return _ocr_engine


def run_ocr(image: np.ndarray) -> list[dict[str, Any]]:
    """Run OCR on `image`, returning one entry per detected text line."""
    engine = get_engine()
    page = engine.predict(image)[0]
    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])
    polys = page.get("rec_polys", [])

    lines = []
    for text, score, poly in zip(texts, scores, polys, strict=True):
        lines.append(
            {
                "text": text,
                "confidence": float(score),
                "box": np.asarray(poly).astype(int).tolist(),
            }
        )
    return lines

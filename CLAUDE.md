# Document Scanner & OCR — H1

**Tier:** 1 · **Category:** H - Computer vision · **Wave:** 2

Root rules in `../CLAUDE.md` apply. This file is project-specific only — keep it under 40 lines.

## What this is

Classical OpenCV document scanner (edge detection, perspective correction, adaptive
thresholding) feeding PaddleOCR for structured text extraction. No deep learning in the
vision pipeline itself — that's the point (CATALOG H1: "CV without deep learning").

## Stack

Python 3.12 · opencv-python-headless · PaddleOCR (PP-OCRv6) · paddlepaddle 3.2.2 (pinned,
see NOTES.md) · pytest · ruff · GitHub Actions.

## Acceptance criteria

- [x] Document detection + perspective correction from scratch with OpenCV (CATALOG H1)
- [x] PaddleOCR → structured JSON output (text + confidence + box per line)
- [x] Measured against real photographed documents, not synthetic-only (README §2, §5)
- [x] Real numbers: 9/9 detection, 5.1px mean corner error, CLAHE-vs-adaptive OCR comparison
- [ ] Ship gate passes (`/ship`)

## Project-specific notes

- **`paddlepaddle` is pinned to `3.2.2`.** 3.3.x has a CPU oneDNN/PIR regression that crashes
  every real inference call on Windows — see NOTES.md.
- **`min_area_ratio` in `detect.py` defaults to 0.05, not a more "obvious" 0.2** — real phone
  photos put the document at 9-15% of frame area. See NOTES.md/README §4 for the story.
- Real test data lives in `data/smartdoc_sample/` (SmartDoc 2015, CC BY 4.0) — see
  `ATTRIBUTION.md` there. Regenerate `results.json` with `uv run python scripts/benchmark.py`.
- No env vars, no accounts. PaddleOCR downloads model weights over the network on first real
  use (cached after).

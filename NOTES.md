# Build Notes — Document Scanner & OCR

Working notes: what broke, what you tried, why you chose X over Y.
Not for recruiters — for you, six months from now, in an interview.

---

## Log

### 2026-09-05

- **Tried:** `paddleocr` + `paddlepaddle` on Python 3.12/Windows, latest versions
  (paddlepaddle 3.3.1) at scaffold time.
- **Broke:** First real inference call crashed with
  `NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
  [pir::ArrayAttribute<pir::DoubleAttribute>]` inside oneDNN's PIR conversion layer — a
  regression in paddlepaddle 3.3.x's CPU inference path, not something wrong in this project's
  code (confirmed via [PaddlePaddle/Paddle#77340](https://github.com/PaddlePaddle/Paddle/issues/77340)
  and [PaddlePaddle/PaddleOCR#18162](https://github.com/PaddlePaddle/PaddleOCR/issues/18162)).
- **Fixed by:** Pinning `paddlepaddle==3.2.2` in `pyproject.toml`. Confirmed with a real
  inference call (a synthetic "HELLO WORLD 12345" image) before writing any wrapper code
  around it — no point designing `ocr.py`'s API against a library call that doesn't run.
- **Learned:** Don't assume "latest" is safe for a fast-moving inference library on Windows;
  a scratch smoke test against the actual API (not just `import`) would have caught this
  before it became a dependency baked into the scaffold.

- **Tried:** `find_document_contour` with `min_area_ratio=0.2`, tuned by hand against synthetic
  test images where the "document" filled most of the frame.
- **Broke:** Ran it against all 9 real SmartDoc photos (§2 of the README) — **zero** were
  detected. Measured the real documents' area ratio directly: 9-15% of frame, well under the
  0.2 floor. The synthetic tests never exercised a realistic phone-camera framing distance, so
  the bug shipped invisibly until real data was available.
- **Fixed by:** Lowering the default to `min_area_ratio=0.05`. Re-ran against all 9 real
  frames: 9/9 detected, mean corner error 5.1px, mean IoU 0.975.
- **Learned:** A synthetic fixture that never varies the one parameter that matters (how much
  of the frame the subject occupies) will pass every test while being wrong on real input.
  This is why `tests/test_detect_real_photos.py` exists as a separate, real-data regression
  test — the synthetic geometry tests in `test_detect.py` couldn't have caught this class of
  bug by construction.

- **Tried:** Scoring OCR accuracy with exact substring match against hand-verified key
  phrases.
- **Broke:** A real run recognized a magazine headline as `"Tackding Tibet."` at 97.6%
  confidence, against the true `"Tackling Tibet"` — one character off (l→d). Exact matching
  scored this as a complete miss, which misrepresents what actually happened: the OCR
  essentially read the headline correctly.
- **Fixed by:** Fuzzy matching via `difflib.SequenceMatcher`, threshold 0.85, in
  `scripts/benchmark.py`. Recall went from 25% (exact) to 50% (fuzzy) for CLAHE — the fuzzy
  number is the honest one; the exact-match number was an artifact of the metric, not the
  OCR's actual behavior.
- **Learned:** For a learned OCR recognizer, "did it get the text right" is not a boolean per
  phrase. A metric that can't express "97.6%-confident and basically correct" will produce
  numbers that actively mislead rather than just being imprecise.

- **Tried/measured:** CLAHE vs. adaptive Gaussian thresholding as the pre-OCR enhancement,
  on all 9 real frames, both methods, via `scripts/benchmark.py`.
- **Result:** CLAHE: 50% key-phrase recall. Adaptive threshold: 2.2%. Not a close call.
- **Learned:** The intuition that "binarize it, OCR likes clean black-and-white text" is
  built for classical engines (Tesseract-era). PaddleOCR's recognizer is a CNN trained on
  natural, grayscale/color scene text; hard thresholding throws away exactly the
  anti-aliasing gradient information it was trained to use. Worth remembering for any future
  OCR project (G7 structured-extract, 24 production-rag if it ever touches scanned PDFs):
  don't binarize before a learned OCR model without measuring first.

---

## Rejected approaches

| Approach | Why rejected |
|---|---|
| Full-page ground-truth transcription for OCR accuracy | Several real frames (the letter series especially) have motion-blurred body text that isn't confidently human-readable either. A "ground truth" transcription of text I can't actually read with confidence would be a guess dressed up as a number — worse than a smaller, honestly-scoped key-phrase metric. |
| Taking my own phone photos for the test set | No camera available in this environment. Used a real, CC BY 4.0 licensed public dataset (SmartDoc 2015) instead of synthesizing "photographed" test images — see README §2. |
| Deep-learning-based page segmentation (e.g. a U-Net document mask) | Explicitly out of scope — CATALOG H1 is "CV without deep learning" for the vision pipeline; a learned detector for the page itself would defeat the point of the project. |
| `use_doc_unwarping=True` (PaddleOCR's own built-in dewarping) | This project's own `detect.py`/`warp.py` already produce an upright, flattened image; enabling PaddleOCR's redundant equivalent would hide whether *this project's* perspective correction actually works. |

## Open questions

- [ ] How does detection accuracy degrade on document types the SmartDoc sample doesn't
      cover (curled receipts, handwriting, non-Latin scripts)? Untested — flagged honestly in
      README §7 rather than assumed to generalize.

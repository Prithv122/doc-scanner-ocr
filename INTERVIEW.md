# Interview Prep — Document Scanner & OCR

**Five questions, five answers.** An unanswered question means this project is not shipped.

---

### Q1. Walk me through the architecture in 90 seconds.

_A:_ A photo goes through four stages. `detect.py` downscales the image, runs Canny edge
detection plus contour finding, and looks for the largest convex 4-point polygon within an
area band (5-95% of frame) — that's the document. `warp.py` orders those four corners
(top-left/top-right/bottom-right/bottom-left) and perspective-corrects them into a flat,
top-down crop, sizing the output from the corners' own geometry rather than a fixed aspect
ratio. `enhance.py` applies CLAHE contrast enhancement (chosen over adaptive thresholding —
more on that in Q2). `ocr.py` wraps PaddleOCR to turn the enhanced image into a list of
`{text, confidence, box}` records. `pipeline.py` chains all four steps and falls back to
running OCR on the un-warped original if no document quadrilateral is found. The CLI
(`cli.py`) is a thin argparse wrapper that prints or writes the resulting JSON.

### Q2. Why did you choose CLAHE over adaptive thresholding for pre-OCR enhancement?

_A:_ I didn't assume it — I measured both against 9 real photographed frames from the
SmartDoc 2015 dataset and ran the actual PaddleOCR pipeline on each. CLAHE got 50% key-phrase
recall; adaptive Gaussian thresholding got 2.2%. The reason is architectural: adaptive
thresholding hard-binarizes the image to pure black and white, which is what you'd want for a
classical engine like Tesseract. But PaddleOCR's recognizer is a CNN trained on natural
grayscale/color scene text — it relies on anti-aliasing and gradient information that hard
thresholding throws away. CLAHE boosts local contrast without destroying that gradient. The
gap (50% vs 2.2%) is large enough that this isn't close — see README §5 and
`data/smartdoc_sample/results.json` for the raw per-frame numbers.

### Q3. What's the weakest part of this, and what would break first under load?

_A:_ Two things. First, `find_document_contour` assumes a clean 4-point boundary — it has no
answer for a curled page, a folded receipt, or a document partially occluded by a hand; a
real production version would need a learned segmentation model for that (explicitly out of
scope here — CATALOG H1 is "CV without deep learning" for this exact reason). Second, my
evaluation set is only 9 frames. That was enough to catch a real bug (see Q5) and measure a
genuine CLAHE-vs-adaptive gap, but it's not enough sample size to trust 50% as a population
recall estimate — under load (thousands of documents), I'd expect meaningfully more variance
than 9 frames can reveal, and I'd want the full ~24,000-frame SmartDoc corpus or an
equivalent before trusting the number in a real product decision.

### Q4. How do you know it works? What did you measure, and against what baseline?

_A:_ Detection: 9/9 real photos correctly found their document quadrilateral, mean corner
error 5.1px on a 1920×1080 frame (0.27% of frame width), mean IoU 0.975 against the SmartDoc
dataset's own XML-annotated ground truth — not my own labels, an independent check. OCR:
50% key-phrase recall with CLAHE vs. 2.2% with adaptive thresholding, using fuzzy matching
(difflib similarity ≥ 0.85) against a small set of key phrases I hand-verified as legible in
each frame. I didn't attempt full-page character-error-rate, because several of the real
frames (motion-blurred letter photos especially) have body text I can't confidently transcribe
myself either — a "ground truth" for text I can't read would just be a guess. All numbers are
reproducible: `uv run python scripts/benchmark.py` regenerates
`data/smartdoc_sample/results.json` from the committed frames and ground truth.

### Q5. What's the hardest bug you hit, and how did you find it?

_A:_ `find_document_contour`'s area-ratio threshold. I built and unit-tested it against
synthetic images where the "document" filled most of the frame, tuned the default to
`min_area_ratio=0.2`, and every synthetic test passed. Then I ran it against 9 real
SmartDoc photos and got zero detections — not degraded accuracy, a complete miss on every
single frame. I measured the real documents' actual area ratio (9-15% of frame — a phone
held at a natural distance photographs a page much smaller than my synthetic close-ups
assumed) and lowered the default to 0.05. That single-line fix took detection from 0/9 to
9/9. The lesson: a synthetic fixture that never varies the one parameter that matters will
pass every test while being systematically wrong on real input — which is why
`tests/test_detect_real_photos.py` exists as a dedicated regression test against real,
ground-truthed photos rather than trusting the synthetic geometry tests alone.

---

## 30-second pitch

Phone photos of documents get perspective-corrected with classical OpenCV (Canny, contour
detection, `getPerspectiveTransform`) — no learned model for the page-finding step — then fed
to PaddleOCR for structured text extraction. Tested against 9 real photographed documents
from a public research dataset (not synthetic images): 100% detection rate, 5.1px mean corner
error, and a measured 50%-vs-2% OCR accuracy gap between two contrast-enhancement choices that
most tutorials treat as interchangeable.

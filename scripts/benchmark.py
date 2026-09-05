"""Measure detection accuracy and OCR recall against the real SmartDoc sample photos.

Produces the numbers cited in README section 5. Run with:
    uv run python scripts/benchmark.py

Two things are measured against real (not synthetic) photographed documents:

1. Detection accuracy: does `find_document_contour` find the page, and how far
   off are its corners from the dataset's own ground truth (mean corner
   distance in pixels, and IoU of the two quadrilaterals)?
2. OCR recall: of a small set of hand-verified, confidently-legible key
   phrases per frame (data/smartdoc_sample/ocr_ground_truth.json), how many
   does the pipeline's OCR output actually contain? Compared across both
   enhancement methods (CLAHE vs adaptive threshold) so enhance.py's claimed
   tradeoff is measured, not assumed.

   Matching is fuzzy (difflib similarity >= MATCH_THRESHOLD against the best
   OCR line), not exact-substring: a real run recognized a magazine headline
   as "Tackding Tibet." (97.6% confidence) against a true "Tackling Tibet" —
   a one-character misread that exact matching would count as a total miss
   despite the OCR having essentially read it correctly.

Full-page character-error-rate isn't attempted: several frames (the letter
series especially) have motion-blurred body text that isn't confidently
human-transcribable either, so a full ground-truth transcription would just
be a guess dressed up as a number. Key-phrase recall only claims what a human
can verify by eye — see NOTES.md.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

from docscannerocr.detect import find_document_contour, order_points
from docscannerocr.enhance import enhance_for_ocr
from docscannerocr.ocr import run_ocr
from docscannerocr.warp import warp_document

DATA_DIR = Path(__file__).parent.parent / "data" / "smartdoc_sample"
ENHANCE_METHODS = ["clahe", "adaptive"]
MATCH_THRESHOLD = 0.85


def _best_match(phrase: str, lines: list[str]) -> tuple[str | None, float]:
    """Best-matching OCR line for `phrase`, by whole-string similarity ratio."""
    best_line, best_ratio = None, 0.0
    for line in lines:
        ratio = SequenceMatcher(None, phrase.upper(), line.upper()).ratio()
        if ratio > best_ratio:
            best_line, best_ratio = line, ratio
    return best_line, best_ratio


def _quad_iou(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> float:
    mask_a = np.zeros(shape, dtype=np.uint8)
    mask_b = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask_a, a.astype(np.int32), 1)
    cv2.fillConvexPoly(mask_b, b.astype(np.int32), 1)
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return intersection / union if union else 0.0


def main() -> None:
    with open(DATA_DIR / "ground_truth.json") as f:
        ground_truth = json.load(f)
    with open(DATA_DIR / "ocr_ground_truth.json") as f:
        ocr_ground_truth = json.load(f)

    per_frame = []
    recall_by_method: dict[str, list[float]] = {m: [] for m in ENHANCE_METHODS}

    for filename, gt in sorted(ground_truth.items()):
        image_path = DATA_DIR / "frames" / filename
        image = cv2.imread(str(image_path))

        corners = gt["corners"]
        true_pts = order_points(
            np.array([corners["tl"], corners["tr"], corners["br"], corners["bl"]], dtype="float32")
        )
        found = find_document_contour(image)

        frame_result: dict = {"file": filename, "detected": found is not None}
        if found is not None:
            ordered_found = order_points(found)
            frame_result["mean_corner_error_px"] = round(
                float(np.linalg.norm(ordered_found - true_pts, axis=1).mean()), 2
            )
            frame_result["iou"] = round(_quad_iou(ordered_found, true_pts, image.shape[:2]), 4)
            working = warp_document(image, found)
        else:
            working = image

        expected_phrases = ocr_ground_truth.get(filename, [])
        frame_result["ocr_by_method"] = {}
        for method in ENHANCE_METHODS:
            enhanced = enhance_for_ocr(working, method=method)
            ocr_lines = [line["text"] for line in run_ocr(enhanced)]

            matches = []
            for phrase in expected_phrases:
                best_line, ratio = _best_match(phrase, ocr_lines)
                matches.append(
                    {
                        "phrase": phrase,
                        "matched": ratio >= MATCH_THRESHOLD,
                        "best_ocr_line": best_line,
                        "similarity": round(ratio, 3),
                    }
                )

            recall = (
                sum(m["matched"] for m in matches) / len(expected_phrases)
                if expected_phrases
                else None
            )
            frame_result["ocr_by_method"][method] = {"recall": recall, "phrases": matches}
            if recall is not None:
                recall_by_method[method].append(recall)

        per_frame.append(frame_result)

    detected_count = sum(1 for r in per_frame if r["detected"])
    mean_corner_error = np.mean([r["mean_corner_error_px"] for r in per_frame if r["detected"]])
    mean_iou = np.mean([r["iou"] for r in per_frame if r["detected"]])

    summary = {
        "n_frames": len(per_frame),
        "detected": detected_count,
        "mean_corner_error_px": round(float(mean_corner_error), 2),
        "mean_iou": round(float(mean_iou), 4),
        "ocr_recall_by_method": {
            m: round(float(np.mean(v)), 3) if v else None for m, v in recall_by_method.items()
        },
    }

    print(f"Detection: {detected_count}/{len(per_frame)} frames")
    print(f"Mean corner error: {summary['mean_corner_error_px']} px")
    print(f"Mean IoU: {summary['mean_iou']}")
    for method, recall in summary["ocr_recall_by_method"].items():
        print(f"OCR key-phrase recall ({method}): {recall}")

    with open(DATA_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_frame": per_frame}, f, indent=2)
    print(f"\nFull results written to {DATA_DIR / 'results.json'}")


if __name__ == "__main__":
    main()

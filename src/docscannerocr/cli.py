"""Console entry point: run the scan pipeline on an image, print/save structured JSON."""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import scan_document


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="doc-scanner-ocr",
        description="Classical OpenCV document scanner + PaddleOCR structured extraction.",
    )
    parser.add_argument("image", help="Path to a photographed document image")
    parser.add_argument(
        "--enhance",
        choices=["clahe", "adaptive", "none"],
        default="clahe",
        help="Contrast/threshold enhancement applied before OCR (default: clahe)",
    )
    parser.add_argument("-o", "--output", help="Write JSON result to this path instead of stdout")
    args = parser.parse_args()

    try:
        result = scan_document(args.image, enhance_method=args.enhance)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    payload = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        print(payload)

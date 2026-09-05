# Resume Bullets — Document Scanner & OCR

Form: **action → technical specifics → measured outcome.** Numbers or it doesn't go on the resume.

---

## Bullets

- Built a classical-CV document scanner (OpenCV Canny/contour detection, perspective
  correction) feeding PaddleOCR for structured text extraction, achieving 100% document
  detection and 5.1px mean corner error on real photographed documents (SmartDoc 2015 dataset).
- Measured a 50% vs. 2.2% OCR key-phrase recall gap between two image-enhancement methods
  (CLAHE vs. adaptive thresholding) against real photos, correcting an assumption most
  tutorials leave unexamined and driving the project's enhancement default.
- Found and fixed a detection-threshold bug invisible to synthetic tests: a document-area
  filter tuned on close-up synthetic images rejected 100% of real phone-camera photos (where
  the page occupies only 9-15% of frame); a one-parameter fix restored full detection.

## Which roles this supports

- [x] Data Scientist / ML
- [x] AI Engineer (LLM/NLP/CV)
- [ ] Data Engineer
- [x] Data Analyst / Python Developer

## Keywords this project earns

OpenCV, perspective transform, contour detection, PaddleOCR, image enhancement (CLAHE,
adaptive thresholding), OCR evaluation, classical computer vision (non-deep-learning),
Python, pytest, CI/CD.

---

### Bad vs good

❌ "Built a machine learning model to predict customer churn using Python."
✅ "Built a churn classifier on 240k accounts (LightGBM, 1:40 class imbalance) with isotonic calibration and cost-sensitive thresholding, lifting precision@10% from 0.31 to 0.58 over the business's existing rules baseline."

The second one is answerable in an interview. The first invites the question you can't answer.

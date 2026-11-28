# Board2Notes: local OCR prototype

This is the student-facing OCR-only application. It turns a whiteboard photograph into editable notes using an existing trained reader and the cached PP-OCRv5 mobile detector. It has no VLM, LLM, automatic spelling correction or slide generator.

## Open it

If the development server is running, open **http://127.0.0.1:8501**.

To start it again, double-click `Assignment/run_ocr.command` or run from `Assignment/`:

```bash
bash run_ocr.command
```

Keep that terminal open. Use Ctrl+C there to stop the server. If port 8501 is already occupied by this prototype, open the address rather than start another copy.

The project Python environment and Streamlit 1.67.0 were installed/tested on 8 November. This is a local app bound to this computer, not a deployed cloud service. See `README_UI.md` and `requirements-ui.txt` for UI details. Existing Paddle/PaddleOCR dependencies and exported model folders must remain available.

## What to do

1. Upload a whiteboard JPG, PNG or HEIC photo, or choose **Try a sample photo**.
2. Check the preview. Optionally rotate it or trim the edges, keeping every letter in frame.
3. Press **Extract notes**. The app reads only when you press this button.
4. Compare the photo and editable notes. Numbered boxes show where writing was found by default. Click one to highlight it and see the extracted handwriting crop plus its original reading, or use the line selector. **How your photo was read** shows the actual image → regions → crop → text path.
5. Correct the notes, then **Copy notes** or **Download text**. Check for missing writing as well as incorrect words.

There is one fixed default reader: **S1 seed 1**, chosen using the existing whiteboard validation results, not Jy test performance. This is a provisional integration default, not a claim of universal superiority. Students never select a model or see experiment names. A developer can set `B2N_OCR_MODEL` before starting the app; model comparisons remain in notebooks 10/11. No automatic per-photo model selection is performed.

Edits survive returning to the photo tools. Re-extraction requires explicit consent before replacing edited notes. Starting another photo asks the user to save or confirm if they have edited the text. Notes remain session-only: download/copy them before closing or refreshing the page. The backend's structured result remains available in Python for later team integration; student-facing JSON/metrics controls have been removed.

## Actual path

EXIF orientation -> existing detector (threshold .20, box threshold .40, unclip 1.2) -> existing small-text re-detection -> line grouping -> simple column/row ordering -> quadrilateral crop -> each model's required resize -> CTC recognition.

This starting version does not apply optional board-perspective correction, lighting enhancement, deskew correction, sharpening or learned SR. A quadrilateral line crop already resamples the detected text region. The simple column-order heuristic can fail on complicated layouts; it is recorded in the result.

The backend uses our verified RunPod resize rounding, embedded character dictionary and inference pass policy on **local Paddle 3.3.1 CPU**. CPU output is newly evaluated and must not be assumed bit-identical to RunPod Paddle 3.2.0 GPU. It loads one crop at a time.

## Jy evaluation and visual notebook

Open `whiteboard-ocr/notebooks/11_jy_ocr_evaluation.ipynb` from the Assignment project. It reads the saved comparison by default; launching inference is an explicit optional action.

Re-run the six-reader original-crop evaluation from `Assignment/`:

```bash
.venv-paddle/bin/python whiteboard-ocr/scripts/evaluate_jy_readers.py
```

This evaluates all 298 originals from both supplied train/test folders. It excludes 2,086 synthetic copies and uses S0/S1/G0 with both seeds. The labels were reported reviewed by the teammate. Per-model caches record image, manifest, model and runtime hashes; modified inputs/settings require a new output directory. The G0-versus-S0 confirmation rule is saved before prediction.

Generate fixed-detection full-photo previews:

```bash
.venv-paddle/bin/python whiteboard-ocr/scripts/run_jy_full_photos.py
```

Outputs: `whiteboard-ocr/outputs/jy_ocr_evaluation/`.

- `report.md`, `summary.json`, `predictions.csv`: crop-level readings and metrics.
- `full_photos/review.html`: 16 full photographs read by S0/S1/G0s2 using identical detected boxes/crops.
- `geometry/reviewed_review.html`: assistant-reviewed correspondence between all 298 supplied crop targets and full-photo regions; original proposals are also retained.
- `full_photo_diagnostics/report.md`: geometric annotated-target diagnostics, including missing/extra regions and split/merge candidates; explicitly not certified full-page accuracy.
- `prototype_photo23.json`: first real full-photo backend smoke result.
- `prototype_screenshot.jpg`: browser proof of the working interface.

## What is and is not verified

- Real backend test: G0s2 read `Lipids` correctly from a supplied original crop.
- Real CLI/browser test: Jy Photo 23 produced nine regions and OCR text. Warm-path processing was approximately 3 seconds on this machine; this is a sample observation, not a benchmark over every photo.
- The revised student flow was exercised with a real Photo 23 upload and the sample route. S1 returned nine regions. The browser verified editing a wrong word, copying the latest edit, returning through photo adjustment without losing the edit, and both cancelling and confirming the new-note dialog. The close-up tool and a phone-sized layout were also checked. The notes download prepares a UTF-8 text blob; the in-app browser's automated download event timed out, so a completed download to disk was not confirmed by that event test. Check this once in the browser used for the presentation.
- Twenty-one automated checks pass: seven pure photo-helper tests and fourteen Streamlit journey tests with OCR mocked. They cover HEIC/EXIF, exact rotated/cropped OCR inputs, tiny crops, preserved crop settings and notes across navigation, linked region selection without losing edits, explicit overwrite consent, confirmation handlers, failures, empty OCR and safe export encoding. They do not establish OCR accuracy or replace testing with actual students.
- Every full-photo comparison reader receives the same saved detections and crop pixels.
- Full-photo ground truth is **not yet certified complete**. The assistant visually checked inferred positions for all 298 supplied targets and recovered three initially unlocated words. Photo 14 still has unlabelled faint writing, and erased-writing scope needs confirmation. Do not present crop-level CER or annotated-target diagnostics as complete-photo OCR accuracy. Confirm scope/completeness before an official full-photo score.
- No model or Jy training was performed. Original data and historical model outputs were preserved.

Future work: review full-photo annotations, evaluate missed/extra text and reading order, confirm the chosen preprocessing policy, then connect OCR JSON to the team's downstream application. VLM integration is later.

## Student usability check still to do

Ask two or three students unfamiliar with the project to add a photo, correct one OCR error and copy/download their notes without coaching. Record where they hesitate, whether the saved text contains their correction, and whether they can recover from an unreadable photo. The UI is designed around that task; the rubric's "outstanding usability" claim still needs observed user evidence.

# Board2Notes student interface

## Start the local app

Double-click `Assignment/run_ocr.command`, or run this from `Assignment/`:

```bash
bash run_ocr.command
```

Open **http://127.0.0.1:8501** and keep the terminal open. Ctrl+C stops it. The launcher uses the existing `.venv-paddle` environment, starts from this app's directory so `.streamlit/config.toml` supplies the light theme/viewer toolbar, and binds only to this computer. It does not reinstall PaddleOCR. This is local deployment; hosted/cloud deployment remains separate work.

Streamlit 1.67.0, Pillow 12.3.0 and pillow-heif 1.8.1 were present during the 8 November check. `requirements-ui.txt` pins only the added UI dependency. If needed, install it using the existing Assignment virtual environment; do not replace that environment's Paddle dependencies.

## Student flow

**Add photo → Check photo → Review notes**

- Upload JPG/JPEG/PNG/HEIC/HEIF, up to 30 MB, or try the included sample.
- Camera EXIF orientation is applied once. Optional 90-degree rotation and percentage edge trimming preserve source pixels. The preview's exact pixels go to OCR; there is no silent enhancement, perspective correction or SR.
- Press **Extract notes** to run detection and recognition. A truthful spinner appears during the operation. Failures preserve the photo and any existing draft.
- Check the photo and editable text side by side. Numbered detection polygons are visible by default. Click a polygon (or use the line selector) to highlight it in orange and show its extracted handwriting crop and original OCR reading. Selection never replaces edited notes. Boxes can be hidden to inspect the unmarked photo. Missing text can be typed manually, including after an empty OCR result.
- **How your photo was read** shows the actual input, detected regions and selected crop-to-text result. The crop uses the same quadrilateral extraction function as OCRService, before recogniser resizing. The interactive board uses a smaller display-only JPEG; OCR still uses the original prepared pixels. No mask, extra correction or successful recognition is fabricated.
- Copy/download actions read the live textarea value at the click, avoiding a stale export while Streamlit is processing the editor's blur event. The text is encoded as data, never executed as HTML or instructions. Clipboard failure gives a selection/download fallback.
- Returning to crop/rotation preserves edits in separate session state. Re-extraction requires an explicit replacement checkbox when notes have changed. Starting another note offers save/cancel/confirm first.
- Notes/photos are session-only. The normalised OCR input is written to a temporary directory then removed. Download/copy notes before closing or refreshing. The app does not create a photo library.

## Fixed reader and integration

The default is `S1` (seed 1), selected using the pre-existing whiteboard validation metric. It is a practical prototype default, not proof of superiority over G0 or all future photos. Students see no experiment IDs, model picker, raw confidence or configuration controls. Set `B2N_OCR_MODEL` before launch only for developer configuration. The existing backend and cached evaluation outputs were left unchanged.

`ocr_backend.OCRService(...).recognise(image_path)` still returns text, lines, polygons, timing, detector settings and model identity. That structured contract remains available for later LLM/slide integration; it is not displayed as technical JSON in the student task. Technical comparisons stay in notebooks 10 and 11. No admin or live technical demonstration was added.

## Verification on 8 November 2026

- **21 automated checks passed:** 7 image-helper checks and 14 mocked Streamlit journeys. Crop settings and notes survive page navigation; re-extraction receives the same exact crop pixels. The added test checks region-event selection, original readings and preservation of edited notes without another OCR call. Run from Assignment:

  ```bash
  .venv-paddle/bin/python -m unittest discover -s whiteboard-ocr/ocr_prototype/tests -p 'test_student*.py'
  ```

- **Actual browser:** uploaded Jy Photo 23; S1 produced nine text regions; corrected `tich` to `rich`; clipboard contained the exact corrected multiline text, including a just-entered edit. Returning via adjustment/rotation retained notes. Cancel and confirm in the new-note dialog both worked. The sample route, region close-up and narrow-screen layout were exercised. No horizontal overflow was observed at the phone-sized viewport.
- The download button prepared a UTF-8 blob and displayed its preparation message. The in-app browser did not report a download event within the test timeout; a completed disk download remains to be confirmed in the presentation browser. The copy operation was independently verified by reading the browser clipboard.
- OCR timings here are sample observations, not benchmarks. Automated journey tests mock the reader and do not measure accuracy. No cloud inference or training was launched.

## Remaining usability evidence

Have a few students complete upload → correct → export without coaching. Record success, hesitation, errors and recovery. This implementation targets the UI rubric; "outstanding" usability should be supported by those observations. Keep further technical-demo requirements for the later discussion.

## Interactive visual check — 8 November follow-up

With the real sample photo, clicking region 4 changed the SVG selection, line selector, crop and original reading to `I want to be tich`. The edited notes remained intact; correcting the notes to `rich` did not rewrite the original prediction. Expanded walkthrough showed the actual photo, overlay and selected crop with that original prediction. No model/backend/evaluation output changed. The local component is packaged under `region_viewer/index.html`; it uses textContent for OCR text and supports Enter/Space selection as well as pointer clicks. Model names remain hidden from students.

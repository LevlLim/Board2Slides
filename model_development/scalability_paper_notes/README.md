# Scalability check: handwritten paper notes (18 Nov 2026)

Question: can Board2Notes take on a new domain (paper notes) without being rebuilt? No training is done here:
every reader already exists.

## What is here

| path | what |
|---|---|
| `writing_sheet.pdf` | What to write (3 pages) and how to photograph it |
| `reference_text/page1-3.txt` | The correct text, one line per written line |
| `photos/` | Put the photos here: `page1_flat.jpg`, `page1_angle.jpg`, `page2_flat.jpg`, ... (HEIC is fine) |
| `tools/run_paper_notes_eval.py` | The evaluation (see below) |
| `tools/app_prepare_wrapper.py` | Runs Board2Slides' own `app/prepare_infer.py` unchanged (thread workaround on Linux aarch64 only) |
| `tools/build_progression_zip.py` | Builds `Assignment/board2notes_reader_progression_20261118.zip` |
| `plumbing_check/` | Check that every reader and every step works, run before any real photo (see below) |
| `outputs/` | Created by the run: `summary.md`, `review.html`, `lines.csv`, `results.json`, overlays, crops |

## How the evaluation works

Per photo: EXIF orientation -> two conditions (`original`, and `app_prepared` = Board2Slides "Automatic
preparation" run unchanged) -> the 9 Nov OCR package detects lines **once** -> the **same crops** go to every
reader -> two scores against the reference text:

- **Reading:** mean line CER over the reference lines the detector found. Crops are matched to lines by the mean
  over all readers, so every reader is scored on the same assignment.
- **End to end:** the whole page; missed lines count as fully wrong, extra boxes as insertions. Reported in the
  package's reading order and order-free.

Also recorded: boxes drawn on each photo, split lines, possible merged boxes, seconds per photo, the
preparation's decisions (perspective, crop, rotation, lighting), the two long lines separately, and page 3 line
by line (downloaded reader = before, final model = after).

| reader | model |
|---|---|
| `downloaded` | PP-OCRv5 server reader as downloaded (Hugging Face, Apache-2.0), read at 48 px padded to 1024 px, as in every experiment |
| `whiteboard` | our whiteboard lines incl. long lines, 64x1024, width pooling (versions_2x2 V2) |
| `paper` | IAM paper lines, same training settings as `whiteboard` (versions_2x2 V1b) |
| `final` | stage-wise LR on the teammate's dataset (S1his) |
| `stagewise` | stage-wise LR on our lines (S1, run 1) |

Run (on the Mac in the app's Paddle environment, or anywhere with paddlepaddle 3.2+ and paddleocr 3.7):

```
python3 tools/run_paper_notes_eval.py run
python3 tools/run_paper_notes_eval.py report      # rebuild summary.md / review.html from the cache
```

`run` caches every step and can be re-run to continue (`--max-seconds N` stops it cleanly).

## Plumbing check (18 Nov, Linux aarch64 VM, Paddle 3.2.0, PaddleOCR 3.7.0)

1. **Readers reproduce their recorded predictions** (20 whiteboard-validation + 20 IAM crops each; the final
   model on 40 of the teammate's val/test crops): downloaded 40/40 identical, whiteboard 39/40, paper 39/40,
   stage-wise 37/40, final 40/40; mean CER difference 0.00 for all. This also confirms that the Hugging Face
   download is the same reader the experiments called "pretrained".
   - Found while checking: read with PaddleOCR's default resize (minimum width 320), the downloaded reader
     reads long lines much worse (14 sample crops: 16.7% vs 7.0% CER). The experiments padded to 1024 px.
2. **Whole pipeline on a synthetic ruled page** (page 1's text in a script font, flat and warped): every step
   ran; flat original: 11 of 11 lines detected. Accuracy on a printed font means nothing for handwriting.
   - **Found:** Board2Slides' Automatic preparation cropped the synthetic ruled page down to 3 of 11 lines (flat)
     and 7 of 11 (angled); perspective abstained on both. On real notebook photos this may need a change
     (for example using the original photo for paper notes). To be confirmed on the real photos.

## Limits of this check

- 3 pages by one writer: a demonstration, not a population estimate.
- This runs the 9 Nov OCR package (the app's detector, our readers), not the final deployed app. Repeat one page
  through the app once it runs the server reader.
- Page 3 is an exploratory retention check of Chinese the downloaded reader already had; dictionary coverage
  alone does not establish recognition ability.

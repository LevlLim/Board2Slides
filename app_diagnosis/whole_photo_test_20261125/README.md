# Whole-photo test of the app's OCR (25 Nov 2026)

Question: does the fine-tuned reader also do better when it reads whole board photos through the app, not only
pre-cut line crops?

Setup: 44 photos of 5 fresh boards (152 lines), the same boards as `../fresh_board_test/`. The app's own OCR path
(`app/ocr_infer.py` settings: PP-OCRv5 mobile detector, `text_det_limit_side_len=1600`), app commit `6226b12`.
Only the reader changes between rows. Photos are left out of the repository, as in `../fresh_board_test/`.

| Reader (folder in `model_development/readers/`) | Characters read right | Lines found | Photos with every line found | Page error |
|---|---|---|---|---|
| Downloaded (`00`) | 78% | 83% (126/152) | 25 | 56.0% |
| Fine-tuned, 320 px, 40 slots (`01`) | 80% | 87% (132/152) | 29 | 50.8% |
| Wide input, still 40 slots (`03`) | 53% | 59% (90/152) | 6 | 124.3% |
| **Final model, full dictionary (`08`)** | **87%** | **89% (135/152)** | **32** | **33.8%** |
| English 94-character reader (app V2, experiment 8) | 90% | 90% (137/152) | 34 | 28.4% |

- Characters read right = 100% minus the mean character error rate of the lines that were found.
- Page error = all edits needed to turn the photo's OCR output into the reference text (missed lines and extra
  boxes count), divided by the reference length. It can pass 100%.
- The app's automatic photo preparation did not help: final model 33.8% → 37.7% page error, English reader
  28.4% → 34.7% (`cond` = `auto` in `scores.json`).
- Only 5 boards, so treat small differences as hints.

Names in `scores.json` (`model` field), mapped to the readers: `downloaded` = 00, `ft40` = 01, `p40` = 03,
`tuned` = 08 (the app's `models/ocr/recognizer` at commit 6226b12), `baseline` = English 94-character reader
(the app's `models/ocr/baseline_rec` at commit 6226b12, now `models/ocr/recognizer`, SHA-256 615ca77e…).

Files: `run.py` (upload, prepare, OCR; paths are those of the machine it ran on), `refs.py` (reference text per
board), `score.py` (matching and scoring), `prepare_f35.py` (preparation with a fixed focal length, a check of the
perspective step), `scores.json` (every photo, reader and condition, with each line's prediction).

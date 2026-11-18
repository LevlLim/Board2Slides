# Fresh-writer line test (photos of 11 Nov 2026)

Test set only. Never add these crops or photos to training, augmentation or development lists.

## What is in it

The 44 phone photos in `Assignment/1111_unfamiliar/` show 5 boards. Each board was photographed several times from
different angles. The photos themselves are not copied or changed. Their SHA-256 hashes are recorded in `manifest.csv`.

| Board | Photos | Lines (writings) | Text |
|---|---|---|---|
| A | IMG_5095–5108 (14) | 6 | bookkeeper / parallel / access / food / weights / ghosts |
| B | IMG_5109–5112 (4) | 3 | book keeper / letter / gradient |
| C | IMG_5113–5115 (3) | 2 | earth / generalization |
| D | IMG_5116–5124 (9) | 3 | three sentences ("The model should preserve …") |
| E | IMG_5125–5138 (14) | 2 | two sentences ("Different writers use …", "Please leave enough space …") |

- **152 line crops** (`images/`) of **16 distinct writings**. Photos of the same writing are not independent, so
  intervals resample whole writings (16 groups), not crops.
- **5 lines excluded** (listed in `manifest.csv` with the reason):
  - IMG_5097, weights and ghosts: the tilted lines overlap.
  - IMG_5119–5121, the third sentence: it runs off the photo edge.

## How the crops were made

1. **Detector.** Line positions come from the app's own detector: PP-OCRv5_mobile_det from the package, with
   unclip 2.0 at 960 / 1600 / 2400 px.
2. **Tightening.** Each line was tightened to its ink, with a margin of 0.25 × line height on the sides and 0.12 ×
   line height above and below. Then it was cut out with the app's `quad_crop`.
3. **Checked by eye.** Every crop was checked against its label (`review_*.jpg`). Where the automatic step failed,
   a documented fix was applied:
   - 5 ghosts lines were located from the line spacing;
   - three "A careful …" crops were extended left to include the faint "A";
   - the sentences ending in a full stop were extended right to include it.

   The scripts are in `tools/`.
4. **Labels.** Labels follow the fresh-photo protocol: what is written, with the writer's own capitals and
   punctuation. Scoring is case-insensitive with spaces normalised, the same as the project scorer.
   - In sentence D1, the word the writer wrote unclearly ("wovds") is read as "words".
   - E2 has no full stop.

## Not yet recorded (needed before calling this "unfamiliar handwriting")

- **Who wrote each board.** The writer column is not filled in yet. If any writer is ZB, JH, DS, Jy or a teammate,
  their handwriting may already be in training, and the result does not count as "new writer".
- **This is not the frozen fresh-photo confirmation** (`board2notes_ocr_package/fresh_photo_confirmation/`):
  - that protocol needs one photo per board and full-page scoring;
  - this set was used on 12 Nov to compare readers, so it is now development evidence for choosing a reader.

## Results, 12 Nov (`scores_server_models.csv`, `analysis_output.txt`)

Server readers, read on CPU with the app's recognition runtime. Mean line CER (%), lower is better:

| Model | All 152 | Board A | B | C | D | E |
|---|---|---|---|---|---|---|
| S0 (2-seed mean) | 12.4 | 16.6 | 8.6 | 3.5 | 8.9 | 6.5 |
| S1 (2-seed mean) | 11.5 | 14.6 | 13.3 | 6.3 | 8.0 | 5.8 |
| G0 guidance off (2-seed mean) | 9.1 | 10.1 | 11.3 | 5.2 | 9.4 | 6.1 |
| V2 (S0's data + 64 teammate crops) | 5.6 | 4.3 | 10.5 | 1.2 | 8.7 | 5.8 |
| S1his (teammate's 80:10:10 split) | 4.6 | 4.1 | 1.4 | 0.0 | 7.5 | 5.7 |

Differences, with 95% intervals resampling whole writings:

| Comparison | Difference (points) | 95% interval | Notes |
|---|---|---|---|
| G0 − S0 | −3.3 | [−6.0, −0.6] | Almost all from board A (−6.5). Seed 1 −5.4, seed 2 −1.1 (not clear). |
| S1 − S0 | −0.9 | [−3.9, +2.9] | No clear difference. |
| V2 − S0 | −6.8 | [−10.7, −2.6] | |
| S1his − S1 | −6.9 | [−14.4, −2.2] | |

Seed noise is as large as the guidance effect. For example, S0 seed 2 − seed 1 = −3.7 points.

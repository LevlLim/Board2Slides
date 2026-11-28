# JH full-photo app evaluation — completed 9 November 2026

The unchanged app pipeline already avoids most of the poor dataset-crop geometry on these seven JH photographs. S1 receives much wider, nearly horizontal writing when the mobile detector supplies the crop. No automatic tightening, additional deskew, retraining or application change was introduced.

## Matched-line recognition result

All 59 supplied dataset lines have a one-to-one geometric association after grouping, so no supplied line was excluded from this comparison. The matching uses frozen image-only bands and a heuristic overlap graph, not recognised text. This is not certified detector recall. Crop labels and full-photo annotation scope remain subject to the earlier review limitations.

All four inputs were read by the same S1 object in the same Mac process, using the app's resize and dictionary.

| Input | Mean line CER | Corpus CER | Deleted characters | Lines with CER >=15% |
|---|---:|---:|---:|---:|
| original | 21.14% | 23.44% | 232 | 29 |
| tight | 16.43% | 18.68% | 187 | 22 |
| straightened | 5.37% | 6.37% | 68 | 4 |
| app | 0.87% | 0.85% | 5 | 0 |

The app reads 46/59 matched lines perfectly under case-insensitive, collapsed-whitespace scoring. There are 20 character edits in 2,355 reference characters. These denominators exclude output with no matched dataset line; this is **not full-page CER** or “99% accurate on arbitrary whiteboards”.

## Detection and grouping

- Raw detector: 58 one-to-one associations, one split line, two unmatched regions.
- After the existing grouping code: 59 one-to-one associations and two unmatched regions; no merge/split/missed-line components under the frozen overlap rule.
- On Photo 2, grouping joins the detector's two pieces of “Sunlight warms the quiet forest” into one line. The result is read correctly. This shows useful existing grouping; no grouping code was changed.
- Photo 2's unmatched box covers the black board eraser and produces an empty reading. Photo 5's unmatched box covers the projected Windows taskbar (magnifier icon and “Search”), not handwriting. S1's saved reading remains “Q 5earch”. It is text-like projected interface content outside the handwritten-note target inventory, and would be unwanted output when the intended scope is handwritten notes. The matched-line score excludes both unmatched regions and is not a full-page score.

**Context correction, 9 November:** the earlier report misidentified the Photo 5 projection as faint handwriting. The wider image `q_search_check/q_search_context.jpg` shows the Windows logo, magnifier and Search box. This correction changes only the visual interpretation; it does not change source images, OCR predictions, matching or scores.

## What changed in the input

Median resized crop widths at height 64: dataset original 342 px; tightened 386 px; straightened 500 px; app crop 814 px. Median absolute residual slope of app crops is 0.4 degrees. Existing quad cropping rectifies the detector's tilted text boxes.

The full photos also contain a higher-resolution source: dataset crops map into them at approximately 1.59x scale and involve different image sampling/compression. Thus this comparison supports the **actual full-photo crop path**, but does not isolate geometry, horizontal scale, neighbouring ink, source resolution or compression as the sole cause.

The four JH crops still difficult after the manual straightening experiment (JH1.4, JH5.3, JH6.4, JH7.3) are all below 15% CER through the app. Inspect the gallery for exact readings and remaining errors; no labels were corrected.

## Uncertainty and provenance

Seven photographs from one already-inspected writer are development evidence. The intervals resample photos, holding this single S1 checkpoint fixed; they do not capture future-writer or training-seed uncertainty. There is no full-photo reading-order evaluation here, consistent with the user's decision to defer that work.

The earlier handoff's “at least 160 matches” claim is inaccurate. Saved placement records have a minimum of 110 matches, 97 RANSAC inliers and 0.8017 pixel correlation. These are sufficient to investigate, but the assistant-reviewed placements are not independent gold annotations. All 59 four-way image comparisons and Photo 2's split/grouping example were visually inspected for this analysis.

Codex completed Claude's existing analysis script with only portable path/font handling changes so it runs on the Mac. Matching thresholds, scoring and the saved detector/reader outputs were not changed. CHECKS_CODEX.json records the current analysis hash and independent checks. All 61 saved app reader-view images were reproduced exactly from saved crop pixels; all 236 line/arm CER records were recomputed. The seven-photo sources, code hashes and frozen placement hashes match the saved run identities. The repeated tight/straightened dataset readings match the earlier S1 readings.

## Decision

Retain the existing mobile-detector/S1 crop path for now. There is no demonstrated need to add another automatic margin or straightening stage on these photos. Do not replace historical benchmark scores: report the original-crop, transformed-crop and actual-app results with their distinct scopes.

The next model-side decision should use this corrected diagnosis, not the earlier claim that S1 cannot read JH handwriting. A practical next milestone is freeze this OCR configuration and check it on fresh photos/writers, while completing deployment. Reading-order changes remain deferred. This result alone does not settle faint short words, other writers, distant capture or complex layouts.

## View and reproduce

- `review.html`: all 59 four-way crop comparisons, full saved readings, exact app input previews and unmatched regions.
- `visuals/`: seven raw/grouped overlays, seven crop/input strips and five four-way contact sheets.
- `RESULTS.json`: original analysis output, per-line records and photo-cluster intervals.
- `CHECKS_CODEX.json`: independent scoring, identity and pixel checks.
- `../whiteboard-ocr/notebooks/16_jh_full_photo_app_evaluation.ipynb`: executed notebook, no model loading.

Run `Assignment/.venv-paddle/bin/python Assignment/claude_jh_fullphoto_20261109/analyse_jh.py` to regenerate the saved-output analysis. Run `build_notebook16.py` afterward to verify and rebuild the notebook. Neither starts inference or training.

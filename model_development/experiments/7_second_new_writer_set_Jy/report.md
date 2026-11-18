# Jy original-crop recognition evaluation

Generated 2026-11-07T20:48:03.841395+00:00 (UTC). **Reader-only evaluation.** No training.

Selected 298 of 298 original crops; synthetic variants are excluded. Labels were reported reviewed by a teammate.

The supplied train/test folder split is ignored: all originals remain evaluation data for these already-trained readers.

| Model | Crops | Mean line CER | Corpus CER | Exact lines (case-sensitive) | Empty readings |
|---|---:|---:|---:|---:|---:|
| S0 | 298 | 7.35% | 6.36% | 52.01% | 0 |
| S0s2 | 298 | 7.33% | 6.44% | 52.35% | 0 |
| S1 | 298 | 5.57% | 5.08% | 56.71% | 0 |
| S1s2 | 298 | 5.48% | 5.08% | 59.73% | 0 |
| G0 | 298 | 5.91% | 5.46% | 56.71% | 0 |
| G0s2 | 298 | 5.38% | 4.67% | 62.08% | 0 |

## Matched two-seed comparisons

Difference is after minus S0: negative means lower error. Intervals resample source photos with the trained seeds held fixed.

| Comparison | Role | Seed 1 change (pp) | Seed 2 change (pp) | Mean change (pp) | 95% interval (pp) | Decision |
|---|---|---:|---:|---:|---|---|
| G0 vs S0 | primary confirmation | -1.44 | -1.95 | -1.70 | -3.02 .. -0.33 | helps |
| S1 vs S0 | secondary | -1.78 | -1.85 | -1.82 | -3.01 .. -0.98 | helps |

## Interpretation limits

- This fresh-collection confirmation does not change the earlier whiteboard-validation primary verdict.
- The 298 crops include 241 individual words and 57 multi-word crops; report length/word groups separately.
- Writer identity/provenance is user supplied. This collection does not establish population-wide handwriting performance.
- Crop-level scores do not include missed detections, merged lines, extra page text or reading-order errors.
- Two-seed averages are statistics, not a deployable ensemble. Do not choose a seed by this test.
- If these mistakes inform a new training or preprocessing change, Jy becomes development evidence for that change.
- Proposed full-photo box positions and completeness require review before claiming full-photo CER.
- Seconds are backend-reported inference time, excluding startup and image decoding; compare only on the same runtime.
- Source files, labels and models were not edited. See protocol.json and per-model metadata for fingerprints.

## Execution note added after the run

Other OCR work could run on the same computer during collection. The saved inference seconds are indicative only and **must not be used as controlled speed benchmarks**. They exclude model startup and image decoding. This qualification changes no predictions, metric calculation or pre-evaluation decision rule.

An independent audit after completion checked all 1,788 prediction rows (298 per model), recomputed character edits/CER with RapidFuzz, verified every mean, confirmed the 241 single-word/57 multi-word counts and the shared backend fingerprint. This is recognition on supplied original crops, not certified complete-photo transcription accuracy.

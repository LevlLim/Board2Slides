# Crop-margin diagnostic — 9 November 2026 (Claude)

**Question.** Do large blank margins around the writing (which make the writing small after the reader resizes every crop to 64 px high) contribute to the JH/DS unfamiliar-writer errors, especially deleted characters?

**Verdict under the rule fixed before inference: supported.** All three pre-declared criteria hold. The effect is large: on crops whose box removes at least 40% of the height, mean CER falls from 35.4% to 6.3%.

## Design (frozen in `PROTOCOL.json` before any new inference)
- **Crops:** all 160 JH/DS new-writer crops. Jy was excluded: 294/298 Jy crops are already tight, and the ink detector missed light first strokes there (e.g. the C of "Ching", the S of "Steroids"), so tightening would have cut letters.
- **Boxes, from images only:** sensitive ink mask (difference from a median-blurred background), all ink components, padded by max(6 px, 0.35 × ink height) vertically and max(10 px, 0.8 × ink height) horizontally. For 4 crops with dark corner shading (JH19.1, JH22.1, JH26.1, JH5.1) border-touching components were ignored. Every changed box was checked visually on contact sheets to keep all target strokes. Coordinates and source SHA-256 per crop: `boxes_final.json`.
- **Groups by dose** (box height ÷ crop height): large margin ≤ 0.6; control ≥ 0.85; middle in between. 4 crops flagged by earlier audits (label punctuation, other writing near the edge) reported separately.
- **Reader:** frozen exported S1 seed-1 model, Mac CPU static reader; both arms read by the same code. The original arm reads the untouched files; the tight arm reads lossless PNG crops of the same decoded pixels.
- **Rule:** supported only if (1) large-margin mean CER change has a photo-cluster bootstrap 95% interval below 0, (2) large-margin deletions decrease, and (3) the large-margin change minus the control change has a 95% interval below 0.

## Results (mean line CER; change in percentage points; S/D/I = substitutions/deletions/insertions)

| group | crops | original | tight | change | 95% interval | deletions | substitutions | insertions | better / worse |
|---|---|---|---|---|---|---|---|---|---|
| large margin | 51 | 35.4% | 6.3% | -29.1 | -36.2 … -19.7 | 323 → 24 | 327 → 51 | 41 → 6 | 40 / 2 |
| middle | 57 | 12.5% | 5.7% | -6.8 | -11.1 … -3.1 | 130 → 34 | 135 → 69 | 17 → 14 | 30 / 5 |
| control | 48 | 24.6% | 19.5% | -5.1 | -7.3 … -2.8 | 223 → 187 | 249 → 191 | 28 → 23 | 29 / 3 |
| flagged | 4 | 43.4% | 37.5% | -5.9 | -17.6 … +0.0 | 21 → 15 | 47 → 48 | 2 → 3 | 1 / 0 |
| all 156 (excl. flagged) | 156 | 23.7% | 10.1% | -13.6 | -18.9 … -8.3 | 676 → 245 | 711 → 311 | 86 → 43 | 99 / 10 |

Difference in differences (large margin − control): **-24.0** points, 95% interval -31.7 … -13.2.

Criteria: CI below 0 = True, deletions down = True, difference-in-differences CI below 0 = True → **supported**.

**Scorer check:** the original arm (Mac CPU) reproduces the saved RunPod GPU S1 readings on 114/160 crops exactly; mean CER is the same to within 0.1 point (GPU 24.1%).

**Visual check** (`regressions_and_top_improvements.jpg`): all 10 regressions are 1–2 character changes on crops that were already read well (e.g. 96HR → 9GHR, Yuzu Juice → VuzuJuice); no regression comes from a cut stroke. The biggest improvements are whole sentences going from unreadable to nearly perfect (e.g. "different regions of an image …" 57% → 0%; "Satsuma" 57% → 0%).

## What this means
1. **A large part of the JH/DS "unfamiliar-writer gap" is a crop-scale effect of how those test crops were cut, not handwriting style alone.** With tight crops, S1 reads JH/DS at about 10% CER instead of about 24%. Familiar-writer sentences are about 2.4% and Jy about 5.5% (Jy crops were already tight), so a real gap remains, but it is much smaller than reported.
2. **Deletions were mostly a scale problem:** large-margin deletions fell from 323 to 24.
3. **The control group also improved (−5.1 points)**, so removing horizontal margins, distracting marks and faint ghost writing helps too, not only vertical scale.
4. **Earlier new-writer comparisons need re-reading.** G0's exploratory new-writer gain over S0 may partly be robustness to small text rather than better handwriting generalisation. Not tested here.

## Limits
- Boxes are an oracle: a rule plus visual checks and 4 manual exceptions, not an automatic app feature. The same rule failed on light strokes in Jy, so an automatic version needs its own test on full photos.
- JH/DS already informed development; one reader (S1 seed 1); CPU scorer.
- Before defining boxes, Claude had seen readings of a few of these crops in earlier notebooks (e.g. Lenovo, Moisture, Tough, Yuzu Juice); the box rule does not use readings.

## Suggested next steps (not done)
1. **Cheap, no training:** read the same tight crops with S0 and G0 (seed 1) to see whether G0's new-writer advantage survives once scale is fixed.
2. **App check:** measure the margins of the detector's crops in the full-photo path (e.g. the 301 Jy detected regions). If they are tight, the app is mostly unaffected; if not, an ink-tightening step before recognition is a justified preprocessing candidate, to test on full photos.
3. **Training option, if wanted:** scale/margin augmentation (random padding around the writing) with the same data, budget and two matched seeds against S0 or S1.
4. **Reporting:** quote new-writer CER on both original and tight crops.

## Files
`PROTOCOL.json` (frozen design), `boxes_final.json`, `labels_original.txt`, `labels_tight.txt`, `crops_tight/`, `pred_original.json`, `pred_tight.json`, `RESULTS.json` (per-crop readings and edit counts), `analyse.py`, `score_static_mac.py`, `regressions_and_top_improvements.jpg`. `_to_delete/` holds superseded first-draft boxes and sheets (safe to delete).


## Addendum (9 Nov, Claude): S0, S1 and G0, both seeds, on the same frozen crops
Same 160 crops, rectangles, labels and Mac scorer (`score_static_mac.py`, CPU, paddle 3.2.0, 64 px high, ≥ 1024 wide). New predictions: `pred_original_models.json`, `pred_tight_models.json` (S0, S0s2, S1s2, G0, G0s2; S1 seed 1 is from the first run). Analysis: `analyse_models.py` → `RESULTS_MODELS.json`, with model and scorer identities. Notebook 14, section 7.

- **Tightening helps every reader** (mean of seeds, CER %): all 160: S0avg 26.3→11.8 | S1avg 25.0→11.0 | G0avg 23.0→10.1; 156 without flagged: S0avg 25.9→11.3 | S1avg 24.5→10.3 | G0avg 22.5→9.6. Every model's interval is below 0, and deletions fall by about two-thirds for each.
- **G0 vs S0 on tightened crops:** mean of seeds -1.65 (-3.08 … -0.52) on all 160, -1.71 (-3.18 … -0.58) on 156. On original crops it was -3.26 (-4.53 … -2.15). Seed 1: -0.91 (-2.59 … +0.60); seed 2: -2.38 (-4.00 … -1.13). **About half of G0's advantage disappears once text size is fixed; a smaller advantage remains** (clear in seed 2, not in seed 1).
- **S1 vs S0 on tightened crops:** -0.82 (-2.40 … +0.32), not clear.
- **Remaining errors:** 26 crops still have ≥ 15% CER for all six readers after tightening. 15 of them are "control" crops that were already tight or untouched, so these are genuine reading failures on unfamiliar handwriting, not a size effect.
- **Regressions per model** (tight worse than original): S0 13, S0s2 9, S1 10, S1s2 11, G0 12, G0s2 14; mostly 1–2 character changes (shown with images in the notebook).
- **Scorer note:** Mac readings of the original crops match the saved RunPod GPU readings on 99–121 of 160 crops per model; mean CER is within about 1 point. All comparisons here use only the Mac scorer.

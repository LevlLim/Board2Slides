# Perspective decision rule, revision 2

30 October 2026. Candidate implementation in notebook 08 and
`preprocessing/board_geometry.py`; no automatic promotion of the dataset chain.
Lighting, rotation, recognition weights, detector thresholds and optional
enlargement are unchanged by this revision.

## What changed

1. Add independent line-segment proposals and merge geometrically consistent
   fragments, retaining separate supported runs. The old merger kept only one
   run from a group and could lose useful frame segments.
2. Search multiple top/bottom/vertical combinations. Prefer observed outer
   frame boundaries. A long internal physical panel joint is also possible,
   but needs an outline-supported lower frame and vertical reference.
3. Permit a short early-ended frame segment to reach a junction only if the
   intervening image pixels show continuous edge evidence. A junction genuinely
   outside a cropped photo is separately constrained and is extrapolated,
   never described as an observed corner.
4. Locate the target panel geometrically. It must contain >=90% of the detected
   box area and at least three boxes. Smaller outside boxes are recorded with
   coordinates for review. They are not proven to be room text; the pipeline
   assumes one target board and must be audited on multiple-board scenes.
5. Replace the blanket median height ratio with per-line preservation. A local
   connected-ink height is a provisional readability proxy. Ink already below
   16 px must retain >=95% of that height; larger ink may shrink towards that
   floor. If glyph estimation fails, preserve the original box height instead.
6. Fit the output to the actual observed panel polygon, not its axis-aligned
   bounding rectangle: rectangle corners outside a slanted board can project
   towards the homography horizon and waste the canvas on white space.
7. Increase homography canvas scale if needed to retain small writing, within
   the existing 4000 px limit. Resample original pixels using the final matrix;
   do not enlarge a previously warped intermediate. Reject impossible size
   preservation or clipped detected writing.

The proposal still needs known focal length, >=4-degree edge convergence,
>=15% height taper over substantial observed overlap, plausible board angle
and support from the board-region estimate. The normal pipeline additionally
requires board angle >= `persp_min_angle` (default 9 degrees). Missing evidence
means abstain, not a declaration that the photo is frontal.

## Configuration in notebook 08

| Field | Default | Meaning |
|---|---:|---|
| `persp_min_angle` | 9 | Minimum estimated board turn, degrees |
| `persp_min_target_area` | 0.90 | Minimum fraction of detected box area in proposed panel |
| `persp_min_vertical` | 0.12 | Minimum observed reference segment / photo long side |
| `persp_ink_floor` | 16 | Provisional ink-height floor in source-resolution pixels |
| `persp_keep_small` | 0.95 | Required retention for small/uncertain text |
| `max_side` | 4000 | Maximum rendered homography canvas side |

`persp_min_text_scale` remains for historical/cache compatibility and stubbed
legacy geometry results. It does not override the revised preservation gate.
Ink height is not a measurement of true character information or OCR success.
These are first-edition heuristics, not certified side-angle detection.

## Evaluation and viewing outputs

`outputs/perspective_rule_v2/review.html` compares baseline, the previous strict
rule and this revision on the same 44 selected photos and fixed label sets.
Each revised result includes the chosen panel, physical edge segments, accepted
and excluded boxes, per-line size estimates and its decision reason.
`summary.json` explicitly says whether all 132 cases completed.

Eight photos reuse the complete-line visibility annotations prepared visually
before the earlier manual diagnostic. Other photos use visibility frozen from
the previous evaluation's OCR readings; this retains selection bias but does
not let the new rule choose its own scoring lines. CER is uncapped, line-matched
and uses identical labels across arms. Stray predictions are reported separately,
not penalised in that CER. Do not compare these absolute numbers directly with
the old capped/OCR-filtered round-4 summary or claim full-page accuracy.

Historical pipeline/geometry/notebook sources are preserved in
`outputs/perspective_rule_v2/source_before/`. Interrupted/superseded preflights
are retained separately and must not be used as completed comparisons.

The standalone comparison is resumable in the same environment:

```bash
../Lab/.venv312/bin/python whiteboard-ocr/scripts/evaluate_perspective_revision.py
```

For the notebook's five-arm comparison, restart the kernel and run sections
0–6 of notebook 08 with `Config(run_order_check=True, order_round=4)`.
That comparison keeps lighting, rotation and enlargement independent and
continues to use its explicitly stated historical visibility metric.
Changed settings/code are fingerprinted; historical records are not silently
reused as current results. The experiment does not rebuild training data.

## Before promotion

Inspect every newly applied warp, especially excluded boxes and faint writing.
Check genuine side views missed, frontal views altered, line detection and OCR
errors. Repeated photographs of familiar writing are not independent boards.
Freeze the candidate settings, then validate on new boards/photos before making
it the default combined preprocessing chain. Good results on the development
photos alone do not establish general reliability.

## Completed evaluation: keep experimental

The 44-photo comparison completed all 132 cases successfully. On the 35 non-B3
photos, uncapped line-matched CER was 32.96% for baseline, 32.94% for the previous
strict gate and 31.96% for this candidate. The baseline-to-candidate difference
is 0.994 percentage points, just below the earlier strict >1-point criterion;
do not round it into a pass. B3 was unchanged at 20.85%.

The candidate applied to four photos: oct17_142 improved 80.20% -> 72.48%,
cnn_11 was unchanged at 15.45%, quiz_17 worsened 16.67% -> 18.42%, and quiz_24
remained 0%. The previous rule's helpful correction on quiz_03 was skipped by
the candidate (1.75% with the previous rule versus 4.39% with baseline/candidate).
The four manually designated near-frontal controls were unchanged.

A second check used eight preselected photos from the newer long-sentence
collection, without retuning the rules: four side-angle photos and four
frontal controls. All eight were skipped, so OCR outputs were unchanged. This
protects the frontal examples but demonstrates inadequate side-view recall.
Only first-pass accepted complete-line labels were scored; no claim of
full-photo accuracy or new-writer generalisation follows from these figures.

Results: `outputs/perspective_rule_new_collection/review.html`, its protocol,
source/package setup, 16 result records and summary. Original HEIC metadata
supplies the camera focal length; conversion to PNG did not replace that with
an assumed default. Source photographs remain unchanged.

The source implementation and notebook 08 are updated, but the default dataset
chain remains the baseline. This revision improves one previously missed wide
side view, not reliable automatic handling of every side-angle capture. Further
work must address geometry evidence in cropped frames or improve proposal
recall; loosening thresholds to force these eight validation photos through
would invalidate this frozen check. No such retuning was performed.

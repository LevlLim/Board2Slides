# Rotation rule implementation and review — 29 October 2026

Updated notebook 08 and `preprocessing/pipeline.py` to use
`rotation_consensus.py`. Jy's original `fix_rotation.py` remains unchanged.
This is a conservative small-roll heuristic, not a general camera-orientation
classifier and not a demonstrated dataset-wide OCR improvement yet.

## Rule

- At least three independent, separated text rows; duplicate/same-row boxes count once.
- At least 70% of eligible independent rows agree within 2 degrees.
- Supporting box directions agree with glyph-component directions within 2.5 degrees.
- At least three agreeing rows overlap horizontally within a column. Separate captions are insufficient.
- Only 2–15-degree residual roll is corrected. No global room-edge fallback.
- Rotate once onto an expanded canvas; verify every detected text corner remains inside.
- Log the decision, angle, row counts, candidate reasons and transformation.

`confidence` means row-agreement fraction, not a calibrated probability.
All these limits are provisional. Consistently sloped handwriting can still
look like camera roll. Joined cursive, faint text, short lines or missing
detections can cause abstention. 90/180-degree orientation is outside scope.

## Self-review fixes

1. Retained large/conflicting horizontal rows in the agreement denominator;
   excluding them could manufacture a majority from a small subset.
2. Required stacked rows in a common column after visual review of `quiz_10.png`;
   independent captions across the board had initially passed.
3. Mapped detection boxes through any perspective homography and ROI offset
   before measuring rotation. The legacy fixed-chain path redetects in its
   current coordinates.
4. Connected the new Config fields to the actual embedded module and settings
   fingerprints. `min_roll_conf` is explicitly a legacy cache field.
5. Invalid geometry causes abstention; no malformed corner list reaches a warp.

## Checks completed

- 16 rotation regressions: both signs, upright/italic text, single/duplicate
  rows, blank regions, conflicting evidence, separate captions, invalid boxes,
  expanded-canvas retention, notebook source parity, Config and coordinate mapping.
- 8 perspective-gate regressions.
- 13 order-check/control-flow regressions.
- All 28 notebook self-tests, including four active rotation-gate checks.
- Notebook code syntax and source synchronisation checked.

Final gate audit used the real cached PP-OCRv5 mobile detector, unclip 1.2,
and all 44 photos from the existing round-4 comparison. No labels/filenames
were used in the decision algorithm:

| Decision | Photos |
|---|---:|
| Independent rows disagree | 23 |
| Insufficient independent reliable rows | 17 |
| Insufficient stacked rows in a column | 2 |
| Below rotation trigger | 1 |
| Rotate | 1 |

The retained candidate is `oct17_137.png`: -3.4 degrees, 11/15 rows agree,
including four stacked rows. All six photos rotated by the old rule now
abstain: `oct17_002`, `oct17_107`, `oct17_122`, `oct17_012`, `oct17_142`,
and `lecture_03`. This establishes changed decisions, not that every abstention
is correct.

A real OCR spot check with PP-OCRv6 medium recognition compared S0 vs S3:

| Photo | S0 CER | S3 CER | Covered label lines | Decision |
|---|---:|---:|---|---|
| oct17_137.png | 0.739 | 0.588 | 39/75 → 44/75 | rotate |
| quiz_10.png | 0.044 | 0.044 | 6/6 → 6/6 | unchanged |

These spot checks used **all supplied label lines** and the existing capped
hurdle metric; they are not directly comparable to round 4's OCR-selected
visible-line aggregate. Missing/unreadable label lines contribute errors.
One improved photo is insufficient evidence of general benefit.

An earlier controlled +7-degree roll of two real handwriting regions, with
their boxes transformed exactly, produced one accepted correction and one
abstention. Conservatism can miss real tilt. The final synthetic roll tests
pass; broad natural-photo sensitivity remains unmeasured.

## Run next

Restart notebook 08's kernel. Use
`CFG = Config(run_order_check=True, order_round=4)` and run sections 0–6.
Section 7 optionally displays previews and compact decision logs.
Compare S3 with S0 on the same photos/visible-line policy, including coverage
and per-condition errors. Inspect skips as well as applied corrections.

No dataset was rebuilt, no historical experiment records were overwritten,
and no combined preprocessing chain was promoted by this work.

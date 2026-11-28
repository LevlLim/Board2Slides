# Preprocessing audit — 30 October 2026

Completed the local test–inspect–fix loop for rotation, lighting and ordinary
bicubic enlargement. A crop-grouping failure discovered during the audit was
also investigated and fixed as an **optional candidate** in notebook 08.
No model was trained, no source photograph or training label was changed,
and no combined chain was promoted. Super-resolution and blemish removal
were excluded.

Perspective retains its earlier experimental status: the revised gate
applied on 4/44 older photos and skipped all four side views in the eight
new-collection captures. The new line-grouping option does not change that
gate or supply missing board boundaries. Its prior controlled diagnostic
is in `docs/PERSPECTIVE_REVISION.md`. No further perspective thresholds
were loosened in this audit, and perspective was not added to a combined
default chain.

## Comparison and limits

The 52 development photos comprise 35 older non-B3 photos, 9 B3 photos and
8 long-sentence captures. The detector is PP-OCRv5_mobile_det, the recognizer
is pretrained PP-OCRv6_medium_rec, and unclip is 1.2. Each comparison holds
the images, selected labels, models and common ROI policy fixed.

CER below is uncapped, case-insensitive, character-weighted over selected
labels. Missing label matches count as deletions; unmatched predictions are
reported separately and are **not** added as insertions. It is therefore
not full-page OCR accuracy. Eight older images use visual complete-line
labels; the remaining older images use visibility selected in a prior OCR
experiment, frozen before this audit. The long-sentence labels are assistant
first-pass accepted annotations, with user review still possible.

These are development results. Repeated photographs and repeated texts are
not independent boards/writers. B3 has been inspected repeatedly and is a
regression check here, not an untouched final test. Further comparisons must
also use the actual deployed/fine-tuned reader before release.

## Independent operations

| Operation | Older non-B3, 35 photos | B3, 9 photos | Long sentences, 8 photos |
|---|---:|---:|---:|
| Baseline | 32.96% | 20.85% | 42.95% |
| Gated rotation | 31.31% | 20.85% | 42.95% |
| Existing gated lighting | 31.94% | 21.02% | 41.95% |
| Gated bicubic enlargement | 32.86% | 20.03% | 42.81% |

All **208/208 cases** completed. These numbers use a different label policy
and uncapped metric from the earlier notebook round-4 table; their absolute
levels should not be compared to that table.

- **Rotation:** applied on only 1/52 natural photos, `oct17_137.png`, where
  selected-line CER fell from 48.58% to 23.77%. Other photos were unchanged.
  This supports that case, but is insufficient evidence of natural-photo
  recall. A separate controlled handwriting montage check accepted 9/10
  inserted ±6° rolls, all within 1.5°, and triggered on 0/5 upright controls.
  Exact supplied boxes and artificial layout isolate the gate/warp; they do
  not test the detector or prove natural-photo OCR benefit. Thresholds were
  not loosened to force corrections.
- **Lighting:** applied on 27/52 photos. Aggregate gains hide regressions:
  non-B3 had 6 better and 5 worse photos; long sentences had 3 better and
  4 worse. On `animal_02.png`, excessive contrast/colour changes changed a
  correct `Animal` to `Anima`, despite high recognition confidence. The
  existing light operator includes more than illumination correction.
- **Enlargement:** applied on 35/52 photos. It is a modest, mixed operation,
  not a reconstruction method. It uses bicubic only, with no sharpening/SR.

## Bounded lighting and combination

`preprocessing/lighting_conservative.py` tests a partial smooth illumination
gain bounded to 0.85–1.35, using the same gain in every colour channel and
limiting highlight clipping. It omits denoising, white balance, contrast
stretch, gamma and CLAHE. The triggering rule was held unchanged so the
comparison isolates the operator.

| Operation | Older non-B3 | B3 | Long sentences |
|---|---:|---:|---:|
| Baseline | 32.96% | 20.85% | 42.95% |
| Bounded lighting | 32.86% | 21.02% | 45.13% |
| Rotation then bounded lighting | 32.11% | 21.02% | 45.13% |

All **156/156 cases** completed. The candidate avoids extreme contrast
changes but does not consistently lower CER. Combining it with rotation
also lost some of rotation's gain. **Neither version was enabled by default.**
The implementation remains available for further controlled investigation.

## Enlargement on fixed labelled lines

To eliminate detector/label-matching effects, the reader also processed all
255 accepted line crops with their exact labels, comparing native pixels
to the existing bicubic rule: enlarge crops under 48px toward 64px, capped
at 4×. No review/rejected crops were included.

| Set | Crops | Native CER | Bicubic CER | Better / worse |
|---|---:|---:|---:|---:|
| All accepted crops | 255 | 60.77% | 60.41% | 23 / 5 |
| Crops under 48px | 57 | 37.15% | 35.48% | 23 / 5 |
| Far frontal | 68 | 47.35% | 45.98% | 23 / 5 |

These represent **36 distinct written lines**, all in one conservative
leakage group, not 255 independent sentences. The fixed-crop CER has no
label-based matching and is not directly comparable to the photo tables.
Large crops were unchanged. Many long handwritten lines still failed even
when visually readable; clearer display is not proof of reader improvement.
A local tensor check confirmed the tested long line had 96 recognition
time steps, so its failure was not evidence of a fixed 40-step output cap.
The active predictor's internal batch size was 1, supporting the exact-crop
reader cache used here.

## Crop-grouping bug and optional fix

The old grouping algorithm compared upright bounding rectangles. Two long,
tilted sentences can have overlapping upright rectangles despite belonging
to different physical rows. On three long-sentence photos, geometric checks
found grouped boxes spanning more than one accepted line while the raw
detector boxes did not.

`preprocessing/line_grouping.py` compares overlap perpendicular to the text
direction, along-line gaps, compatible orientations and whole-group row
alignment. This preserves separate tilted rows and limits transitive chains
of fragments from bridging different lines. No OCR text, confidence, labels
or photo names enter the algorithm.

| Set | Old grouping CER | Text-direction grouping CER | Better / worse photos |
|---|---:|---:|---:|
| Older non-B3, 35 photos | 32.96% | 25.33% | 10 / 3 |
| B3, 9 photos | 20.85% | 21.35% | 0 / 2 |
| Long sentences, 8 photos | 42.95% | 39.49% | 3 / 0 |

All **104/104 cases** completed. This was the most useful new development
candidate, but the B3 regression prevents claiming a universal win. It is
integrated in the package and the self-contained notebook as an explicit
option, with the default kept off:

A narrower follow-up preserved old groups unless their members had
overlapping along-line spans and clearly separated baselines. Another
**104/104 cases** completed: non-B3 CER was 25.52%, long sentences 39.49%,
and B3 still 21.35%. It did not remove the B3 regression and was not selected
over the first candidate. Its code and results remain available as an
explicit unsuccessful follow-up, not another default behavior.

```python
CFG = Config(group_directional=True, photo_limit=5, run_order_check=False)
```

Restart the notebook kernel before using a changed setting. A `photo_limit`
trial uses a separate output folder. The option and embedded source enter
the fingerprint, so cached old/new grouping is not silently mixed. Do not
rebuild training data or combine this with other changes to judge it.

## Saved reports and verification

- `outputs/remaining_preprocessing_v1/review.html`: completed independent arms.
- `outputs/remaining_preprocessing_v2/review.html`: bounded lighting/combination.
- `outputs/remaining_preprocessing_v2/fixed_lines.json`: all 255 labelled crop readings.
- `outputs/line_grouping_revision/review.html`: crop-grouping comparison.
- `outputs/line_grouping_guarded/review.html`: narrower follow-up; not selected.
- `outputs/handwriting_roll_stress/results.json`: known-angle functional checks.
- Each experiment contains its frozen protocol, records and run log.
- `outputs/remaining_preprocessing_v1/source_before_integration/` preserves
  the source modules, scripts and notebook used before optional integration.

The first cache preflight rounded detection boxes to float32, causing one
baseline ROI to differ. It was archived as
`outputs/remaining_preprocessing_precision_preflight/`, marked superseded,
and the entire independent-arm comparison was repeated with original dtypes.
All 44 older baseline readings then reproduced the previous baseline.
Reader outputs were reused only for identical crop pixels and shapes.

Code checks cover tilted adjacent lines, same-row fragments, separate
columns, conflicting slopes, transitive bridges, default/opt-in behavior,
notebook source parity, settings identity, illumination bounds and cache
geometry precision. Existing rotation, perspective and order-check tests
were also rerun. No further parameter search was performed after these
results; a genuinely fresh board/writer evaluation is the next safeguard.

Frozen protocols intentionally reject changed source/settings. For a future
audit use a fresh output name, for example
`evaluate_remaining_preprocessing.py --output remaining_preprocessing_v3`,
then pass `--baseline remaining_preprocessing_v3 --output <new-name>` to
the bounded-lighting or grouping runner. Do not overwrite a completed
experiment to make it accept new code. This audit's original sources are
preserved in the snapshot folder above.

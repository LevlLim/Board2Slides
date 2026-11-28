# `preprocessing/` — the hurdle-slide preprocessing as one importable package

Used by `notebooks/07_preprocessing_pipeline.ipynb` (builds the training set) and
`scripts/preprocess_order_check.py` (measures which steps help, and in which order).
Notebook `08_board2notes_preprocessing_pipeline.ipynb` embeds the same current
modules for its independent preprocessing comparison (round 4).

| File | What | Where it came from |
|---|---|---|
| `fix_lighting.py` | lighting correction (`fix(img, strength="light"/"strong")`), never binarises | **Jian Yan**, GitHub `TYW02/PaddleOCR-fine-tune`, branch `Jy`, commit `35b9dc2` (15 Oct 2026). Copied **unchanged** (md5 `2cd05802a3a3e329f92652fcd53e93ea`). |
| `fix_rotation.py` | in-plane rotation (`fix_rotation(img)`), projection-profile sweep | **Jian Yan**, same branch and commit. Copied **unchanged** (md5 `500a4350a3019f94f573ccf0ca23383b`). |
| `rotation_consensus.py` | active pipeline small-roll gate (`estimate`, `correct`): independent text rows plus glyph-pixel evidence, expanded canvas | new; wraps the rotation decision without modifying Jy's original helper |
| `blemish_removal.py` | marker residue removal (`remove_residue`, `clean_text_region`) | `Computer Vision & Deep Learning/BlemishRemovalTest.py` (md5 `ee7738c3...`). The three functions are the original's; the blob filter's size limits became parameters and its loop was vectorised (same rule). `clean_text_region` is new: limits from the line height, BGR output. |
| `board_geometry.py` | perspective correction (`auto_rectify(img, boxes, f35)`) | `Assignment/section8_auto_rectify_cells.ipynb`, sections 8.1-8.3, globals removed (boxes passed in). Changes: the largest-text-box fallback is off by default; a board angle over 75 deg is rejected; HoughLinesP output reshaped for OpenCV 5. |
| `text_lines.py` | detection the app's way (`Detector.detect`), line grouping, reading order, `quad_crop`, enlargement (`prepare_for_reader`), `Reader` | same notebook, section 9 (`smart_ocr` steps 1-4). `group_into_lines` and `reading_order` are new. |
| `labels.py` | ground truth loading, crop-to-label pairing (`match_lines`), hurdle metrics | new |
| `pipeline.py` | the chain and its variants (`preprocess_photo`, `VARIANTS`) | new |
| `dataset.py` | split by board, train-only augmentation, PaddleOCR label files, leak checks | new |

The files from teammates are kept byte-identical so that a fix in their branch can be copied over again
without merging: if you change them, change them in the `Jy` branch and copy them here.

Only `text_lines.Detector` / `Reader` need PaddleOCR; everything else is OpenCV + NumPy (+ openpyxl, Pillow, SciPy).

## Conservative rotation rule (29 October 2026)

The pipeline no longer uses global projection-profile confidence to decide
whether to rotate. It requires at least three separated detected text rows,
70% agreement within 2 degrees, and agreement between each supporting box and
its glyph-pixel direction within 2.5 degrees. At least three agreeing rows must
also overlap horizontally within a column. Same-row fragments count once.
The correction range is 2–15 degrees; uncertain cases retain their input pixels.
One expanded-canvas warp is used, and all detected text corners must fit.

`signals` records the angle, row agreement, independent/agreeing/stacked row
counts, candidate reasons and, when used, the transformation. The `confidence`
field is an agreement fraction, not a calibrated probability or OCR confidence.
This does not handle 90/180-degree orientation. Consistently sloped handwriting
can still resemble camera roll, and joined cursive can cause abstention.

Notebook Config exposes `rot_min_rows`, `rot_min_agreement`, `rot_angle_tol`,
`rot_pixel_box_tol`, `rot_min_angle` and `max_roll`. `min_roll_conf` remains only
for legacy cache compatibility. Restart the kernel and run sections 0–6 for
round 4; historical notebook outputs are not results of this new rule.

Offline regression tests: `scripts/test_rotation_consensus.py`,
`scripts/test_side_view_gate.py`, `scripts/test_ordercheck_logging.py` and
notebook section 3's self-tests. Geometry checks establish implementation
behavior; the complete OCR comparison is still needed to establish benefit.

# Claude handoff: visual capture-quality assistant for Board2Notes

Prepared 22 October 2026. This is a self-contained implementation plan, not a claim that the proposed model already works.

## 1. Objective and scope

Build and evaluate a small computer-vision helper that examines photographed handwriting and identifies text regions at risk of poor PaddleOCR recognition. Use those regions to offer review or recapture guidance. Keep the existing PaddleOCR reader frozen throughout this experiment.

The user is Zhao Bao, working on a university project called Board2Notes. A phone photo of a whiteboard is converted into editable notes, with a later text-to-diagram/notes stage. A teammate owns recognition fine-tuning. Zhao Bao previously developed preprocessing, perspective correction, selective enlargement and super-resolution experiments. SR has not consistently improved recognition, and the user is changing direction. Do not restart SR experiments or take over the teammate's reader training.

Start with a single uploaded photo. The user should not have to submit several photos for the assistant to function. A replacement photo is an optional response to a warning. Real-time camera/video analysis and multi-photo fusion are outside version 1.

Version 1 has a narrow, measurable task: predict OCR error risk from an image of a text line. A high score means the line should be checked. It does not certify that the image is blurry, that the reading is wrong, or that a retake will help. Validate those distinctions explicitly.

## 2. Existing project and verified starting points

Local project root:

`/Users/zhaobao/Documents/SIT/SIT-Y2-Tri1/Computer Vision & Deep Learning/Assignment/whiteboard-ocr`

Its parent `Assignment/` contains many datasets. Use repository-relative paths and configurable data roots in new code. The local absolute root above is orientation for this handoff, not a value to hard-code into scripts.

Read applicable repository instructions and inspect current files before editing. The working tree already contains many modified and untracked files owned by the user. Preserve them. Do not reset, delete, overwrite trained weights, or commit/push changes without authorization. If another person is running a notebook, keep implementation in new files and avoid modifying their notebook.

Useful files:

- `notebooks/04_whiteboard_ocr_pipeline.ipynb`: main pipeline. Inspection found `PP-OCRv5_mobile_det`, `PP-OCRv6_medium_rec`, `smart_ocr`, and `read_whiteboard`. Verify current code because teammates may change it.
- `scripts/capture_check.py`: existing classical checker. Uses Laplacian sharpness, gradient coherence and original-photo ink-band height. It already exposes `capture_advice` and `photo_advice`.
- `scripts/capture_check_thresholds.json`: existing calibrated thresholds. Preserve as historical artifacts. They were informed by existing data; do not present results on that same data as untouched test performance.
- `scripts/reader_evidence.py`: extracts character/word OCR evidence and checks decoding against PaddleOCR. Available for an optional OCR-signal comparison; it is not a prerequisite for the first image-only model. Existence of this script does not prove that its generated outputs exist or passed validation.
- `scripts/fusion_check.py`, `scripts/reading_fusion.py`: previous multi-photo OCR selection experiment. Not version 1's main direction.
- `scripts/pair_data.py`: existing board normalization and pair-processing utilities. Reuse only after checking their effect on visual quality signals.
- `scripts/finetune_reader.py`: identifies existing crop sources and reader configuration. Read for provenance; do not use it to train the main reader.
- `docs/SUPER_RESOLUTION_RUNS.md`: prior results and failure analysis. Treat historical claims as context, verify numbers before reusing them.
- `requirements.txt`: existing environment. Avoid replacing it or upgrading Paddle/PyTorch unnecessarily. Local scripts have used `Assignment/.venv-paddle/bin/python`. Inspect the active environment before running.

The README is not necessarily current. Prefer executable code and generated manifests over dated summaries.

### Existing data, inspected on 22 October

These counts describe files at handoff time, not final training eligibility:

| Source, relative to Assignment | Rows | Photos | Groups/status |
| --- | ---: | ---: | --- |
| `airdrop_rec_paddle/manifest.csv` | 601 | 67 | B1-B5; 353 accepted, 63 duplicate, 185 review |
| `17oct_rec_paddle/manifest.csv` | 392 | 31 | One group `17oct_board01`; 271 accepted, 121 review |
| `whiteboard-ocr/outputs/fusion_check/readings.csv` | 573 | 82 | B1-B5 and 17oct, with OCR readings and scores |

The Airdrop condition field contains 143 perspective, 242 scale/resolution, 174 clean/control, 41 blemish/background entries and only one explicit glare entry. Condition names are metadata, not verified quality labels. The 17 October field describes close/medium/far/side-angle roles rather than blur or glare. Inspect `Assignment/PhotosToUse/` as well; the old checker references a small set of real shake examples there. Audit their source identity and labels.

The 573-row cache is a subset selected for earlier work. Do not assume it covers every usable line. Equally, do not train on every row of the two original manifests: review rows, duplicates, partial crops and incorrect transcripts require handling.

There are only six recorded board groups across the principal sources. Hundreds of crops, repeated photographs or synthetic variants are not hundreds of independent board examples. More varied boards/writers may matter more than more shots of the same text.

## 3. Scientific question and success criteria

Main question: does a learned visual model identify poor OCR inputs better than simple image measurements and the existing capture rules on unseen board-writing setups?

Secondary question: when flagged captures are replaced with better photographs, does OCR improve enough to justify the user's extra effort?

Distinguish three outputs:

1. OCR-risk score: the primary learned output. It estimates recognition difficulty for a particular frozen reader/pipeline.
2. Visible capture problems: small text, blur or glare, assessed by validated measurements or separately labelled models.
3. Retake action: a product decision based on risk and evidence that capture is the issue. Otherwise suggest checking the transcription.

Examples: a clear unfamiliar term may be misread but not need recapture. An unknown camera frame with no detected text must not be labelled "good" merely because no line-level risks were computed. Use an explicit unknown/no-text state.

Do not promise an accuracy improvement, a novelty claim or a rubric grade in advance. Establish baselines and a practically useful improvement target on development data before inspecting final test results.

## 4. Phase A: audit and reuse the existing dataset

Create a read-only audit first. Inspect manifest status, label correctness, crop completeness, original resolution, photo IDs, board versions and session information. Review contact sheets rather than assuming every accepted row is suitable.

Audit outputs should include:

- Counts of source photos, crops, distinct texts, physical board-writing setups, writers/sessions if known, and existing quality conditions.
- Duplicate/near-duplicate provenance and suspected source overlaps.
- Crops with unknown original scale, incomplete/multiline content, questionable labels or no readable ink.
- Presence of both low-risk and high-risk examples within each group.
- Whether saved OCR outputs match the selected reader and processing version.
- A collection gap list, especially glare/shake and independent boards.

Exclude `review`/`needs_review`/rejected examples from supervised recognition-error labels unless explicitly corrected. A row marked duplicate may be another valid photo of the same writing; determine what the status means rather than blindly deleting or including it. Deduplicate identical images and preserve grouping for related captures. A crop with no trustworthy transcription can still be retained in a separate cause-label review pool, not assigned an invented CER.

Do not edit original manifests. Save a versioned derived manifest with exclusions and reasons.

## 5. Phase B: define provenance and leakage-safe splits

For each example store at least:

`sample_id, source_photo, crop_path, board_group, writing_instance_id, session_id_if_known, transcript, annotation_status, crop_box_or_polygon_if_known, native_crop_width, native_crop_height, original_scale_known, original_text_height_if_known, parent_sample_id, synthetic_transform, reader_id, pipeline_version, split`.

Use one derived format, such as JSONL, plus a CSV export if useful. Make unknown fields explicit. Do not infer a new independent board from a new filename, camera angle or crop.

Split at the board-writing/session level before making augmentations. All crops, close/far photos and synthetic variants of a shared writing setup must stay together. Inspect repeated text across groups and report it; avoid claiming generalization to new vocabulary when phrases repeat.

B3 has been consulted across many experiments. It is development evidence now, not a pristine final test. With the existing few groups, use a small grouped development comparison and report its limitations. Do not reuse the old photo-based 17 October train/validation split for this model.

Reserve newly collected boards for a final locked evaluation. If no fresh test data exists, finish the development study and say final generalization is pending; do not disguise a random crop split as a rigorous test. Do not manufacture metrics where a split has no positive or negative examples.

## 6. Phase C: build OCR-risk labels

Choose one exact frozen reader and one input-processing contract. Initially use the current PP-OCRv6 reader with the same line processing as the application. Record checkpoint/version and normalization. The quality model examines native information before enlargement; the label records OCR after the selected application processing.

Cached readings may save work only if their crop and processing contract match. `fusion_check.py` reads saved crops directly, which is not automatically identical to the full app's recognition path. Make this explicit.

For reviewed complete line crops:

1. Run the frozen reader and save its output and confidence.
2. Compute character edit distance against the transcript.
3. Save standard CER (including values above 1 if insertions warrant it), and clearly name any separate legacy capped/normalized metric used for comparability.
4. Use a configurable meaningful-error threshold to define a binary high-risk target. Decide it from development needs, not the final test. Preserve continuous CER so the target can be audited.
5. Inspect examples near the threshold, clear images with OCR errors, and visually poor images that OCR nevertheless reads correctly.

Do not call this label "blur" or "retake needed." It is an OCR-error proxy. Ordinary reader mistakes and capture failures are mixed in it; that is a limitation to measure.

For recapture validation, use only trusted same-writing pairs or a new capture study. Compare the original and replacement with the same reader. Do not assume a photo named `close_target` improves CER. Check identical textual extent and handle partially overlapping crops by aligned content or exclude them. Any idealized existing-pair result must be labelled a retrospective estimate, not an observed user-followed retake benefit.

## 7. Phase D: establish baselines before a neural model

Evaluate on identical development samples:

- Existing `capture_check.py` policy, unchanged, as a historical application baseline.
- Simple blur/contrast/text-height rules fitted on training/development groups only.
- A small logistic-regression model using classical visual features; a useful low-data baseline.
- PaddleOCR line confidence as an after-recognition reference. It incurs recognition computation; do not call it a before-recognition method.

Retain native text size separately. A crop enlarged to 64 pixels tall does not mean the photographed characters were originally 64 pixels tall. Missing scale metadata must not be filled with an assumed value.

Do not import a helper that launches training, downloads models or recalibrates on test data as a side effect. Separate reusable functions from command-line entry points where necessary.

## 8. Phase E: first learned visual model

Use a lightweight pretrained vision backbone, such as ResNet-18 or MobileNet, with a small binary-classification head. Start by freezing the backbone and training the head. Compare a limited fine-tuning variant only if development evidence supports it. This is transfer learning for a helper model, not main-reader fine-tuning.

The input is a line image, not the entire classroom image reduced to 224 pixels. Preserve aspect ratio using padding/bucketing or documented line windows. Avoid squeezing long text into distorted square images. Native blur/stroke information can be normalized for the CNN, but original text-height measurements must survive as explicit features in a second hybrid variant. For very long lines define window aggregation and evaluate it; don't silently truncate writing.

Start with binary cross-entropy on logits. Use training-only class weighting or balanced sampling if necessary. Record sampling weights, since rebalancing can distort raw probability calibration. Use a modest pretrained-backbone learning rate if unfreezing, mini-batch Adam/AdamW, validation-based early stopping, seeds and logged settings. These are starting choices, not claimed optimal settings.

Select CUDA when available, then MPS, then CPU. Smoke-test a small batch and checkpoint reload before a longer run. The first pilot should not require renting a GPU. Don't silently change dependencies or run expensive sweeps.

Version 1 comparisons should stay small: classical features; image CNN; image CNN plus native-size/visual features. OCR-internal evidence is an optional later comparison. Don't make character-probability extraction or a large transformer a prerequisite.

### Augmentation rules

Use realistic mild blur, downsampling, compression and illumination variations on training images only. Preserve parent/group IDs. A quality-changing transform can change the target: rerun the frozen reader to generate the transformed sample's OCR-risk label, or use it only for explicitly defined cause supervision. Do not retain "good" automatically after strong blur or label every synthetic image "bad."

Brightness changes may leave a sample readable; severity is not itself an error label. Preserve readable controls. Avoid horizontal flips, arbitrary crop truncation and geometric changes that misalign the transcript. Evaluate primarily on real untouched captures from unseen groups. Synthetic expansion does not replace source diversity.

## 9. Phase F: evaluate whether the model helps

Report the real error prevalence as well as accuracy. Main evaluation:

- Error-class average precision/precision-recall curve, with prevalence as the chance reference.
- Recall of poor OCR inputs and unnecessary flags on good inputs at validation-selected operating thresholds.
- Review/retake rate: how much extra work is requested.
- Calibration/reliability and Brier score if probabilities are displayed.
- Latency, model size and memory on the intended local/cloud devices.

Do not choose a threshold on final test labels to force a desired recall. Report actual test recall at the locked threshold. A raw sigmoid output is a score, not an established calibrated probability. Select/calibrate on a separate development partition or an explicitly documented grouped procedure.

Report paired method comparisons on the same cases, by board and pooled. Repeated crops from one photo/board are correlated; uncertainty estimates must respect the groups. With very few independent boards, acknowledge instability. Report board and writer/session coverage instead of treating thousands of synthetic crops as independent evidence.

For recapture benefit: evaluate initial photo versus replacement using identical writing and reader. Report CER changes, improvements/regressions, unnecessary recaptures and extra time. Compare a simple retake policy with the learned policy. Account for rejected/flagged examples; lower CER on only the accepted subset is not automatically better overall OCR.

Inspect failure galleries: unusually clear handwriting the reader misreads; blur read correctly; glare; small words; dark/coloured markers; diagrams; merged/clipped lines. A visually interesting heatmap is not a substitute for held-out results.

Proceed to integration only if the learned approach has a practical development advantage over simple rules, or clearly present it as an experimental option. Preserve the rule-based fallback if the CNN does not win. Do not keep increasing depth or data blindly to produce a preferred conclusion.

## 10. Phase G: guidance and app integration

Expose a standalone API that accepts text regions plus provenance and returns per-region risk, supporting measurements, bounding polygon, model version and an optional advice code. Ground-truth text, file names, board IDs and labels must never be model inputs.

Suggested output fields:

`region_id, polygon, risk_score, score_calibrated, native_text_height_px, visible_problem, advice, status`.

Use `visible_problem=unknown` when evidence is insufficient. Initially the CNN predicts risk only. Validated size measurements can support "move closer"; blur/glare-specific prompts need audited cause evidence. Keep learned risk and heuristic diagnosis separately named in logs/UI.

Suggested UI behaviour:

- Low risk: continue; this does not certify correctness.
- High risk with very small native writing: highlight region and suggest moving closer.
- High risk with validated shake/blur evidence: suggest stabilizing/refocusing.
- High risk with independently validated glare evidence: suggest changing angle/lighting.
- High risk without identified capture problem: suggest reviewing the reading rather than asserting retaking will fix it.
- No detected text / uncertain region: state that text could not be assessed and allow retry/manual review.

Show a coloured border over the actual text region. Allow "continue anyway" and "retake". Keep unaffected text available where possible. Version 1 can use upload/preview; do not require a live video camera pipeline.

Use the existing detector and original image crop coordinates without retraining it. Handle resize, rectification and crop coordinate mappings explicitly. If a reliable overlay on the original photo needs a transform not available yet, first show crop thumbnails; do not draw misleading boxes.

Integrate with `read_whiteboard` through a minimal optional hook once the standalone evaluation works. Keep helper optional so the existing app still runs without its checkpoint. Adding a PyTorch helper to a Paddle pipeline has a startup/memory cost; measure it, and prefer separate model loading on demand. Export/optimization is optional and must pass prediction-parity checks before use.

## 11. Data collection after the audit

Use existing data for the first pilot. A planning target, not a scientific minimum, is approximately 15-20 varied board-writing setups, 6-10 photos per setup, roughly 100-200 original photos and 1,000-2,000 line crops. Existing source manifests already contain about 98 photos, so don't interpret that target as 200 additional photos. The main gap may be independent writing and conditions.

For each new setup write several complete lines, include different writers where possible, and collect a clear reference plus realistic distance, focus/shake and lighting variations. Include moderately difficult and readable examples, not just extremes. Do not require every condition on every board. Store transcripts and same-writing IDs; labels must match each crop's visible content.

Keep some new boards untouched for final testing. Ask for targeted collection only after documenting the actual gaps. Never invent cause labels or independent board identities from filenames.

## 12. Deliverables and suggested file layout

Prefer new scoped files, adapting names to the repo if needed:

- `scripts/quality_data.py`: audit, provenance, splits and target construction.
- `scripts/quality_model.py`: image preprocessing, model definition and inference contract.
- `scripts/train_capture_quality.py`: reproducible smoke/train CLI.
- `scripts/evaluate_capture_quality.py`: baselines, comparisons, thresholds and reports.
- `scripts/capture_quality.py`: lightweight app-facing API/fallback.
- `notebooks/experiments/capture_quality_assistant.ipynb`: audit, model, evaluation and demo sections, runnable top to bottom.
- `configs/capture_quality.yaml`: versioned model/target/split/processing settings if a config fits existing conventions.
- `outputs/capture_quality/<run_id>/`: derived manifests, audit, predictions, plots, examples, metrics and logs.
- `models/capture_quality/<run_id>/`: best checkpoint and preprocessing/label/calibration metadata.
- `docs/CAPTURE_QUALITY_RESULTS.md`: what was tried, comparisons, limitations and reproduction commands.

Keep data/weights out of Git. Don't overwrite historical caches. Record exact commands for audit, smoke training, full training, evaluation and demo. The user likes notebooks they can run themselves, so expose expensive stages explicitly and avoid automatically retraining on every Run All.

Meaningful checks: group/parent split leakage; label/provenance consistency; safe handling of empty/no-text crops; checkpoint reload; identical frozen-reader configuration across comparisons; coordinate and scale handling; and successful smoke execution. Use proportionate tests, not a large unrelated infrastructure rewrite.

## 13. How the user can present it

Explain: "I trained a visual capture-quality model to identify text regions likely to cause OCR errors. It complements our recognizer by guiding review or better capture."

Show the data protocol and board split, visual model and loss, baseline comparison, held-out example gallery, and one complete initial-photo/warning/replacement-photo example with correct transcription. Clearly mark illustrative UI scores versus real predictions. Include cases where a retake did not help.

Connect to course topics: image classification, CNN feature extraction, transfer learning, binary cross-entropy, gradient optimization, training-only augmentation, regularization, early stopping and unseen-data evaluation. Multiple causes can co-occur, so any later blur/glare/small-text classifier would be multi-label rather than forced into one exclusive cause.

For the rubric, show own/curated annotations, justified training strategies and ablations, local/cloud use, useful UI and extension to other cameras/writers. Existing architectures and standard losses are not automatically novel. Do not describe a threshold change or a standard CNN as an invented architecture. Preserve the negative SR findings as project evidence rather than hiding them.

## 14. Execution order and stop conditions

1. Audit files/data and write the provenance/gap report.
2. Define the frozen-reader processing contract and grouped development split.
3. Generate and inspect labels; evaluate simple baselines.
4. Smoke-test one small image model; then run a limited grouped comparison.
5. If it beats practical baselines, freeze its development-selected settings and add the notebook demo/API.
6. Validate recapture benefit on real matched captures, ideally new boards. Keep this separate from risk-prediction performance.
7. Integrate upload/preview guidance, test local/cloud behaviour and prepare final report.

Do not request a large collection first. If labels/groups are insufficient, complete unaffected audit/baseline work and request a precise missing set. If the learned model offers no benefit, document the result and keep classical advice. If final test boards are unavailable, leave final-test claims pending rather than relabelling development data.

No phase should silently launch paid cloud resources, retrain the main reader, contact teammates, or restart SR. This handoff authorizes implementation of the capture-quality work; coordinate with the user on real data collection and expensive training when needed.

## 15. Research and rubric references

- Project rubric: `../../ProjectBriefing.pptx` from this repository root (the briefing is at the course workspace root). Verify slide 3 rather than relying on memory.
- [Text-line-based document quality prediction, Li et al. (2019)](https://arxiv.org/abs/1906.01907): CNN quality assessment at text-line level; not proof of whiteboard performance.
- [CG-DIQA, Li et al. (2018)](https://arxiv.org/abs/1807.04047): character-gradient quality with OCR accuracy as target; useful classical reference.
- [ConfidNet, Corbiere et al. (NeurIPS 2019)](https://arxiv.org/abs/1910.04851): learned failure confidence; not a ready-made whiteboard retake system.
- [PERO quality](https://github.com/DCGM/pero-quality): local readability heatmaps using OCR evidence; an alternative reference, with different reader/dependencies.
- [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html): implementation guidance, not a minimum-data guarantee.
- Existing broader research comparison: course workspace `output/research/paddleocr_assistance_research_2026-10-21.md`.

### First action for Claude

Begin with Phase A using the actual current files. Report the audited usable data and group counts, then implement a small reproducible baseline/pilot. Do not assume every historical count or result remains current, and do not start with a complex multi-task network. Proceed through the plan using evidence, preserving the user's running work.

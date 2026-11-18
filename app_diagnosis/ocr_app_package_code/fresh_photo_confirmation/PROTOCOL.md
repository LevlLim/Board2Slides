# Fresh-photo confirmation — protocol

**Status: DRAFT. Not started.** No fresh photos have been supplied, transcribed or run. Until `results/RESULTS.md` exists from a frozen run, nobody may say this configuration has been "confirmed".

The machine-readable rules are in `protocol.json`. This page explains them. Both files are hashed when the set is frozen.

## Question

Does the frozen package (`MANIFEST.json`, unchanged) give usable notes on whole whiteboard photos from **writers the project has never used**?

This is a confirmation of the provisional development result. It is not a new experiment: nothing is tuned on these photos.

## Sample

- **10–15 full photos from 2–3 writers**, at most 7 per writer.
- **Writers must be genuinely new.** They must not be ZB, JH, DS or Jy, and must not be any Board2Notes teammate. Teammates are excluded because the 64 uncertain-writer teammate crops make every teammate a possible training writer. They must also not be the writer of any lecture/CNN/whiteboard training photo or of the familiar-writer sentence collection. Record each writer's agreement in `consent_for_evaluation`.
- **Photos must be new captures.** Not a photo, crop or edit of anything in an existing project folder. `freeze.py` checks byte identity against every image under the Assignment folder. That check cannot detect a re-photographed board; that part is the collector's responsibility.
- **Content.** Ordinary lecture-style notes, written for this purpose or from a real class. Do not choose boards because they look easy or hard: photograph every board you planned to, in order, and keep failures.
- **Spread of conditions** to aim for across the set:
  - mostly single-column, plus 2–4 multi-column or diagram-with-labels boards;
  - two or more marker colours on some boards;
  - at least two lighting conditions (glare and non-glare);
  - straight-on and moderately angled shots (up to about 30°);
  - 1–3 m distance;
  - each writer's usual phone, with the default camera app, HEIC or JPEG.
- **Kept out of everything else.** These photos must never be added to training, development, augmentation or prompt material, either before or after the run. Store them only in `photos/`.

## Reference transcriptions

The reference is fixed **before** any prediction is seen.

1. The **transcriber** looks only at the photo and writes `transcriptions/<photo_id>.txt`:
   - one line of text per written line on the board, in the human reading order (column by column, top to bottom);
   - the writer's own spelling, capitals and punctuation;
   - crossed-out or erased words, printed text, screens and posters are left out;
   - a line nobody can read is written as `[illegible]`.
2. A different person, the **reviewer**, checks every line against the photo. The two agree one reading for any doubtful character and note the doubt in the manifest `notes` column. Then set `transcription_status` to `reviewed`.
3. **Optional geometry:** `transcriptions/<photo_id>.quads.json` holds one 4-point quad per written line, in the order of the `.txt` lines, in EXIF-oriented full-resolution pixels. An example is in `templates/`.
   - Geometry enables the reading-order-free and region diagnostics.
   - Quads must be drawn from the photo alone, never from the app's boxes.
   - Claude can locate them from the photo and transcription before the run, as was done for Jy, if the team prefers.

## Metrics, fixed now

- **Normalisation** is that of notebook 15: lower-case, whitespace runs become one space, nothing else changes. Punctuation counts; case does not.
- **Main metric: pooled page CER in the app's own reading order.** This is the edit distance between the reference page (lines joined) and the app's text (regions joined in output order), summed over photos and divided by the total reference characters.
  - This is what the student receives, and it is the same main metric as notebook 15.
  - It includes reading-order mistakes.
- **Always reported alongside:**
  - per-photo, per-writer and per-layout page CER;
  - a photo bootstrap 95 % interval (10,000 resamples, seed 0). It holds the writers fixed, so it says nothing about other writers.
- **With quads:**
  - page CER after putting the app's components in reference order; the difference from the main metric is the **reading-order cost**, which is deferred work;
  - one-to-one **matched-line CER**;
  - counts of one-to-one, missed, extra, merged, split, many-to-many and illegible-region components.
  - The overlap rule is IoU ≥ 0.25 or intersection/smaller ≥ 0.50, and text is never used for matching. `scoring.py --self-test` checks it against the notebook 15 originals.
- **Rejected photos** (the app cannot open them) score as complete misses: all reference characters count as deletions.
- **Not comparable:** the development figure (0.87 % mean *line* CER on 59 matched JH lines) uses a different unit, metric and selection. Do not use it as a pass mark.

## Decision rule

Default: **descriptive**, with no pass/fail. If the team wants a threshold, it must be written into `protocol.json → decision_rule` before freezing.

## Procedure

1. Agree these rules. In `protocol.json`, set `status: "agreed"`, `agreed_by` and `agreed_on`.
2. Collect the photos. Fill in `photos_manifest.csv` (see README), the transcriptions and, optionally, the quads.
3. Run `python fresh_photo_confirmation/tools/freeze.py` until it reports no problems, then run it again with `--freeze`. This writes `FROZEN.json` with the hash of every input, the rules, the scorer and the package manifest.
4. Run `python fresh_photo_confirmation/tools/run_and_score.py` **once**. It refuses to run if anything frozen has changed, if the package fails its integrity check, or if results already exist.
5. Report `results/RESULTS.md` as written, including bad photos. Any change made afterwards (to settings, crops or models) needs a **new** fresh set to confirm it.

If the run crashes for a tooling reason, record the error and fix only the tooling. Then delete the partial `results/` folder (or move it to `_to_delete/`) and run again on the same frozen inputs. Editing `run_and_score.py` or `scoring.py` after freezing invalidates the freeze; the run refuses, and both the change and a re-freeze must be recorded.

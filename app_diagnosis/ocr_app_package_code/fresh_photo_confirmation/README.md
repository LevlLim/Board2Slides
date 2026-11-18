# Fresh-photo confirmation kit

**Not complete.** No photos have been collected yet. Read `PROTOCOL.md` first. The rules are fixed before anyone sees a prediction.

## Folders

| path | what goes there |
|---|---|
| `photos/` | the original photo files, unedited and straight from the phone (HEIC or JPEG) |
| `photos_manifest.csv` | one row per photo (columns below) |
| `transcriptions/<photo_id>.txt` | the reviewed reference text (format in `PROTOCOL.md`; example in `templates/`) |
| `transcriptions/<photo_id>.quads.json` | optional line geometry (example in `templates/`) |
| `protocol.json`, `PROTOCOL.md` | the rules; agree them first |
| `tools/freeze.py` | checks the set; `--freeze` locks it (`FROZEN.json`) |
| `tools/run_and_score.py` | the single prediction and scoring run, after freezing |
| `results/` | created by the run: predictions, `RESULTS.json`, `RESULTS.md` |

Keep this folder out of every training, augmentation and development folder. The Docker image excludes it (see `.dockerignore`).

## Manifest columns

| column | fill in |
|---|---|
| `photo_id` | e.g. `W1_01` (writer 1, photo 1); also the transcription file name |
| `file` | file name inside `photos/` |
| `writer_id` | a pseudonym such as W1, W2, W3. **Do not use real names.** Keep the key to who is who outside this folder |
| `writer_new_to_project` | `yes` only if the person is none of the excluded writers in `protocol.json` |
| `consent_for_evaluation` | `yes` once the writer has agreed to their handwriting photo being used for this evaluation |
| `captured_on` | date (YYYY-MM-DD) |
| `device` | phone model and camera app |
| `distance_m` | approximate distance to the board |
| `angle` | `straight` / `slight` (< 15°) / `angled` (15–30°) |
| `lighting` | e.g. classroom fluorescent, daylight, mixed |
| `glare` | `none` / `some` / `strong` |
| `board_type` | whiteboard / glass board / other |
| `marker_colours` | e.g. `black+blue` |
| `layout` | `single-column` / `multi-column` / `diagram` |
| `approx_lines` | rough count of written lines |
| `in_any_training_or_dev_set` | must be `no` |
| `transcriber`, `reviewer` | two different people |
| `transcription_status` | `draft` → `reviewed` |
| `notes` | agreed readings of doubtful characters, anything unusual |

## Commands

Run these from the package folder, with the package's environment:

```
.venv/bin/python fresh_photo_confirmation/tools/scoring.py --self-test
.venv/bin/python fresh_photo_confirmation/tools/freeze.py
.venv/bin/python fresh_photo_confirmation/tools/freeze.py --freeze
.venv/bin/python fresh_photo_confirmation/tools/run_and_score.py
```

## Capture checklist (per photo)

- The whole writing area is in frame. Do not crop, filter, enhance or "scan" with an app.
- Take one photo per board, and keep it even if it looks hard to read. Retake only for a reason that has nothing to do with legibility, such as a face in the frame.
- Avoid people's faces in the frame. If a face appears, it is personal data: get consent or retake before writing anything down.
- Copy the original file from the phone as it is. AirDrop or a cable keeps HEIC/EXIF; messaging apps recompress it.

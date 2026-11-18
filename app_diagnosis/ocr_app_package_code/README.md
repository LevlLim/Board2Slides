# Board2Notes OCR package (frozen 9 November 2026)

This package holds the current working OCR pipeline with its own weights, settings manifest and checks. It needs nothing else from the Assignment folder at run time.

```
photo (EXIF-oriented) -> PP-OCRv5_mobile_det -> existing line grouping + order -> quad crop -> S1 seed-1 reader -> editable OCR text
```

**Status.** This is a provisional development configuration. Accuracy evidence so far is development evidence. Notebook 15 measured this exact configuration on 16 provisionally annotated Jy full photos at **27.5 % page CER**, most of it from reading order (diagnosed at about +21.9 points; about 5.6 % once the reference order is applied). Notebook 16 measured 0.87 % mean line CER on 59 matched lines from JH's full photos. Earlier crop-level evaluations are listed in `PROJECT_PROGRESS.md`. None of these is fresh-writer confirmation. Reading order is a simple heuristic: it was provisionally evaluated and diagnosed in notebook 15, and improving it is deferred.

**Not included or changed:**
- no training, detector change, automatic tightening, extra deskew, VLM or slide generation;
- the original app (`whiteboard-ocr/ocr_prototype`) and the notebooks are untouched.

## Contents

| path | what |
|---|---|
| `models/PP-OCRv5_mobile_det/` | detector inference files (4.7 MB), the exact cached copy every recorded app run used |
| `models/PP-OCRv5_server_rec_board2notes_S1/` | S1 seed-1 exported reader (81 MB), with `verified.json` from its export check |
| `b2n_ocr/` | vendored app code: Streamlit UI, backend, `preprocessing/text_lines.py`, UI tests (see patches below) |
| `MANIFEST.json` | settings (detection, grouping, crop, resize, decode, dictionary), source-file hashes, patch list, SHA-256 of every model/code/fixture file |
| `fixtures/photos/` | 4 fixed regression photos (Jy 8, Jy 23, JH 2, JH 5) |
| `fixtures/expected/` | reference outputs per platform (`linux-aarch64.json` now; `mac-arm64.json` after the Mac step) |
| `tools/` | `verify_package.py`, `run_fixtures.py`, `ocr_image.py` (command line), `e2e_app_check.py`, `benchmark.py`, `compare_with_saved_app_runs.py`, `build_package.py` |
| `requirements/` | pinned inference/UI dependencies and the platform notes |
| `deploy/` | `Dockerfile`, `start.sh`, `DEPLOYMENT.md` |
| `fresh_photo_confirmation/` | collection and annotation kit for 10–15 new photos, with evaluation rules fixed in advance |
| `VERIFICATION.md`, `STATUS.md` | what has been checked, and what is still open |

**Excluded:** training checkpoints, optimizer state, other readers, datasets (except the 4 fixture photos), notebooks, credentials, API keys and private configuration. The package contains no network credentials and never downloads models.

## Patches to the vendored code

Everything else in `b2n_ocr/` is byte-identical to the live app. Each patch is listed in `MANIFEST.json`. The builder refuses to build if the live code no longer matches a patch.

- **P1–P2, P4–P5** — model paths are package-relative:
  - only S1 is offered;
  - the detector loads from `models/` (no download);
  - the detector runs with `device='cpu', enable_mkldnn=False`, the plain Paddle engine the Mac reference uses.
- **P3** — `import torch` is attempted only if torch is installed. The original Mac environment used it to avoid a libiomp clash; torch is not an inference dependency.
- **P6** — the "Try a sample photo" image is the packaged copy of Jy Photo 23.
- **Optional `B2N_DET_CPU_THREADS`** — unset by default, which keeps PaddleOCR's default as tested. Set it to `1` on Linux aarch64 with Paddle 3.2.2, where multithreaded detection crashes.

Rebuild from the live app with:

```
python3 board2notes_ocr_package/tools/build_package.py --detector-dir ~/.paddlex/official_models/PP-OCRv5_mobile_det
```

## Install

Python 3.12. Do **not** copy `.venv-paddle`: it also holds training and experiment packages.

macOS (Apple Silicon), from this folder:

```
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements/base.txt
.venv/bin/pip install --no-deps --force-reinstall -r requirements/opencv-override.txt
```

Linux x86_64: the same three commands with `python3.12`, or use the container (`deploy/DEPLOYMENT.md`).

Linux aarch64: `paddlepaddle==3.3.1` has no aarch64 wheel. For a local check, use 3.2.2 and set `B2N_DET_CPU_THREADS=1`; see `requirements/README.md`.

## Verify

```
.venv/bin/python tools/verify_package.py
.venv/bin/python tools/verify_package.py --smoke
```

- The first command checks that every model, code and fixture file is present and unchanged (SHA-256 against `MANIFEST.json`), and that `models/`, `b2n_ocr/` and `fixtures/photos/` contain no unexpected files (Python caches and `.DS_Store` are ignored). Docs, `tools/`, `deploy/` and `requirements/` are not hash-checked. It also notes any package version that differs from the tested ones, and exits with 1 if anything is missing, changed or unexpected.
- `--smoke` also reads the fixtures, plus derived PNG, HEIC, EXIF-rotated, unreadable, truncated, blank and tiny inputs, and compares the result with this platform's reference.
- On the development Mac, adding `--against-app` also runs the live app in `.venv-paddle` and requires identical boxes and text.

## Launch

```
./run_local.command
```

This opens http://127.0.0.1:8501 on this computer only. The container and Linux server setup is in `deploy/DEPLOYMENT.md`.

## Command line

```
.venv/bin/python tools/ocr_image.py PHOTO.jpg --text notes.txt --json result.json
```

Exit codes:
- `0` — read; "No writing was found" is reported but is not an error;
- `2` — the file is not a readable image;
- `1` — any other failure.

The JSON is the backend's structured result: lines, polygons, reading order, timings, detector settings and model identity.

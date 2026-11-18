# Verification record — package 2026-11-09

This file keeps two kinds of evidence apart.

- **Implementation verification** asks whether the package produces the same output as the current app, handles every kind of input safely, and has intact files. This is answered by checks with fixed inputs and exact expected outputs.
- **Accuracy evaluation** asks how good the readings are. Nothing in this file measures accuracy. The accuracy evidence for this configuration is development evidence: notebook 15 (27.5 % page CER on 16 provisionally annotated Jy full photos, mostly reading order) and notebook 16 (0.87 % mean line CER on 59 matched JH lines, with JH 2 and JH 5 among the photos), plus earlier crop-level evaluations in `PROJECT_PROGRESS.md`. Fresh-photo confirmation is set up in `fresh_photo_confirmation/` and has **not** been carried out yet.

The four fixture photos are development photos. Their role here is regression only: the package must reproduce the app's saved output on them. Whether that output is correct is a separate question.

## Results so far

| check | where | result |
|---|---|---|
| Integrity: 22 model, code and fixture files (SHA-256 against `MANIFEST.json`) | Linux aarch64 VM | pass |
| Tamper test: one changed byte, one deleted file, one extra file | Linux aarch64 VM | all three reported; exit 1 |
| Unexpected-file check after the 9 Nov fix: extra file in `b2n_ocr/` and in `fixtures/photos/`; `__pycache__` and `.DS_Store` | Linux aarch64 VM | both extras reported, exit 1; caches ignored |
| `--record … --record-dir outputs/expected`, then `--reference` reading it back | Linux aarch64 VM | recorded and compared: 11/11 identical |
| Permission normalisation used in the Dockerfile's first stage (`chmod -R u=rwX,go=rX`), applied to an owner-only copy | Linux aarch64 VM | all files 644, all folders 755 |
| Fixture photos match the **saved Mac app outputs** (`compare_with_saved_app_runs.py`) | Linux aarch64 VM vs Mac app runs from notebooks 15/16 | **identical** on all 4 photos: same region count, boxes within 6e-5 px, 63/63 texts identical |
| Phone formats: PNG, HEIC (pillow-heif, q95) and JPEG with EXIF orientation 6 made from Jy 23 | Linux aarch64 VM | read correctly, 9 regions each; PNG and EXIF-rotated text identical to the JPEG; HEIC differs in 2 single characters (lossy re-encoding changes pixels) |
| Unreadable bytes, truncated JPEG (first 2 KB) | Linux aarch64 VM | rejected with the app's message ("This photo could not be opened…"); no crash |
| Blank board (1200×900 white), 12×12 px image | Linux aarch64 VM | status ok, 0 lines; the UI and command line show "No writing was found" |
| Text and JSON export round-trip (UTF-8, line order kept) | Linux aarch64 VM | pass |
| Existing app unit tests (21, mocked reader) against the vendored code | Linux aarch64 VM | 21/21 pass |
| End-to-end app run with real models (`e2e_app_check.py`): sample → extract → notes equal OCR text → edit kept → original reading unchanged | Linux aarch64 VM | pass (9 regions) |
| Server start (`deploy/start.sh`) and `/_stcore/health` | Linux aarch64 VM | ok |
| `verify_package.py --smoke --record linux-aarch64` | Linux aarch64 VM | pass; reference saved as `fixtures/expected/linux-aarch64.json` |

**VM environment:**
- Ubuntu 22.04 aarch64, 4 vCPU, Python 3.12.15;
- paddlepaddle **3.2.2** (3.3.1 has no aarch64 wheel), paddleocr 3.7.0, paddlex 3.7.2, numpy 2.3.5;
- cv2 5.0.0 (headless override), pillow 12.3.0, pillow-heif 1.8.1;
- `B2N_DET_CPU_THREADS=1`.

The detector **segfaulted** on aarch64 with Paddle 3.2.2 when multithreaded. One thread fixed it, and the output was still identical to the Mac.

Even with a different Paddle version and CPU architecture, boxes and text came out identical. That is good evidence that the package is not tied to the Mac environment. It is not a substitute for checks on the target platform.

## Not yet verified

| check | why it is still open | command |
|---|---|---|
| Packaged code vs the **live** app on the development Mac (Paddle 3.3.1, same photos, same run) | requires the Mac; the VM compared against saved outputs | `verify_package.py --smoke --against-app --record mac-arm64` |
| Fresh package environment on the Mac (`.venv` from `requirements/`, not `.venv-paddle`) | requires the Mac | see README "Install" |
| Linux x86_64 container with Paddle 3.3.1 (the cloud target), including the non-root permission fix | Docker is not available in this VM. The Dockerfile now checks at build time that user 10001 can read every file, but no build has run yet | `deploy/DEPLOYMENT.md` steps 1–3 |
| Copy button and the browser download event | AppTest does not run browser JavaScript; the export *content* is tested | manual browser check (below) |
| Real phone HEIC files | the HEIC fixture is generated, not a real iPhone file | upload one real iPhone photo in the browser check |

### Manual browser check

Takes about 5 minutes and is done once per platform.

1. Run `./run_local.command`, or the container, and open the app.
2. Click **Try a sample photo**, then **Extract notes**. Expect 9 regions with boxes drawn.
3. Change one word in the notes. Click **Copy notes** and paste somewhere: the edit is there. Click **Save my notes**: the downloaded `.txt` contains the edit.
4. Upload a real iPhone HEIC photo, a PNG screenshot and a non-image file renamed to `.jpg`. The first two read; the third shows the "could not be opened" message.
5. Upload a file over 30 MB. Streamlit refuses it before OCR runs.

## Where results are stored

- `fixtures/expected/<platform>.json` holds the reference outputs that `--smoke` compares against.
- `outputs/` holds per-run fixture results and benchmarks (`fixtures_*.json`, `benchmark_*.json`).

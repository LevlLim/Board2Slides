# Status — 9 November 2026

**Configuration (frozen in this package):** `PP-OCRv5_mobile_det → existing grouping/order → quad crop → S1 seed-1 → editable text`.

This is provisional. Accuracy evidence so far is development evidence. Notebook 15 measured this exact configuration on 16 provisionally annotated Jy full photos at **27.5 % page CER**, most of it from reading order (diagnosed at about +21.9 points; about 5.6 % once the reference order is applied). Notebook 16 measured 0.87 % mean line CER on 59 matched lines from JH's full photos. Earlier crop-level evaluations are listed in `PROJECT_PROGRESS.md`. None of these is fresh-writer confirmation. Fresh-photo confirmation has **not** been done.

## Completed

- **Inference-only package** (`board2notes_ocr_package/`, about 107 MB):
  - the detector and S1 weights;
  - vendored app code with 6 listed path/runtime patches; the original app is untouched;
  - pinned requirements, `MANIFEST.json` and 4 regression photos;
  - no checkpoints, datasets, notebooks, credentials or keys.
- **Tools:**
  - `verify_package.py` checks integrity and reports any missing, changed or extra file;
  - `--smoke` runs the fixture checks;
  - `--against-app` compares with the live app;
  - plus a command-line reader, an end-to-end UI check, a benchmark and a saved-output comparison.
- **Container configuration:** linux/amd64 `Dockerfile` with packaged weights, no downloads and a build-time integrity check; non-root; health check. After Codex's 9 Nov review, a first build stage normalises the owner-only file modes, and a build step proves that the runtime user can read everything. References recorded in the container go to the writable `outputs/expected/`. `deploy/DEPLOYMENT.md` covers upload limits, model loading, the serialised request handling, scaling and the benchmark/compare steps.
- **Fresh-photo kit** (`fresh_photo_confirmation/`):
  - manifest, transcription and geometry templates;
  - pre-registered rules (`protocol.json`, `PROTOCOL.md`);
  - `freeze.py`, which runs the checks and hash-locks the inputs before any prediction;
  - `run_and_score.py`, a single-shot run scored with the notebook 15 rules.

## Verified (Linux aarch64 VM; details in `VERIFICATION.md`)

- Integrity check passes, and a tamper test (changed, missing and extra files) is detected.
- **The 4 fixture photos match the saved Mac app outputs exactly:** 63/63 texts identical, boxes within 6e-5 px. This held on a different CPU architecture and Paddle 3.2.2.
- PNG, HEIC and EXIF-rotated inputs are read. Unreadable and truncated files are rejected with the app's message. Blank and tiny images return no lines without error. Text and JSON export round-trips pass.
- The 21 existing app tests and the real-model end-to-end UI check pass. The server starts and its health endpoint answers.
- **Benchmark (4 vCPU VM):**
  - cold first request 5.5 s;
  - warm 2.8–15 s per photo;
  - 3 simultaneous requests are processed one after another (23 s wall);
  - peak 1.85 GB.
- **Fresh-photo tooling, exercised in a throwaway copy with fake data** (development photos standing in as "fresh"; the results mean nothing and nothing was kept in the real kit):
  - `freeze.py` rejects a draft protocol and byte-identical project photos;
  - freezing works;
  - the run refuses after edits to frozen inputs and refuses a second run;
  - rejected photos score as full misses;
  - illegible-line regions, reference-order page CER and the region diagnostics compute as specified;
  - `scoring.py --self-test` is identical to the notebook 15 functions.

## Not yet verified

1. **Mac:** a fresh package `.venv`, packaged vs **live** app on the same run (`--against-app`), the `mac-arm64` reference, the benchmark and the browser check. These are the commands handed over below.
2. **Linux x86_64 container with Paddle 3.3.1:** build, smoke against `mac-arm64`, benchmark. Docker was not available here.
3. **Browser-only behaviour:** the copy button, the download event, a real iPhone HEIC, and the upload over 30 MB.
4. **Any cloud deployment.** Nothing was provisioned or published.

## Limitations to state plainly

- **Accuracy is unconfirmed beyond development data:**
  - one development writer (JH) for the 0.87 % figure;
  - line-level, matched lines only;
  - provisional assistant-reviewed bands.
- **Reading order is a provisional heuristic.** It was provisionally evaluated in notebook 15, where it was the largest error source on multi-column Jy photos. Improving it is deferred.
- **Re-encoding changes the output.** In the plumbing test, re-saving the dense multi-column Jy 8 photo as JPEG q90 changed the reading order and several word readings (e.g. "Lecithin" → "Lecithih"; 36 page edits on 342 characters). The single-column JH 2 copy was unchanged.
  - This is one incidental observation, not a measured robustness result.
  - It is why the fresh-photo kit asks for original, unrecompressed files.
  - It also suggests that small box shifts can reorder multi-column output.
- **Throughput is serial per process** (lock). Concurrency means more instances, with sticky sessions.
- **Linux aarch64** needs Paddle 3.2.2 and a single-threaded detector. The supported targets are macOS arm64 and Linux x86_64 with 3.3.1.
- **No access control in the app.** A public URL would let anyone upload.

## Input needed from the team

1. Run the Mac commands and send back their output. This records `fixtures/expected/mac-arm64.json`.
2. **Deployment decision:** host, region, instance size, expected cost, who can access it, and whether a registry push is allowed. Nothing will be provisioned before this.
3. **Fresh photos:** agree the protocol (`status: "agreed"`, with names and date), then collect 10–15 photos from 2–3 genuinely new writers and have them transcribed and reviewed by two people.
4. **Resolved:** the Photo 5 extra region is a projected Windows 11 taskbar (Windows logo, magnifier, "Search"), not handwriting. Checked on the full photo, and Codex agrees. It does not affect the 59-line figure. `claude_jh_fullphoto_20261109/REPORT.md`, `RESULTS.json`, `review.html` and Codex's notebook 16 still say "Q Search" writing, and need the same correction by their author.
5. **Two notebook 16 files exist:** `16_jh_full_photo_app_evaluation.ipynb` (Codex, executed, canonical) and `16_jh_full_photo_app_check.ipynb` (Claude). Suggest keeping the Codex notebook as the record and moving the other to `_to_delete/`.

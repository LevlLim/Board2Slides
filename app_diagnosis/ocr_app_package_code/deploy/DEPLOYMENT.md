# Cloud / container deployment (prepared, not provisioned)

**Nothing has been deployed.** No account, server, registry or public URL has been created.

These steps prepare and check the container on your own machine. Choosing the hosting destination, and accepting its cost, is a decision for the team. Make that decision before any step that pushes an image or opens a public address.

## What the container is

| item | value |
|---|---|
| base | `python:3.12-slim-bookworm`, **linux/amd64** (paddlepaddle 3.3.1 has no Linux aarch64 wheel) |
| dependencies | `requirements/base.txt` and the OpenCV override, pinned to the versions tested on the Mac |
| models | copied into the image from `models/`. `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`, and the detector is given a local folder, so **no model is downloaded** at build or run time |
| build-time check | `RUN python tools/verify_package.py`: the build fails if any model, code or fixture file is missing, altered or unexpected. It is run again as the runtime user, after a check that this user can read every file |
| permissions | host files may be owner-only (600/700). The first build stage sets files to 644 and folders to 755 before they are copied in, so the root-owned app is readable but not writable by uid 10001 |
| process | `deploy/start.sh` → Streamlit on `0.0.0.0:${PORT:-8501}`, headless, usage statistics off, upload limit 30 MB |
| user | uid 10001; only `/app/outputs` (results, benchmarks, recorded references) and `/tmp` are writable |
| health | `GET /_stcore/health` (Docker `HEALTHCHECK`) |
| secrets | none needed. Pass no API keys or credentials into the image or environment |
| image size | roughly 1.3–1.6 GB expected (Paddle, OpenCV, 86 MB of models); not yet measured |

## How it handles requests

These points matter when choosing a host.

- **Uploads.** One photo at a time. JPEG, PNG and HEIC/HEIF are accepted, up to **30 MB**, and Streamlit rejects larger files before OCR. The photo is decoded in memory, EXIF orientation is applied, and it is written to a temporary PNG that is deleted after reading.
  - Nothing is written to disk after the request, and photos are not logged.
  - The photo and its notes stay in that browser session's server memory until the session ends.
- **Model loading.** Lazy, once per process. Both models load on the first **Extract notes**: the reader in about 1 s, the detector in about 0.5 s. The first photo pays the cold-start cost.
  - Instances that scale to zero will therefore have a slow first request.
  - Keep one instance warm, or accept the extra 5–10 s.
- **Concurrency.** All browser sessions in a process share **one** `OCRService` (`st.cache_resource`), and its lock **serialises recognition**.
  - Two students who press Extract together are processed one after the other.
  - The second waits for the first, and the measured total equals the sum of the single runs.
  - This is deliberate: the predictors are never run in parallel, and memory stays bounded.
  - **To serve more people at once, run more instances (processes or containers). Do not add threads.**
- **Sessions.** Streamlit keeps each browser session on a WebSocket. Behind a load balancer with more than one instance, enable **sticky sessions** (session affinity) and WebSocket support. Otherwise a session can land on an instance that does not hold its state.
- **Memory.** Plan for **2 GB per instance**. The VM benchmark peaked at 1.85 GB RSS, and a 4000 px small-text retry is the expensive case.
- **CPU.** Recognition is CPU-only: oneDNN is off on both models, and the reader uses 4 threads. Per-photo time scales with the number of regions; dense boards are the slow case.
  - On the 4-vCPU aarch64 VM: 3–5 s for a typical photo, 15 s for the 38-region Jy 8.
  - On x86 these figures still need measuring (step 3).

## Step 1 — build (local, free)

From the package folder, with Docker Desktop or any Docker engine:

```
docker build --platform linux/amd64 -f deploy/Dockerfile -t board2notes-ocr:2026-11-09 .
```

On an Apple Silicon Mac this builds an amd64 image under emulation. The build works, but timings inside it are **not representative**. Use an x86 machine or the chosen host for step 3.

## Step 2 — run and verify (local, free)

```
docker run --rm -p 127.0.0.1:8501:8501 --name b2n board2notes-ocr:2026-11-09
```

Then open http://127.0.0.1:8501 and do the manual browser check in `VERIFICATION.md`. In a second terminal:

```
docker exec b2n python tools/verify_package.py --smoke --reference mac-arm64
docker exec b2n python tools/e2e_app_check.py
```

`--reference mac-arm64` compares the container's boxes and text with the Mac reference. The command needs `fixtures/expected/mac-arm64.json`, which the Mac step creates; rebuild the image after it exists. To keep the container's own outputs as a reference, record into the writable `outputs/expected/` and copy it out:

```
docker exec b2n python tools/verify_package.py --smoke --record linux-x86_64 --record-dir outputs/expected
docker cp b2n:/app/outputs/expected/linux-x86_64.json fixtures/expected/
```

`fixtures/expected/` inside the image is read-only for the runtime user, by design.

## Step 3 — benchmark (repeatable)

Run the same command locally and on the target host, on an otherwise idle machine:

```
docker exec b2n python tools/benchmark.py --repeats 3 --threads 4 --out outputs/benchmark_container.json
docker cp b2n:/app/outputs/benchmark_container.json outputs/
```

It reports four things:
- **cold start**, in a fresh process: imports, reader load, detector load and first photo, timed separately;
- **warm** median seconds per fixture photo;
- **peak memory**;
- **4 simultaneous requests** on one shared service: the wall time, each request's latency, and the sum of the single runs.

Record the machine's vCPU count and RAM alongside it. Only compare results across machines when the vCPU count is also reported.

The VM reference figures (aarch64, 4 vCPU, Paddle 3.2.2, 1 repeat, 3 threads) are in `outputs/benchmark_linux_aarch64_vm.json`:
- cold first request: 5.5 s in total;
- warm: Jy 23 4.0 s, Jy 8 15.0 s, JH 2 2.8 s, JH 5 5.1 s;
- 3 at once: 23.1 s wall, against 21.8 s summed;
- peak 1.85 GB.

## Step 4 — compare local and cloud outputs

Accuracy is not re-measured here. The check is that the cloud returns *the same output* as the local run. After deploying to the chosen host, run the smoke check **inside** the running instance:

```
python tools/verify_package.py --smoke --reference mac-arm64
```

If the host does not allow a shell, upload the 4 photos in `fixtures/photos/` through the UI. Then compare each "Save my notes" file with the `text` fields in `fixtures/expected/mac-arm64.json`.

Expected result: identical text and region counts. Small box differences (at most 0.01 px) are allowed. Any text difference must be explained before the deployment is used, for example by a different Paddle build or a oneDNN path.

## Step 5 — choose a destination (needs a team decision; may cost money)

These are not run, and there is no recommendation yet. Points to settle first:

- **Who can reach it.** The app has no login. A public URL means anyone can upload photos and use CPU time. For a class demo, prefer a private or expiring link, or an authenticated proxy offered by the host.
- **Instance size.** At least 2 vCPU and 2 GB RAM (4 vCPU recommended), x86_64. **No GPU** is needed: the package is CPU-only.
- **Scale-to-zero** saves money but adds the cold start. A single always-on small instance is simpler for a demo.
- **WebSockets and sticky sessions** must be supported (see above).
- **Image registry.** Pushing the image to a registry is a publishing step. Confirm where it goes and whether it is private.
- **Photos are personal data** (handwriting, possibly faces in the background). Confirm the host's region, and confirm that request logs do not keep uploads.

Before running anything that creates paid resources, write down the chosen host, region, instance size, expected monthly cost and who can access it.

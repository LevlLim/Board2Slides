# Deploy Board2Slides to a Hugging Face GPU Space

This keeps the phone website, photo preparation, OCR, Qwen slide planner, and PowerPoint renderer in one container. The same repository still runs locally with `python scripts/run.py`. A cloud build and end-to-end cloud run must be completed before claiming a successful deployment.

## 1. Create the account and private Space

1. Create and verify a [Hugging Face account](https://huggingface.co/join). A personal [PRO subscription](https://huggingface.co/pro) is currently required to create a Docker Space; add a payment method in your own account. The subscription and GPU usage are billed separately.
2. Open [Create a Space](https://huggingface.co/new-space). Set **Owner** to your account, **SDK** to **Docker**, **Visibility** to **Private**, and hardware to **CPU Basic** initially. The current Space is `Levl07/Board2Notes`. A private Space is accessible only to its owner and permitted collaborators; decide how the team will access it before relying on independent teammate testing.
3. Do not select paid GPU hardware until the application upload and image build are ready. The Dockerfile includes a CUDA version of PyTorch; the pinned public Qwen base downloads when the container starts so it does not inflate the Docker image.

The deployment metadata (`sdk: docker`, `app_port: 8000`) is already at the top of this repository's `README.md`.

## 2. Review and commit the application

Use GitHub Desktop to inspect the changes, commit them, and push to `LevlLim/Board2Slides`. Do not add `models/llm/base`, `data`, `.venv-*`, `node_modules`, `.cache`, or the raw `board2notes_*` folders. The deployment uploader only selects the app, exported OCR models, LoRA adapter, requirements, renderer, and Docker files.

The upload command requires a clean Git working tree. It previews the exact tracked files before writing to the Space. Its Space commit substitutes an LFS-enabled `.gitattributes` for the local GitHub version so the Docker build receives real OCR and adapter weights instead of pointer text.

## 3. Log in and upload from Windows

Open PowerShell and run:

```powershell
cd 'C:\Users\levll\SIT\Year 2\Tri 1\3001 CVDL (6)\Assignment\Board2Slides-App'
.\.venv-llm\Scripts\hf.exe auth login
.\.venv-llm\Scripts\python.exe scripts\deploy_space.py
.\.venv-llm\Scripts\python.exe scripts\deploy_space.py Levl07/Board2Notes --upload
```

The login command opens an authorization flow; do not place a token in a command, Git commit, or chat message. If `.venv-llm` is missing, complete the local `python scripts\setup.py` first.

For a teammate deploying from a Mac clone, run the same last three commands with `.venv-llm/bin/hf`, `.venv-llm/bin/python`, and their own clone path. They must have upload rights to the Space. The uploader rejects a public Space unless `--allow-public` is deliberately supplied.

The first upload is about 264 MiB of application files and exported weights. The 4.1 GB Qwen base downloads when a new cloud container starts, not from your laptop or GitHub repository. Watch **Build logs** and then **Run logs** on `https://huggingface.co/spaces/Levl07/Board2Notes`. The initial build installs PaddleOCR, CUDA PyTorch, and Node packages; startup waits for the Qwen download. Docker Desktop is optional on this Windows computer; the Space builds the image remotely.

## 4. Turn on GPU and set billing controls

When the image builds, open the Space **Settings → Hardware**, choose **1× L4 GPU**, and check the displayed rate before confirming. The listed rate is US$0.80 per running hour as of November 2026. In **Settings → Variables**, set `BOARD2SLIDES_LLM_DEVICE` to `cuda` so the cloud request fails clearly if the GPU is not available rather than silently using the CPU. Set a custom **sleep time** (for example, one hour of inactivity). GPU time is billed while the Space is starting or running; sleeping and manually paused time are not billed. The PRO subscription remains separate.

For a planned demo, open the Space early, wait for it to wake, then run a complete rehearsal. When testing ends, use **Settings → Pause**. A paused Space requires an owner or collaborator to restart it; ordinary visitors cannot wake it. Automatic sleep can be woken by a visit.

## 5. Verify the full cloud pipeline

1. Open the Space while signed in. Check `/api/health`; `ocr_ready`, `llm_ready`, and `renderer_ready` should be `true`, and all three OCR choices should be available. This is a file and executable check, not an inference test.
2. Upload a board photo from a phone, prepare it, select an OCR version, run OCR, review the coloured boxes, and correct the joined text.
3. Run the Qwen planner and export a PowerPoint. Download the deck. Repeat with two ordered photos to verify the multi-photo path.
4. Record the cloud URL, the hardware badge, OCR time, slide-plan time, and a screenshot or short video. For the rubric's local-and-cloud demonstration, also record the same flow from the local laptop server. Stop the laptop server and use phone mobile data to prove the cloud URL is independent of the laptop.

If `/api/health` is ready but an actual request fails, inspect **Run logs** on the Space. Check CUDA availability, model download, memory, and the app error before retrying. The first photo or LLM request tests the actual runtimes.

## Draft persistence and later updates

The current app stores drafts, photos, and generated slides in `data/jobs` on the server. A Space's default disk is **ephemeral**: pausing, sleeping, or restarting it can erase those files. For an initial class demo, create a new draft after waking and download the deck before pausing. If drafts must survive restarts, attach persistent storage and verify its file behaviour with the app before using it for important work.

After a model or code change: commit and push the tested change to GitHub, rerun `scripts/deploy_space.py ... --upload`, watch the Space rebuild, and repeat the health plus full-flow checks. Keep one web worker because the current job revision lock is in process.

Official references: [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker), [GPU prices, sleep, and pause](https://huggingface.co/docs/hub/spaces-gpus), [ephemeral and persistent storage](https://huggingface.co/docs/hub/spaces-storage).

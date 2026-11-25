# Deploy Board2Slides to a Hugging Face Docker Space

This hosts the phone website, photo preparation, OCR, Qwen slide planner, and PowerPoint renderer in one container. Follow the local setup in [README.md](README.md) first. Uploading files starts a rebuild; verify an actual OCR-to-PowerPoint run before reporting the new revision as deployed.

## 1. Create the account and private Space

1. Create and verify a [Hugging Face account](https://huggingface.co/join). Hugging Face currently requires a paid plan to create a Docker Space on CPU Basic; a personal account uses [PRO](https://huggingface.co/pro). Check the current options and billing terms in your account before choosing hardware. GPU usage is charged separately from a subscription. See the [Spaces overview](https://huggingface.co/docs/hub/spaces-overview).
2. Open [Create a Space](https://huggingface.co/new-space). Set **Owner** to your account, **SDK** to **Docker**, **Visibility** to **Private** for team testing, and hardware to **CPU Basic** initially. The existing team Space is `Levl07/Board2Notes`; use it if you have write access. For a different Space, substitute its `USERNAME/SPACE_NAME` in the commands below. A private Space is visible only to its owner and permitted collaborators; a link alone cannot grant a teammate or examiner access.
3. Do not select paid GPU hardware until the application upload and image build are ready. The Dockerfile includes a CUDA version of PyTorch; the pinned public Qwen base downloads when the container starts so it does not inflate the Docker image.

The deployment metadata (`sdk: docker`, `app_port: 8000`) is already at the top of this repository's `README.md`.

## 2. Review and commit the application

From the cloned repository, use GitHub Desktop or Git to inspect, commit, and push tested changes to the team's [GitHub repository](https://github.com/LevlLim/Board2Slides). Do not add `models/llm/base`, `data`, `.venv-*`, `node_modules`, `.cache`, student photos, or the raw `board2notes_*` folders. The deployment uploader selects only the app, exported OCR models, LoRA adapter, requirements, renderer, and Docker files.

The upload command requires all **tracked changes to be committed**. Check `git status --short` and commit the app changes you intend to publish. Untracked local photos are excluded from the deployment manifest and do not block an upload; inspect them separately before any GitHub commit. The command previews the exact tracked files before writing to the Space. Its Space commit substitutes an LFS-enabled `.gitattributes` for the GitHub version so the Docker build receives real OCR and adapter weights instead of pointer text.

## 3. Log in and upload from Windows

Open PowerShell in your clone. Replace the example folder and Space ID with your own values. If you already opened a terminal in the repository folder, skip the `cd` line.

```powershell
cd 'C:\path\to\Board2Slides'
.\.venv-llm\Scripts\hf.exe auth login
.\.venv-llm\Scripts\python.exe scripts\deploy_space.py
$spaceId = 'YOUR_USERNAME/YOUR_SPACE_NAME'
.\.venv-llm\Scripts\python.exe scripts\deploy_space.py $spaceId --upload
```

The login command opens an authorization flow; do not place a token in a command, Git commit, or chat message. If `.venv-llm` is missing, complete `python scripts\setup.py` from [README.md](README.md) first. The preview command prints the deployment manifest and does not upload anything. Use `--upload` only when ready to update the Space.

On a Mac, open Terminal in the cloned repository and use:

```bash
./.venv-llm/bin/hf auth login
./.venv-llm/bin/python scripts/deploy_space.py
./.venv-llm/bin/python scripts/deploy_space.py YOUR_USERNAME/YOUR_SPACE_NAME --upload
```

The person uploading must have write access to the Space. The uploader rejects a public Space unless `--allow-public` is deliberately supplied.

Expect an upload of several hundred MiB of application files and exported weights. The 4.1 GB Qwen base downloads when a new cloud container starts, not from your computer or GitHub repository. Watch **Build logs** and then **Run logs** at `https://huggingface.co/spaces/USERNAME/SPACE_NAME`. The first build installs PaddleOCR, CUDA PyTorch, and Node packages; startup waits for the Qwen download. Docker Desktop is optional because Hugging Face builds the image remotely.

## 4. Turn on GPU and set billing controls

When the image builds, open the Space **Settings → Hardware** and choose a GPU with enough memory. The team previously completed OCR, LLM planning, and PowerPoint export on **1× A10G small**; availability and price can change, so check the current selection before confirming. In **Settings → Variables**, set `BOARD2SLIDES_LLM_DEVICE` to `cuda` so the cloud request fails clearly if the GPU is unavailable rather than silently using the CPU. Set a custom **sleep time** (for example, one hour of inactivity). Hugging Face bills GPU time while the Space is Starting or Running, by the minute; sleeping and manually paused time are not billed. See the [GPU billing guide](https://huggingface.co/docs/hub/spaces-gpus).

For a planned demo, open the Space early, wait for it to wake, then run a complete rehearsal. When testing ends, use **Settings → Pause**. A paused Space requires an owner or collaborator to restart it; ordinary visitors cannot wake it. Automatic sleep can be woken by a visit.

## 5. Verify the full cloud pipeline

1. Open the Space's **App** tab while signed in. Copy its direct App URL and append `/api/health`. `ocr_ready`, `llm_ready`, and `renderer_ready` should be `true`, and all three OCR choices should be available. This is a file and executable check, not an inference test.
2. Upload a board photo from a phone, prepare it, select an OCR version, run OCR, review the coloured boxes, and correct the joined text.
3. Run the Qwen planner and export a PowerPoint. Download the deck. Repeat with two ordered photos to verify the multi-photo path.
4. Record the Space App URL, the hardware badge, OCR time, slide-plan time, and a screenshot or short video. For the local-and-cloud demonstration, also record the same flow from the local laptop server. Stop the laptop server and use phone mobile data to show that the cloud App URL is independent of the laptop. A private Space requires a signed-in collaborator; arrange examiner access or change visibility deliberately before sharing the link.

If `/api/health` is ready but an actual request fails, inspect **Run logs** on the Space. Check CUDA availability, model download, memory, and the app error before retrying. The first photo or LLM request tests the actual runtimes.

## Draft persistence and later updates

The current app stores drafts, photos, and generated slides in `data/jobs` on the server. A Space's default disk is **ephemeral**: pausing, sleeping, or restarting it can erase those files. For an initial class demo, create a new draft after waking and download the deck before pausing. If drafts must survive restarts, attach persistent storage and verify its file behaviour with the app before using it for important work.

After a model or code change: commit and push the tested change to GitHub, rerun `scripts/deploy_space.py ... --upload`, watch the Space rebuild, and repeat the health plus full-flow checks. Keep one web worker because the current job revision lock is in process.

Official references: [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker), [GPU prices, sleep, and pause](https://huggingface.co/docs/hub/spaces-gpus), [ephemeral and persistent storage](https://huggingface.co/docs/hub/spaces-storage).

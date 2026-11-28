---
title: Board2Slides
emoji: 📷
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
---

# Board2Slides

A phone-friendly browser app that turns whiteboard photos from one lesson into an editable PowerPoint. The working path is **photos → reviewed OCR text → Qwen slide-plan JSON → editable slides**. A teacher or student can correct the text and slide draft before export.

This repository contains the runnable application, inference exports, deployment files, and the team's OCR research records. LLM training notebooks and their raw evaluation outputs are kept in the team's separate research workspace; the inference adapter used by this app is included here.

**Start here:** Install Python 3.12 or 3.13 and Node.js 22 or newer, clone this repository, run `scripts/setup.py` once, then run `scripts/run.py`. Use the platform-specific commands below. Setup needs internet access and about 10 GB of free disk space for dependencies and the Qwen base model.

- [Windows: exact setup and run commands](#windows-setup-and-run)
- [Apple Silicon Mac: exact setup and run commands](#apple-silicon-mac-setup-and-run)
- [Phone or second computer: open the local website](#open-on-the-phone-or-another-computer)
- [Hugging Face: deploy after local changes are ready](DEPLOY.md)
- [Why this OCR + LLM design?](#why-ocr--llm-instead-of-only-a-vision-language-model)
- [How the slide planner was tuned and evaluated](#slide-planner-development-and-evaluation)
- [How JSON becomes an editable PowerPoint](#from-slide-plan-json-to-powerpoint)

## What happens in the app

1. **Capture:** On a phone or laptop, take or upload one to four ordered JPG, PNG, or WebP photos from the same lesson. Reorder or remove photos before continuing; start another draft for a different lesson.
2. **Prepare:** Rotate a sideways photo with the arrow buttons if needed. Automatic preparation checks perspective, small tilt (2–15°), lighting, and a crop around writing. The original and prepared photos appear side by side with reasons for applied or skipped steps. Individual modes let the team compare one correction at a time. These checks are experimental; blemish removal is not enabled.
3. **Read:** Choose V1, V2, or V3 before running OCR. All use the official PP-OCRv5 **mobile detector**. V1 has an untuned server recognizer; V2 has the team's fine-tuned 94-character English server recognizer; V3 has a larger-character-dictionary recognizer. V2 cannot produce characters outside its dictionary, and V3's training record alone does not establish accuracy in other languages.
4. **Check text:** Each coloured OCR box matches an editable line. Correct mistakes and add missing lines. The displayed percentage is the recognizer's **original score**, even after an edit; it is not a measured probability that the whole line is correct. In **Joined text**, review the order and wording sent to the LLM; groups can be edited, split, merged, or moved within the same photo while retaining source box IDs. The sample lesson and manual entry also work without OCR.
5. **Review slides:** Qwen3-1.7B with the adaptive-visual LoRA v3 converts the saved text from all photos into **one** slide plan. Review the cover, key points, optional visual, and activity. Edit content, regenerate an LLM draft, or remove/restore a slide. If no supported visual exists, the redundant visual slide starts removed. A separate text-only demo planner is available explicitly; it is not silently substituted for an LLM failure.
6. **Export:** Save the plan and download one editable `.pptx`. Charts, tables, processes, and formulas are generated only when the reviewed text supports them; the user still has to check academic accuracy.

Drafts live on the server in `data/jobs`. A link or QR code opens the same draft on another device connected to that server. Save on one device and press **Refresh from server** on the other; editing is not live-synced. Anyone with a reachable draft link can edit it, so use this prototype only with your team.

## OCR dataset

The team's OCR training dataset is hosted as [whiteboard-notes-v1 on Hugging Face](https://huggingface.co/datasets/TYW02/whiteboard-notes-v1). The slide-planner training pairs are a separate dataset, described below.

## Model development: the OCR reader

The app's text reader is PaddleOCR's `PP-OCRv5_server_rec`, fine-tuned for whiteboard handwriting. Training scripts,
every run's `config.yml`, logs and scored predictions are in [`model_development/`](https://github.com/LevlLim/Board2Slides/blob/main/model_development/README.md),
which also maps every poster number to its source file. Checks of the app's OCR on whole photos are in
[`app_diagnosis/`](https://github.com/LevlLim/Board2Slides/blob/main/app_diagnosis/README.md). Reader weights: Release
[Reader_V1](https://github.com/LevlLim/Board2Slides/releases/tag/Reader_V1).

Error = mean line character error rate, lower is better. The pooling and learning-rate changes were each tested
against a control that differs only in that change.

| Change from PaddleOCR's setup | Why | Result |
|---|---|---|
| Fine-tune on our whiteboard line crops | The downloaded reader was not trained on whiteboard handwriting | Whiteboard validation 15.5% → 8.9% |
| Wider input (48×320 → 64×1024) and width pooling in training: one CTC column per 8 px (128) instead of a fixed 40 | With 40 columns, long lines have too few positions for CTC | Long lines 45.3% → 4.5% (same lines and input, only the pooling differs) |
| Label limit `max_text_length` 25 → 100 (all runs) | Training skips labels longer than the limit | Long board lines can be trained on |
| Stage-wise learning rate for the backbone stages `[0.1, 0.1, 0.25, 0.5, 1.0]` | Early layers keep their general features | New writers 26.8% → 24.9% (2 runs each) |
| Final model: the same settings on the team dataset | More writers and lines | Held-out test lines 16.9% (downloaded) → 7.0% |
| English 94-character dictionary instead of 18,383 characters (app V2) | Smaller reader that only outputs English | 7.4% on the same test lines (final model 7.0% on those lines); weights 75.5 MB instead of 84.4 MB |

Tried and not adopted: letting the CTC loss train the backbone (by default only the NRTR loss does). Routing it into
the late layers read new writers and fresh boards slightly better (26.7% → 26.0%, 11.2% → 10.4%), but within the
difference between two runs of the same recipe (`model_development/experiments/3_*` and `4_*`).

Whole photos through the app (44 photos, five boards): among the lines found, characters read right were **78% with the downloaded reader**, **87% with the final full-dictionary reader**, and **90% with the English V2 reader**. Missed lines are measured separately in the [whole-photo report](https://github.com/LevlLim/Board2Slides/blob/main/app_diagnosis/whole_photo_test_20261125/README.md); these figures are not whole-page accuracy.

To reproduce a run: each experiment's `source/` folder holds the exact training and scoring script, and each
`runs/<run>/config.yml` is the PaddleOCR config it trained with. Training needs a GPU (we used RunPod) and the
dataset above; our own earlier whiteboard crops are not public. The runner scripts expect the RunPod volume layout
described in each script's header, so they record exactly what ran rather than being a one-command setup.

The [whole-photo app check](https://github.com/LevlLim/Board2Slides/blob/main/app_diagnosis/whole_photo_test_20261125/README.md) also tests the preprocessing decision. On its 44 photos from five boards, V2's page error was **28.4% with the original photos** and **34.7% after automatic preparation**. This small test does not establish that preparation is always harmful, but it is why the app shows both images, logs each decision, and lets users keep the original or choose an individual correction. The automatic mode is experimental rather than a guaranteed accuracy improvement.

## Why OCR + LLM instead of only a vision-language model?

An early research comparison transcribed the **same 38 images and 65 manually labelled lines** with these exact models:

- **OCR baseline:** PaddleOCR `PP-OCRv5_server_det` (detector) + `PP-OCRv5_server_rec` (recognizer), both pretrained. This was the research baseline, not the mobile detector and fine-tuned reader now used in the app.
- **Vision-language model:** `Qwen/Qwen3-VL-2B-Instruct`, run in BF16 with deterministic decoding.

The experiment used the same raw images, ground truth, and scoring rules for both systems. It measured transcription, not slide quality or the current fine-tuned V2 reader:

| Transcription metric | Pretrained PP-OCRv5 server detector + reader | Qwen3-VL-2B-Instruct |
| --- | ---: | ---: |
| Exact lines | 47.7% | 73.8% |
| Character error rate (lower is better) | 40.9% | 27.9% |

The VLM was the stronger transcriber in that experiment. We kept a modular OCR → LLM design because the app needs **per-region polygons, reading order, and an editable link from each text box to its source photo**. PaddleOCR exposes those directly, and its detector and recognizer can be evaluated or replaced separately. The LLM then works from reviewed text rather than guessing both handwriting and slide content in one opaque step. This is a design choice for traceability and human correction, **not a claim that our OCR is more accurate than a VLM**. A VLM remains a possible fallback for difficult regions. These early baseline numbers must not be compared directly with the teammate's later V2 whole-photo results above. The raw VLM experiment and evaluation report remain in the separate research workspace, outside this deployable repository.

## Slide-planner development and evaluation

The application uses the included **Qwen3-1.7B adaptive-visual LoRA v3** adapter. We fine-tuned the text model with supervised pairs of OCR-style `regions` and target slide-plan JSON. The adapter has rank 16 and trains about **6.42 million of 1.73 billion parameters**; the base model is unchanged. The recorded v3 experiment used **352 curated training pairs, 47 validation pairs, and an 18-case held-out set** with three cases for each of six visual choices: no visual, process, comparison table, formula, line chart, and bar chart. These are mostly authored teaching examples, not a corpus of instructor-approved full-board plans. The paired examples and evaluation records are kept in the team's research workspace rather than this application repository.

The model is asked to produce a title, short summary, two to four distinct key points, an activity/example, a visual decision, and uncertain OCR terms. A visual must cite real source-region IDs; charts require the board's actual label–value pairs, and a formula needs an explicit matching equation. If the input does not support a useful visual, the correct choice is `none`.

The following results compare v2 and v3 on the *same new, balanced 18-case holdout*. The first three rows measure raw model output; the last measures how many unmodified outputs passed the content check. Runtime repairs are excluded from every row:

| Held-out check | LoRA v2 | Shipped LoRA v3 |
| --- | ---: | ---: |
| Valid raw JSON | 18/18 | 18/18 |
| Strict slide-plan schema | 13/18 | 16/18 |
| Intended visual type | 17/18 | 18/18 |
| Passed content checks without repair | 9/18 | 15/18 |

For v3, deterministic repairs handled the other three cases, giving **18/18 passes for the guarded pipeline** on this set; that is **not** 18/18 raw model accuracy. On a *different earlier* 18-case holdout, untuned base Qwen passed the strict schema in 0/18 cases and chose the intended visual in 4/18; v2 improved those to 17/18 and 15/18. The two holdouts should not be combined into one accuracy claim.

These controlled cases exposed structural progress, but real board content is harder. On **11 photo-level annotation inputs** (not final OCR predictions or approved slide-plan labels), v3 passed a basic schema check in **8/11** cases and chose `none` for all 11. Human review found generic or repeated explanations and some misleading claims. Content-focused v4 and v5 experiments improved individual cases but introduced unsupported numbers or visuals, so **v3 remains the shipped adapter**. The app consequently keeps OCR and slide editing, warnings, evidence checks, and user approval in the workflow. Future tuning needs more independent, human-reviewed full-lesson examples and a held-out test that measures factual usefulness, photo coverage, repetition, and edit time, not JSON validity alone.

## From slide-plan JSON to PowerPoint

1. The app keeps each OCR region's ID, corrected text, photo, and bounding box. It joins nearby lines into readable lesson text while retaining their source IDs; users can edit that joined view before planning.
2. Qwen receives the saved text from all photos **once** and returns one JSON slide plan. `app/app.py` checks its structure and source references. Syntax repair is labelled separately; unsupported visual details are repaired only when they follow from cited evidence or the visual is removed.
3. [`renderer/adaptive_visual_renderer.js`](renderer/adaptive_visual_renderer.js) uses **PptxGenJS** to build an editable cover, key-points slide, optional visual slide, and activity slide. [`renderer/slide_plan_safety.js`](renderer/slide_plan_safety.js) checks explicit chart pairs, prose numbers, and equations against the reviewed text before export. The preview lets users edit or omit slides; exporting applies those saved choices.

These checks prevent some unsupported numbers and malformed plans. They cannot prove that every academic explanation is correct, so the final deck remains a draft for human review.

## Requirements

- **Windows 64-bit** or **Apple Silicon macOS**. The Mac setup has not yet been physically tested by the team. Intel Macs can use the browser against a Windows or deployed server; local model installation may need a different runtime.
- **Python 3.12 or 3.13**, **Node.js 22+** with npm, and internet for the first setup.
- About **10 GB free disk space** for Python packages and the Qwen base model. A CUDA GPU on Windows or Apple GPU on Mac is preferred. CPU LLM generation can be very slow and needs substantial RAM; use the demo planner if the LLM is unavailable.
- The `models/ocr/detector`, `models/ocr/baseline_rec`, `models/ocr/recognizer`, `models/ocr/augmented_rec`, and `models/llm/adapter` folders supplied in this repository.

The Qwen base weights are about 4.1 GB and exceed GitHub's normal file limit. They are **not included in the repository**. `scripts/setup.py` downloads the pinned [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) base into the ignored `models/llm/base` folder. The exported OCR and LoRA weights are included. Do not force-add the base weights, virtual environments, `node_modules`, or student photos to Git.

Before committing, run `git status --short` **from this folder** and check the staged file list. `models/llm/base`, `.venv-*`, `node_modules`, `.cache`, and `data` should stay out of commits. `.gitignore` handles this for normal Git commands; uploading the entire folder through a browser file picker can bypass it.

Teammates should **clone the repository**, then follow the Windows or Mac setup below. The documented `model_development/` and `app_diagnosis/` folders contain the team's existing OCR research; they are **not loaded by the app**. Keep new training datasets, large checkpoints, private photos, and temporary outputs out of the application runtime paths. Commit only a tested inference export when updating a model. For UI or model changes, use a branch so teammates can review the change before merging:

```bash
git clone https://github.com/LevlLim/Board2Slides.git
cd Board2Slides
git switch -c feature/my-change
# Edit files, then inspect exactly what will be shared.
git status --short
git add app/static    # or the specific model/code files you changed
git commit -m "Describe the change"
git push -u origin feature/my-change
```

Open a pull request on GitHub to merge the branch into `main`. Before starting new work later, run `git switch main` and `git pull`. A teammate needs repository access to push a branch to this GitHub repository; otherwise they can fork it and propose a pull request.

## Windows setup and run

Open PowerShell and run:

```powershell
git clone https://github.com/LevlLim/Board2Slides.git
cd Board2Slides
python --version
node --version
npm --version
python scripts\setup.py
python scripts\run.py
```

If you already cloned with GitHub Desktop, open PowerShell in that clone's folder and start at `python --version`; do not clone a second copy. The first setup creates `.venv-web`, `.venv-ocr`, and `.venv-llm`, installs the locked Node packages, and downloads the pinned Qwen base if it is absent. It can take a while. Subsequent starts need only `python scripts\run.py` from the repository folder. Use Ctrl+C to stop.

When the server says it is ready, open `http://localhost:8000` on that computer.

For a quick website and slide-renderer run without installing the models, use `python scripts\setup.py --demo-only`, then `python scripts\run.py`. The health page will show OCR and LLM as unavailable until full setup is completed. For uploaded photos, choose **Use original upright photo**, press **Prepare photos**, then **Enter board text manually instead**. Use **Demo planner** to export slides.

If PowerShell says `python` is not found, install Python 3.12 or 3.13 and select **Add Python to PATH**. Check with `python --version`. Check Node with `node --version` and `npm --version`. `setup.py` installs CUDA PyTorch on a Windows computer with `nvidia-smi`; otherwise it installs CPU PyTorch.

## Apple Silicon Mac setup and run

Open Terminal and run:

```bash
git clone https://github.com/LevlLim/Board2Slides.git
cd Board2Slides
python3 --version
node --version
npm --version
python3 scripts/setup.py
python3 scripts/run.py
```

If you already cloned with GitHub Desktop, open Terminal in that clone's folder and start at `python3 --version`. Use a Python 3.12 or 3.13 executable. If `python3 --version` reports another version, use the appropriate executable (for example `python3.12`) for both setup and run. The setup installs CPU PaddleOCR and Mac PyTorch; the app selects MPS for Qwen when available. The first setup downloads the same pinned Qwen base. The Mac path still needs a real run on a teammate's Mac; if a package install or MPS operation fails, run the website on the Windows laptop and open it from the Mac browser while the team resolves that platform issue.

When the server says it is ready, open `http://localhost:8000` on that Mac.

## Open on the phone or another computer

The server prints `http://localhost:8000` for the computer running it. A second device needs the **server computer's Wi-Fi IP address**, not its own. Both devices must be on a network that allows them to reach each other.

- Windows server: run `ipconfig` and find **Wi-Fi → IPv4 Address**.
- Mac server: run `ipconfig getifaddr en0` in Terminal. If that is blank, check the active network interface in System Settings.
- On the phone, open `http://SERVER-IP:8000`, replacing `SERVER-IP` with that address. If direct camera capture is unavailable over local HTTP, use **Choose a photo**. A public deployment should use HTTPS.

Open `http://localhost:8000/api/health` on the server computer. For the full local pipeline, `ocr_ready`, `llm_ready`, and `renderer_ready` should all be `true`. These checks confirm files and executables; the first real photo and LLM request also test their runtimes.

## Folder map and team handoff

| Path | Change it when… |
| --- | --- |
| `app/static/` | You redesign the phone and laptop website. No frontend build step is needed. |
| `app/app.py` | You change uploads, jobs, review, API routes, or model orchestration. |
| `app/ocr_infer.py` and `app/ocr_mapping.py` | You change model selection, PaddleOCR inference, or its output mapping. |
| `app/lesson_lines.py` | You change how OCR boxes and manual lines are ordered and joined for the adaptive LLM. The original box IDs and corrected text stay in the job. |
| `app/preprocessing/` and `app/prepare_infer.py` | You change photo preparation and decision logs. The individual operations came from the team's `pipeline.ipynb`; this repository adds the automatic sequence that gates and combines them. The notebook is not needed to run the application. |
| `models/ocr/baseline_rec/` | V1: the downloaded, untuned [PaddlePaddle PP-OCRv5 server reader](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec), copied from the team's `00_downloaded_PP-OCRv5_server_rec` export. The three inference files are included here for teammates. |
| `models/ocr/recognizer/` | V2: the 94-character English server reader exported from the teammate's `board2notes_reader_en94_20261119` package. Only the inference files are needed here. |
| `models/ocr/augmented_rec/` | V3: the supplied reader exported from `board2notes_multilang_reader`, distinct from V2. |
| `models/ocr/detector/` | The official untuned PP-OCRv5 mobile detector export. |
| `app/llm_infer.py` and `models/llm/adapter/` | You replace the LoRA model. Keep its prompt and JSON schema aligned with training. |
| `renderer/` | You change the actual PowerPoint slide design or evidence checks. |
| `requirements/`, `package.json`, `package-lock.json` | You change runtime dependencies. |
| `scripts/` | You change setup, base-model download, or startup behavior. |
| `model_development/`, `app_diagnosis/`, `preprocessing_and_super_resolution/` | You inspect the team's OCR training, whole-photo checks, and preprocessing experiments. These are research records, not runtime dependencies or part of the Space upload. |

The shipped adapter expects the pinned **Qwen3-1.7B** base and the JSON fields described in [slide-planner development](#slide-planner-development-and-evaluation). A replacement adapter must keep its training prompt, output schema, and all six renderer layouts aligned. JSON syntax repair and warnings about repetition or photo coverage are review aids, not proof of correctness.

To try a newly exported recognizer, stop the server, replace `inference.json`, `inference.pdiparams`, and `inference.yml` together in the corresponding `models/ocr/` folder, restart, then choose **Change model or preparation** on an existing draft or upload new photos. All three current readers are PP-OCRv5 **server** recognition exports; the detector remains the official **mobile** model. A training checkpoint (`.pdparams`) is not an inference export (`.pdiparams`). Rerunning OCR replaces that draft's old text and invalidates its old slide plan. Keep a backup of a working model before replacing team files.

For a fair model comparison, prepare the **same photos in the same mode**, run V1, record the result, then run V2 or V3. Switching versions replaces the draft's OCR text. V1 is the untuned downloaded server reader, V2 is the fine-tuned 94-character English reader, and V3 is the supplied reader with a larger character dictionary. V2 cannot produce characters outside its English dictionary. Report accuracy only against human-checked labels on held-out photos; high OCR confidence is not proof that a word is correct. The teammate's 94-character reader report lists 16.9% test CER for the downloaded reader, 7.4% for that 94-character tuned reader, and 7.0% for the 10 November final reader on eligible line crops; those are recognizer crop results, not whole-photo pipeline accuracy.

For other model locations or Python environments, set `BOARD2SLIDES_OCR_PYTHON`, `BOARD2SLIDES_OCR_DET_DIR`, `BOARD2SLIDES_OCR_REC_DIR` (V2 only), `BOARD2SLIDES_OCR_DEVICE`, `BOARD2SLIDES_LLM_PYTHON`, `BOARD2SLIDES_LLM_BASE`, `BOARD2SLIDES_LLM_ADAPTER`, or `BOARD2SLIDES_LLM_DEVICE` before starting. OCR defaults to CPU. LLM device `auto` chooses CUDA, then Apple MPS, then CPU.

## Cloud deployment and measured checks

The team deployed the same phone website and server pipeline in a [Hugging Face Docker Space](https://levl07-board2notes.hf.space/). The Docker container hosts FastAPI and the static website, the exported PaddleOCR models, the Qwen adapter, and the Node/PptxGenJS renderer. The 4.1 GB pinned Qwen base is downloaded at container startup instead of being committed to Git or baked into the image. On the tested **A10G small** setting, Qwen uses CUDA automatically; OCR currently stays on CPU. **CPU Basic** keeps the website available without a paid GPU, but LLM generation can be slow. GPU availability, price, and startup time depend on Hugging Face's current service.

One cloud smoke check on **26 November 2026** completed the following stages. These are user-facing request times from *different test inputs*, not a controlled local-versus-cloud benchmark or an end-to-end latency total:

| Cloud check on A10G small | Time | Device / outcome |
| --- | ---: | --- |
| Automatic preparation, one demo photo | 9.3 s | CPU; completed |
| V2 OCR, same prepared photo | 13.3 s | CPU; three text regions returned |
| Qwen v3 slide plan, separate sample lesson | 18.8 s | A10G GPU; plan created |
| PowerPoint export, sample lesson | Completed | Editable `.pptx` downloaded; time not separately recorded |

The app and its models were tested on the hosted server; this is stronger evidence than a healthy `/api/health` response alone. It does not establish average latency or OCR/slide accuracy. Hardware switches and restarts can erase drafts because the current `data/jobs` storage is ephemeral. Start a new draft after a restart, and download the deck before pausing the Space.

The Space's **Protected** access setting exposes the running app through its direct HTTPS link while keeping its Space source repository private. A shared draft link opens one server-side job on another device; the laptop must open that draft link or refresh a draft it already has open. Editing is not live-synced. The Space does **not** automatically follow GitHub commits: follow [DEPLOY.md](DEPLOY.md) for a deliberate upload and full workflow check after code or model changes. The uploader includes the runtime files and this README, but excludes the research folders; the research links above therefore point to GitHub.

This remains a **single-server prototype**. It stores jobs as JSON files, uses an in-process revision lock, and has no user accounts, retention policy, or storage quota. Run one web worker. For broader public use, add access control for uploaded photos and drafts, storage limits and cleanup, persistent shared storage, and runtime monitoring. The renderer currently does not embed uploaded photos in the slides. Loading the OCR and LLM models as persistent workers is a possible later speed improvement; the current JSON interfaces in `ocr_infer.py` and `llm_infer.py` leave room for that separation.

## What has been checked

The Windows development copy completed a two-photo OCR-to-PowerPoint flow, and the photo-rotation controls were checked at desktop and phone widths. V2's three exported files match the teammate's 94-character reader by SHA-256; a local OCR run loaded it and returned text from a demo photo. The shipped LoRA adapter also matches the saved v3 research export by SHA-256. The cloud smoke checks above exercised preparation, OCR, LLM planning, and export. These are functionality checks, **not** proof of academic accuracy, reliable automatic correction, or Mac compatibility. Perspective correction can skip when camera geometry is insufficient; small-tilt correction can skip without enough independent text rows. The LoRA can repeat an explanation or underuse one photo, so review the text and every slide before export. Recheck the complete flow after changing a model or deploying.

Official model sources: [PP-OCRv5 mobile detector](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det), [PP-OCRv5 server recognizer](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec), [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). Verify the licenses when publishing your team's repository.

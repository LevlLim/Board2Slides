---
title: Board2Slides
emoji: 📷
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
---

# Board2Slides

A phone-friendly browser app that turns one lesson's board photos into one editable PowerPoint. This repository contains the standalone application, exported inference models, and deployment files. Training experiments and the teammate's source model folders are not needed to run it.

**Start here:** Install Python 3.12 or 3.13 and Node.js 22 or newer, clone this repository, run `scripts/setup.py` once, then run `scripts/run.py`. Use the platform-specific commands below. Setup needs internet access and about 10 GB of free disk space for dependencies and the Qwen base model.

- [Windows: exact setup and run commands](#windows-setup-and-run)
- [Apple Silicon Mac: exact setup and run commands](#apple-silicon-mac-setup-and-run)
- [Phone or second computer: open the local website](#open-on-the-phone-or-another-computer)
- [Hugging Face: deploy after local changes are ready](DEPLOY.md)

## What happens in the app

1. Open the website on a phone or laptop and take or upload **one to four ordered JPG, PNG, or WebP photos from the same lesson**. Reorder or remove photos before continuing. Start a separate draft for a different lesson.
2. In **Prepare photos**, use the arrows below each photo to rotate sideways or upside-down writing until it reads upright. The preview turns immediately; press **Prepare photos** to apply the choice before OCR. The default **Automatic preparation** then checks each photo for supported perspective, small tilt (2–15°), and lighting corrections, and crops around detected writing. Small tilt straightens slightly slanted upright writing; it does not replace the 90° photo rotation buttons. More than one correction may be applied to a photo. Compare each oriented photo with the prepared result and read the applied/skipped decisions. The individual modes remain available for the class demo; each runs independently from the oriented photo, rather than stacking corrections. These checks are experimental, and blemish removal is not enabled.
3. Select an OCR recognizer **before running OCR**. All three choices use the official PP-OCRv5 mobile detector. V1 uses the downloaded, untuned PP-OCRv5 server reader as the baseline. V2 uses the team's fine-tuned 94-character English server reader; it cannot output Chinese or characters outside its dictionary. V3 uses the supplied reader with a larger character dictionary. V2 and V3 are now different inference exports. V3's supplied training record describes English handwriting, not measured performance in other languages.
4. Review each photo's coloured text boxes beside matching editable lines. Add missing lines and correct recognition mistakes. The percentage remains the **original OCR confidence**, even after editing. The separate **joined text** section is the exact text and order passed to the LLM: edit a group, split it into source boxes, merge it with the previous group, or move it within the same photo, then save. Every group keeps its source box IDs. Saving later OCR corrections resets the joined groups so they can be checked again. The sample lesson and manual text entry still work when OCR is unavailable.
5. Choose the **Qwen3-1.7B + adaptive-visual LoRA v3** planner or the explicit text-only demo planner. The LLM receives the saved joined text from all photos for **one** slide plan and **one** editable PowerPoint. Review the proposed content and visual choice before export. In **Review slides**, use the slide tabs or Previous/Next buttons to inspect the cover, key points, visual, and activity. **Remove slide** omits the selected slide from the downloaded PowerPoint; **Restore slide** reverses it. At least one slide must remain. When no supported visual exists, the repetitive visual recap starts removed, giving a three-slide deck by default. Save the draft to share these choices with another device. The warning card calls out repeated explanations, possible missing photo coverage, and any format or visual repair. After editing and saving, the quality warnings are recalculated. **Regenerate LLM draft** gives the model feedback from the prior draft and replaces it only if generation and safety checks pass. It can still produce a weak or similar draft. Recoverable JSON syntax errors are repaired and flagged; an unsupported visual is replaced with an explanation-only choice and flagged. The app never substitutes the demo planner for failed LLM output. The renderer may use a process, comparison table, formula, line chart, or bar chart only when the reviewed text supports it.

Drafts live on the server in `data/jobs`. A link or QR code opens the same draft on another device connected to that server. Save on one device and press **Refresh from server** on the other; editing is not live-synced. Anyone with a reachable draft link can edit it, so use this prototype only with your team.

## Dataset 
Our training dataset hosted [here](https://huggingface.co/datasets/TYW02/whiteboard-notes-v1) on hugging face.

## Requirements

- **Windows 64-bit** or **Apple Silicon macOS**. The Mac setup has not yet been physically tested by the team. Intel Macs can use the browser against a Windows or deployed server; local model installation may need a different runtime.
- **Python 3.12 or 3.13**, **Node.js 22+** with npm, and internet for the first setup.
- About **10 GB free disk space** for Python packages and the Qwen base model. A CUDA GPU on Windows or Apple GPU on Mac is preferred. CPU LLM generation can be very slow and needs substantial RAM; use the demo planner if the LLM is unavailable.
- The `models/ocr/detector`, `models/ocr/baseline_rec`, `models/ocr/recognizer`, `models/ocr/augmented_rec`, and `models/llm/adapter` folders supplied in this repository.

The Qwen base weights are about 4.1 GB and exceed GitHub's normal file limit. They are **not included in the repository**. `scripts/setup.py` downloads the pinned [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) base into the ignored `models/llm/base` folder. The exported OCR and LoRA weights are included. Do not force-add the base weights, virtual environments, `node_modules`, or student photos to Git.

Before committing, run `git status --short` **from this folder** and check the staged file list. `models/llm/base`, `.venv-*`, `node_modules`, `.cache`, and `data` should stay out of commits. `.gitignore` handles this for normal Git commands; uploading the entire folder through a browser file picker can bypass it.

Teammates should **clone the repository**, then follow the Windows or Mac setup below. Keep experiments, training datasets, checkpoints, and evaluation outputs outside this clone. Commit only a tested inference export when updating a model. For UI or model changes, use a branch so teammates can review the change before merging:

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

The current LoRA adapter expects the pinned **Qwen3-1.7B** base and produces JSON with `title`, `summary`, `key_points`, `example`, `visual`, and `uncertain_ocr_terms`. The renderer supports `none`, `process`, `comparison_table`, `formula`, `line_chart`, and `bar_chart`. Replacing the adapter with another training run can require updating the prompt and checking every renderer layout. The JSON repair step only fixes formatting when it can recover all required fields; the schema and PowerPoint evidence checks still run. The app flags repeated explanations and possible missing coverage of a photo, but these are review prompts rather than proof of correctness. Review the plan even when it passes: formatting checks cannot guarantee good explanations. To improve the model, collect human-reviewed multi-photo OCR input paired with strong slide plans, keep entire lessons separated between training and evaluation, and compare base Qwen against each adapter on JSON validity, photo coverage, support for numerical claims, repetition, and human edit time.

To try a newly exported recognizer, stop the server, replace `inference.json`, `inference.pdiparams`, and `inference.yml` together in the corresponding `models/ocr/` folder, restart, then choose **Change model or preparation** on an existing draft or upload new photos. All three current readers are PP-OCRv5 **server** recognition exports; the detector remains the official **mobile** model. A training checkpoint (`.pdparams`) is not an inference export (`.pdiparams`). Rerunning OCR replaces that draft's old text and invalidates its old slide plan. Keep a backup of a working model before replacing team files.

For a fair model comparison, prepare the **same photos in the same mode**, run V1, record the result, then run V2 or V3. Switching versions replaces the draft's OCR text. V1 is the untuned downloaded server reader, V2 is the fine-tuned 94-character English reader, and V3 is the supplied reader with a larger character dictionary. V2 cannot produce characters outside its English dictionary. Report accuracy only against human-checked labels on held-out photos; high OCR confidence is not proof that a word is correct. The teammate's 94-character reader report lists 16.9% test CER for the downloaded reader, 7.4% for that 94-character tuned reader, and 7.0% for the 10 November final reader on eligible line crops; those are recognizer crop results, not whole-photo pipeline accuracy.

For other model locations or Python environments, set `BOARD2SLIDES_OCR_PYTHON`, `BOARD2SLIDES_OCR_DET_DIR`, `BOARD2SLIDES_OCR_REC_DIR` (V2 only), `BOARD2SLIDES_OCR_DEVICE`, `BOARD2SLIDES_LLM_PYTHON`, `BOARD2SLIDES_LLM_BASE`, `BOARD2SLIDES_LLM_ADAPTER`, or `BOARD2SLIDES_LLM_DEVICE` before starting. OCR defaults to CPU. LLM device `auto` chooses CUDA, then Apple MPS, then CPU.

## Deploying later

Follow [DEPLOY.md](DEPLOY.md) for the private Hugging Face Docker Space setup, exact upload commands, GPU billing controls, and cloud smoke test. The Dockerfile builds separate web, OCR, and CUDA-enabled LLM environments. The pinned Qwen base downloads when the cloud container starts.

This is a **single-server prototype**. It currently stores jobs as JSON files, uses an in-process revision lock, and has no accounts, retention policy, or storage quota. Run one web worker. Before public class use, add authentication, storage limits and cleanup, a shared database for multiple workers, HTTPS, and model service monitoring.

Review dependency advisories before any public deployment. The current renderer does not embed uploaded photos in slides.

The 4.1 GB Qwen base is downloaded at container startup rather than committed to Git or baked into the image. A later optimization can run OCR and LLM as persistent workers to avoid loading models for each request. The current repo is prepared for that separation through the `ocr_infer.py` and `llm_infer.py` JSON interfaces. Verify the full cloud pipeline after each deployment.

## What has been checked

The Windows development copy has completed a two-photo OCR-to-PowerPoint flow, and the photo-rotation controls were checked at desktop and phone widths. V2's three exported files match the teammate's 94-character reader by SHA-256; a local OCR run loaded it and returned text from a demo photo. These checks do not establish OCR accuracy or prove that macOS and the next cloud release work. The current automatic perspective correction lacks the camera metadata needed to trigger its strict gate, and small-tilt correction skips photos without enough independent text rows. The LoRA can repeat an explanation or underuse one photo, so review the draft before export. Recheck the complete flow after changing a model or deploying.

Official model sources: [PP-OCRv5 mobile detector](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det), [PP-OCRv5 server recognizer](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec), [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). Verify the licenses when publishing your team's repository.

"""Board2Slides demo server.

Run from the repository root: python scripts/run.py
Uploaded photos use a mobile PaddleOCR detector with the team's recognizer.
Planning uses the saved Qwen LoRA or an explicit deterministic demo mode.
"""

from __future__ import annotations

import io
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from threading import RLock
from pathlib import Path
from uuid import uuid4

import qrcode
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from llm_infer import model_paths
from lesson_lines import assemble_ocr
from ocr_infer import MODEL_NAMES, model_paths as ocr_model_paths, recognizer_path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
JOBS = PROJECT / "data" / "jobs"
STATIC = HERE / "static"
RENDERER = PROJECT / "renderer" / "adaptive_visual_renderer.js"
NODE_MODULE = PROJECT / "node_modules" / "pptxgenjs"
LLM_SCRIPT = HERE / "llm_infer.py"
OCR_SCRIPT = HERE / "ocr_infer.py"
PREPARE_SCRIPT = HERE / "prepare_infer.py"
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000
MAX_PHOTOS = 4
MAX_REGIONS = 60
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
SAMPLE_LINES = [
    "Cell division: mitosis",
    "Interphase: DNA is copied",
    "Metaphase: chromosomes align at the cell centre",
    "Anaphase: chromosomes move to opposite poles",
    "Cytokinesis: one cell divides into two daughter cells",
]
JOB_LOCK = RLock()
LOGGER = logging.getLogger(__name__)

JOBS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Board2Slides", version="0.2.0")
app.mount("/assets", StaticFiles(directory=STATIC), name="assets")


class Region(BaseModel):
    id: str = Field(min_length=1, max_length=20)
    text: str = Field(min_length=1, max_length=180)
    confidence: float | None = None
    photo_id: str | None = None


class OcrUpdate(BaseModel):
    revision: int = Field(ge=0)
    regions: list[Region] = Field(min_length=1, max_length=MAX_REGIONS)


class KeyPoint(BaseModel):
    heading: str = Field(min_length=1, max_length=70)
    explanation: str = Field(min_length=1, max_length=240)


class Example(BaseModel):
    title: str = Field(min_length=1, max_length=70)
    description: str = Field(min_length=1, max_length=280)


class PlanUpdate(BaseModel):
    revision: int = Field(ge=0)
    title: str = Field(min_length=1, max_length=90)
    summary: str = Field(min_length=1, max_length=300)
    key_points: list[KeyPoint] = Field(min_length=2, max_length=4)
    example: Example
    visual: dict
    uncertain_ocr_terms: list[str] = Field(default_factory=list, max_length=30)


class RevisionRequest(BaseModel):
    revision: int = Field(ge=0)


class PlanRequest(RevisionRequest):
    mode: str = "llm"
    regenerate: bool = False


class PlannerRegion(BaseModel):
    id: str = Field(min_length=1, max_length=20)
    text: str = Field(min_length=1, max_length=350)
    photo_id: str | None = None
    source_region_ids: list[str] = Field(min_length=1, max_length=MAX_REGIONS)


class PlannerInputUpdate(RevisionRequest):
    regions: list[PlannerRegion] = Field(min_length=1, max_length=MAX_REGIONS)


class PrepareRequest(RevisionRequest):
    mode: str = "auto"


class OcrRunRequest(RevisionRequest):
    model: str = "tuned"


def llm_python() -> Path:
    configured = os.environ.get("BOARD2SLIDES_LLM_PYTHON")
    if configured:
        return Path(configured)
    local = PROJECT / ".venv-llm" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return local if local.is_file() else Path(sys.executable)


def llm_ready() -> bool:
    base, adapter = model_paths()
    if llm_python().resolve() == Path(sys.executable).resolve() and not os.environ.get("BOARD2SLIDES_LLM_PYTHON"):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
            return False
    return (llm_python().is_file() and (base / "config.json").is_file()
            and (adapter / "adapter_model.safetensors").is_file())


def ocr_python() -> Path:
    configured = os.environ.get("BOARD2SLIDES_OCR_PYTHON")
    if configured:
        return Path(configured)
    local = PROJECT / ".venv-ocr" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return local if local.is_file() else Path(sys.executable)


def ocr_ready(model: str = "tuned") -> bool:
    if model not in MODEL_NAMES:
        return False
    _, detector, _ = ocr_model_paths()
    recognizer = recognizer_path(model)
    if ocr_python().resolve() == Path(sys.executable).resolve() and not os.environ.get("BOARD2SLIDES_OCR_PYTHON"):
        if importlib.util.find_spec("paddle") is None or importlib.util.find_spec("paddleocr") is None:
            return False
    needed = ("inference.json", "inference.pdiparams", "inference.yml")
    return (ocr_python().is_file() and all((detector / name).is_file() for name in needed)
            and all((recognizer / name).is_file() for name in needed))


def run_ocr(folder: Path, image_path: Path | None = None, model: str = "tuned") -> tuple[list[dict], list[dict], str]:
    if not ocr_ready(model):
        raise RuntimeError("OCR model files or Paddle Python are unavailable")
    image_path = image_path or folder / "preview.jpg"
    output_file = folder / f"ocr-output-{uuid4().hex}.json"
    try:
        try:
            result = subprocess.run(
                [str(ocr_python()), str(OCR_SCRIPT), str(image_path), str(output_file), "--model", model],
                cwd=PROJECT, capture_output=True, text=True, timeout=120, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("OCR took longer than two minutes") from exc
        if result.returncode or not output_file.is_file():
            raise RuntimeError((result.stderr or result.stdout or "No OCR output").strip()[-600:])
        normalized = json.loads(output_file.read_text(encoding="utf-8"))
        regions = normalized["ocr"]["regions"]
        evidence = normalized["region_evidence"]
        if len(regions) != len(evidence):
            raise RuntimeError("OCR regions and evidence lengths differ")
        notes = []
        if len(regions) > MAX_REGIONS:
            raise RuntimeError(f"This photo has more than {MAX_REGIONS} lines; use fewer or clearer photos.")
        for region in regions:
            if len(region["text"]) > 180:
                region["text"] = region["text"][:180]
                notes.append("Some long lines were shortened for review.")
        return regions, evidence, " ".join(dict.fromkeys(notes))
    finally:
        output_file.unlink(missing_ok=True)


def prepare_photo(folder: Path, photo: dict, mode: str) -> dict:
    original = folder / photo["id"] / "preview.jpg"
    processed = folder / photo["id"] / "processed.jpg"
    decision_file = folder / photo["id"] / "preparation.json"
    if mode == "original":
        shutil.copyfile(original, processed)
        with Image.open(processed) as image:
            return {"mode": "original", "applied": False,
                    "reason": "Original upright photo retained.",
                    "width": image.width, "height": image.height}
    result = subprocess.run(
        [str(ocr_python()), str(PREPARE_SCRIPT), str(original), str(processed),
         str(decision_file), "--mode", mode], cwd=PROJECT,
        capture_output=True, text=True, timeout=180, check=False,
    )
    if result.returncode or not processed.is_file() or not decision_file.is_file():
        raise RuntimeError((result.stderr or result.stdout or "Photo preparation failed").strip()[-800:])
    return json.loads(decision_file.read_text(encoding="utf-8"))


def validate_plan(plan: dict, ocr: dict) -> dict:
    try:
        clean = PlanUpdate.model_validate({"revision": 0, **plan}).model_dump(exclude={"revision"})
    except Exception as exc:
        raise HTTPException(422, f"The model returned an incomplete slide plan: {exc}") from exc
    visual = clean["visual"]
    allowed = {"none", "process", "comparison_table", "formula", "line_chart", "bar_chart"}
    if not isinstance(visual, dict) or visual.get("type") not in allowed:
        raise HTTPException(422, "The model returned an unsupported slide layout.")
    if not isinstance(visual.get("data"), dict) or not isinstance(visual.get("source_region_ids"), list):
        raise HTTPException(422, "The model returned invalid visual data.")
    if not all(isinstance(region_id, str) for region_id in visual["source_region_ids"]):
        raise HTTPException(422, "The visual cites invalid board regions.")
    ids = {region["id"] for region in ocr["regions"]}
    if not set(visual["source_region_ids"]).issubset(ids):
        raise HTTPException(422, "The visual cites board regions that do not exist.")
    return clean


def repair_visual_claims(plan: dict, ocr: dict) -> tuple[dict, list[str]]:
    """Apply the notebook's deterministic v3 chart repair after a guardrail failure."""
    from copy import deepcopy
    fixed = deepcopy(plan)
    visual = fixed["visual"]
    if visual["type"] not in {"bar_chart", "line_chart"}:
        return fixed, []
    data = visual["data"]
    try:
        labels = [str(value) for value in data["labels"]]
        values = [float(value) for value in data["series"][0]["values"]]
        if len(labels) != len(values) or len(labels) < 3:
            return fixed, []
        unit = str(data.get("unit", "")).strip()
    except (KeyError, TypeError, ValueError, IndexError):
        return fixed, []
    suffix = " " + unit if unit else ""
    maximum, minimum = max(values), min(values)
    max_label, min_label = labels[values.index(maximum)], labels[values.index(minimum)]
    fmt = lambda number: f"{number:g}"
    if visual["type"] == "line_chart":
        fixed["summary"] = (f"{visual['title']} changes from {fmt(values[0])}{suffix} at {labels[0]} "
                            f"to {fmt(values[-1])}{suffix} at {labels[-1]}.")
        fixed["key_points"] = [
            {"heading": "Highest recorded value", "explanation": f"{max_label} has the highest recorded value at {fmt(maximum)}{suffix}."},
            {"heading": "Lowest recorded value", "explanation": f"{min_label} has the lowest recorded value at {fmt(minimum)}{suffix}."},
        ]
    else:
        fixed["summary"] = f"{max_label} has the largest value at {fmt(maximum)}{suffix}."
        fixed["key_points"] = [
            {"heading": "Largest category", "explanation": f"{max_label} has the largest value at {fmt(maximum)}{suffix}."},
            {"heading": "Smallest category", "explanation": f"{min_label} has the smallest value at {fmt(minimum)}{suffix}."},
        ]
    return validate_plan(fixed, ocr), ["Chart teaching claims were recomputed from the cited chart values."]


def omit_unsupported_visual(plan: dict, ocr: dict) -> tuple[dict, list[str]]:
    """Keep the LLM prose while removing a visual rejected by evidence checks."""
    from copy import deepcopy
    fixed = deepcopy(plan)
    fixed["visual"] = {"type": "none", "title": "Explanation only",
                       "reason": "The proposed visual did not pass the board-evidence check.",
                       "source_region_ids": [], "data": {}}
    return validate_plan(fixed, ocr), ["The LLM's visual was removed because its details were not supported by the reviewed board text."]


def planner_ocr(job: dict) -> dict:
    return job.get("reviewed_planner_input") or assemble_ocr(
        job["ocr"], job.get("region_evidence", []), job.get("photos", []))


def plan_quality_notes(plan: dict, source_ocr: dict, photos: list[dict]) -> list[str]:
    explanations = [" ".join(point["explanation"].lower().split()) for point in plan["key_points"]]
    notes = []
    if len(explanations) != len(set(explanations)):
        notes.append("The draft repeats a key-point explanation. Edit it or regenerate the draft.")
    notes.extend(photo_coverage_notes(plan, source_ocr, photos))
    return notes


def photo_coverage_notes(plan: dict, source_ocr: dict, photos: list[dict]) -> list[str]:
    """Flag a photo whose distinctive words are absent from the slide draft."""
    if len(photos) < 2:
        return []
    stop = {"about", "after", "also", "and", "are", "for", "from", "has", "have",
            "into", "its", "more", "not", "that", "the", "their", "this", "type",
            "was", "were", "what", "when", "which", "with"}
    tokens = lambda value: {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", value)
                            if word.lower() not in stop}
    by_photo = {photo["id"]: tokens(" ".join(
        region["text"] for region in source_ocr["regions"] if region.get("photo_id") == photo["id"]))
                for photo in photos}
    prose = " ".join([plan["title"], plan["summary"],
                      *[point["heading"] + " " + point["explanation"] for point in plan["key_points"]],
                      plan["example"]["description"]])
    if plan["visual"]["type"] != "none":
        prose += " " + json.dumps(plan["visual"], ensure_ascii=False)
    mentioned = tokens(prose)
    notes = []
    for index, photo in enumerate(photos, 1):
        distinct = by_photo[photo["id"]] - set().union(
            *(terms for other_id, terms in by_photo.items() if other_id != photo["id"]))
        if distinct and not distinct.intersection(mentioned):
            notes.append(f"Check Photo {index}: its distinct terms do not appear in the slide draft.")
    return notes


def render_deck(folder: Path, job: dict, plan: dict, output_file: Path, demo: bool,
                source_ocr: dict | None = None) -> None:
    node = shutil.which("node")
    if not node or not NODE_MODULE.is_dir():
        raise HTTPException(503, "PowerPoint renderer is not installed. Run python scripts/setup.py.")
    envelope = {
        "generation_mode": "reviewed_demo" if demo else "adaptive_llm_v3",
        "model_condition": "Reviewed board text · demo planner" if demo else "Qwen3-1.7B · adaptive visual LoRA v3",
        "ocr": source_ocr if source_ocr is not None else job["ocr"], "model_output": plan,
    }
    input_file = folder / f"renderer-input-{uuid4().hex}.json"
    input_file.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        result = subprocess.run([node, str(RENDERER), str(input_file), str(output_file)],
                                cwd=PROJECT, capture_output=True, text=True, timeout=90, check=False)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "PowerPoint generation timed out.") from exc
    finally:
        input_file.unlink(missing_ok=True)
    if result.returncode != 0 or not output_file.is_file():
        detail = (result.stderr or result.stdout or "Unknown rendering error").strip()
        raise HTTPException(422, f"PowerPoint safety check stopped this draft: {detail[-1200:]}")


def job_dir(job_id: str) -> Path:
    if not JOB_ID.fullmatch(job_id):
        raise HTTPException(404, "Job not found")
    folder = JOBS / job_id
    if not (folder / "job.json").is_file():
        raise HTTPException(404, "Job not found")
    return folder


def read_job(job_id: str) -> dict:
    job = json.loads((job_dir(job_id) / "job.json").read_text(encoding="utf-8"))
    # Drafts made by the earlier text-only prototype had no visual field.
    if job.get("plan") and "visual" not in job["plan"]:
        job["plan"]["visual"] = {
            "type": "none", "title": "Explanation only",
            "reason": "Created with the earlier demo planner.",
            "source_region_ids": [r["id"] for r in job["ocr"]["regions"]], "data": {},
        }
        job["plan"].setdefault("uncertain_ocr_terms", [])
        job["planner_mode"] = "demo"
    return job


def save_job(folder: Path, job: dict) -> None:
    temporary = folder / "job.tmp"
    temporary.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(folder / "job.json")


def response_job(job: dict) -> dict:
    return {
        "id": job["id"],
        "revision": job.get("revision", 0),
        "mode": job["mode"],
        "planner_mode": job.get("planner_mode"),
        "image_url": job["image_url"],
        "photos": job.get("photos", []),
        "selected_model": job.get("selected_model", "tuned"),
        "ocr": job["ocr"],
        "ocr_status": job.get("ocr_status", "manual"),
        "ocr_message": job.get("ocr_message", ""),
        "region_evidence": job.get("region_evidence", []),
        "planner_input": planner_ocr(job),
        "plan": job.get("plan"),
        "plan_notes": job.get("plan_notes", []),
        "download_url": job.get("download_url"),
    }


def check_revision(job: dict, revision: int) -> None:
    if job.get("revision", 0) != revision:
        raise HTTPException(
            409,
            "This draft changed on another device. Refresh it before saving again.",
        )


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "ocr": "PP-OCRv5 mobile detector + team recognizer" if ocr_ready() else "manual input (OCR files unavailable)",
        "ocr_ready": ocr_ready(),
        "planner": "Qwen3-1.7B + adaptive visual LoRA v3" if llm_ready() else "demo only (LLM files unavailable)",
        "llm_ready": llm_ready(),
        "renderer_ready": bool(shutil.which("node") and RENDERER.is_file() and NODE_MODULE.is_dir()),
        "ocr_models": [{"id": name, "label": label, "available": ocr_ready(name)}
                       for name, label in MODEL_NAMES.items()],
    }


async def save_uploaded_photo(upload: UploadFile, folder: Path, photo_id: str) -> dict:
    raw = await upload.read(MAX_IMAGE_BYTES + 1)
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Each photo must be under 15 MB.")
    try:
        picture = Image.open(io.BytesIO(raw))
        picture.verify()
        picture = Image.open(io.BytesIO(raw))
        image_format = (picture.format or "").upper()
        if image_format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Use JPG, PNG or WebP photos.")
        if picture.width * picture.height > MAX_IMAGE_PIXELS:
            raise ValueError("Photo exceeds 30 megapixels.")
        picture = ImageOps.exif_transpose(picture).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, f"Could not read {upload.filename or 'photo'}: {exc}") from exc
    target = folder / photo_id
    target.mkdir()
    extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[image_format]
    (target / f"original.{extension}").write_bytes(raw)
    picture.thumbnail((4000, 4000))
    picture.save(target / "preview.jpg", "JPEG", quality=92, optimize=True)
    safe_name = Path(upload.filename or f"Photo {photo_id}").name[:100]
    return {"id": photo_id, "name": safe_name,
            "original_url": f"/api/jobs/{{job_id}}/images/{photo_id}/original",
            "processed_url": f"/api/jobs/{{job_id}}/images/{photo_id}/processed",
            "preparation": None, "ocr_status": "pending", "ocr_message": "",
            "line_count": 0}


@app.post("/api/jobs")
async def create_job(
    sample: bool = Form(False),
    image: UploadFile | None = File(None),
    images: list[UploadFile] | None = File(None),
):
    images = images if isinstance(images, list) else []
    if images:
        if sample or image is not None:
            raise HTTPException(400, "Choose photos or the sample lesson, not both.")
        if len(images) > MAX_PHOTOS:
            raise HTTPException(400, f"Choose at most {MAX_PHOTOS} photos for one lesson.")
        job_id = uuid4().hex
        folder = JOBS / job_id
        folder.mkdir()
        try:
            photos = []
            for index, upload in enumerate(images, 1):
                photo = await save_uploaded_photo(upload, folder, f"p{index}")
                photo["original_url"] = photo["original_url"].format(job_id=job_id)
                photo["processed_url"] = photo["processed_url"].format(job_id=job_id)
                photos.append(photo)
        except Exception:
            shutil.rmtree(folder)
            raise
        job = {
            "id": job_id, "revision": 0, "mode": "photos", "planner_mode": None,
            "photos": photos, "selected_model": "tuned",
            "image_url": photos[0]["original_url"], "ocr": {"regions": []},
            "ocr_status": "pending", "ocr_message": "Prepare the photos, then choose an OCR model.",
            "region_evidence": [], "plan": None, "plan_notes": [], "download_url": None,
        }
        save_job(folder, job)
        return response_job(job)
    if not sample and image is None:
        raise HTTPException(400, "Choose a photo or use the sample lesson.")
    if sample and image is not None:
        raise HTTPException(400, "Choose either a photo or the sample lesson.")

    job_id = uuid4().hex
    folder = JOBS / job_id
    folder.mkdir(parents=True)
    if sample:
        image_url = "/assets/sample-board.svg"
        regions = [{"id": f"r{i + 1}", "text": line, "confidence": None}
                   for i, line in enumerate(SAMPLE_LINES)]
        mode = "sample"
        ocr_status = "sample"
        ocr_message = "Sample board text is ready to review."
        region_evidence = []
    else:
        raw = await image.read(MAX_IMAGE_BYTES + 1)
        if len(raw) > MAX_IMAGE_BYTES:
            shutil.rmtree(folder)
            raise HTTPException(413, "Photo is too large. Use an image under 15 MB.")
        try:
            picture = Image.open(io.BytesIO(raw))
            picture.verify()
            picture = Image.open(io.BytesIO(raw))
            image_format = (picture.format or "").upper()
            if image_format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Use a JPG, PNG or WebP photo.")
            if picture.width * picture.height > MAX_IMAGE_PIXELS:
                raise ValueError("Photo exceeds 30 megapixels.")
            picture = ImageOps.exif_transpose(picture).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            shutil.rmtree(folder)
            raise HTTPException(400, f"Could not read this photo: {exc}") from exc
        extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[image_format]
        (folder / f"original.{extension}").write_bytes(raw)
        picture.thumbnail((2000, 2000))
        picture.save(folder / "preview.jpg", "JPEG", quality=86, optimize=True)
        image_url = f"/api/jobs/{job_id}/image"
        mode = "photo"
        regions = []
        region_evidence = []
        if ocr_ready():
            try:
                regions, region_evidence, note = await run_in_threadpool(run_ocr, folder)
                ocr_status = "auto" if regions else "empty"
                ocr_message = (f"Found {len(regions)} lines. Check every line before creating slides. " + note
                               if regions else "No text was detected. Type the board lines manually.")
            except Exception as exc:
                LOGGER.warning("OCR failed for job %s: %s", job_id, exc)
                ocr_status = "error"
                ocr_message = "Automatic reading failed. Type the board lines manually or try OCR again."
        else:
            ocr_status = "unavailable"
            ocr_message = "OCR models are unavailable on this server. Type the board lines manually."

    job = {
        "id": job_id,
        "revision": 0,
        "mode": mode,
        "planner_mode": None,
        "image_url": image_url,
        "ocr": {"regions": regions},
        "ocr_status": ocr_status,
        "ocr_message": ocr_message,
        "region_evidence": region_evidence,
        "plan": None,
        "plan_notes": [],
        "download_url": None,
    }
    save_job(folder, job)
    return response_job(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return response_job(read_job(job_id))


@app.get("/api/jobs/{job_id}/qr")
def get_job_qr(job_id: str, request: Request):
    job_dir(job_id)
    link = str(request.base_url) + "?job=" + job_id
    image = qrcode.make(link)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return Response(
        content=output.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/jobs/{job_id}/image")
def get_image(job_id: str):
    image = job_dir(job_id) / "preview.jpg"
    if not image.is_file():
        raise HTTPException(404, "Photo not found")
    return FileResponse(image, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/images/{photo_id}/{kind}")
def get_photo_image(job_id: str, photo_id: str, kind: str):
    job = read_job(job_id)
    if photo_id not in {photo["id"] for photo in job.get("photos", [])} or kind not in {"original", "processed"}:
        raise HTTPException(404, "Photo not found")
    folder = job_dir(job_id) / photo_id
    if kind == "original":
        image = folder / "preview.jpg"
    else:
        image = folder / "processed.jpg"
        if not image.is_file():
            image = folder / "preview.jpg"
    return FileResponse(image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/jobs/{job_id}/prepare")
def prepare_images(job_id: str, payload: PrepareRequest):
    if payload.mode not in {"auto", "original", "baseline", "rotation", "lighting", "perspective"}:
        raise HTTPException(400, "Choose a supported photo preparation mode.")
    with JOB_LOCK:
        job = read_job(job_id)
        check_revision(job, payload.revision)
        if job["mode"] != "photos":
            raise HTTPException(400, "Upload photos before preparing them.")
        photos = job["photos"]
    if payload.mode != "original" and not ocr_ready():
        raise HTTPException(503, "The detector and OCR environment are needed for photo preparation.")
    folder = job_dir(job_id)
    decisions = []
    for photo in photos:
        try:
            decisions.append(prepare_photo(folder, photo, payload.mode))
        except Exception as exc:
            LOGGER.warning("Preparation failed for %s: %s", photo["id"], exc)
            raise HTTPException(503, f"Could not prepare {photo['name']}: {str(exc)[-400:]}") from exc
    with JOB_LOCK:
        job = read_job(job_id)
        check_revision(job, payload.revision)
        for photo, decision in zip(job["photos"], decisions):
            photo["preparation"] = decision
            photo["ocr_status"] = "pending"
            photo["ocr_message"] = ""
            photo["line_count"] = 0
        job["ocr"] = {"regions": []}
        job["reviewed_planner_input"] = None
        job["region_evidence"] = []
        job["ocr_status"] = "pending"
        job["ocr_message"] = "Photos prepared. Choose a model and run OCR."
        job["plan"] = None
        job["plan_notes"] = []
        job["planner_mode"] = None
        job["download_url"] = None
        job["revision"] += 1
        save_job(folder, job)
    return response_job(job)


@app.put("/api/jobs/{job_id}/ocr")
def update_ocr(job_id: str, payload: OcrUpdate):
    cleaned = [region.text.strip() for region in payload.regions]
    if any(not line for line in cleaned):
        raise HTTPException(400, "Remove empty lines before continuing.")
    with JOB_LOCK:
        job = read_job(job_id)
        check_revision(job, payload.revision)
        if job["mode"] == "photos":
            photo_ids = {photo["id"] for photo in job["photos"]}
            ids = [region.id for region in payload.regions]
            if len(ids) != len(set(ids)):
                raise HTTPException(400, "Board line IDs must be unique.")
            if any(region.photo_id not in photo_ids for region in payload.regions):
                raise HTTPException(400, "Each board line must belong to an uploaded photo.")
            old_evidence = {item["id"]: item for item in job.get("region_evidence", [])}
            if any(region.id in old_evidence and old_evidence[region.id]["photo_id"] != region.photo_id
                   for region in payload.regions):
                raise HTTPException(400, "A text box cannot be moved to another photo.")
            job["ocr"] = {"regions": [
                {"id": region.id, "text": text, "photo_id": region.photo_id}
                for region, text in zip(payload.regions, cleaned)
            ]}
            job["region_evidence"] = [old_evidence[rid] for rid in ids if rid in old_evidence]
            for photo in job["photos"]:
                photo["line_count"] = sum(region.photo_id == photo["id"] for region in payload.regions)
                photo["ocr_status"] = "reviewed"
        else:
            job["ocr"] = {"regions": [
                {"id": f"r{i + 1}", "text": text, "confidence": None}
                for i, text in enumerate(cleaned)
            ]}
            job["region_evidence"] = []
        job["ocr_status"] = "reviewed"
        job["ocr_message"] = "Board text was corrected and saved."
        job["reviewed_planner_input"] = None
        job["plan"] = None
        job["plan_notes"] = []
        job["planner_mode"] = None
        job["download_url"] = None
        job["revision"] = payload.revision + 1
        save_job(job_dir(job_id), job)
    return response_job(job)


@app.post("/api/jobs/{job_id}/ocr/run")
def rerun_ocr(job_id: str, payload: OcrRunRequest):
    model = getattr(payload, "model", "tuned")
    if model not in MODEL_NAMES:
        raise HTTPException(400, "Choose a supported OCR model.")
    with JOB_LOCK:
        job = read_job(job_id)
        check_revision(job, payload.revision)
        folder = job_dir(job_id)
        if job["mode"] == "photos":
            if any(photo.get("preparation") is None for photo in job["photos"]):
                raise HTTPException(400, "Prepare every photo before running OCR.")
            photos = job["photos"]
        else:
            photos = None
    if photos is not None:
        if not ocr_ready(model):
            raise HTTPException(503, "The selected OCR model is not installed on this server.")
        all_regions, all_evidence, outcomes = [], [], []
        for photo in photos:
            try:
                regions, evidence, note = run_ocr(
                    folder / photo["id"], folder / photo["id"] / "processed.jpg", model)
                if len(all_regions) + len(regions) > MAX_REGIONS:
                    raise HTTPException(400, f"One deck supports at most {MAX_REGIONS} review lines. Use fewer photos.")
                for index, (region, item) in enumerate(zip(regions, evidence), 1):
                    rid = f"{photo['id']}r{index}"
                    region["id"] = rid
                    region["photo_id"] = photo["id"]
                    item["id"] = rid
                    item["photo_id"] = photo["id"]
                    item["image_width"] = photo["preparation"]["width"]
                    item["image_height"] = photo["preparation"]["height"]
                all_regions.extend(regions)
                all_evidence.extend(evidence)
                outcomes.append(("auto" if regions else "empty", len(regions), note))
            except HTTPException:
                raise
            except Exception as exc:
                LOGGER.warning("OCR failed for photo %s: %s", photo["id"], exc)
                outcomes.append(("error", 0, "OCR failed for this photo; add its text manually."))
        with JOB_LOCK:
            job = read_job(job_id)
            check_revision(job, payload.revision)
            job["ocr"] = {"regions": all_regions}
            job["reviewed_planner_input"] = None
            job["region_evidence"] = all_evidence
            job["selected_model"] = model
            for photo, (status, count, note) in zip(job["photos"], outcomes):
                photo["ocr_status"] = status
                photo["ocr_message"] = note
                photo["line_count"] = count
            failures = sum(status == "error" for status, _, _ in outcomes)
            job["ocr_status"] = "auto" if all_regions and not failures else "partial" if all_regions else "error" if failures else "empty"
            job["ocr_message"] = (f"Found {len(all_regions)} lines across {len(photos)} photo(s). "
                                  "Review each photo before creating slides."
                                  + (f" {failures} photo(s) need manual text." if failures else ""))
            job["plan"] = None
            job["plan_notes"] = []
            job["planner_mode"] = None
            job["download_url"] = None
            job["revision"] += 1
            save_job(folder, job)
        return response_job(job)
    with JOB_LOCK:
        job = read_job(job_id)
        check_revision(job, payload.revision)
        if job["mode"] == "sample" or not (folder / "preview.jpg").is_file():
            raise HTTPException(400, "Upload a board photo to run OCR.")
    if not ocr_ready(model):
        raise HTTPException(503, "OCR models are unavailable on this server.")
    try:
        regions, evidence, note = run_ocr(folder, model=model)
    except Exception as exc:
        LOGGER.warning("OCR retry failed for job %s: %s", job_id, exc)
        raise HTTPException(503, "OCR failed. Your saved board text was kept; try again or edit it manually.") from exc
    with JOB_LOCK:
        job = read_job(job_id)
        check_revision(job, payload.revision)
        job["ocr"] = {"regions": regions}
        job["reviewed_planner_input"] = None
        job["selected_model"] = model
        job["region_evidence"] = evidence
        job["ocr_status"] = "auto" if regions else "empty"
        job["ocr_message"] = (f"Found {len(regions)} lines. Check every line before creating slides. " + note
                              if regions else "No text was detected. Type the board lines manually.")
        job["plan"] = None
        job["plan_notes"] = []
        job["planner_mode"] = None
        job["download_url"] = None
        job["revision"] = payload.revision + 1
        save_job(folder, job)
    return response_job(job)


@app.put("/api/jobs/{job_id}/planner-input")
def update_planner_input(job_id: str, payload: PlannerInputUpdate):
    with JOB_LOCK:
        job = read_job(job_id)
        check_revision(job, payload.revision)
        source = {region["id"]: region for region in job["ocr"]["regions"]}
        if not source:
            raise HTTPException(400, "Save the board text before editing joined text.")
        seen: list[str] = []
        groups = []
        photo_order = {photo["id"]: index for index, photo in enumerate(job.get("photos", []))}
        last_photo = -1
        for region in payload.regions:
            ids = region.source_region_ids
            if region.id != ids[0] or any(rid not in source for rid in ids):
                raise HTTPException(400, "Each joined line must use its first real board box as its ID.")
            if any(source[rid].get("photo_id") != region.photo_id for rid in ids):
                raise HTTPException(400, "Joined lines cannot combine different photos.")
            if photo_order:
                index = photo_order.get(region.photo_id, -1)
                if index < last_photo or index < 0:
                    raise HTTPException(400, "Keep the joined lines in photo order.")
                last_photo = index
            seen.extend(ids)
            groups.append({"id": region.id, "text": region.text.strip(),
                           "photo_id": region.photo_id, "source_region_ids": ids})
        if len(seen) != len(source) or set(seen) != set(source):
            raise HTTPException(400, "Every saved board box must appear exactly once in the joined text.")
        if any(not region["text"] for region in groups):
            raise HTTPException(400, "Joined lines cannot be empty.")
        job["reviewed_planner_input"] = {"regions": groups}
        job["plan"] = None
        job["plan_notes"] = []
        job["plan_system_notes"] = []
        job["planner_mode"] = None
        job["download_url"] = None
        job["revision"] += 1
        save_job(job_dir(job_id), job)
    return response_job(job)


@app.post("/api/jobs/{job_id}/plan")
def create_plan(job_id: str, payload: PlanRequest):
    if payload.mode not in {"llm", "demo"}:
        raise HTTPException(400, "Choose the LLM or demo planner.")
    if payload.regenerate and payload.mode != "llm":
        raise HTTPException(400, "Only an LLM draft can be regenerated.")
    with JOB_LOCK:
        job = read_job(job_id)
        check_revision(job, payload.revision)
        regions = job["ocr"]["regions"]
        if len(regions) < 2:
            raise HTTPException(400, "Add at least two board lines to create slides.")
        if job["mode"] == "photos":
            covered = {region.get("photo_id") for region in regions}
            missing = [photo["name"] for photo in job["photos"] if photo["id"] not in covered]
            if missing:
                raise HTTPException(400, "Add at least one reviewed line for each photo. Missing: " + ", ".join(missing))
        ocr = planner_ocr(job) if payload.mode == "llm" else job["ocr"]
        if payload.regenerate and (not job.get("plan") or job.get("planner_mode") != "llm"):
            raise HTTPException(400, "Create an LLM draft before regenerating it.")
    if payload.mode == "llm":
        if not llm_ready():
            raise HTTPException(503, "The saved LLM is unavailable here. Install the model environment or select Demo planner.")
        folder = job_dir(job_id)
        input_file = folder / f"llm-input-{uuid4().hex}.json"
        output_file = folder / f"llm-output-{uuid4().hex}.json"
        llm_input = dict(ocr)
        if payload.regenerate:
            feedback = job.get("plan_notes", [])
            old_points = [point["heading"] + ": " + point["explanation"]
                          for point in job["plan"].get("key_points", [])]
            llm_input["revision_feedback"] = (
                "Write a new draft with distinct key points. Cover every photo and keep claims "
                "grounded in the reviewed board text. Avoid repeating these previous points: "
                + " | ".join(old_points)[:350] + ". Review flags: "
                + " ".join(feedback)[:250])
        input_file.write_text(json.dumps(llm_input, ensure_ascii=False), encoding="utf-8")
        try:
            result = subprocess.run([str(llm_python()), str(LLM_SCRIPT), str(input_file), str(output_file)],
                                    cwd=PROJECT, capture_output=True, text=True, timeout=240, check=False)
            if result.returncode or not output_file.is_file():
                detail = (result.stderr or result.stdout or "No model output").strip()[-900:]
                if "JSONDecodeError" in detail:
                    raise HTTPException(422, "The LLM returned incomplete JSON. Try again or select Demo planner.")
                raise HTTPException(503, f"LLM generation failed: {detail}")
            raw_plan = json.loads(output_file.read_text(encoding="utf-8"))
            format_repaired = bool(raw_plan.pop("_format_repaired", False))
            plan = validate_plan(raw_plan, ocr)
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(504, "LLM generation timed out. Try again or select Demo planner.") from exc
        except json.JSONDecodeError as exc:
            raise HTTPException(422, "The model did not return valid slide-plan JSON. Try again or use Demo planner.") from exc
        finally:
            input_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)
        notes = (["The LLM's JSON formatting was repaired. Check the draft before sharing it."]
                 if format_repaired else [])
        preview_file = folder / f"plan-check-{uuid4().hex}.pptx"
        try:
            try:
                render_deck(folder, job, plan, preview_file, demo=False, source_ocr=ocr)
            except HTTPException as first_error:
                if first_error.status_code != 422:
                    raise
                if plan["visual"]["type"] == "none":
                    raise first_error
                repaired_plan, repair_notes = repair_visual_claims(plan, ocr)
                if repair_notes:
                    try:
                        render_deck(folder, job, repaired_plan, preview_file,
                                    demo=False, source_ocr=ocr)
                        plan = repaired_plan
                        notes.extend(repair_notes)
                    except HTTPException as retry_error:
                        if retry_error.status_code != 422:
                            raise
                        plan, visual_notes = omit_unsupported_visual(plan, ocr)
                        render_deck(folder, job, plan, preview_file,
                                    demo=False, source_ocr=ocr)
                        notes.extend(visual_notes)
                else:
                    plan, visual_notes = omit_unsupported_visual(plan, ocr)
                    render_deck(folder, job, plan, preview_file,
                                demo=False, source_ocr=ocr)
                    notes.extend(visual_notes)
        finally:
            preview_file.unlink(missing_ok=True)
        system_notes = notes[:]
        notes.extend(plan_quality_notes(plan, ocr, job.get("photos", [])))
        if len(regions) <= 3 and sum(len(region["text"]) for region in regions) < 120:
            notes.append("These board notes are short. Check that the slides add useful, supported explanations.")
    else:
        lines = [region["text"] for region in regions]
        title = lines[0].split(":", 1)[0].strip()[:90]
        selected = lines[1:5] if len(lines) >= 3 else lines[:2]
        key_points = []
        for i, line in enumerate(selected):
            parts = line.split(":", 1)
            heading = parts[0].strip() if len(parts) > 1 else f"Board point {i + 1}"
            key_points.append({"heading": heading[:70], "explanation": line[:240]})
        plan = {
            "title": title,
            "summary": "Reviewed notes from the board: " + "; ".join(lines[1:3])[:240],
            "key_points": key_points,
            "example": {
                "title": "Check your understanding",
                "description": "Explain this board note in your own words: " + selected[0][:210],
            },
            "uncertain_ocr_terms": [],
            "visual": {"type": "none", "title": "Explanation only",
                       "reason": "Demo planner uses reviewed board text only.",
                       "source_region_ids": [r["id"] for r in regions], "data": {}},
        }
    with JOB_LOCK:
        job = read_job(job_id)
        check_revision(job, payload.revision)
        job["plan"] = plan
        job["plan_notes"] = notes if payload.mode == "llm" else []
        job["plan_system_notes"] = system_notes if payload.mode == "llm" else []
        job["planner_mode"] = payload.mode
        job["download_url"] = None
        job["revision"] = payload.revision + 1
        save_job(job_dir(job_id), job)
    return response_job(job)


@app.put("/api/jobs/{job_id}/plan")
def update_plan(job_id: str, payload: PlanUpdate):
    with JOB_LOCK:
        job = read_job(job_id)
        check_revision(job, payload.revision)
        if not job["ocr"]["regions"]:
            raise HTTPException(400, "Review the board text first.")
        source_ocr = planner_ocr(job) if job.get("planner_mode") == "llm" else job["ocr"]
        job["plan"] = validate_plan(payload.model_dump(exclude={"revision"}), source_ocr)
        job["plan_notes"] = (job.get("plan_system_notes", []) +
                             plan_quality_notes(job["plan"], source_ocr, job.get("photos", []))
                             if job.get("planner_mode") == "llm" else [])
        job["download_url"] = None
        job["revision"] = payload.revision + 1
        save_job(job_dir(job_id), job)
    return response_job(job)


@app.post("/api/jobs/{job_id}/export")
def export_pptx(job_id: str, payload: RevisionRequest):
    if not shutil.which("node") or not NODE_MODULE.is_dir():
        raise HTTPException(503, "PowerPoint renderer is not installed. Run python scripts/setup.py.")
    with JOB_LOCK:
        job = read_job(job_id)
        check_revision(job, payload.revision)
        if not job.get("plan"):
            raise HTTPException(400, "Create a slide plan first.")
        folder = job_dir(job_id)
        source_ocr = planner_ocr(job) if job.get("planner_mode") == "llm" else job["ocr"]
        model_output = validate_plan(job["plan"], source_ocr)
        demo = job.get("planner_mode", "demo") == "demo"
        output_file = folder / "study-slides.pptx"
        output_file.unlink(missing_ok=True)
        render_deck(folder, job, model_output, output_file, demo, source_ocr=source_ocr)
        job["download_url"] = f"/api/jobs/{job_id}/download"
        job["revision"] = payload.revision + 1
        save_job(folder, job)
        return response_job(job)


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    path = job_dir(job_id) / "study-slides.pptx"
    if not path.is_file():
        raise HTTPException(404, "Export the deck first.")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename="board2slides-study-deck.pptx",
    )


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    shutil.rmtree(job_dir(job_id))
    return {"deleted": True}

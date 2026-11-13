"""Run the official PP-OCRv5 mobile detector with our exported recogniser.

Usage: python ocr_infer.py IMAGE OUTPUT_JSON
The model runs in the separate Paddle environment; the web server reads only
the normalized JSON output. Paths can be supplied through BOARD2SLIDES_OCR_*.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CACHE = PROJECT / ".cache" / "paddlex"
DEFAULT_RECOGNIZER = PROJECT / "models" / "ocr" / "recognizer"
MODEL_NAMES = {
    "baseline": "V1 · Official PP-OCRv5 mobile baseline",
    "tuned": "V2 · Team fine-tuned recognizer",
    "augmented": "V3 · Fine-tuned with augmented data",
}


def model_paths() -> tuple[Path, Path, Path]:
    cache = Path(os.environ.get("BOARD2SLIDES_OCR_CACHE_DIR", str(DEFAULT_CACHE)))
    detector = Path(os.environ.get("BOARD2SLIDES_OCR_DET_DIR",
                                   str(PROJECT / "models" / "ocr" / "detector")))
    recognizer = Path(os.environ.get("BOARD2SLIDES_OCR_REC_DIR", str(DEFAULT_RECOGNIZER)))
    return cache, detector, recognizer


def recognizer_path(model: str) -> Path:
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown OCR model: {model}")
    if model == "tuned":
        return model_paths()[2]
    return PROJECT / "models" / "ocr" / ("baseline_rec" if model == "baseline" else "augmented_rec")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", choices=MODEL_NAMES, default="tuned")
    args = parser.parse_args()
    cache, detector, _ = model_paths()
    recognizer = recognizer_path(args.model)
    if not (detector / "inference.pdiparams").is_file():
        raise FileNotFoundError(f"Mobile detector missing: {detector}")
    if not (recognizer / "inference.pdiparams").is_file():
        raise FileNotFoundError(f"Selected recognizer missing: {recognizer}")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache)
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    from paddleocr import PaddleOCR

    from ocr_mapping import normalize

    device = os.environ.get("BOARD2SLIDES_OCR_DEVICE", "cpu")
    pipeline = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_detection_model_dir=str(detector),
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        text_recognition_model_dir=str(recognizer),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=1600,
        text_det_limit_type="max",
        device=device,
        # Paddle 3.3's Windows CPU oneDNN path raises a PIR attribute error.
        enable_mkldnn=False if device == "cpu" else True,
    )
    result = list(pipeline.predict(str(args.image)))
    if len(result) != 1:
        raise RuntimeError(f"Expected one OCR result, got {len(result)}")
    record = result[0].json["res"]
    try:
        normalized = normalize(record)
    except ValueError as exc:
        if "No nonblank recognized text" not in str(exc):
            raise
        normalized = {"ocr": {"regions": []}, "region_evidence": [],
                      "source_type": "paddleocr_prediction", "source_image": str(args.image)}
    normalized["models"] = {
        "detector": "PP-OCRv5_mobile_det (official, unfine-tuned)",
        "recognizer": MODEL_NAMES[args.model],
        "model_id": args.model,
        "device": device,
    }
    args.output.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

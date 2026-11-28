"""Prepare one upright board photo using the team's gated notebook operations.

The prepared image is the coordinate space for subsequent OCR polygons.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output_image", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--mode", choices=("auto", "original", "baseline", "rotation", "lighting", "perspective"),
                        default="auto")
    args = parser.parse_args()
    import cv2

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError("Could not open the photo for preparation")
    log: dict = {"mode": args.mode, "applied": False, "reason": "Original upright photo retained."}
    prepared = image
    if args.mode != "original":
        from preprocessing.pipeline import Steps, preprocess_auto, preprocess_isolated
        from preprocessing.text_lines import Detector

        root = Path("/home/claude/b2s")
        model_dir = Path(os.environ.get("BOARD2SLIDES_OCR_DET_DIR",
                                        str(root / "models/ocr/detector")))
        detector = Detector(model_dir=model_dir)
        result = (preprocess_auto(image, detector, Steps(auto=True, unclip=1.2), f35=float(os.environ["B2N_F35"]))
                  if args.mode == "auto" else
                  preprocess_isolated(image, detector, Steps(isolated=args.mode, unclip=1.2)))
        prepared = result["read_image"]
        log = result["log"]
        log["mode"] = args.mode
        if args.mode == "baseline" and prepared.shape != image.shape:
            log.update(applied=True, reason="Cropped around detected writing; no other correction applied.")
    if prepared.size == 0:
        raise ValueError("Preparation produced an empty image")
    ok, encoded = cv2.imencode(".jpg", prepared, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError("Could not encode prepared photo")
    args.output_image.write_bytes(encoded.tobytes())
    log["width"] = int(prepared.shape[1])
    log["height"] = int(prepared.shape[0])
    args.output_json.write_text(json.dumps(log, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

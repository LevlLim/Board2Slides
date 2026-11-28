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
    parser.add_argument("--turn", type=int, choices=(0, 90, 180, 270), default=0,
                        help="Manual clockwise orientation before preparation")
    args = parser.parse_args()
    import cv2

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError("Could not open the photo for preparation")
    if args.turn:
        image = cv2.rotate(image, {90: cv2.ROTATE_90_CLOCKWISE,
                                   180: cv2.ROTATE_180,
                                   270: cv2.ROTATE_90_COUNTERCLOCKWISE}[args.turn])
    log: dict = {"mode": args.mode, "applied": False, "reason": "Original upright photo retained."}
    prepared = image
    if args.mode != "original":
        from preprocessing.pipeline import Steps, preprocess_auto, preprocess_isolated
        from preprocessing.text_lines import Detector

        root = Path(__file__).resolve().parent.parent
        model_dir = Path(os.environ.get("BOARD2SLIDES_OCR_DET_DIR",
                                        str(root / "models/ocr/detector")))
        detector = Detector(model_dir=model_dir)
        result = (preprocess_auto(image, detector, Steps(auto=True, unclip=1.2))
                  if args.mode == "auto" else
                  preprocess_isolated(image, detector, Steps(isolated=args.mode, unclip=1.2)))
        prepared = result["read_image"]
        log = result["log"]
        log["mode"] = args.mode
        if args.mode == "baseline" and prepared.shape != image.shape:
            log.update(applied=True, reason="Cropped around detected writing; no other correction applied.")
    log["turn"] = args.turn
    if args.turn:
        turn_reason = {90: "Turned right 90°.", 180: "Turned 180°.",
                       270: "Turned left 90°."}[args.turn]
        log["applied"] = True
        log["reason"] = turn_reason + " " + log.get("reason", "").strip()
        if isinstance(log.get("decisions"), list):
            log["decisions"].insert(0, {"name": "orientation", "applied": True,
                                        "reason": turn_reason})
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

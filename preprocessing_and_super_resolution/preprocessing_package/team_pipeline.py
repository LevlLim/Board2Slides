"""Small teammate-facing interface; the research notebook remains separate.

One explicitly selected preprocessing mode is allowed per run. Default:
audited isolated baseline, legacy grouping, native line crops, no SR/blemish.
Detection and recognition models are created once per TeamPipeline instance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import board_geometry as geometry, line_grouping, pipeline, text_lines

MODES = ('baseline', 'rotation', 'enlargement', 'perspective', 'lighting')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.tif', '.tiff', '.bmp', '.webp'}


@dataclass(frozen=True)
class Settings:
    detector_name: str = 'PP-OCRv5_mobile_det'
    recognizer_name: str = 'PP-OCRv6_medium_rec'
    recognizer_dir: str | None = None
    preprocess: str = 'baseline'
    directional_grouping: bool = False
    unclip: float = 1.2

    def validate(self):
        if self.preprocess not in MODES:
            raise ValueError(f'preprocess must be one of {MODES}')
        if not isinstance(self.directional_grouping, bool):
            raise ValueError('directional_grouping must be True or False')
        if not np.isfinite(self.unclip) or self.unclip <= 0:
            raise ValueError('unclip must be positive and finite')
        if self.recognizer_dir is not None:
            folder = Path(self.recognizer_dir).expanduser()
            if not folder.is_dir():
                raise ValueError('recognizer_dir must point to the exported inference MODEL FOLDER')
            required = ['inference.pdiparams', 'inference.yml']
            if not all((folder/name).is_file() for name in required) or not any(
                (folder/name).is_file() for name in ('inference.json', 'inference.pdmodel')):
                raise ValueError('Model folder needs inference.pdiparams, inference.yml and inference.json (or inference.pdmodel)')


def resolve_inputs(entries):
    """Resolve explicit files or directories, with stable ordering and no duplicates."""
    if isinstance(entries, (str, Path)):
        entries = [entries]
    paths = []
    for entry in entries:
        path = Path(entry).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f'Photo/folder does not exist: {path}')
        candidates = sorted(path.iterdir()) if path.is_dir() else [path]
        selected = [p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        if not selected:
            raise ValueError(f'No supported photos in: {path}')
        paths.extend(selected)
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise ValueError('Add a photo path or folder to PHOTO_PATHS in the settings cell first.')
    return unique


def load_photo(path):
    """Native pixels with EXIF orientation applied; no preview downscaling."""
    path = Path(path)
    if path.suffix.lower() in {'.heic', '.heif'}:
        try:
            import pillow_heif
        except ImportError as exc:
            raise ImportError('Install pillow-heif to open HEIC/HEIF photos.') from exc
        pillow_heif.register_heif_opener()
    with Image.open(path) as image:
        focal35 = image.getexif().get_ifd(0x8769).get(0xA405)
        rgb = np.array(ImageOps.exif_transpose(image).convert('RGB'))
    focal35 = float(focal35) if focal35 else None
    if focal35 is not None and (not np.isfinite(focal35) or focal35 <= 0):
        focal35 = None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), focal35


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class _RecordingDetector:
    def __init__(self, detector):
        self.detector = detector
        self.last_boxes = []

    def detect(self, *args, **kwargs):
        boxes, info = self.detector.detect(*args, **kwargs)
        self.last_boxes = boxes
        return boxes, info


def _write_image(path, image):
    # imencode + Path works with non-ASCII paths on platforms where imwrite may not.
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f'Could not encode {path.name}')
    path.write_bytes(encoded.tobytes())


class TeamPipeline:
    def __init__(self, settings=None, *, detector=None, reader=None):
        self.settings = settings or Settings()
        self.settings.validate()
        # Avoid the macOS mixed PyTorch/Paddle import-order issue when Torch
        # happens to be installed. Torch is not a dependency of this notebook.
        if detector is None or reader is None:
            import importlib.util
            if importlib.util.find_spec('torch') is not None:
                import torch
        self.detector = detector if detector is not None else text_lines.Detector(
            model_name=self.settings.detector_name)
        self.reader = reader if reader is not None else text_lines.Reader(
            model_name=self.settings.recognizer_name,
            model_dir=str(Path(self.settings.recognizer_dir).expanduser().resolve())
            if self.settings.recognizer_dir is not None else None)

    def process_photo(self, photo, output_dir):
        """Save text, JSON, processed/annotated images and the actual OCR crops.

        Exported boxes refer to processed.png, NOT the original photograph.
        Every recognized line is retained; confidence does not certify accuracy.
        """
        started = time.perf_counter()
        source = Path(photo).expanduser().resolve()
        image, focal35 = load_photo(source)
        detector = _RecordingDetector(self.detector)
        result = pipeline.preprocess_photo(image, detector, pipeline.Steps(
            isolated=self.settings.preprocess, unclip=self.settings.unclip,
            safety_net=False), f35=focal35)
        # Reuse the final detection boxes. This makes grouping explicit without
        # mutating pipeline globals or repeating neural-network detection.
        group = line_grouping.group_into_lines if self.settings.directional_grouping else text_lines.group_into_lines
        lines, _ = group(detector.last_boxes)
        result['lines'] = [lines[i] for i in text_lines.reading_order(lines)]
        if self.settings.preprocess == 'enlargement':
            small = sum(text_lines.line_height(p) < text_lines.REC_MIN_H for p in result['lines'])
            result['log'].update(applied=bool(small), small_lines=int(small),
                                 reason=f'bicubic enlargement requested for {small} small line crops')
        result['log']['lines'] = len(result['lines'])
        crops = pipeline.line_crops(result)
        readings = self.reader.read([c['crop'] for c in crops]) if crops else []
        if len(readings) != len(crops):
            raise RuntimeError('Reader returned a different number of results than line crops')

        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r'[^A-Za-z0-9_-]+', '_', source.stem)[:70] or 'photo'
        name = f'{stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}'
        target = output/name
        temp = Path(tempfile.mkdtemp(prefix='.writing_', dir=output))
        try:
            (temp/'crops').mkdir()
            annotated = result['read_image'].copy()
            rows = []
            for i, (crop, (text, confidence)) in enumerate(zip(crops, readings), 1):
                crop_file = f'crops/line_{i:03d}.png'
                _write_image(temp/crop_file, crop['crop'])
                q = np.asarray(crop['box'], np.float32)
                cv2.polylines(annotated, [np.int32(q)], True, (0, 190, 0), 2)
                anchor = tuple(np.maximum(q.min(0).astype(int), [0, 18]))
                cv2.putText(annotated, str(i), anchor, cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 220), 2)
                rows.append(dict(line_id=i, text=str(text), confidence=float(confidence),
                    box=q.tolist(), crop=crop_file, source_box_height=float(crop['height']),
                    reader_crop_shape=list(crop['crop'].shape)))
            text = '\n'.join(row['text'] for row in rows)
            versions = {}
            for package in ('paddlepaddle','paddlepaddle-gpu','paddleocr','paddlex','numpy','Pillow'):
                try: versions[package] = version(package)
                except PackageNotFoundError: pass
            data = _json_safe(dict(schema='board2notes.ocr.v1', source=str(source),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                settings=asdict(self.settings), versions=versions,
                original_upright_shape=list(image.shape), processed_shape=list(result['read_image'].shape),
                coordinates='Line boxes are in processed.png pixels; not original photo coordinates.',
                preprocessing=result['log'], line_count=len(rows), text=text, lines=rows,
                seconds=round(time.perf_counter()-started,3)))
            _write_image(temp/'processed.png', result['read_image'])
            _write_image(temp/'annotated.jpg', annotated)
            (temp/'ocr.txt').write_text(text, encoding='utf-8')
            (temp/'ocr.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            temp.rename(target)
        except Exception:
            shutil.rmtree(temp)
            raise
        return dict(output_dir=str(target), text_path=str(target/'ocr.txt'),
                    json_path=str(target/'ocr.json'), annotated_path=str(target/'annotated.jpg'),
                    text=text, line_count=len(rows), decision=data['preprocessing'], seconds=data['seconds'])
